"""Full REST API expected by the Flutter app + admin/elder auth + ingest + escalation."""

from __future__ import annotations

import json
import logging
import os
import uuid

import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, WebSocket
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel, Field

from flask_backend.app.auth_jwt import create_token, decode_token, hash_password, verify_password
from flask_backend.app.database import get_connection, init_schema, iso_now, seed_default_admin
from flask_backend.app.elder_credential_allocator import (
    ElderUsernameAllocationFailed,
    elder_name_slug,
    pick_unique_elder_username_for_patient,
    temporary_password_for_patient,
)
from flask_backend.app.detector_state import build_detection_payload
from flask_backend.app.ml_bridge import (
    VoteBuffer,
    acc_gyro_ori_to_window_lists,
    build_enhanced_features_numpy,
    samples_to_feature_vector,
)
from flask_backend.app.schemas_fall_feedback import FallFeedbackAck, FallFeedbackEvent
from flask_backend.app.schemas_motion import MotionInferenceRequest, MotionInferenceResponse
from flask_backend.app.services.motion_xgb_service import InferenceArtifacts, run_inference

logger = logging.getLogger(__name__)

RESPONSE_DEADLINE_SEC = int(os.environ.get("FALL_RESPONSE_DEADLINE_SEC", "30"))
EMERGENCY_DEADLINE_SEC = int(os.environ.get("FALL_EMERGENCY_DEADLINE_SEC", "90"))

DETECTOR_CFG = {
    "medium_risk_score": 0.35,
    "high_risk_score": 0.58,
    "fall_score": 0.92,
}
DEBUG_SENSOR_LOGS = os.environ.get("EMS_DEBUG_SENSOR_LOGS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# MobiAct fall-type codes (when optional fall-type artifacts are configured).
_FALL_CODE_TO_NAME: dict[str, str] = {
    "FOL": "Forward lying fall",
    "FKL": "Front knees lying fall",
    "BSC": "Backward sitting-chair fall",
    "SDL": "Sideward lying fall",
}

# MobiAct ADL codebook used by training data loaders.
_ADL_CODE_TO_NAME: dict[str, str] = {
    "STD": "Standing",
    "WAL": "Walking",
    "JOG": "Jogging",
    "JUM": "Jumping",
    "STU": "Stairs Up",
    "STN": "Stairs Down",
    "SCH": "Sit to Stand",
    "SIT": "Sitting",
    "CHU": "Stand to Sit",
    "CSI": "Car Step In",
    "CSO": "Car Step Out",
    "LYI": "Lying",
}

# Legacy/derived ADL encoder path can emit numeric IDs for the original codebook.
# These indices map back to code tokens, then to user-facing names above.
_ADL_INDEX_TO_CODE: dict[int, str] = {
    0: "CHU",
    1: "CSI",
    2: "CSO",
    4: "JOG",
    5: "JUM",
    6: "LYI",
    7: "SCH",
    8: "SIT",
    9: "STD",
    10: "STN",
    11: "STU",
    12: "WAL",
}


def _humanize_fall_type_label(raw_label: Any) -> str | None:
    if raw_label is None:
        return None
    s = str(raw_label).strip()
    if not s:
        return None
    upper = s.upper()
    if upper in _FALL_CODE_TO_NAME:
        return _FALL_CODE_TO_NAME[upper]
    return s


def _humanize_activity_label(raw_label: Any) -> str | None:
    if raw_label is None:
        return None
    s = str(raw_label).strip()
    if not s:
        return None

    upper = s.upper()
    if upper in _ADL_CODE_TO_NAME:
        return _ADL_CODE_TO_NAME[upper]

    try:
        idx = int(s)
    except ValueError:
        return s

    code = _ADL_INDEX_TO_CODE.get(idx)
    if code is None:
        return s
    return _ADL_CODE_TO_NAME.get(code, code)


def _preview_vector(values: list[float], limit: int = 8) -> str:
    head = values[:limit]
    return ", ".join(f"{v:.4f}" for v in head)


def _preview_rows(rows: list[list[float]], limit: int = 2) -> str:
    if not rows:
        return "[]"
    picked = rows[:limit]
    chunks = []
    for r in picked:
        chunks.append("[" + ", ".join(f"{v:.4f}" for v in r[:3]) + "]")
    return "[" + ", ".join(chunks) + (" ...]" if len(rows) > limit else "]")


# ── Sensor diagnostic log ─────────────────────────────────────────────────────
_SENSOR_DIAG_LOG = os.path.join(
    os.environ.get("EMS_DIAG_DIR", os.path.join(os.path.dirname(__file__), "..", "logs")),
    "sensor_diag.jsonl",
)
os.makedirs(os.path.dirname(_SENSOR_DIAG_LOG), exist_ok=True)

_SEP = "=" * 70
_DIAG_COUNTER = 0   # batch sequence number (module-level, not thread-safe but fine for diag)


def _diag_sensor_batch(samples: list[dict[str, Any]], patient_id: str) -> None:
    """
    Print a clearly visible sensor diagnostic banner to the terminal and append
    a JSON record to sensor_diag.jsonl for every ingest batch.

    Checks:
      • Accelerometer  — mean/min/max per axis + magnitude mean (m/s²)
      • Gyroscope      — mean/min/max per axis + magnitude mean (rad/s)
      • Orientation    — presence flag + mean azimuth/pitch/roll (degrees)
    """
    global _DIAG_COUNTER
    _DIAG_COUNTER += 1
    seq = _DIAG_COUNTER
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = len(samples)
    if n == 0:
        return

    acc_x  = [float(s.get("acc_x",  0.0)) for s in samples]
    acc_y  = [float(s.get("acc_y",  0.0)) for s in samples]
    acc_z  = [float(s.get("acc_z",  0.0)) for s in samples]
    gyro_x = [float(s.get("gyro_x", 0.0)) for s in samples]
    gyro_y = [float(s.get("gyro_y", 0.0)) for s in samples]
    gyro_z = [float(s.get("gyro_z", 0.0)) for s in samples]

    ori_samples  = [s for s in samples if s.get("azimuth") is not None]
    ori_present  = len(ori_samples)
    ori_coverage = ori_present / n * 100.0
    az_vals  = [float(s["azimuth"]) for s in ori_samples] if ori_samples else []
    pit_vals = [float(s.get("pitch", 0.0)) for s in ori_samples] if ori_samples else []
    rol_vals = [float(s.get("roll",  0.0)) for s in ori_samples] if ori_samples else []

    acc_mag  = [float(np.sqrt(x**2 + y**2 + z**2)) for x, y, z in zip(acc_x, acc_y, acc_z)]
    gyro_mag = [float(np.sqrt(x**2 + y**2 + z**2)) for x, y, z in zip(gyro_x, gyro_y, gyro_z)]

    def _s(vals: list[float]) -> str:
        if not vals:
            return "N/A"
        return f"mean={np.mean(vals):+.3f}  min={min(vals):+.3f}  max={max(vals):+.3f}"

    ori_status = (
        f"PRESENT  ({ori_present}/{n} samples = {ori_coverage:.0f}%)"
        if ori_present > 0
        else "MISSING  *** no orientation data in this batch ***"
    )

    # ── terminal banner ────────────────────────────────────────────────────────
    banner = (
        f"\n{_SEP}\n"
        f"  SENSOR DIAGNOSTIC  |  batch #{seq:04d}  |  {ts}  |  patient={patient_id}\n"
        f"  samples={n}\n"
        f"{_SEP}\n"
        f"  ACCELEROMETER  (m/s²)\n"
        f"    acc_x : {_s(acc_x)}\n"
        f"    acc_y : {_s(acc_y)}\n"
        f"    acc_z : {_s(acc_z)}\n"
        f"    |mag| : {_s(acc_mag)}\n"
        f"{'-' * 70}\n"
        f"  GYROSCOPE  (rad/s)\n"
        f"    gyr_x : {_s(gyro_x)}\n"
        f"    gyr_y : {_s(gyro_y)}\n"
        f"    gyr_z : {_s(gyro_z)}\n"
        f"    |mag| : {_s(gyro_mag)}\n"
        f"{'-' * 70}\n"
        f"  ORIENTATION  (degrees)\n"
        f"    status  : {ori_status}\n"
    )
    if ori_samples:
        banner += (
            f"    azimuth : {_s(az_vals)}\n"
            f"    pitch   : {_s(pit_vals)}\n"
            f"    roll    : {_s(rol_vals)}\n"
        )
    banner += _SEP

    print(banner, flush=True)

    # ── append JSON record to log file ─────────────────────────────────────────
    record: dict[str, Any] = {
        "seq": seq,
        "ts": ts,
        "patient_id": patient_id,
        "n_samples": n,
        "acc": {
            "x": {"mean": round(float(np.mean(acc_x)), 4), "min": round(min(acc_x), 4), "max": round(max(acc_x), 4)},
            "y": {"mean": round(float(np.mean(acc_y)), 4), "min": round(min(acc_y), 4), "max": round(max(acc_y), 4)},
            "z": {"mean": round(float(np.mean(acc_z)), 4), "min": round(min(acc_z), 4), "max": round(max(acc_z), 4)},
            "mag_mean": round(float(np.mean(acc_mag)), 4),
        },
        "gyro": {
            "x": {"mean": round(float(np.mean(gyro_x)), 4), "min": round(min(gyro_x), 4), "max": round(max(gyro_x), 4)},
            "y": {"mean": round(float(np.mean(gyro_y)), 4), "min": round(min(gyro_y), 4), "max": round(max(gyro_y), 4)},
            "z": {"mean": round(float(np.mean(gyro_z)), 4), "min": round(min(gyro_z), 4), "max": round(max(gyro_z), 4)},
            "mag_mean": round(float(np.mean(gyro_mag)), 4),
        },
        "orientation": {
            "present": ori_present > 0,
            "coverage_pct": round(ori_coverage, 1),
            "n_with_ori": ori_present,
            "azimuth_mean":  round(float(np.mean(az_vals)),  3) if az_vals  else None,
            "pitch_mean":    round(float(np.mean(pit_vals)), 3) if pit_vals else None,
            "roll_mean":     round(float(np.mean(rol_vals)), 3) if rol_vals else None,
        },
    }
    try:
        with open(_SENSOR_DIAG_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass  # never crash ingest over a diag write failure


def _is_alarm_eligible_alert(
    *,
    severity: str | None,
    message: str | None,
    status: str | None = None,
) -> bool:
    """Single source of truth for caregiver alarm eligibility."""
    if (status or "").strip().lower() == "resolved":
        return False
    if (severity or "").strip().lower() != "fall_detected":
        return False
    msg = (message or "").strip().lower()
    if (
        "did not confirm" in msg
        or "no elder response" in msg
        or ("no elder" in msg and "response" in msg)
        or "emergency: elder" in msg
    ):
        return False
    return True


def _heuristic_fall_probability(samples_dict: list[dict[str, Any]]) -> float:
    """Accelerometer-magnitude fallback when ML stack is unavailable or ``run_inference`` fails."""
    mags: list[float] = []
    for s in samples_dict:
        ax, ay, az = s["acc_x"], s["acc_y"], s["acc_z"]
        mags.append(float((ax * ax + ay * ay + az * az) ** 0.5))
    return float(min(1.0, max(0.0, (np.max(mags) / 25.0) if mags else 0.0)))


def _heuristic_activity_label(samples_dict: list[dict[str, Any]]) -> str:
    """Best-effort ADL label when ML inference is unavailable.

    This is intentionally simple and deterministic so caregiver UI does not show
    empty activity during fallback mode.
    """
    if not samples_dict:
        return "unknown"

    mags: list[float] = []
    gyro_mags: list[float] = []
    for s in samples_dict:
        ax, ay, az = float(s["acc_x"]), float(s["acc_y"]), float(s["acc_z"])
        gx, gy, gz = float(s["gyro_x"]), float(s["gyro_y"]), float(s["gyro_z"])
        mags.append(float((ax * ax + ay * ay + az * az) ** 0.5))
        gyro_mags.append(float((gx * gx + gy * gy + gz * gz) ** 0.5))

    mean_mag = float(np.mean(mags))
    std_mag = float(np.std(mags))
    peak_mag = float(np.max(mags))
    peak_gyro = float(np.max(gyro_mags)) if gyro_mags else 0.0

    # Heuristic buckets tuned for phone IMU magnitude around gravity (m/s^2).
    if peak_mag > 22.0 or std_mag > 3.5 or peak_gyro > 4.0:
        return "jogging"
    if std_mag < 0.30 and 8.5 <= mean_mag <= 10.8 and peak_gyro < 0.8:
        return "standing"
    if std_mag < 0.55 and peak_gyro < 1.2:
        return "sitting"
    if std_mag < 2.0:
        return "walking"
    return "moving"


router = APIRouter()

# Set by ``main.py`` lifespan — avoids circular imports.
_RUNTIME: dict[str, Any] = {}

# Per-patient sliding-window ADL vote buffer (falls bypass and reset the buffer).
_patient_vote_buffers: dict[str, VoteBuffer] = {}


def set_inference_runtime(state: dict[str, Any]) -> None:
    global _RUNTIME
    _RUNTIME = state


def _get_art() -> InferenceArtifacts | None:
    return _RUNTIME.get("art")


def _claims_opt(authorization: str | None) -> dict[str, Any] | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    raw = authorization.split(" ", 1)[1].strip()
    try:
        return decode_token(raw)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def _need_role(claims: dict[str, Any] | None, roles: set[str]) -> dict[str, Any]:
    if claims is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    if claims.get("role") not in roles:
        raise HTTPException(status_code=403, detail="Insufficient role")
    return claims


def _caregiver_id_for_patient(patient_id: str) -> str | None:
    init_schema()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT caregiver_id FROM patients WHERE id = ?", (patient_id,))
        row = c.fetchone()
        if row and row["caregiver_id"]:
            return str(row["caregiver_id"])
        c.execute("SELECT caregiver_id FROM caregiver_patient WHERE patient_id = ? LIMIT 1", (patient_id,))
        row2 = c.fetchone()
        if row2 and row2["caregiver_id"]:
            return str(row2["caregiver_id"])
    return None


def _assert_manual_alert_authorized(body: Any, authorization: str | None) -> None:
    claims = _claims_opt(authorization)
    if claims is None:
        raise HTTPException(status_code=401, detail="Authorization required for manual alerts")
    role = claims.get("role")
    sub = str(claims["sub"])
    init_schema()
    with get_connection() as conn:
        c = conn.cursor()
        if role == "elder":
            c.execute(
                "SELECT id FROM patients WHERE elder_user_id = ? AND id = ?",
                (sub, body.patient_id),
            )
            if not c.fetchone():
                raise HTTPException(status_code=403, detail="Patient does not match elder token")
        elif role == "caregiver":
            c.execute(
                """
                SELECT p.id FROM patients p
                WHERE p.id = ? AND (
                  p.caregiver_id = ? OR EXISTS (
                    SELECT 1 FROM caregiver_patient cp
                    WHERE cp.patient_id = p.id AND cp.caregiver_id = ?
                  )
                )
                """,
                (body.patient_id, sub, sub),
            )
            if not c.fetchone():
                raise HTTPException(status_code=403, detail="Caregiver cannot create alert for this patient")
        else:
            raise HTTPException(status_code=403, detail="Only elder or caregiver tokens may trigger manual alerts")


async def _broadcast_alert_ws(caregiver_id: str | None, payload: dict[str, Any]) -> None:
    if not caregiver_id:
        return
    from flask_backend.app.realtime_hub import hub

    await hub.broadcast_to_caregiver(caregiver_id, payload)


# --- Schemas ---


class CaregiverSignupBody(BaseModel):
    full_name: str
    email: str
    password: str


class CaregiverLoginBody(BaseModel):
    email: str
    password: str


class ElderLoginBody(BaseModel):
    username: str
    password: str


class AdminLoginBody(BaseModel):
    email: str
    password: str


class PatientLocationBody(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accuracy_m: float | None = Field(default=None, ge=0)
    heading_degrees: float | None = Field(
        default=None,
        description="Device compass / course over ground in degrees (0–360) for map direction.",
    )


class PatientCreateBody(BaseModel):
    full_name: str
    age: int | None = None


class DeviceCreateBody(BaseModel):
    label: str
    platform: str = "flutter_mobile"
    owner_name: str | None = None
    patient_id: str | None = None


class SessionCreateBody(BaseModel):
    patient_id: str
    device_id: str
    sample_rate_hz: float = 50.0
    started_by: str = "flutter_app"


class SessionStopBody(BaseModel):
    stopped_by: str = "flutter_app"
    note: str | None = None


class SamplePayload(BaseModel):
    timestamp_ms: int
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    # MobiAct *_ori_*.txt convention: Azimuth, Pitch, Roll in degrees (optional).
    azimuth: float | None = None
    pitch: float | None = None
    roll: float | None = None


class IngestLiveBody(BaseModel):
    patient_id: str
    device_id: str
    session_id: str
    source: str = "flutter_mobile"
    sampling_rate_hz: float = 50.0
    acceleration_unit: str = "m_s2"
    gyroscope_unit: str = "rad_s"
    battery_level: float | None = None
    samples: list[SamplePayload]


class ManualAlertBody(BaseModel):
    patient_id: str
    device_id: str | None = None
    session_id: str | None = None
    severity: str = "fall_detected"
    message: str = "Emergency alert triggered from mobile app."
    actor: str = "flutter_app"


class AckBody(BaseModel):
    actor: str = "caregiver_app"
    note: str | None = None


class DetectorConfigBody(BaseModel):
    medium_risk_score: float
    high_risk_score: float
    fall_score: float


class PatientCredBody(BaseModel):
    caregiver_token: str
    full_name: str
    age: int | None = None
    home_address: str = ""
    emergency_contact: str | None = None
    notes: str | None = None


class AdminCaregiverCreateBody(BaseModel):
    full_name: str
    email: str
    password: str


class AdminPatientCreateBody(BaseModel):
    full_name: str
    age: int | None = None
    # If set, must be an existing caregiver user id; also inserts caregiver_patient.
    caregiver_id: str | None = None


class PatientSignupBody(BaseModel):
    full_name: str
    username: str
    password: str
    age: int | None = None
    email: str | None = None


def _delete_patient_cascade(c: Any, patient_id: str) -> bool:
    """Remove patient row and dependents; delete linked elder login if present. Returns False if patient missing."""
    c.execute("SELECT elder_user_id FROM patients WHERE id = ?", (patient_id,))
    pr = c.fetchone()
    if not pr:
        return False
    elder_uid = pr["elder_user_id"]
    c.execute("DELETE FROM alerts WHERE patient_id = ?", (patient_id,))
    c.execute("DELETE FROM fall_incidents WHERE patient_id = ?", (patient_id,))
    c.execute("DELETE FROM sessions WHERE patient_id = ?", (patient_id,))
    c.execute("DELETE FROM devices WHERE patient_id = ?", (patient_id,))
    c.execute("DELETE FROM patient_live WHERE patient_id = ?", (patient_id,))
    c.execute("DELETE FROM caregiver_patient WHERE patient_id = ?", (patient_id,))
    c.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    if elder_uid:
        c.execute("DELETE FROM users WHERE id = ? AND role = 'elder'", (str(elder_uid),))
    return True


def _collect_patient_ids_for_caregiver(c: Any, caregiver_id: str) -> list[str]:
    out: list[str] = []
    c.execute("SELECT id FROM patients WHERE caregiver_id = ?", (caregiver_id,))
    for row in c.fetchall():
        out.append(str(row["id"]))
    c.execute("SELECT patient_id FROM caregiver_patient WHERE caregiver_id = ?", (caregiver_id,))
    for row in c.fetchall():
        pid = str(row["patient_id"])
        if pid not in out:
            out.append(pid)
    return out


def tick_fall_escalations(conn) -> None:
    now = datetime.now(timezone.utc)
    c = conn.cursor()
    c.execute(
        "SELECT * FROM fall_incidents WHERE stage IN ('awaiting_response','alarm_local')",
    )
    rows = c.fetchall()
    for row in rows:
        rid = row["id"]
        pid = row["patient_id"]
        stage = row["stage"]
        resp_dead = row["response_deadline_at"]
        alarm_dead = row["alarm_deadline_at"]
        try:
            rdt = datetime.fromisoformat(resp_dead.replace("Z", "+00:00")) if resp_dead else None
            adt = datetime.fromisoformat(alarm_dead.replace("Z", "+00:00")) if alarm_dead else None
        except ValueError:
            continue
        if stage == "awaiting_response" and rdt and now > rdt:
            c.execute(
                "UPDATE fall_incidents SET stage = ? WHERE id = ?",
                ("alarm_local", rid),
            )
            # Do not generate a timeout-based high_risk alert when elder feedback
            # is missing. This avoids labeling non-response as additional risk.
        elif stage == "alarm_local" and adt and now > adt:
            c.execute(
                "UPDATE fall_incidents SET stage = ? WHERE id = ?",
                ("emergency", rid),
            )
            aid = uuid.uuid4().hex
            c.execute(
                """INSERT INTO alerts (id, patient_id, device_id, session_id, severity, status, message, score, created_at, manually_triggered)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    aid,
                    pid,
                    None,
                    row["session_id"],
                    "fall_detected",
                    "open",
                    "EMERGENCY: elder did not confirm after alarm — escalate to caretaker.",
                    1.0,
                    iso_now(),
                    0,
                ),
            )


def _persist_feedback_db(body: FallFeedbackEvent) -> None:
    with get_connection() as conn:
        c = conn.cursor()
        eid = uuid.uuid4().hex
        c.execute(
            "INSERT INTO app_events (id, event_type, payload_json, created_at) VALUES (?,?,?,?)",
            (eid, "fall_feedback", json.dumps(body.model_dump()), iso_now()),
        )


@router.post("/api/v1/auth/caregiver/signup")
def caregiver_signup(body: CaregiverSignupBody):
    init_schema()
    seed_default_admin()
    full_name = body.full_name.strip()
    email = body.email.strip().lower()
    password = body.password
    if not full_name:
        raise HTTPException(status_code=422, detail="full_name is required")
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=422, detail="Valid email is required")
    if not password or len(password.strip()) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters")
    uid = uuid.uuid4().hex
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE email = ?", (email,))
        if c.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")
        c.execute(
            """INSERT INTO users (id, email, username, password_hash, role, full_name, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                uid,
                email,
                None,
                hash_password(password),
                "caregiver",
                full_name,
                iso_now(),
            ),
        )
    token = create_token(user_id=uid, role="caregiver", email=email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "caregiver": {"id": uid, "full_name": full_name, "email": email},
    }


@router.post("/api/v1/auth/caregiver/login")
def caregiver_login(body: CaregiverLoginBody):
    init_schema()
    seed_default_admin()
    em = body.email.strip().lower()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ? AND role = 'caregiver'", (em,))
        row = c.fetchone()
        if not row or not verify_password(body.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        uid = row["id"]
        token = create_token(user_id=uid, role="caregiver", email=em)
        return {
            "access_token": token,
            "token_type": "bearer",
            "caregiver": {"id": uid, "full_name": row["full_name"], "email": em},
        }


@router.get("/api/v1/caregiver/my-patients")
def caregiver_my_patients(authorization: Annotated[str | None, Header()] = None):
    """Patients assigned to the signed-in caretaker (one per account)."""
    claims = _need_role(_claims_opt(authorization), {"caregiver"})
    cid = str(claims["sub"])
    init_schema()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT id, full_name, age FROM patients WHERE caregiver_id = ? ORDER BY full_name COLLATE NOCASE",
            (cid,),
        )
        rows = c.fetchall()
    return {
        "patients": [
            {"id": row["id"], "full_name": row["full_name"], "age": row["age"]}
            for row in rows
        ],
    }


@router.delete("/api/v1/caregiver/my-patients/{patient_id}")
def caregiver_delete_my_patient(patient_id: str, authorization: Annotated[str | None, Header()] = None):
    """Signed-in caregiver removes a patient they manage (devices, elder login, alerts cascade)."""
    claims = _need_role(_claims_opt(authorization), {"caregiver"})
    cid = str(claims["sub"])
    init_schema()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            """
            SELECT p.id FROM patients p
            WHERE p.id = ?
              AND (
                p.caregiver_id = ?
                OR EXISTS (
                  SELECT 1 FROM caregiver_patient cp
                  WHERE cp.patient_id = p.id AND cp.caregiver_id = ?
                )
              )
            """,
            (patient_id, cid, cid),
        )
        if not c.fetchone():
            raise HTTPException(status_code=404, detail="Patient not found or not linked to this caregiver")
        if not _delete_patient_cascade(c, patient_id):
            raise HTTPException(status_code=404, detail="Patient not found")
    return {"ok": True, "patient_id": patient_id}


@router.post("/api/v1/auth/admin/login")
def admin_login(body: AdminLoginBody):
    init_schema()
    seed_default_admin()
    em = body.email.strip().lower()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ? AND role = 'admin'", (em,))
        row = c.fetchone()
        if not row or not verify_password(body.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_token(user_id=row["id"], role="admin", email=em)
        return {"access_token": token, "token_type": "bearer", "role": "admin", "email": em}


@router.post("/api/v1/auth/patient/signup")
def patient_signup(body: PatientSignupBody):
    """Independent patient self-registration. No caregiver required."""
    init_schema()
    un = body.username.strip()
    if not un:
        raise HTTPException(status_code=422, detail="Username is required.")
    if len(body.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters.")
    pid = uuid.uuid4().hex
    eid = uuid.uuid4().hex
    email = (body.email or "").strip() or f"{un}@patients.local"
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ?", (un,))
        if c.fetchone():
            raise HTTPException(status_code=409, detail="Username already taken.")
        c.execute("SELECT id FROM users WHERE email = ?", (email,))
        if c.fetchone():
            raise HTTPException(status_code=409, detail="Email already registered.")
        c.execute(
            """INSERT INTO users (id, email, username, password_hash, role, full_name, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (eid, email, un, hash_password(body.password), "elder", body.full_name.strip(), iso_now()),
        )
        c.execute(
            """INSERT INTO patients (id, full_name, age, caregiver_id, elder_user_id)
               VALUES (?,?,?,NULL,?)""",
            (pid, body.full_name.strip(), body.age, eid),
        )
    token = create_token(user_id=eid, role="elder", email=email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": "elder",
        "user_id": eid,
        "patient_id": pid,
        "display_name": body.full_name.strip(),
    }


@router.post("/api/v1/auth/elder/login")
def elder_login(body: ElderLoginBody):
    init_schema()
    un = body.username.strip()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND role = 'elder'", (un,))
        row = c.fetchone()
        if not row or not verify_password(body.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        elder_uid = str(row["id"])
        c.execute("SELECT id FROM patients WHERE elder_user_id = ?", (elder_uid,))
        prow = c.fetchone()
        if not prow:
            raise HTTPException(
                status_code=409,
                detail="No patient record linked to this account. Please sign up again.",
            )
        patient_id = str(prow["id"])
        token = create_token(user_id=row["id"], role="elder", email=row["email"] or un)
        return {
            "access_token": token,
            "token_type": "bearer",
            "role": "elder",
            "user_id": row["id"],
            "patient_id": patient_id,
            "display_name": row["full_name"] or un,
        }


@router.post("/api/v1/auth/caregiver/patient-credentials")
def patient_credentials(body: PatientCredBody):
    init_schema()
    try:
        claims = decode_token(body.caregiver_token.strip())
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid caregiver token")
    if claims.get("role") != "caregiver":
        raise HTTPException(status_code=403, detail="Caregiver token required")
    caregiver_id = claims["sub"]

    with get_connection() as conn:
        tick_fall_escalations(conn)
        pid = uuid.uuid4().hex
        eid = uuid.uuid4().hex
        c = conn.cursor()
        display = body.full_name.strip()
        name_slug = elder_name_slug(display)
        try:
            username = pick_unique_elder_username_for_patient(c, display)
        except ElderUsernameAllocationFailed as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        temp_pass = temporary_password_for_patient(name_slug)
        c.execute(
            """INSERT INTO patients (id, full_name, age, caregiver_id, home_address, emergency_contact, notes)
               VALUES (?,?,?,?,?,?,?)""",
            (
                pid,
                body.full_name.strip(),
                body.age,
                caregiver_id,
                body.home_address.strip(),
                body.emergency_contact or "",
                body.notes or "",
            ),
        )
        c.execute(
            """INSERT INTO users (id, email, username, password_hash, role, full_name, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                eid,
                f"{username}@patients.local",
                username,
                hash_password(temp_pass),
                "elder",
                body.full_name.strip(),
                iso_now(),
            ),
        )
        c.execute("UPDATE patients SET elder_user_id = ? WHERE id = ?", (eid, pid))
        c.execute(
            "INSERT INTO caregiver_patient (caregiver_id, patient_id) VALUES (?,?)",
            (caregiver_id, pid),
        )

    return {
        "patient_id": pid,
        "patient_name": body.full_name.strip(),
        "home_address": body.home_address.strip(),
        "username": username,
        "temporary_password": temp_pass,
    }


@router.post("/api/v1/patients")
def create_patient(body: PatientCreateBody, authorization: Annotated[str | None, Header()] = None):
    init_schema()
    claims = _claims_opt(authorization)
    caregiver_id = claims["sub"] if claims and claims.get("role") == "caregiver" else None
    pid = uuid.uuid4().hex
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO patients (id, full_name, age, caregiver_id) VALUES (?,?,?,?)",
            (pid, body.full_name.strip(), body.age, caregiver_id),
        )
    return {"id": pid, "full_name": body.full_name.strip(), "age": body.age}


@router.get("/api/v1/patients/{patient_id}")
def get_patient(patient_id: str):
    init_schema()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
        row = c.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Patient not found")
        return {"id": row["id"], "full_name": row["full_name"], "age": row["age"]}


@router.post("/api/v1/devices")
def create_device(body: DeviceCreateBody):
    init_schema()
    if not body.patient_id or not body.patient_id.strip():
        raise HTTPException(status_code=422, detail="patient_id is required (create patient first)")
    did = _create_device_for_patient(body.patient_id.strip(), body.label, body.platform)
    return {"id": did, "label": body.label, "platform": body.platform}


@router.get("/api/v1/devices/{device_id}")
def get_device(device_id: str):
    init_schema()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM devices WHERE id = ?", (device_id,))
        row = c.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="device not found")
        return {"id": row["id"], "label": row["label"], "platform": row["platform"]}


def _create_device_for_patient(patient_id: str, label: str, platform: str) -> str:
    did = uuid.uuid4().hex
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO devices (id, patient_id, label, platform) VALUES (?,?,?,?)",
            (did, patient_id, label, platform),
        )
    return did


@router.post("/api/v1/sessions")
def start_session(body: SessionCreateBody):
    init_schema()
    sid = uuid.uuid4().hex
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM patients WHERE id = ?", (body.patient_id,))
        if not c.fetchone():
            raise HTTPException(status_code=404, detail="patient not found")
        c.execute(
            """INSERT INTO sessions (id, patient_id, device_id, status, sample_rate_hz, started_at)
               VALUES (?,?,?,?,?,?)""",
            (sid, body.patient_id, body.device_id, "active", body.sample_rate_hz, iso_now()),
        )
    return {
        "id": sid,
        "patient_id": body.patient_id,
        "device_id": body.device_id,
        "status": "active",
        "sample_rate_hz": body.sample_rate_hz,
    }


@router.post("/api/v1/sessions/{session_id}/stop")
def stop_session(session_id: str, body: SessionStopBody):
    init_schema()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE sessions SET status = ?, stopped_at = ? WHERE id = ?",
            ("stopped", iso_now(), session_id),
        )
    return {"ok": True}


@router.post("/api/v1/ingest/live")
def ingest_live(body: IngestLiveBody, background_tasks: BackgroundTasks):
    init_schema()
    art = _get_art()
    # Use live detector config so caregiver sensitivity slider affects ingest behavior.
    # Manifest threshold is used only as a startup fallback.
    thr = float(DETECTOR_CFG.get("fall_score", 0.8))
    if thr <= 0:
        thr = float(art.fall_threshold) if art is not None else 0.8

    samples_dict = [s.model_dump(exclude_none=True) for s in body.samples]
    _diag_sensor_batch(samples_dict, body.patient_id)
    feat_vec, acc300, gyro300, ori300 = samples_to_feature_vector(samples_dict)
    acc_w, gyro_w, ori_w = acc_gyro_ori_to_window_lists(acc300, gyro300, ori300)
    if DEBUG_SENSOR_LOGS:
        # Use warning level so logs are visible even when app logger INFO is filtered.
        # Avoid duplicate lines by keeping a single logging path (no extra print()).
        logger.warning(
            "[ingest/live] sensors patient_id=%s session_id=%s raw_samples=%d acc300=%s gyro300=%s ori300=%s",
            body.patient_id,
            body.session_id,
            len(samples_dict),
            _preview_rows(acc_w),
            _preview_rows(gyro_w),
            _preview_rows(ori_w),
        )
        logger.warning(
            "[ingest/live] features patient_id=%s session_id=%s dim=%d head=[%s]",
            body.patient_id,
            body.session_id,
            len(feat_vec),
            _preview_vector(feat_vec.tolist()),
        )

    inferred_activity: str | None = None
    ml_ok = False
    p_fall = 0.0
    branch = "unknown"
    inference_source = "heuristic"

    if art is not None:
        try:
            raw = run_inference(
                art,
                feat_vec.tolist(),
                None,
                predict_fall_type=True,
                acc_window=acc_w,
                gyro_window=gyro_w,
                ori_window=ori_w,
            )
            p_fall = float(raw["fall_probability"])
            branch = str(raw.get("branch", ""))
            is_fall_ml = bool(raw.get("is_fall", False))
            if raw.get("branch") == "adl":
                raw_activity = _humanize_activity_label(raw.get("activity_label"))
                # Smooth ADL label with per-patient majority-vote buffer.
                buf = _patient_vote_buffers.setdefault(body.patient_id, VoteBuffer())
                inferred_activity = buf.push(raw_activity or "unknown", is_fall=False) if raw_activity else None
            elif raw.get("branch") == "fall":
                # Falls bypass voting and reset the buffer for this patient.
                buf = _patient_vote_buffers.get(body.patient_id)
                if buf is not None:
                    buf.reset()
                inferred_activity = _humanize_fall_type_label(
                    raw.get("fall_type_label") or raw.get("fall_type_code")
                ) or "Fall"
            else:
                inferred_activity = None
            ml_ok = True
            inference_source = "model"
            logger.info(
                "[ingest/live] inference=model patient_id=%s session_id=%s samples=%d branch=%s p_fall=%.4f",
                body.patient_id,
                body.session_id,
                len(samples_dict),
                branch,
                p_fall,
            )
        except Exception as exc:
            logger.warning("run_inference failed; using heuristic fall probability: %s", exc, exc_info=True)
            p_fall = _heuristic_fall_probability(samples_dict)
            branch = "unknown"
            inferred_activity = _heuristic_activity_label(samples_dict)
            ml_ok = False
            inference_source = "heuristic"
            logger.info(
                "[ingest/live] inference=heuristic patient_id=%s session_id=%s samples=%d reason=run_inference_failed p_fall=%.4f",
                body.patient_id,
                body.session_id,
                len(samples_dict),
                p_fall,
            )
    else:
        p_fall = _heuristic_fall_probability(samples_dict)
        inferred_activity = _heuristic_activity_label(samples_dict)
        logger.info(
            "[ingest/live] inference=heuristic patient_id=%s session_id=%s samples=%d reason=artifacts_not_loaded p_fall=%.4f",
            body.patient_id,
            body.session_id,
            len(samples_dict),
            p_fall,
        )

    detection = build_detection_payload(
        samples=samples_dict,
        fall_probability=p_fall,
        inferred_activity=inferred_activity,
        ml_ok=ml_ok,
        threshold=thr,
    )
    logger.info(
        "[ingest/live] detection source=%s patient_id=%s severity=%s score=%.4f fall_probability=%.4f",
        inference_source,
        body.patient_id,
        detection.get("severity", "unknown"),
        float(detection.get("score", 0.0)),
        p_fall,
    )

    active_alert = None
    with get_connection() as conn:
        tick_fall_escalations(conn)
        c = conn.cursor()
        c.execute("SELECT full_name FROM patients WHERE id = ?", (body.patient_id,))
        prow = c.fetchone()
        pname = prow["full_name"] if prow else "Unknown"

        alert_ids: list[str] = []
        if detection["severity"] == "fall_detected" and p_fall >= thr:
            c.execute(
                "SELECT id FROM fall_incidents WHERE patient_id = ? AND stage IN ('awaiting_response','alarm_local')",
                (body.patient_id,),
            )
            has_open_incident = c.fetchone() is not None

            # Avoid duplicate fall notifications while the same incident is active.
            if not has_open_incident:
                aid = uuid.uuid4().hex
                created_at = iso_now()
                c.execute(
                    """INSERT INTO alerts (id, patient_id, device_id, session_id, severity, status, message, score, created_at, manually_triggered)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        aid,
                        body.patient_id,
                        body.device_id,
                        body.session_id,
                        "fall_detected",
                        "open",
                        detection["message"],
                        float(detection["score"]),
                        created_at,
                        0,
                    ),
                )
                alert_ids.append(aid)
                active_alert = {
                    "id": aid,
                    "patient_id": body.patient_id,
                    "severity": "fall_detected",
                    "status": "open",
                    "message": detection["message"],
                    "score": float(detection["score"]),
                    "created_at": created_at,
                    "manually_triggered": False,
                    "alarm_eligible": _is_alarm_eligible_alert(
                        severity="fall_detected",
                        message=detection["message"],
                        status="open",
                    ),
                }

                iid = uuid.uuid4().hex
                now = datetime.now(timezone.utc)
                c.execute(
                    """INSERT INTO fall_incidents (id, patient_id, session_id, stage, created_at, response_deadline_at, alarm_deadline_at, fall_probability, metadata_json)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        iid,
                        body.patient_id,
                        body.session_id,
                        "awaiting_response",
                        iso_now(),
                        (now + timedelta(seconds=RESPONSE_DEADLINE_SEC)).isoformat(),
                        (now + timedelta(seconds=EMERGENCY_DEADLINE_SEC)).isoformat(),
                        p_fall,
                        json.dumps({"branch": branch, "ml_ok": ml_ok}),
                    ),
                )
                # Push to caregiver WebSocket immediately (same path as manual alerts).
                cg = _caregiver_id_for_patient(body.patient_id)
                background_tasks.add_task(_broadcast_alert_ws, cg, active_alert)

        c.execute("SELECT id FROM alerts WHERE patient_id = ? AND status = 'open'", (body.patient_id,))
        alert_ids = [r[0] for r in c.fetchall()]

        live = {
            "patient_id": body.patient_id,
            "patient_name": pname,
            "session_id": body.session_id,
            "device_id": body.device_id,
            "severity": detection["severity"],
            "score": float(detection["score"]),
            "fall_probability": float(p_fall),
            "predicted_activity_class": inferred_activity,
            "last_message": detection["message"],
            "sample_rate_hz": body.sampling_rate_hz,
            "latest_metrics": {"branch": branch, "ml_ok": float(ml_ok)},
            "active_alert_ids": alert_ids,
        }
        c.execute(
            """INSERT INTO patient_live (patient_id, patient_name, session_id, device_id, severity, score, fall_probability, predicted_activity_class, last_message, sample_rate_hz, active_alert_ids, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(patient_id) DO UPDATE SET
               patient_name=excluded.patient_name, session_id=excluded.session_id, device_id=excluded.device_id,
               severity=excluded.severity, score=excluded.score, fall_probability=excluded.fall_probability,
               predicted_activity_class=excluded.predicted_activity_class, last_message=excluded.last_message,
               sample_rate_hz=excluded.sample_rate_hz, active_alert_ids=excluded.active_alert_ids, updated_at=excluded.updated_at""",
            (
                body.patient_id,
                pname,
                body.session_id,
                body.device_id,
                live["severity"],
                live["score"],
                float(p_fall),
                inferred_activity,
                live["last_message"],
                body.sampling_rate_hz,
                json.dumps(alert_ids),
                iso_now(),
            ),
        )

    telemetry = {
        "patient_id": body.patient_id,
        "patient_name": pname,
        "session_id": body.session_id,
        "device_id": body.device_id,
        "source": body.source,
        "sampling_rate_hz": body.sampling_rate_hz,
        "acceleration_unit": body.acceleration_unit,
        "gyroscope_unit": body.gyroscope_unit,
        "battery_level": body.battery_level,
        "received_at": iso_now(),
        "samples_in_last_batch": len(body.samples),
        "latest_samples": samples_dict[-min(32, len(samples_dict)) :],
    }

    return {
        "ingested_samples": len(body.samples),
        "detection": detection,
        "live_status": live,
        "active_alert": active_alert,
        "telemetry": telemetry,
    }


@router.post("/api/v1/alerts/manual")
def manual_alert(
    body: ManualAlertBody,
    background_tasks: BackgroundTasks,
    authorization: Annotated[str | None, Header()] = None,
):
    _assert_manual_alert_authorized(body, authorization)
    init_schema()
    aid = uuid.uuid4().hex
    created = iso_now()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            """INSERT INTO alerts (id, patient_id, device_id, session_id, severity, status, message, score, created_at, manually_triggered)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                aid,
                body.patient_id,
                body.device_id,
                body.session_id,
                body.severity,
                "open",
                body.message,
                1.0,
                created,
                1,
            ),
        )
    payload = {
        "id": aid,
        "patient_id": body.patient_id,
        "severity": body.severity,
        "status": "open",
        "message": body.message,
        "score": 1.0,
        "created_at": created,
        "manually_triggered": True,
        "alarm_eligible": _is_alarm_eligible_alert(
            severity=body.severity,
            message=body.message,
            status="open",
        ),
    }
    cg = _caregiver_id_for_patient(body.patient_id)
    background_tasks.add_task(_broadcast_alert_ws, cg, payload)
    return payload


@router.get("/api/v1/alerts")
def list_alerts(
    status: str | None = None,
    patient_id: str | None = None,
    authorization: Annotated[str | None, Header()] = None,
):
    init_schema()
    claims = _claims_opt(authorization)
    caregiver_filter_ids: list[str] | None = None
    if claims and claims.get("role") == "caregiver":
        with get_connection() as conn:
            c = conn.cursor()
            caregiver_filter_ids = _collect_patient_ids_for_caregiver(c, str(claims["sub"]))
        if not caregiver_filter_ids:
            return []

    with get_connection() as conn:
        tick_fall_escalations(conn)
        c = conn.cursor()
        q = "SELECT * FROM alerts WHERE 1=1"
        args: list[Any] = []
        if status:
            q += " AND status = ?"
            args.append(status)
        if patient_id:
            q += " AND patient_id = ?"
            args.append(patient_id)
        if caregiver_filter_ids is not None:
            placeholders = ",".join("?" for _ in caregiver_filter_ids)
            q += f" AND patient_id IN ({placeholders})"
            args.extend(caregiver_filter_ids)
        q += " ORDER BY created_at DESC LIMIT 200"
        c.execute(q, args)
        rows = c.fetchall()
    out = []
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "patient_id": row["patient_id"],
                "severity": row["severity"],
                "status": row["status"],
                "message": row["message"],
                "score": row["score"],
                "created_at": row["created_at"],
                "acknowledged_at": row["acknowledged_at"],
                "resolved_at": row["resolved_at"],
                "manually_triggered": bool(row["manually_triggered"]),
                "alarm_eligible": _is_alarm_eligible_alert(
                    severity=row["severity"],
                    message=row["message"],
                    status=row["status"],
                ),
            }
        )
    return out


@router.post("/api/v1/alerts/{alert_id}/acknowledge")
def ack_alert(alert_id: str, body: AckBody):
    init_schema()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE alerts SET status = ?, acknowledged_at = ? WHERE id = ?",
            ("acknowledged", iso_now(), alert_id),
        )
        c.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
        row = c.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="alert not found")
    return {
        "id": row["id"],
        "patient_id": row["patient_id"],
        "severity": row["severity"],
        "status": row["status"],
        "message": row["message"],
        "score": row["score"],
        "created_at": row["created_at"],
        "acknowledged_at": row["acknowledged_at"],
        "resolved_at": row["resolved_at"],
        "manually_triggered": bool(row["manually_triggered"]),
        "alarm_eligible": _is_alarm_eligible_alert(
            severity=row["severity"],
            message=row["message"],
            status=row["status"],
        ),
    }


@router.post("/api/v1/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: str, body: AckBody):
    init_schema()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE alerts SET status = ?, resolved_at = ? WHERE id = ?",
            ("resolved", iso_now(), alert_id),
        )
        c.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
        row = c.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="alert not found")
    return {
        "id": row["id"],
        "patient_id": row["patient_id"],
        "severity": row["severity"],
        "status": row["status"],
        "message": row["message"],
        "score": row["score"],
        "created_at": row["created_at"],
        "acknowledged_at": row["acknowledged_at"],
        "resolved_at": row["resolved_at"],
        "manually_triggered": bool(row["manually_triggered"]),
        "alarm_eligible": _is_alarm_eligible_alert(
            severity=row["severity"],
            message=row["message"],
            status=row["status"],
        ),
    }


@router.get("/api/v1/summary")
def summary(authorization: Annotated[str | None, Header()] = None):
    init_schema()
    claims = _claims_opt(authorization)
    with get_connection() as conn:
        tick_fall_escalations(conn)
        c = conn.cursor()
        if claims and claims.get("role") == "caregiver":
            ids = _collect_patient_ids_for_caregiver(c, str(claims["sub"]))
            if not ids:
                tp = 0
                ac = 0
                oa = 0
            else:
                placeholders = ",".join("?" for _ in ids)
                c.execute(f"SELECT COUNT(*) FROM patients WHERE id IN ({placeholders})", ids)
                tp = c.fetchone()[0]
                c.execute(
                    f"SELECT COUNT(*) FROM sessions WHERE status = 'active' AND patient_id IN ({placeholders})",
                    ids,
                )
                ac = c.fetchone()[0]
                c.execute(
                    f"SELECT COUNT(*) FROM alerts WHERE status = 'open' AND patient_id IN ({placeholders})",
                    ids,
                )
                oa = c.fetchone()[0]
        else:
            c.execute("SELECT COUNT(*) FROM patients")
            tp = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM sessions WHERE status = 'active'")
            ac = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM alerts WHERE status = 'open'")
            oa = c.fetchone()[0]
    return {
        "total_patients": tp,
        "active_sessions": ac,
        "open_alerts": oa,
        "last_event_at": iso_now(),
    }


@router.post("/api/v1/patients/me/location")
def post_my_location(
    body: PatientLocationBody,
    authorization: Annotated[str | None, Header()] = None,
):
    """Elder device shares GPS — stored on ``patient_live`` for caregiver map."""
    claims = _need_role(_claims_opt(authorization), {"elder"})
    elder_uid = str(claims["sub"])
    init_schema()
    now = iso_now()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT id, full_name FROM patients WHERE elder_user_id = ?",
            (elder_uid,),
        )
        row = c.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="No patient linked to this elder account")
        pid = str(row["id"])
        pname = str(row["full_name"])

        c.execute("SELECT patient_id FROM patient_live WHERE patient_id = ?", (pid,))
        if c.fetchone():
            c.execute(
                """UPDATE patient_live SET latitude=?, longitude=?, location_accuracy_m=?,
                   location_updated_at=?, updated_at=?, heading_degrees=? WHERE patient_id=?""",
                (
                    body.latitude,
                    body.longitude,
                    body.accuracy_m,
                    now,
                    now,
                    body.heading_degrees,
                    pid,
                ),
            )
        else:
            c.execute(
                """INSERT INTO patient_live (
                    patient_id, patient_name, session_id, device_id,
                    severity, score, fall_probability, predicted_activity_class,
                    last_message, sample_rate_hz, active_alert_ids, updated_at,
                    latitude, longitude, location_accuracy_m, location_updated_at,
                    heading_degrees
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    pid,
                    pname,
                    None,
                    None,
                    "low",
                    0.0,
                    0.0,
                    None,
                    "Location shared",
                    None,
                    "[]",
                    now,
                    body.latitude,
                    body.longitude,
                    body.accuracy_m,
                    now,
                    body.heading_degrees,
                ),
            )
    return {"ok": True, "patient_id": pid, "location_updated_at": now}


@router.get("/api/v1/monitor/patients/live")
def live_patients(authorization: Annotated[str | None, Header()] = None):
    init_schema()
    claims = _claims_opt(authorization)
    with get_connection() as conn:
        tick_fall_escalations(conn)
        c = conn.cursor()
        if claims and claims.get("role") == "caregiver":
            ids = _collect_patient_ids_for_caregiver(c, str(claims["sub"]))
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            c.execute(f"SELECT * FROM patient_live WHERE patient_id IN ({placeholders})", ids)
        else:
            c.execute("SELECT * FROM patient_live")
        rows = c.fetchall()
    out = []
    for row in rows:
        aids = []
        raw = row["active_alert_ids"]
        if raw:
            try:
                aids = json.loads(raw)
            except json.JSONDecodeError:
                aids = []
        out.append(
            {
                "patient_id": row["patient_id"],
                "patient_name": row["patient_name"],
                "session_id": row["session_id"],
                "device_id": row["device_id"],
                "severity": row["severity"],
                "score": row["score"],
                "fall_probability": row["fall_probability"],
                "predicted_activity_class": row["predicted_activity_class"],
                "last_message": row["last_message"],
                "sample_rate_hz": row["sample_rate_hz"],
                "latest_metrics": {},
                "active_alert_ids": aids,
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "location_accuracy_m": row["location_accuracy_m"],
                "location_updated_at": row["location_updated_at"],
                "heading_degrees": row["heading_degrees"] if "heading_degrees" in row.keys() else None,
            }
        )
    return out


@router.put("/api/v1/detector/config")
def detector_config(body: DetectorConfigBody):
    DETECTOR_CFG["medium_risk_score"] = body.medium_risk_score
    DETECTOR_CFG["high_risk_score"] = body.high_risk_score
    DETECTOR_CFG["fall_score"] = body.fall_score
    return {"ok": True, "config": DETECTOR_CFG}


@router.get("/api/v1/admin/dashboard")
def admin_dashboard(authorization: Annotated[str | None, Header()] = None):
    _need_role(_claims_opt(authorization), {"admin"})
    init_schema()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users WHERE role='caregiver'")
        nc = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM users WHERE role='elder'")
        ne = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM patients")
        np = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM alerts WHERE status='open'")
        oa = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM app_events WHERE event_type='fall_feedback'")
        nf = c.fetchone()[0]
    return {
        "schema": "sisfall_monitoring_v1",
        "caretakers": nc,
        "elders_registered": ne,
        "patients": np,
        "open_alerts": oa,
        "fall_feedback_events": nf,
        "datasets": ["SisFall", "MobiAct"],
        "note": "Train scripts under scripts/; SisFall loader: scripts/sisfall/",
    }


@router.get("/api/v1/admin/caregivers")
def admin_list_caregivers(authorization: Annotated[str | None, Header()] = None):
    _need_role(_claims_opt(authorization), {"admin"})
    init_schema()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT id, email, full_name, created_at FROM users WHERE role = 'caregiver' ORDER BY created_at DESC"
        )
        rows = c.fetchall()
    return [
        {
            "id": row["id"],
            "email": row["email"],
            "full_name": row["full_name"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


@router.post("/api/v1/admin/caregivers")
def admin_create_caregiver(
    body: AdminCaregiverCreateBody,
    authorization: Annotated[str | None, Header()] = None,
):
    _need_role(_claims_opt(authorization), {"admin"})
    init_schema()
    seed_default_admin()
    em = body.email.strip().lower()
    uid = uuid.uuid4().hex
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE email = ?", (em,))
        if c.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")
        c.execute(
            """INSERT INTO users (id, email, username, password_hash, role, full_name, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                uid,
                em,
                None,
                hash_password(body.password),
                "caregiver",
                body.full_name.strip(),
                iso_now(),
            ),
        )
    return {"id": uid, "email": em, "full_name": body.full_name.strip()}


@router.delete("/api/v1/admin/caregivers/{user_id}")
def admin_delete_caregiver(user_id: str, authorization: Annotated[str | None, Header()] = None):
    _need_role(_claims_opt(authorization), {"admin"})
    init_schema()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT role FROM users WHERE id = ?", (user_id,))
        row = c.fetchone()
        if not row or row["role"] != "caregiver":
            raise HTTPException(status_code=404, detail="Caretaker not found")
        for pid in _collect_patient_ids_for_caregiver(c, user_id):
            _delete_patient_cascade(c, pid)
        c.execute("DELETE FROM caregiver_patient WHERE caregiver_id = ?", (user_id,))
        c.execute("DELETE FROM users WHERE id = ? AND role = 'caregiver'", (user_id,))
    return {"ok": True}


@router.get("/api/v1/admin/patients")
def admin_list_patients(authorization: Annotated[str | None, Header()] = None):
    _need_role(_claims_opt(authorization), {"admin"})
    init_schema()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            """
            SELECT p.id, p.full_name, p.age, p.caregiver_id, p.elder_user_id, u.username AS elder_username
            FROM patients p
            LEFT JOIN users u ON u.id = p.elder_user_id AND u.role = 'elder'
            ORDER BY p.full_name
            """
        )
        rows = c.fetchall()
    return [
        {
            "id": row["id"],
            "full_name": row["full_name"],
            "age": row["age"],
            "caregiver_id": row["caregiver_id"],
            "elder_user_id": row["elder_user_id"],
            "elder_username": row["elder_username"],
        }
        for row in rows
    ]


@router.post("/api/v1/admin/patients")
def admin_create_patient(
    body: AdminPatientCreateBody,
    authorization: Annotated[str | None, Header()] = None,
):
    _need_role(_claims_opt(authorization), {"admin"})
    init_schema()
    raw_cg = (body.caregiver_id or "").strip()
    cid: str | None = raw_cg or None
    pid = uuid.uuid4().hex
    with get_connection() as conn:
        c = conn.cursor()
        if cid:
            c.execute("SELECT id FROM users WHERE id = ? AND role = 'caregiver'", (cid,))
            if not c.fetchone():
                raise HTTPException(status_code=400, detail="caregiver_id is not a valid caretaker")
        c.execute(
            """INSERT INTO patients (id, full_name, age, caregiver_id, home_address, emergency_contact, notes)
               VALUES (?,?,?,?,?,?,?)""",
            (
                pid,
                body.full_name.strip(),
                body.age,
                cid,
                "",
                "",
                "",
            ),
        )
        if cid:
            c.execute(
                "INSERT OR IGNORE INTO caregiver_patient (caregiver_id, patient_id) VALUES (?,?)",
                (cid, pid),
            )
    return {"id": pid, "full_name": body.full_name.strip(), "caregiver_id": cid}


@router.delete("/api/v1/admin/patients/{patient_id}")
def admin_delete_patient(patient_id: str, authorization: Annotated[str | None, Header()] = None):
    _need_role(_claims_opt(authorization), {"admin"})
    init_schema()
    with get_connection() as conn:
        c = conn.cursor()
        if not _delete_patient_cascade(c, patient_id):
            raise HTTPException(status_code=404, detail="Patient not found")
    return {"ok": True}


@router.post("/api/v1/events/fall-feedback")
def fall_feedback_db(body: FallFeedbackEvent):
    from flask_backend.app.settings import repo_root as _repo_root

    _persist_feedback_db(body)
    log_dir = _repo_root() / "data" / "feedback"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "fall_events.jsonl"
    row = body.model_dump()
    row["_server_logged_at"] = FallFeedbackAck().logged_at
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return FallFeedbackAck()


@router.websocket("/api/v1/ws/caregiver")
async def caregiver_alerts_ws(websocket: WebSocket, token: str = Query(..., min_length=8)):
    """Caregiver subscribes with `?token=<JWT>`; receives JSON `{"type":"alert","data":{...}}` on new manual/system alerts."""
    try:
        claims = decode_token(token)
    except Exception:
        await websocket.close(code=4401)
        return
    if claims.get("role") != "caregiver":
        await websocket.close(code=4403)
        return
    cid = str(claims["sub"])
    from flask_backend.app.realtime_hub import hub

    await hub.register(cid, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.unregister(cid, websocket)


def _apply_deadband(arr: np.ndarray, threshold: float) -> np.ndarray:
    if arr is None or arr.size == 0:
        return arr
    out = arr.copy()
    for col in range(out.shape[1]):
        prev = out[0, col]
        for i in range(1, out.shape[0]):
            if abs(out[i, col] - prev) < threshold:
                out[i, col] = prev
            else:
                prev = out[i, col]
    return out


@router.post("/api/v1/inference/motion", response_model=MotionInferenceResponse)
def inference_motion(body: MotionInferenceRequest):
    art = _get_art()
    if art is None:
        err = _RUNTIME.get("load_error", "not loaded")
        raise HTTPException(503, detail=f"Inference not loaded: {err}")

    if body.acc_window is not None:
        acc = np.asarray(body.acc_window, dtype=np.float64)
        gyro = np.asarray(body.gyro_window, dtype=np.float64) if body.gyro_window is not None else None
        
        # Apply deadband to flatten micro-tremors (mimics phone resting in pocket)
        # Increased to 0.4 to aggressively filter out hand tremors that trick the
        # model into predicting "Walking"
        acc = _apply_deadband(acc, 0.4)
        if gyro is not None:
            gyro = _apply_deadband(gyro, 0.4)

        ori = np.asarray(body.ori_window, dtype=np.float64) if body.ori_window is not None else None
        enhanced_in = build_enhanced_features_numpy(acc, gyro, ori).tolist()
        logger.info(
            "[inference/motion] server-built 144-D features from raw windows (rows=%s)",
            acc.shape[0],
        )
    else:
        enhanced_in = list(body.enhanced_features)  # type: ignore[arg-type]
        acc = None
        gyro = None

    try:
        raw = run_inference(
            art,
            enhanced_in,
            body.fall_type_features,
            predict_fall_type=body.predict_fall_type,
            acc_window=acc.tolist() if acc is not None else body.acc_window,
            gyro_window=gyro.tolist() if gyro is not None else body.gyro_window,
            ori_window=body.ori_window,

        )
        return MotionInferenceResponse(**raw)
    except ValueError as e:
        logger.warning("[inference/motion] request rejected: %s", e)
        raise HTTPException(422, detail=str(e)) from e
