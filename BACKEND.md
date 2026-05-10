# Backend — Complete Reference

## Overview

The EMS backend is a **FastAPI** application (`flask_backend/`) that serves:
- REST API for the Flutter mobile app (elderly patients + caregiver monitoring)
- ML inference pipeline (fall detection + ADL classification)
- Real-time WebSocket push for caregiver alerts
- MongoDB persistence for production (users, patients, sessions, alerts, fall incidents)
- SQLite fallback for local/dev when Mongo is not configured
- JWT authentication for three roles: `admin`, `caregiver`, `elder`

**Framework:** FastAPI 0.110+  
**Server:** Uvicorn (standard, with WebSocket support)  
**Database:** SQLite (thread-safe, file-based, `data/elder_monitor.db`)  
**ML Backend:** scikit-learn / XGBoost / LightGBM via joblib  
**Auth:** JWT HS256 + bcrypt password hashing

---

## Project Structure

```
flask_backend/
├── __init__.py
├── requirements.txt
├── captain-definition              (CapRover deployment config)
├── app/
│   ├── __init__.py
│   ├── main.py                     (FastAPI app, lifespan, health endpoints)
│   ├── monitoring_routes.py        (ALL REST routes + WebSocket, 1750+ lines)
│   ├── auth_jwt.py                 (JWT encode/decode + bcrypt hashing)
│   ├── database.py                 (SQLite schema, connection, seeding)
│   ├── detector_state.py           (Severity mapping + IMU heuristics)
│   ├── elder_credential_allocator.py (Username/password generation)
│   ├── ml_bridge.py                (Sensor → 144-D features + VoteBuffer)
│   ├── motion_enhanced_features.py (144-D feature extraction math)
│   ├── realtime_hub.py             (WebSocket fan-out for caregiver alerts)
│   ├── schemas_fall_feedback.py    (Pydantic: fall feedback events)
│   ├── schemas_motion.py           (Pydantic: motion inference request/response)
│   ├── settings.py                 (Path config with env-var overrides)
│   └── services/
│       ├── __init__.py
│       └── motion_xgb_service.py   (Re-exports: InferenceArtifacts, load_artifacts, run_inference)
├── models/
│   ├── inference_manifest.json
│   └── baseline_adl&fall/
│       ├── fall_xgb_model.pkl
│       ├── fall_scaler.pkl
│       ├── adl_xgb_model.pkl
│       ├── adl_scaler.pkl
│       ├── adl_label_encoder.pkl
│       └── results_v2.json
└── tests/
    └── test_pipeline.py
```

---

## Application Entry Point

**File:** [`flask_backend/app/main.py`](flask_backend/app/main.py)

```python
app = FastAPI(title="SisFall Elder Monitoring", version="2.0.0", lifespan=lifespan)
```

### Lifespan (startup/shutdown)

```python
@asynccontextmanager
async def lifespan(_app):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    init_schema()           # create SQLite tables if not exist
    seed_default_admin()    # create admin@local if no admin row
    try:
        _state["art"] = load_artifacts(inference_manifest_path(), model_root())
    except Exception as exc:
        _state["load_error"] = str(exc)
    set_inference_runtime(_state)  # inject into routes module
    yield
```

Thread control: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` (prevents over-subscription in containers)

### Middleware

**CORS:** `allow_origins=["*"]`, all methods, all headers (development config)

**Timing:** Custom middleware adds `X-Process-Time-Ms` header to every response

---

## Configuration & Paths

**File:** [`flask_backend/app/settings.py`](flask_backend/app/settings.py)

All paths configurable via environment variables:

| Env Var | Default | Purpose |
|---|---|---|
| `MODEL_ROOT` | `<repo>/flask_backend/models` | Directory containing model pkl files |
| `REPO_ROOT` | Detected from `__file__` (3 levels up) | Repo root for DB and feedback file paths |
| `INFERENCE_MANIFEST` | `<MODEL_ROOT>/inference_manifest.json` | Path to inference manifest |
| `DB_PATH` | `<REPO_ROOT>/data/elder_monitor.db` | SQLite database location |
| `ADMIN_PASSWORD` | `admin123` | Default admin password (override in prod) |
| `ADMIN_EMAIL` | `admin@local` | Default admin email |
| `JWT_SECRET` | `sisfall-dev-change-me-in-production` | JWT signing key (**must override in prod**) |
| `FALL_RESPONSE_DEADLINE_SEC` | `30` | Seconds before fall → alarm escalation |
| `FALL_EMERGENCY_DEADLINE_SEC` | `90` | Seconds before alarm → emergency escalation |
| `EMS_DEBUG_SENSOR_LOGS` | `1` | Log raw sensor previews (disable in prod) |

---

## Authentication

**File:** [`flask_backend/app/auth_jwt.py`](flask_backend/app/auth_jwt.py)

### JWT Tokens

- Algorithm: **HS256**
- TTL: **7 days**
- Secret: `JWT_SECRET` env var (default is dev-only placeholder)
- Payload: `{ sub: user_id, role, email, iat, exp }`

```python
def create_token(*, user_id, role, email) -> str
def decode_token(token) -> dict  # raises jwt.InvalidTokenError on bad/expired token
```

### Password Hashing

- Library: **bcrypt** (4.1+)
- Salt: auto-generated per password
- Functions: `hash_password(password)`, `verify_password(password, hash)`

### Role Enforcement

```python
def _claims_opt(authorization) -> dict | None:
    # Parses "Bearer <token>" header; raises 401 on invalid token

def _need_role(claims, roles: set[str]) -> dict:
    # Raises 401 if no claims, 403 if role not in allowed set
```

### User Roles
| Role | Login Endpoint | Can Access |
|---|---|---|
| `admin` | `/api/v1/auth/admin/login` | Admin dashboard, manage caregivers/patients |
| `caregiver` | `/api/v1/auth/caregiver/login` | Own patients, alerts, live monitoring |
| `elder` | `/api/v1/auth/elder/login` | Own patient data, location push, manual alerts |

---

## Database Schema

**File:** [`flask_backend/app/database.py`](flask_backend/app/database.py)  
**Location:** `data/elder_monitor.db` (SQLite)  
**Thread safety:** Single `threading.Lock()` guards all connections

### Tables

#### `users`
| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID hex |
| email | TEXT UNIQUE | — |
| username | TEXT UNIQUE | Elders use username; caregivers use email |
| password_hash | TEXT NOT NULL | bcrypt hash |
| role | TEXT NOT NULL | `admin`, `caregiver`, `elder` |
| full_name | TEXT | — |
| created_at | TEXT NOT NULL | ISO 8601 UTC |

### Database Persistence

Production should use MongoDB. Configure either `MONGO_URI` or `MONGODB_URI`:

```bash
EMS_DB_BACKEND=mongo
MONGO_URI=mongodb+srv://USER:PASSWORD@HOST/ems?retryWrites=true&w=majority
MONGO_DB_NAME=ems
```

If `MONGO_URI`/`MONGODB_URI` is present, the backend automatically uses MongoDB.
The route layer writes these Mongo collections:

- `users`
- `patients`
- `caregiver_patient`
- `devices`
- `sessions`
- `alerts`
- `fall_incidents`
- `app_events`
- `patient_live`

To migrate existing local SQLite data into MongoDB:

```bash
python scripts/migrate_sqlite_to_mongo.py ^
  --sqlite-path data/elder_monitor.db ^
  --mongo-uri "mongodb+srv://USER:PASSWORD@HOST/ems" ^
  --db ems
```

SQLite is still available as a dev fallback when Mongo is not configured.

By default the database is stored at `REPO_ROOT/data/elder_monitor.db`.
You can override it with `EMS_DB_PATH` or `DATABASE_PATH`, for example:

```bash
EMS_DB_PATH=/app/data/elder_monitor.db
```

In Docker/CapRover deployments, mount that directory as persistent storage.
If `/app/data` is not mounted, caretakers and patients will appear to vanish
after container restart/redeploy because a fresh SQLite file is created.

#### `patients`
| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID hex |
| full_name | TEXT NOT NULL | — |
| age | INTEGER | Optional |
| caregiver_id | TEXT FK→users | Primary caregiver |
| elder_user_id | TEXT FK→users | Linked elder login |
| home_address | TEXT | — |
| emergency_contact | TEXT | — |
| notes | TEXT | — |

#### `caregiver_patient`
Many-to-many bridge for additional caregiver assignments:
| Column | Type |
|---|---|
| caregiver_id | TEXT FK→users (PK part) |
| patient_id | TEXT FK→patients (PK part) |

#### `devices`
| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID hex |
| patient_id | TEXT FK→patients | |
| label | TEXT | Device name |
| platform | TEXT | e.g. `flutter_mobile` |

#### `sessions`
| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID hex |
| patient_id | TEXT FK→patients | |
| device_id | TEXT FK→devices | |
| status | TEXT | `active` / `stopped` |
| sample_rate_hz | REAL | Default 50.0 |
| started_at | TEXT NOT NULL | ISO 8601 UTC |
| stopped_at | TEXT | Null until stopped |

#### `alerts`
| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID hex |
| patient_id | TEXT FK→patients | |
| device_id | TEXT | May be null (manual alerts) |
| session_id | TEXT | — |
| severity | TEXT | `low`, `medium`, `high_risk`, `fall_detected` |
| status | TEXT | `open`, `acknowledged`, `resolved` |
| message | TEXT | Human-readable alert message |
| score | REAL | 0.0–1.0 confidence score |
| created_at | TEXT NOT NULL | ISO 8601 UTC |
| acknowledged_at | TEXT | Null until acked |
| resolved_at | TEXT | Null until resolved |
| manually_triggered | INTEGER | 0 = ML/auto, 1 = manual |

#### `fall_incidents`
State machine for fall escalation timeline:
| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID hex |
| patient_id | TEXT FK→patients | |
| session_id | TEXT | — |
| stage | TEXT | `awaiting_response` → `alarm_local` → `emergency` |
| created_at | TEXT NOT NULL | ISO 8601 UTC |
| response_deadline_at | TEXT | NOW + 30s (FALL_RESPONSE_DEADLINE_SEC) |
| alarm_deadline_at | TEXT | NOW + 90s (FALL_EMERGENCY_DEADLINE_SEC) |
| fall_probability | REAL | Raw ML probability |
| fall_type_code | TEXT | `FOL`, `FKL`, `BSC`, `SDL` (optional) |
| response | TEXT | Elder feedback text |
| metadata_json | TEXT | JSON: branch, ml_ok |

#### `app_events`
Generic event log (used for fall feedback):
| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | UUID hex |
| event_type | TEXT | e.g. `fall_feedback` |
| payload_json | TEXT | Full JSON payload |
| created_at | TEXT NOT NULL | ISO 8601 UTC |

#### `patient_live`
Live inference state per patient (upserted on every ingest):
| Column | Type | Notes |
|---|---|---|
| patient_id | TEXT PK | — |
| patient_name | TEXT | Denormalized |
| session_id | TEXT | Current session |
| device_id | TEXT | Current device |
| severity | TEXT | Latest detection severity |
| score | REAL | Latest score |
| fall_probability | REAL | Latest ML probability |
| predicted_activity_class | TEXT | Latest ADL or fall-type label |
| last_message | TEXT | Latest detection message |
| sample_rate_hz | REAL | — |
| active_alert_ids | TEXT | JSON array of open alert IDs |
| updated_at | TEXT NOT NULL | ISO 8601 UTC |
| latitude | REAL | GPS (added via migration) |
| longitude | REAL | GPS |
| location_accuracy_m | REAL | GPS accuracy |
| location_updated_at | TEXT | When GPS was last updated |
| heading_degrees | REAL | Compass/course |

### Schema Migration

`_migrate_schema()` runs on every `init_schema()` call. Uses `PRAGMA table_info` to check existing columns; adds missing ones with `ALTER TABLE ADD COLUMN`. Currently handles: `latitude`, `longitude`, `location_accuracy_m`, `location_updated_at`, `heading_degrees`.

### Default Admin Seeding

`seed_default_admin()` creates `admin@local` (password: `admin123`) if no admin row exists. Reads `ADMIN_PASSWORD` and `ADMIN_EMAIL` env vars.

---

## REST API Endpoints

**Router:** All routes via `APIRouter` in [`flask_backend/app/monitoring_routes.py`](flask_backend/app/monitoring_routes.py)

### Health & Status

#### `GET /api/v1/health`
No auth required.
```json
{
  "status": "ok",
  "inference_ready": true,
  "load_error": null,
  "versions": {"numpy": "...", "sklearn": "...", "xgboost": "..."},
  "product": "SisFall_dataset_monitoring",
  "credential_rules_version": "20260203-name-slug-username"
}
```

#### `GET /api/v1/inference/status`
No auth. Returns 503 if inference not loaded.
```json
{
  "loaded": true,
  "schema_version": "2.0",
  "enhanced_feature_dim": 144,
  "fall_type_raw_dim": 263,
  "fall_type_enabled": false,
  "fall_threshold": 0.55,
  "model_root": "/path/to/models"
}
```

---

### Authentication

#### `POST /api/v1/auth/caregiver/signup`
Body: `{ full_name, email, password (min 6 chars) }`  
Validates email format. Rejects duplicate email (400).  
Returns: `{ access_token, token_type: "bearer", caregiver: {id, full_name, email} }`

#### `POST /api/v1/auth/caregiver/login`
Body: `{ email, password }`  
Returns: `{ access_token, token_type, caregiver: {id, full_name, email} }`

#### `POST /api/v1/auth/admin/login`
Body: `{ email, password }`  
Returns: `{ access_token, token_type, role: "admin", email }`

#### `POST /api/v1/auth/elder/login`
Body: `{ username, password }`  
Returns: `{ access_token, token_type, role: "elder", user_id, patient_id, display_name }`  
Raises 409 if elder login exists but has no linked patient record.

#### `POST /api/v1/auth/patient/signup`
Self-registration by patient (no caregiver required).  
Body: `{ full_name, username, password, age?, email? }`  
Creates both an elder user and a linked patient row.  
Email defaults to `{username}@patients.local` if not provided.  
Returns: `{ access_token, role: "elder", user_id, patient_id, display_name }`

#### `POST /api/v1/auth/caregiver/patient-credentials`
Caregiver enrolls a patient and gets back auto-generated elder login credentials.  
Body (inline, not header): `{ caregiver_token, full_name, age?, home_address, emergency_contact?, notes? }`  
- Generates unique elder username (slug → slug_2 … slug_9999 → random)
- Generates temp password: `{slug}_{3letters}{3digits}`
- Creates patient row, elder user row, links both, inserts caregiver_patient
- Also ticks fall escalation check

Returns: `{ patient_id, patient_name, home_address, username, temporary_password }`

---

### Patient Management

#### `POST /api/v1/patients`
Auth: optional (caregiver token → links patient to that caregiver)  
Body: `{ full_name, age? }`  
Returns: `{ id, full_name, age }`

#### `GET /api/v1/patients/{patient_id}`
No auth. Returns `{ id, full_name, age }` or 404.

#### `GET /api/v1/caregiver/my-patients`
Auth: caregiver JWT required.  
Returns patients where `caregiver_id = claims.sub` (ordered by name).

#### `DELETE /api/v1/caregiver/my-patients/{patient_id}`
Auth: caregiver JWT. Must own the patient (via `caregiver_id` or `caregiver_patient`).  
Cascade deletes: alerts, fall_incidents, sessions, devices, patient_live, caregiver_patient, patient row, linked elder user.

---

### Device & Session Management

#### `POST /api/v1/devices`
Body: `{ label, platform?, patient_id (required), owner_name? }`  
Returns: `{ id, label, platform }`  
Raises 422 if `patient_id` missing.

#### `GET /api/v1/devices/{device_id}`
Returns device or 404.

#### `POST /api/v1/sessions`
Body: `{ patient_id, device_id, sample_rate_hz?, started_by? }`  
Creates session with `status="active"`. Validates patient exists.  
Returns: `{ id, patient_id, device_id, status, sample_rate_hz }`

#### `POST /api/v1/sessions/{session_id}/stop`
Body: `{ stopped_by?, note? }`  
Sets `status="stopped"`, `stopped_at=now`.

---

### Sensor Ingest & Inference

#### `POST /api/v1/ingest/live`

The primary real-time endpoint. Accepts a batch of IMU samples, runs ML inference, creates alerts, and upserts `patient_live`.

**Body:**
```json
{
  "patient_id": "str",
  "device_id": "str",
  "session_id": "str",
  "source": "flutter_mobile",
  "sampling_rate_hz": 50.0,
  "acceleration_unit": "m_s2",
  "gyroscope_unit": "rad_s",
  "battery_level": null,
  "samples": [
    {
      "timestamp_ms": 1234567890,
      "acc_x": 0.1, "acc_y": 9.8, "acc_z": 0.2,
      "gyro_x": 0.01, "gyro_y": 0.02, "gyro_z": 0.01,
      "azimuth": null, "pitch": null, "roll": null
    }
  ]
}
```

**Processing flow:**
1. Convert samples → 144-D feature vector + (300,3) raw windows via `samples_to_feature_vector()`
2. Log raw sensor preview (if `EMS_DEBUG_SENSOR_LOGS=1`)
3. Run `run_inference()` if ML artifacts loaded:
   - ADL branch → smooth via per-patient `VoteBuffer`
   - Fall branch → reset VoteBuffer, set `inferred_activity` to fall-type label
4. On `run_inference()` failure → heuristic fallback (acc magnitude / rule-based activity)
5. `build_detection_payload()` → severity mapping with stationary motion guard
6. If `severity == "fall_detected"` AND `p_fall >= thr` AND no open fall_incident:
   - Insert new alert row
   - Insert new fall_incident row (stage=`awaiting_response`)
   - Schedule deadline: response=NOW+30s, alarm=NOW+90s
   - Queue WebSocket push to caregiver (background task)
7. Tick fall escalations (`tick_fall_escalations()`)
8. Upsert `patient_live` row (full live state)

**Threshold:** `DETECTOR_CFG["fall_score"]` (default 0.80, live-adjustable via `PUT /api/v1/detector/config`)

**Returns:**
```json
{
  "ingested_samples": 128,
  "detection": {
    "severity": "low",
    "score": 0.12,
    "fall_probability": 0.08,
    "fall_probability_ml": 0.08,
    "predicted_activity_class": "Walking",
    "peak_acc_g": 1.1,
    "peak_gyro_dps": 45.2,
    "peak_jerk_g_per_s": 0.3,
    "stillness_ratio": 0.85,
    "samples_analyzed": 128,
    "message": "Fall risk 0.08 (low). Activity hint: Walking",
    "reasons": ["ml_stack"]
  },
  "live_status": { ... },
  "active_alert": null,
  "telemetry": { ... }
}
```

#### `POST /api/v1/inference/motion`

Direct inference endpoint (bypasses alert creation). Accepts pre-computed enhanced features OR raw windows.

**Body:**
```json
{
  "enhanced_features": [144 floats],
  "fall_type_features": null,
  "predict_fall_type": true,
  "acc_window": null,
  "gyro_window": null,
  "ori_window": null
}
```

- If `acc_window` provided: server rebuilds 144-D features server-side (training parity)
- If `enhanced_features` dimension mismatches and `acc_window` present: auto-rebuilds (backward compatibility)
- Returns `MotionInferenceResponse` matching `run_inference()` output schema

**Response model:**
```python
class MotionInferenceResponse(BaseModel):
    is_fall: bool
    fall_probability: float
    fall_threshold: float
    schema_version: str
    branch: str
    activity_class_index: int | None
    activity_label: str | None
    fall_type_code: str | None
    fall_type_label: str | None
    fall_type_class_index: int | None
    fall_type_skipped_reason: str | None
```

---

### Alert Management

#### `GET /api/v1/alerts`
Query params: `status` (open/acknowledged/resolved), `patient_id`  
Auth: optional. Caregiver token restricts to own patients.  
Returns: last 200 alerts ordered by `created_at DESC`.  
Each alert includes `alarm_eligible` (bool): true iff severity=fall_detected AND not resolved AND message is not a "no elder response" timeout escalation.

#### `POST /api/v1/alerts/manual`
Auth: `elder` or `caregiver` JWT required (role-checked, ownership verified).  
Body: `{ patient_id, device_id?, session_id?, severity?, message?, actor? }`  
Creates alert with `score=1.0`, `manually_triggered=1`. Broadcasts to caregiver WebSocket.

#### `POST /api/v1/alerts/{alert_id}/acknowledge`
Body: `{ actor?, note? }`  
Sets `status="acknowledged"`, `acknowledged_at=now`.

#### `POST /api/v1/alerts/{alert_id}/resolve`
Body: `{ actor?, note? }`  
Sets `status="resolved"`, `resolved_at=now`.

---

### Fall Escalation State Machine

`tick_fall_escalations(conn)` is called on every alert list, ingest, and credential endpoint. It checks open `fall_incidents` and advances their stage:

```
[awaiting_response]
    ↓ response_deadline_at exceeded (30s)
[alarm_local]
    ↓ alarm_deadline_at exceeded (90s)
[emergency]
    → NEW alert inserted: severity="fall_detected", score=1.0
      message="EMERGENCY: elder did not confirm after alarm — escalate to caretaker."
```

- Stage `awaiting_response → alarm_local`: no new alert (avoids labeling non-response as additional risk)
- Stage `alarm_local → emergency`: inserts new `fall_detected` alert into database

**Alarm eligibility filter** (`_is_alarm_eligible_alert`):
```python
# An alert is NOT alarm-eligible if:
# - status == "resolved"
# - severity != "fall_detected"
# - message contains: "did not confirm", "no elder response", "no elder...response", "emergency: elder"
```
This prevents timeout-escalation alerts from triggering the caregiver alarm sound again.

---

### Live Monitoring

#### `GET /api/v1/monitor/patients/live`
Auth: caregiver → own patients only; admin/none → all patients.  
Returns `patient_live` rows with full live state including GPS.

Each row:
```json
{
  "patient_id": "...",
  "patient_name": "...",
  "session_id": "...",
  "device_id": "...",
  "severity": "low",
  "score": 0.05,
  "fall_probability": 0.04,
  "predicted_activity_class": "Walking",
  "last_message": "...",
  "sample_rate_hz": 50.0,
  "latest_metrics": {},
  "active_alert_ids": [],
  "latitude": 33.7215,
  "longitude": 73.0433,
  "location_accuracy_m": 12.0,
  "location_updated_at": "2026-05-10T...",
  "heading_degrees": 180.0
}
```

#### `GET /api/v1/summary`
Auth: caregiver → own patients; others → global.  
Returns: `{ total_patients, active_sessions, open_alerts, last_event_at }`

#### `POST /api/v1/patients/me/location`
Auth: `elder` JWT required.  
Body: `{ latitude [-90,90], longitude [-180,180], accuracy_m?, heading_degrees? }`  
Upserts GPS coordinates in `patient_live` for the elder's linked patient.  
Returns: `{ ok: true, patient_id, location_updated_at }`

---

### Detector Configuration

#### `PUT /api/v1/detector/config`
No auth. Live-adjusts the caregiver sensitivity thresholds.  
Body: `{ medium_risk_score, high_risk_score, fall_score }`  
Modifies in-memory `DETECTOR_CFG` dict (resets on server restart).

Defaults:
```python
DETECTOR_CFG = {
    "medium_risk_score": 0.35,
    "high_risk_score": 0.58,
    "fall_score": 0.80,
}
```

---

### Fall Feedback

#### `POST /api/v1/events/fall-feedback`
No auth. Elder submits feedback on detected falls.  
Body: `FallFeedbackEvent` (Pydantic):
```python
class FallFeedbackEvent(BaseModel):
    alert_id: str
    patient_id: str
    event_type: str  # okay, need_help, false_alarm, wrong_fall_type,
                     # correct_fall_type, no_help_needed
    fall_type_code: str | None
    notes: str | None
    occurred_at: str  # ISO 8601
```

Side effects:
1. Inserts into `app_events` table (event_type=`fall_feedback`)
2. Appends to JSONL file: `data/feedback/fall_events.jsonl`

Returns: `FallFeedbackAck { ok: true, logged_at: str }`

---

### Admin Endpoints

All require `admin` JWT.

#### `GET /api/v1/admin/dashboard`
Returns system-wide counts:
```json
{
  "schema": "sisfall_monitoring_v1",
  "caretakers": 3,
  "elders_registered": 12,
  "patients": 12,
  "open_alerts": 2,
  "fall_feedback_events": 45,
  "datasets": ["SisFall", "MobiAct"]
}
```

#### `GET /api/v1/admin/caregivers`
Returns all caregivers: `[{id, email, full_name, created_at}]` ordered by `created_at DESC`

#### `POST /api/v1/admin/caregivers`
Body: `{ full_name, email, password }`  
Creates caregiver user. Rejects duplicate email.

#### `DELETE /api/v1/admin/caregivers/{user_id}`
Cascade deletes: all patients of that caregiver → cascade each patient (alerts, incidents, sessions, devices, elder user), then caregiver_patient rows, then the caregiver user.

#### `GET /api/v1/admin/patients`
Returns all patients joined with elder username:
```json
[{
  "id", "full_name", "age", "caregiver_id",
  "elder_user_id", "elder_username"
}]
```

#### `POST /api/v1/admin/patients`
Body: `{ full_name, age?, caregiver_id? }`  
Creates patient; if `caregiver_id` given, validates it exists and inserts `caregiver_patient`.

#### `DELETE /api/v1/admin/patients/{patient_id}`
Cascade deletes patient and all dependents.

---

### WebSocket

#### `WS /api/v1/ws/caregiver?token=<JWT>`

Caregiver subscribes for real-time alert push.

**Authentication:**
- `token` query parameter (JWT)
- Closes with code 4401 if token invalid
- Closes with code 4403 if token role ≠ `caregiver`

**Message format received by client:**
```json
{
  "type": "alert",
  "data": {
    "id": "...",
    "patient_id": "...",
    "severity": "fall_detected",
    "status": "open",
    "message": "...",
    "score": 0.92,
    "created_at": "...",
    "manually_triggered": false,
    "alarm_eligible": true
  }
}
```

**Lifecycle:**
1. `hub.register(caregiver_id, ws)` — accept + add to room
2. Loop: `await websocket.receive_text()` (keeps connection alive)
3. On disconnect: `hub.unregister(caregiver_id, ws)`

**Fanout implementation** (`realtime_hub.py`):
```python
class CaregiverRealtimeHub:
    _rooms: dict[str, set[WebSocket]]  # keyed by caregiver_id
    _lock: asyncio.Lock

    async def register(caregiver_id, ws)     # accept + add to room
    async def unregister(caregiver_id, ws)   # remove from room, clean empty rooms
    async def broadcast_to_caregiver(caregiver_id, payload)
        # Sends JSON to all WebSocket connections for that caregiver
        # Auto-unregisters disconnected clients on send error
```

---

## Elder Credential Allocator

**File:** [`flask_backend/app/elder_credential_allocator.py`](flask_backend/app/elder_credential_allocator.py)

Rules version: `20260203-name-slug-username` (exposed on `/api/v1/health`)

### Username Generation

```python
def elder_name_slug(full_name) -> str:
    # "John Smith" → "john", "Mary-Jane" → "maryjane"
    # Takes first word, strips non-alphanumeric, max 32 chars

def pick_unique_elder_username_for_patient(c, full_name) -> str:
    # Try: slug → slug_2 → slug_3 … slug_9999
    # Then: 64 random attempts of 5-letter + 5-digit tokens
    # Raises ElderUsernameAllocationFailed if all taken
```

### Password Generation

```python
def temporary_password_for_patient(name_slug) -> str:
    # Format: {slug}_{3_random_letters}{3_random_digits}
    # Example: "john_abc123"
    # Uses secrets module (cryptographically secure)
```

---

## Heuristic Fallback (No ML)

When `_state["art"]` is None (artifacts failed to load) or `run_inference()` raises:

### Fall Probability
```python
p_fall = min(1.0, max(0.0, np.max(acc_magnitudes) / 25.0))
```
Max magnitude normalized to 25 m/s² (≈2.55 G), clamped to [0, 1].

### Activity Label
```python
if peak_mag > 22.0 or std_mag > 3.5 or peak_gyro > 4.0:  → "running"
if std_mag < 0.30 and 8.5 <= mean_mag <= 10.8 and peak_gyro < 0.8:  → "standing"
if std_mag < 0.55 and peak_gyro < 1.2:  → "sitting"
if std_mag < 2.0:  → "walking"
else:  → "moving"
```

---

## Pydantic Request/Response Models

### Ingest Body Models (`monitoring_routes.py`)

```python
class SamplePayload(BaseModel):
    timestamp_ms: int
    acc_x: float; acc_y: float; acc_z: float
    gyro_x: float; gyro_y: float; gyro_z: float
    azimuth: float | None = None
    pitch: float | None = None
    roll: float | None = None

class IngestLiveBody(BaseModel):
    patient_id: str; device_id: str; session_id: str
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

class PatientLocationBody(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accuracy_m: float | None = Field(default=None, ge=0)
    heading_degrees: float | None = None
```

### Inference Schemas (`schemas_motion.py`)

```python
class MotionInferenceRequest(BaseModel):
    enhanced_features: list[float]     # exactly 144 floats
    fall_type_features: list[float] | None = None  # 263 floats if provided
    predict_fall_type: bool = True
    acc_window: list[list[float]] | None = None    # (128 or 300, 3)
    gyro_window: list[list[float]] | None = None
    ori_window: list[list[float]] | None = None
```

### Fall Feedback Schemas (`schemas_fall_feedback.py`)

```python
class FallFeedbackEvent(BaseModel):
    alert_id: str
    patient_id: str
    event_type: str  # okay | need_help | false_alarm | wrong_fall_type |
                     # correct_fall_type | no_help_needed
    fall_type_code: str | None = None
    notes: str | None = None
    occurred_at: str  # ISO 8601

class FallFeedbackAck(BaseModel):
    ok: bool = True
    logged_at: str  # ISO 8601 UTC
```

---

## ADL & Fall-Type Label Humanization

**Source:** `monitoring_routes.py`

### Fall-Type Codes → Human Names
| Code | Display Name |
|---|---|
| FOL | Forward lying fall |
| FKL | Front knees lying fall |
| BSC | Backward sitting-chair fall |
| SDL | Sideward lying fall |

### ADL Codes → Human Names
| Code | Display Name |
|---|---|
| STD | Standing |
| WAL | Walking |
| JOG | Jogging |
| JUM | Jumping |
| STU | Stairs Up |
| STN | Stairs Down |
| SCH | Sit to Stand |
| SIT | Sitting |
| CHU | Stand to Sit |
| CSI | Car Step In |
| CSO | Car Step Out |
| LYI | Lying |

### ADL Index → Code Mapping (legacy encoder fallback)
```python
_ADL_INDEX_TO_CODE = {
    0: "CHU", 1: "CSI", 2: "CSO", 4: "JOG", 5: "JUM",
    6: "LYI", 7: "SCH", 8: "SIT", 9: "STD", 10: "STN",
    11: "STU", 12: "WAL"
}
```

---

## Authorization Rules

### Manual Alerts
```python
def _assert_manual_alert_authorized(body, authorization):
    # elder: patient must be linked to their elder_user_id
    # caregiver: patient must be in their caregiver_id or caregiver_patient
    # admin: not allowed (raises 403)
```

### Caregiver Patient Collection
```python
def _collect_patient_ids_for_caregiver(c, caregiver_id) -> list[str]:
    # Collects from both patients.caregiver_id AND caregiver_patient table
    # Deduplicates; used for scoped alert/summary/live queries
```

### Cascade Delete
```python
def _delete_patient_cascade(c, patient_id) -> bool:
    # Deletes in order:
    # alerts → fall_incidents → sessions → devices → patient_live
    # → caregiver_patient → patients → elder user (if linked)
    # Returns False if patient not found
```

---

## Concurrency & State

### Per-Patient Vote Buffers
```python
_patient_vote_buffers: dict[str, VoteBuffer] = {}
```
In-memory, keyed by `patient_id`. Created on first ingest; reset on fall; cleared on server restart.

### Inference Runtime Injection
```python
_RUNTIME: dict = {}

def set_inference_runtime(state: dict) -> None:
    # Called by main.py lifespan with {"art": InferenceArtifacts | None, "load_error": str | None}

def _get_art() -> InferenceArtifacts | None:
    return _RUNTIME.get("art")
```

Avoids circular imports between `main.py` and `monitoring_routes.py`.

### Detector Config (Live-Adjustable)
```python
DETECTOR_CFG = {
    "medium_risk_score": 0.35,
    "high_risk_score": 0.58,
    "fall_score": 0.80,
}
```
Modified by `PUT /api/v1/detector/config`. Falls back to manifest threshold if `fall_score <= 0`.

---

## Sensor Preprocessing in Backend

**File:** [`flask_backend/app/ml_bridge.py`](flask_backend/app/ml_bridge.py)

### samples_to_feature_vector()
```
samples: list[dict]  →  (feat_144, acc_300, gyro_300, ori_300)

1. Extract acc (n,3), gyro (n,3), ori (n,3) from sample dicts
2. Linear resample → 128 rows (WINDOW_ENHANCED)
3. extract_enhanced_features() → (1, 144) → feat_144[0]
4. Linear resample → 300 rows (WINDOW_FALL_TYPE)
   → acc_300, gyro_300, ori_300  (for optional fall-type branch)
```

### build_enhanced_features_numpy()
```
acc (n,3), gyro (n,3)|None, ori (n,3)|None
→ validate shapes
→ linear resample each to 128 rows
→ extract_enhanced_features() → (144,) vector
```

### acc_gyro_ori_to_window_lists()
```
Converts numpy arrays → nested Python lists (for JSON serialization or run_inference() call)
```

---

## Testing

### Backend Tests (`flask_backend/tests/test_pipeline.py`)

10 tests covering:
1. Feature extraction produces exactly 144 features per window
2. Gravity separation (low-pass) and jerk signals are finite
3. Models load from `inference_manifest.json`
4. ADL scaler/model dimensions match manifest (144)
5. Fall scaler/model dimensions match manifest (144)
6. `run_inference()` returns correct schema for ADL window
7. `run_inference()` returns `is_fall=True` for synthetic high-impact window
8. `samples_to_feature_vector()` produces (144,) from raw sample dicts
9. `build_enhanced_features_numpy()` produces (144,) from (128,3) arrays
10. `VoteBuffer` stabilises ADL predictions and resets on fall

### Project-Level Tests (`tests/`)

| File | What it tests |
|---|---|
| `test_feature_dimensions.py` | 144-D feature shape + finite values |
| `test_fall_detection_fall_type_models.py` | Fall + fall-type model end-to-end |
| `test_elder_credentials.py` | Username slug, uniqueness, temp password format |
| `test_inference_http.py` | HTTP inference endpoint (httpx client) |
| `test_motion_artifacts.py` | Artifact loading + inference on synthetic windows |
| `test_baseline_adl_unit.py` | ADL feature extraction unit tests |
| `test_baseline_fall_imports.py` | Fall detection module imports |
| `test_baseline_falltype_imports.py` | Fall-type module imports |
| `conftest.py` | Shared fixtures |

### Running Tests
```bash
# All tests
pytest

# Backend pipeline tests only
pytest flask_backend/tests/

# Feature dimension tests
pytest tests/test_feature_dimensions.py

# With verbose output
pytest -v --tb=short
```

**`pytest.ini`:** Located at repo root, sets Python path for all test discovery.

---

## Key Environment Variables Summary

| Variable | Default | Security Note |
|---|---|---|
| `JWT_SECRET` | `sisfall-dev-change-me-in-production` | **Must override in production** |
| `ADMIN_PASSWORD` | `admin123` | **Must override in production** |
| `ADMIN_EMAIL` | `admin@local` | Change for production |
| `MODEL_ROOT` | `<repo>/flask_backend/models` | Override to mount models externally |
| `REPO_ROOT` | Auto-detected from `__file__` | Override in containers |
| `INFERENCE_MANIFEST` | `<MODEL_ROOT>/inference_manifest.json` | Override to use different manifest |
| `FALL_RESPONSE_DEADLINE_SEC` | `30` | Seconds until `awaiting_response → alarm_local` |
| `FALL_EMERGENCY_DEADLINE_SEC` | `90` | Seconds until `alarm_local → emergency` |
| `EMS_DEBUG_SENSOR_LOGS` | `1` | Set to `0` in production to silence sensor noise |
| `OMP_NUM_THREADS` | `1` (set at startup) | Prevent ML thread over-subscription |
| `OPENBLAS_NUM_THREADS` | `1` (set at startup) | Same |

---

## Data Flow: End to End

```
Flutter App
    │  POST /api/v1/ingest/live
    │  { patient_id, device_id, session_id, samples: [...128 rows...] }
    ▼
monitoring_routes.ingest_live()
    │
    ├─ samples_to_feature_vector(samples)
    │    └─ parse acc/gyro/ori → (n,3) arrays
    │    └─ resample → 128 rows (acc_e, gyro_e, ori_e)
    │    └─ extract_enhanced_features() → feat_144 (144-D)
    │    └─ resample → 300 rows → acc_300, gyro_300, ori_300
    │
    ├─ run_inference(art, feat_144.tolist(), None,
    │                predict_fall_type=True,
    │                acc_window=acc_300, gyro_window=gyro_300, ori_window=ori_300)
    │    │
    │    ├─ Stage 1: fall_scaler.transform(x) → fall_model.predict_proba
    │    │           p_fall = proba[1] ; is_fall = (p_fall >= 0.55)
    │    │
    │    ├─ Stage 2a (not fall): adl_scaler.transform → adl_model.predict
    │    │                        adl_encoder.inverse_transform → "WAL"
    │    │
    │    └─ Stage 2b (fall): extract 263-D from acc_300/gyro_300/ori_300
    │                         → scaler → select 150 features
    │                         → fall_type_model.predict → "FOL"
    │
    ├─ VoteBuffer.push(activity_label, is_fall=False) → smoothed label
    │
    ├─ build_detection_payload(samples, p_fall, activity, ml_ok, threshold)
    │    ├─ simple_signal_metrics(samples) → peak_acc_g, peak_gyro_dps, stillness
    │    ├─ _effective_fall_probability() → dampen if stationary
    │    ├─ _severity_from_fall_prob() → "low"/"medium"/"high_risk"/"fall_detected"
    │    └─ Impact evidence guard (downgrade if no physical impact signature)
    │
    ├─ If fall_detected + p_fall >= threshold + no open incident:
    │    ├─ INSERT alerts (severity="fall_detected", status="open")
    │    ├─ INSERT fall_incidents (stage="awaiting_response",
    │    │                         response_deadline=NOW+30s, alarm_deadline=NOW+90s)
    │    └─ background_tasks: broadcast_to_caregiver() via WebSocket
    │
    ├─ UPSERT patient_live (full live state)
    │
    └─ Return { ingested_samples, detection, live_status, active_alert, telemetry }

Caregiver App (WebSocket /api/v1/ws/caregiver?token=<JWT>)
    ← { "type": "alert", "data": { severity, score, message, alarm_eligible, ... } }
```

---

## Dependencies

**File:** `flask_backend/requirements.txt`

| Package | Version | Purpose |
|---|---|---|
| fastapi | ≥0.110, <1 | Web framework + routing |
| uvicorn[standard] | ≥0.27, <1 | ASGI server + WebSocket |
| pydantic | ≥2.5, <3 | Request/response validation |
| pyjwt | ≥2.8, <3 | JWT encoding/decoding |
| bcrypt | ≥4.1, <6 | Password hashing |
| numpy | ≥1.26, <3 | Feature math |
| scikit-learn | ≥1.6.1, <2 | Scaler, encoder, model inference |
| xgboost | ≥2.0, <4 | Primary classifier |
| lightgbm | ≥4.0, <5 | Optional comparison classifier |
| joblib | ≥1.3, <2 | Model deserialization |
| scipy | ≥1.11, <2 | Fall-type Butterworth filter (optional branch) |
