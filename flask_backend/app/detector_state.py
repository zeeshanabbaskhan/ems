"""Map ML output + heuristics to `DetectionResultModel` / live severity (server-side)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Default "medium" profile (see Flutter updateDetectorSensitivity)
MEDIUM = {
    "medium_risk_score": 0.35,
    "high_risk_score": 0.58,
    "fall_score": 0.80,
}

# Guardrail against false alarms from tiny handset motion.
# Both must clear for "fall_detected" — a real fall generates ground-impact acceleration
# AND rotational velocity from the body tumbling. Slight jolts or jogging typically
# produce one but not both simultaneously.
FALL_MIN_PEAK_ACC_G = 2.5       # raised from 1.70 — tiny jolts produce <2.5g
FALL_MIN_PEAK_GYRO_DPS = 150.0  # kept permissive so pocket-muffled falls aren't missed

# Stationary guard: quiet stance / pocket noise (phone barely moving).
_STATIONARY_GYRO_PEAK_DPS = 95.0
_STATIONARY_PEAK_ACC_G = 2.0
_STATIONARY_STILLNESS_MIN = 0.58  # slightly looser so micro-tremors are caught

# Locomotion guard: sustained rhythmic motion (walking / jogging / running).
# KEY: do NOT gate on peak_acc — vigorous jogging footstrike can exceed 4–5 g,
# which previously caused the guard to silently fail and let jogging trigger fall alerts.
# The impulse ratio (peak/mean) is the real discriminator: falls produce a single
# brief spike (ratio > 4×), whereas jogging keeps ratio ≈ 1.5–2.5× even at high peaks.
_LOCOMOTION_MIN_MEAN_ACC_G = 1.10   # sustained above-gravity = person is actively moving
_LOCOMOTION_MAX_GYRO_DPS = 260.0    # hard fall tumble usually exceeds this
_LOCOMOTION_MAX_IMPULSE_RATIO = 3.8  # raised: vigorous jogging footstrike can hit 3.0–3.5; falls stay ≈ 4+


def _effective_fall_probability(p: float, sig: dict[str, float]) -> tuple[float, bool]:
    """Blend down inflated ML fall prob when the batch looks stationary or like locomotion."""
    if p <= 0.35:
        return p, False

    # --- stationary guard (phone barely moving) ---
    looks_stationary = (
        sig["peak_gyro_dps"] < _STATIONARY_GYRO_PEAK_DPS
        and sig["peak_acc_g"] < _STATIONARY_PEAK_ACC_G
        and sig["stillness"] >= _STATIONARY_STILLNESS_MIN
    )
    if looks_stationary:
        dampened = min(p, 0.14 + p * 0.22)
        return dampened, True

    # --- locomotion guard (sustained walking / jogging / running rhythm) ---
    if p > 0.50:
        mean_acc = sig.get("mean_acc_g", 0.0)
        # Use peak_acc / max(mean_acc, 0.5) so near-zero mean doesn't inflate ratio.
        impulse_ratio = sig["peak_acc_g"] / max(mean_acc, 0.5)
        looks_like_locomotion = (
            mean_acc >= _LOCOMOTION_MIN_MEAN_ACC_G           # person is actively moving
            and sig["peak_gyro_dps"] < _LOCOMOTION_MAX_GYRO_DPS   # no violent tumble
            and impulse_ratio < _LOCOMOTION_MAX_IMPULSE_RATIO     # rhythmic, not spike
            # NOTE: peak_acc is intentionally NOT checked here — vigorous jogging
            # footstrike can exceed 4–5 g, which previously bypassed this guard.
        )
        if looks_like_locomotion:
            # Dampen aggressively: locomotion can never fire a fall alert.
            # max possible dampened value = min(1.0, 0.32 + 1.0*0.18) = 0.50 < 0.80 alert thr.
            dampened = min(p, 0.32 + p * 0.18)
            return dampened, True

    return p, False


def _severity_from_fall_prob(p: float, thr: float) -> str:
    if p >= thr:
        return "fall_detected"
    if p >= MEDIUM["high_risk_score"]:
        return "high_risk"
    if p >= MEDIUM["medium_risk_score"]:
        return "medium"
    return "low"


def simple_signal_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
    """When ML unavailable — coarse stats from raw batch."""
    if not samples:
        return {
            "peak_acc_g": 0.0,
            "peak_gyro_dps": 0.0,
            "peak_jerk": 0.0,
            "stillness": 1.0,
        }
    peaks_acc = []
    peaks_gyro = []
    prev_mag = None
    jerks = []
    mags = []
    for s in samples:
        ax, ay, az = float(s["acc_x"]), float(s["acc_y"]), float(s["acc_z"])
        gx, gy, gz = float(s["gyro_x"]), float(s["gyro_y"]), float(s["gyro_z"])
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        mags.append(mag)
        peaks_acc.append(mag / 9.80665)
        peaks_gyro.append(math.sqrt(gx * gx + gy * gy + gz * gz) * 180.0 / math.pi)
        if prev_mag is not None:
            jerks.append(abs(mag - prev_mag))
        prev_mag = mag
    peak_acc_g = max(peaks_acc) if peaks_acc else 0.0
    mean_acc_g = float(np.mean(peaks_acc)) if peaks_acc else 0.0
    peak_gyro_dps = max(peaks_gyro) if peaks_gyro else 0.0
    peak_jerk = max(jerks) if jerks else 0.0
    stillness = float(np.std(np.asarray(mags))) if mags else 0.0
    stillness_ratio = max(0.0, min(1.0, 1.0 - stillness / max(np.mean(mags), 1e-6)))
    return {
        "peak_acc_g": peak_acc_g,
        "mean_acc_g": mean_acc_g,
        "peak_gyro_dps": peak_gyro_dps,
        "peak_jerk": peak_jerk,
        "stillness": stillness_ratio,
    }


def build_detection_payload(
    *,
    samples: list[dict[str, Any]],
    fall_probability: float,
    inferred_activity: str | None,
    ml_ok: bool,
    threshold: float,
) -> dict[str, Any]:
    sig = simple_signal_metrics(samples)
    p_eff, stationary_guard = _effective_fall_probability(fall_probability, sig)
    severity = _severity_from_fall_prob(p_eff, threshold)
    if severity == "fall_detected":
        # Require BOTH high acceleration (ground impact) AND rotation (body tumbling).
        # A real fall produces both; a table drop or jogging spike produces only one.
        has_impact_evidence = (
            sig["peak_acc_g"] >= FALL_MIN_PEAK_ACC_G
            and sig["peak_gyro_dps"] >= FALL_MIN_PEAK_GYRO_DPS
        )
        if not has_impact_evidence:
            severity = "high_risk"
    score = max(p_eff, sig["peak_acc_g"] / 5.0 * 0.3 + p_eff * 0.7)
    reasons = []
    if ml_ok:
        reasons.append("ml_stack")
    else:
        reasons.append("heuristic_fallback")
    if stationary_guard:
        reasons.append("stationary_motion_guard")
    if sig["peak_acc_g"] > 2.5:
        reasons.append("high_peak_acceleration")
    if (
        p_eff >= threshold
        and sig["peak_acc_g"] < FALL_MIN_PEAK_ACC_G
        and sig["peak_gyro_dps"] < FALL_MIN_PEAK_GYRO_DPS
    ):
        reasons.append("fall_suppressed_low_impact_motion")
    msg = (
        f"Fall risk {p_eff:.2f} ({severity}). "
        + (f"Activity hint: {inferred_activity}" if inferred_activity else "")
    ).strip()
    return {
        "severity": severity,
        "score": min(1.0, float(score)),
        # UX / alerts: use calibrated prob; raw ML available as fall_probability_ml when needed.
        "fall_probability": float(p_eff),
        "fall_probability_ml": float(fall_probability),
        "predicted_activity_class": inferred_activity,
        "frailty_proxy_score": None,
        "gait_stability_score": None,
        "movement_disorder_score": None,
        "peak_acc_g": float(sig["peak_acc_g"]),
        "mean_acc_g": float(sig["mean_acc_g"]),
        "peak_gyro_dps": float(sig["peak_gyro_dps"]),
        "peak_jerk_g_per_s": float(sig["peak_jerk"]),
        "stillness_ratio": float(sig["stillness"]),
        "samples_analyzed": len(samples),
        "message": msg,
        "reasons": reasons,
    }
