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
# A "true" fall should usually show at least one noticeable impact/rotation cue.
FALL_MIN_PEAK_ACC_G = 1.35
FALL_MIN_PEAK_GYRO_DPS = 120.0


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
    peak_gyro_dps = max(peaks_gyro) if peaks_gyro else 0.0
    peak_jerk = max(jerks) if jerks else 0.0
    stillness = float(np.std(np.asarray(mags))) if mags else 0.0
    stillness_ratio = max(0.0, min(1.0, 1.0 - stillness / max(np.mean(mags), 1e-6)))
    return {
        "peak_acc_g": peak_acc_g,
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
    severity = _severity_from_fall_prob(fall_probability, threshold)
    if severity == "fall_detected":
        has_impact_evidence = (
            sig["peak_acc_g"] >= FALL_MIN_PEAK_ACC_G
            or sig["peak_gyro_dps"] >= FALL_MIN_PEAK_GYRO_DPS
        )
        if not has_impact_evidence:
            # Keep elevated risk visible, but suppress hard fall alert.
            severity = "high_risk"
    score = max(fall_probability, sig["peak_acc_g"] / 5.0 * 0.3 + fall_probability * 0.7)
    reasons = []
    if ml_ok:
        reasons.append("ml_stack")
    else:
        reasons.append("heuristic_fallback")
    if sig["peak_acc_g"] > 2.5:
        reasons.append("high_peak_acceleration")
    if (
        fall_probability >= threshold
        and sig["peak_acc_g"] < FALL_MIN_PEAK_ACC_G
        and sig["peak_gyro_dps"] < FALL_MIN_PEAK_GYRO_DPS
    ):
        reasons.append("fall_suppressed_low_impact_motion")
    msg = (
        f"Fall risk {fall_probability:.2f} ({severity}). "
        + (f"Activity hint: {inferred_activity}" if inferred_activity else "")
    ).strip()
    return {
        "severity": severity,
        "score": min(1.0, float(score)),
        "fall_probability": float(fall_probability),
        "predicted_activity_class": inferred_activity,
        "frailty_proxy_score": None,
        "gait_stability_score": None,
        "movement_disorder_score": None,
        "peak_acc_g": float(sig["peak_acc_g"]),
        "peak_gyro_dps": float(sig["peak_gyro_dps"]),
        "peak_jerk_g_per_s": float(sig["peak_jerk"]),
        "stillness_ratio": float(sig["stillness"]),
        "samples_analyzed": len(samples),
        "message": msg,
        "reasons": reasons,
    }
