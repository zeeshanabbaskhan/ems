"""Convert ingest samples → 128-D features (Colab `WINDOW_SIZE=128` @ 50 Hz) + 300×3 windows for optional fall-type."""

from __future__ import annotations

from typing import Any

import numpy as np

from flask_backend.app.motion_enhanced_features import extract_enhanced_features

# Must match MobiAct Colab training / Flutter `MotionFeatureExtractor.windowLength`.
_WINDOW_ENHANCED = 128
# Fall-type pipeline (`scripts/baseline_falltype`) expects ~6 s @ 50 Hz.
_WINDOW_FALL_TYPE = 300


def _resample_rows(data: np.ndarray, target_len: int) -> np.ndarray:
    """data: (n, 3) -> (target_len, 3)"""
    n = data.shape[0]
    if n == target_len:
        return data
    if n < 2:
        return np.zeros((target_len, 3), dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, 3), dtype=np.float64)
    for j in range(3):
        out[:, j] = np.interp(x_new, x_old, data[:, j])
    return out


def _sample_ori_degrees(s: dict[str, Any]) -> tuple[float, float, float]:
    """MobiAct ori columns: azimuth (z), pitch (x), roll (y) in degrees — optional per axis."""

    def g(key: str) -> float:
        v = s.get(key)
        if v is None:
            return 0.0
        return float(v)

    return g("azimuth"), g("pitch"), g("roll")


def samples_to_feature_vector(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns enhanced 128-D vector (same length scale as training), plus acc/gyro/ori resampled
    to (300,3) for the optional 263-D fall-type branch when artifacts are present.

    Orientation defaults to zeros when [azimuth, pitch, roll] are absent (degrees).
    """
    if not samples:
        raise ValueError("empty samples")
    n = len(samples)
    acc = np.zeros((n, 3), dtype=np.float64)
    gyro = np.zeros((n, 3), dtype=np.float64)
    ori = np.zeros((n, 3), dtype=np.float64)
    for i, s in enumerate(samples):
        acc[i, 0] = float(s.get("acc_x", 0.0))
        acc[i, 1] = float(s.get("acc_y", 0.0))
        acc[i, 2] = float(s.get("acc_z", 0.0))
        gyro[i, 0] = float(s.get("gyro_x", 0.0))
        gyro[i, 1] = float(s.get("gyro_y", 0.0))
        gyro[i, 2] = float(s.get("gyro_z", 0.0))
        az, pit, rol = _sample_ori_degrees(s)
        ori[i, 0] = az
        ori[i, 1] = pit
        ori[i, 2] = rol

    acc_e = _resample_rows(acc, _WINDOW_ENHANCED)
    gyro_e = _resample_rows(gyro, _WINDOW_ENHANCED)
    ori_e = _resample_rows(ori, _WINDOW_ENHANCED)

    xb = acc_e[np.newaxis, ...]
    yb = gyro_e[np.newaxis, ...]
    zb = ori_e[np.newaxis, ...]
    feat = extract_enhanced_features(xb, yb, zb)

    acc_300 = _resample_rows(acc, _WINDOW_FALL_TYPE)
    gyro_300 = _resample_rows(gyro, _WINDOW_FALL_TYPE)
    ori_300 = _resample_rows(ori, _WINDOW_FALL_TYPE)

    return feat[0], acc_300, gyro_300, ori_300


def acc_gyro_ori_to_window_lists(
    acc: np.ndarray, gyro: np.ndarray, ori: np.ndarray
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    return acc.tolist(), gyro.tolist(), ori.tolist()
