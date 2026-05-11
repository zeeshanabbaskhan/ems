"""Convert ingest samples → 144-D features (v2 training parity, WINDOW_SIZE=128 @ 50 Hz).

Also provides ``VoteBuffer`` for sliding-window ADL majority-vote stabilisation across
consecutive inference calls (falls bypass voting for immediate alert).
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

import numpy as np

from flask_backend.app.motion_enhanced_features import extract_enhanced_features

# Must match MobiAct v2 training WINDOW_SIZE.
_WINDOW_ENHANCED = 128
# Fall-type pipeline expects ~6 s @ 50 Hz.
_WINDOW_FALL_TYPE = 300

# Raised from 0.55: reduces false fall-branch routing for high-intensity activities (jogging).
# Real falls typically score 0.85+; 0.65 still catches them while routing jogging to ADL branch.
FALL_THRESHOLD_DEFAULT = 0.65

# Majority-vote window: 11 consecutive inference windows ≈ 14 s at 50% overlap.
# Raised from 7 to dampen noisy STD→WAL and WAL→JOG label flips from brief phone movement.
VOTE_BUFFER_SIZE = 11


class VoteBuffer:
    """
    Sliding majority-vote buffer for ADL predictions.

    Falls are returned immediately and reset the buffer.
    ADL labels are smoothed across the last ``size`` windows.
    """

    def __init__(self, size: int = VOTE_BUFFER_SIZE) -> None:
        self._buf: deque[str] = deque(maxlen=size)
        self._size = size

    def push(self, label: str, is_fall: bool) -> str:
        """Push a new prediction; return the (possibly smoothed) label."""
        if is_fall:
            self._buf.clear()
            return label
        self._buf.append(label)
        if len(self._buf) < self._size:
            return label
        return Counter(self._buf).most_common(1)[0][0]

    def reset(self) -> None:
        self._buf.clear()

    @property
    def confidence(self) -> float:
        """Fraction of buffer occupied by the current majority label."""
        if not self._buf:
            return 0.0
        majority = Counter(self._buf).most_common(1)[0][0]
        return Counter(self._buf)[majority] / len(self._buf)


def _resample_rows(data: np.ndarray, target_len: int) -> np.ndarray:
    """data: (n, 3) → (target_len, 3) via linear interpolation."""
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
    """MobiAct ori columns: azimuth, pitch, roll in degrees — optional per axis."""

    def g(key: str) -> float:
        v = s.get(key)
        return 0.0 if v is None else float(v)

    return g("azimuth"), g("pitch"), g("roll")


def samples_to_feature_vector(
    samples: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a list of raw sensor samples to a 144-D feature vector.

    Returns (feat_144, acc_300, gyro_300, ori_300):
      - feat_144: (144,) float64 — primary enhanced feature vector
      - acc_300 / gyro_300 / ori_300: (300, 3) resampled windows for the optional
        fall-type branch (263-D) when artifacts are present.

    Orientation defaults to zeros when azimuth/pitch/roll are absent.
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

    feat = extract_enhanced_features(
        acc_e[np.newaxis, ...],
        gyro_e[np.newaxis, ...],
        ori_e[np.newaxis, ...],
    )

    acc_300 = _resample_rows(acc, _WINDOW_FALL_TYPE)
    gyro_300 = _resample_rows(gyro, _WINDOW_FALL_TYPE)
    ori_300 = _resample_rows(ori, _WINDOW_FALL_TYPE)

    return feat[0], acc_300, gyro_300, ori_300


def build_enhanced_features_numpy(
    acc: np.ndarray,
    gyro: np.ndarray | None = None,
    ori: np.ndarray | None = None,
) -> np.ndarray:
    """
    144-D feature vector from pre-assembled (n, 3) arrays.

    Linearly resamples each modality to ``_WINDOW_ENHANCED`` rows (training parity),
    then runs ``extract_enhanced_features``.

    Shapes: acc (n, 3), gyro (n, 3)|None, ori (n, 3)|None — n ≥ 2.
    """
    acc = np.asarray(acc, dtype=np.float64)
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("acc must be (n, 3)")
    n = acc.shape[0]
    if gyro is None:
        gyro = np.zeros((n, 3), dtype=np.float64)
    else:
        gyro = np.asarray(gyro, dtype=np.float64)
        if gyro.shape != (n, 3):
            raise ValueError("gyro shape must match acc")
    if ori is None:
        ori = np.zeros((n, 3), dtype=np.float64)
    else:
        ori = np.asarray(ori, dtype=np.float64)
        if ori.shape != (n, 3):
            raise ValueError("ori shape must match acc")

    acc_e = _resample_rows(acc, _WINDOW_ENHANCED)
    gyro_e = _resample_rows(gyro, _WINDOW_ENHANCED)
    ori_e = _resample_rows(ori, _WINDOW_ENHANCED)
    feat = extract_enhanced_features(
        acc_e[np.newaxis, ...],
        gyro_e[np.newaxis, ...],
        ori_e[np.newaxis, ...],
    )
    return feat[0]


def acc_gyro_ori_to_window_lists(
    acc: np.ndarray, gyro: np.ndarray, ori: np.ndarray
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    return acc.tolist(), gyro.tolist(), ori.tolist()
