"""128-D multi-sensor features for MobiAct ADL/fall pipelines.

Feature order matches the training/inference notebook contract:
- Accelerometer: 45 time + 3 corr + 5 magnitude + 6 frequency = 59
- Gyroscope:     45 time + 3 corr + 5 magnitude + 6 frequency = 59
- Orientation:   mean/std/range per axis + azimuth MRL = 10
Total = 128 features.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


def extract_enhanced_features(
    acc_windows: np.ndarray,
    gyro_windows: np.ndarray | None = None,
    ori_windows: np.ndarray | None = None,
) -> np.ndarray:
    n = len(acc_windows)
    if gyro_windows is None:
        gyro_windows = np.zeros((n, acc_windows.shape[1], 3), dtype=np.float64)
    if ori_windows is None:
        ori_windows = np.zeros((n, acc_windows.shape[1], 3), dtype=np.float64)

    features: list[list[float]] = []

    for idx in tqdm(range(n), desc="Enhanced features"):
        acc = np.asarray(acc_windows[idx], dtype=np.float64)
        gyro = np.asarray(gyro_windows[idx], dtype=np.float64)
        ori = np.asarray(ori_windows[idx], dtype=np.float64)
        feat: list[float] = []

        feat.extend(_time_domain_features(acc))
        feat.extend(_cross_axis_correlations(acc))
        feat.extend(_magnitude_features(acc))
        feat.extend(_frequency_domain_features(acc))

        feat.extend(_time_domain_features(gyro))
        feat.extend(_cross_axis_correlations(gyro))
        feat.extend(_magnitude_features(gyro))
        feat.extend(_frequency_domain_features(gyro))

        feat.extend(_orientation_stats(ori))

        features.append(feat)

    return np.asarray(features, dtype=np.float64)


def _time_domain_features(data: np.ndarray) -> list[float]:
    out: list[float] = []
    for axis in range(data.shape[1]):
        x = data[:, axis]
        out.extend(
            [
                float(np.mean(x)),
                float(np.std(x)),
                float(np.median(x)),
                float(np.min(x)),
                float(np.max(x)),
                float(np.ptp(x)),
                float(np.percentile(x, 5)),
                float(np.percentile(x, 25)),
                float(np.percentile(x, 75)),
                float(np.percentile(x, 95)),
                float(np.sqrt(np.mean(x**2))),
                float(np.mean(np.abs(np.diff(x)))),
                float(np.sum(np.abs(np.diff(x)))),
                float(np.var(x)),
                float(np.sum(x**2) / len(x)),
            ]
        )
    return out


def _cross_axis_correlations(data: np.ndarray) -> list[float]:
    out: list[float] = []
    for i_idx, j_idx in [(0, 1), (0, 2), (1, 2)]:
        c = float(np.corrcoef(data[:, i_idx], data[:, j_idx])[0, 1])
        out.append(0.0 if np.isnan(c) else c)
    return out


def _magnitude_features(data: np.ndarray) -> list[float]:
    mag = np.sqrt(np.sum(data**2, axis=1))
    return [
        float(np.mean(mag)),
        float(np.std(mag)),
        float(np.max(mag)),
        float(np.percentile(mag, 95)),
        float(np.sum(mag)),
    ]


def _frequency_domain_features(data: np.ndarray, fs: float = 50.0) -> list[float]:
    out: list[float] = []
    freqs = np.fft.rfftfreq(data.shape[0], d=1.0 / fs)
    for axis in range(data.shape[1]):
        fft_mag = np.abs(np.fft.rfft(data[:, axis]))
        if fft_mag.size <= 1:
            out.extend([0.0, 0.0])
            continue
        dom_idx = int(np.argmax(fft_mag[1:]) + 1)
        dom_freq = float(freqs[dom_idx])
        spectral_energy = float(np.sum(fft_mag**2) / len(fft_mag))
        out.extend([dom_freq, spectral_energy])
    return out


def _orientation_stats(data: np.ndarray) -> list[float]:
    out: list[float] = []
    for axis in range(data.shape[1]):
        x = data[:, axis]
        out.extend([float(np.mean(x)), float(np.std(x)), float(np.ptp(x))])
    azimuth_rad = np.deg2rad(data[:, 0])
    mrl = float(np.sqrt(np.mean(np.cos(azimuth_rad)) ** 2 + np.mean(np.sin(azimuth_rad)) ** 2))
    out.append(mrl)
    return out


ENHANCED_FEATURE_DIM = 128
