"""144-D orientation-invariant features for live ingest — aligned with v2 training code.

Feature design: magnitude-first, gravity-separated (FIR low-pass), orientation-invariant.
Matches ``extract_window_features()`` from the MobiAct v2 training notebook exactly.

Feature layout (total = 144):
  84  — 6 signals × 14 mag_stats  (acc_mag, lin_mag, gyro_mag, acc_jerk, lin_jerk, gyro_jerk)
  18  — 3 signals × 6  mag_freq   (acc_mag, lin_mag, gyro_mag)
  36  — 2 sensors × 18 axis_stats (linear_acc, gyro)
   3  — cross-axis correlations   (linear_acc pairs 01, 02, 12)
   3  — orientation means         (azimuth, pitch, roll)
"""

from __future__ import annotations

import warnings

import numpy as np

ENHANCED_FEATURE_DIM = 144
SAMPLE_RATE = 50


def _lowpass(signal: np.ndarray, cutoff_hz: float = 0.3, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Hamming-windowed FIR low-pass — gravity separation without scipy."""
    sig_len = signal.shape[0]
    n = min(int(fs / cutoff_hz), sig_len // 2)
    n = n if n % 2 == 1 else n + 1
    kernel = np.hamming(n)
    kernel /= kernel.sum()
    out = np.zeros_like(signal, dtype=np.float64)
    for ax in range(signal.shape[1]):
        conv = np.convolve(signal[:, ax], kernel, mode="full")
        trim = (len(conv) - sig_len) // 2
        out[:, ax] = conv[trim : trim + sig_len]
    return out


def _mag_stats(sig: np.ndarray) -> list[float]:
    """14 statistics from a 1-D magnitude signal."""
    return [
        float(np.mean(sig)),
        float(np.std(sig)),
        float(np.median(sig)),
        float(np.min(sig)),
        float(np.max(sig)),
        float(np.ptp(sig)),
        float(np.percentile(sig, 5)),
        float(np.percentile(sig, 95)),
        float(np.sqrt(np.mean(sig**2))),          # RMS
        float(np.mean(np.abs(np.diff(sig)))),      # mean abs diff
        float(np.sum(np.abs(np.diff(sig)))),       # arc-length
        float(np.var(sig)),
        float(np.sum(sig**2) / len(sig)),          # power
        float(np.max(np.abs(sig))),                # peak
    ]


def _mag_freq(sig: np.ndarray, fs: int = SAMPLE_RATE) -> list[float]:
    """6 frequency features from a 1-D magnitude signal."""
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(sig))
    dom_f = float(freqs[np.argmax(fft_mag[1:]) + 1]) if fft_mag.size > 1 else 0.0
    spec_e = float(np.sum(fft_mag**2) / len(fft_mag))
    psd = fft_mag**2
    psd_norm = psd / (psd.sum() + 1e-10)
    spec_ent = float(-np.sum(psd_norm * np.log(psd_norm + 1e-10)))
    band_slow = float(np.sum(fft_mag[(freqs >= 0) & (freqs < 1)] ** 2))
    band_mid = float(np.sum(fft_mag[(freqs >= 1) & (freqs < 3)] ** 2))
    band_high = float(np.sum(fft_mag[freqs >= 3] ** 2))
    return [dom_f, spec_e, spec_ent, band_slow, band_mid, band_high]


def _axis_stats(data: np.ndarray) -> list[float]:
    """6 stats × 3 axes = 18 features."""
    f: list[float] = []
    for ax in range(data.shape[1]):
        x = data[:, ax]
        f += [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.max(x)),
            float(np.min(x)),
            float(np.sqrt(np.mean(x**2))),    # RMS
            float(np.percentile(x, 95)),
        ]
    return f


def extract_window_features(acc: np.ndarray, gyro: np.ndarray, ori: np.ndarray) -> list[float]:
    """
    144-D orientation-invariant feature vector for a single window.

    Inputs: acc (128, 3), gyro (128, 3), ori (128, 3) — float arrays (m/s², rad/s, degrees).
    Returns a flat list of 144 floats.
    """
    acc = np.asarray(acc, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    ori = np.asarray(ori, dtype=np.float64)

    gravity = _lowpass(acc)
    linear_acc = acc - gravity

    acc_mag = np.sqrt(np.sum(acc**2, axis=1))
    lin_mag = np.sqrt(np.sum(linear_acc**2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro**2, axis=1))

    acc_jerk = np.diff(acc_mag, prepend=acc_mag[0])
    lin_jerk = np.diff(lin_mag, prepend=lin_mag[0])
    gyro_jerk = np.diff(gyro_mag, prepend=gyro_mag[0])

    feat: list[float] = []

    # 6 × 14 = 84 magnitude stats
    feat += _mag_stats(acc_mag)
    feat += _mag_stats(lin_mag)
    feat += _mag_stats(gyro_mag)
    feat += _mag_stats(acc_jerk)
    feat += _mag_stats(lin_jerk)
    feat += _mag_stats(gyro_jerk)

    # 3 × 6 = 18 magnitude frequency features
    feat += _mag_freq(acc_mag)
    feat += _mag_freq(lin_mag)
    feat += _mag_freq(gyro_mag)

    # 2 × 18 = 36 axis stats (gravity-removed acc + gyro)
    feat += _axis_stats(linear_acc)
    feat += _axis_stats(gyro)

    # 3 cross-axis correlations of linear_acc (zero-std axes give NaN → 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            c = np.corrcoef(linear_acc[:, i], linear_acc[:, j])[0, 1]
            feat.append(0.0 if np.isnan(c) else float(c))

    # 3 orientation means (useful for LYI vs STD separation)
    for ax in range(ori.shape[1]):
        feat.append(float(np.mean(ori[:, ax])))

    return feat  # 144 features total


def extract_enhanced_features(
    acc_windows: np.ndarray,
    gyro_windows: np.ndarray | None = None,
    ori_windows: np.ndarray | None = None,
) -> np.ndarray:
    """
    Batch version: (n, window_size, 3) arrays → (n, 144) feature matrix.

    Replaces the old 128-D extractor; parity with v2 training ``extract_window_features``.
    """
    n = len(acc_windows)
    if gyro_windows is None:
        gyro_windows = np.zeros((n, acc_windows.shape[1], 3), dtype=np.float64)
    if ori_windows is None:
        ori_windows = np.zeros((n, acc_windows.shape[1], 3), dtype=np.float64)

    features = [
        extract_window_features(acc_windows[i], gyro_windows[i], ori_windows[i])
        for i in range(n)
    ]
    out = np.asarray(features, dtype=np.float64)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
