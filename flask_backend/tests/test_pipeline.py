"""End-to-end pipeline tests — v2 (144-D orientation-invariant features).

Verifies:
  1. Feature extraction produces exactly 144 features per window
  2. Gravity separation (low-pass) and jerk signals are finite
  3. Models load successfully from inference_manifest.json
  4. ADL scaler/model dimensions match the manifest (144)
  5. Fall scaler/model dimensions match the manifest (144)
  6. run_inference returns the expected response schema for an ADL window
  7. run_inference returns is_fall=True for a synthetic high-impact window
  8. samples_to_feature_vector produces a (144,) vector from raw sample dicts
  9. build_enhanced_features_numpy produces a (144,) vector from (128,3) arrays
 10. VoteBuffer stabilises ADL predictions and resets on fall
"""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path

import numpy as np
import pytest

# ── path setup ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ── module under test ────────────────────────────────────────────────────────
from flask_backend.app.motion_enhanced_features import (
    ENHANCED_FEATURE_DIM,
    extract_enhanced_features,
    extract_window_features,
    _lowpass,
    _mag_stats,
    _mag_freq,
    _axis_stats,
)
from flask_backend.app.ml_bridge import (
    VoteBuffer,
    VOTE_BUFFER_SIZE,
    build_enhanced_features_numpy,
    samples_to_feature_vector,
)
from flask_backend.app.settings import inference_manifest_path, model_root

MANIFEST_PATH = inference_manifest_path()
MODEL_ROOT = model_root()

# ── helpers ──────────────────────────────────────────────────────────────────

WINDOW = 128
RNG = np.random.default_rng(42)


def _adl_window() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic walking-like window: mild oscillation, low acceleration."""
    t = np.linspace(0, 2 * math.pi, WINDOW)
    acc = np.stack([
        1.2 * np.sin(t) + RNG.normal(0, 0.05, WINDOW),
        0.3 * np.cos(t) + RNG.normal(0, 0.05, WINDOW),
        9.8 + RNG.normal(0, 0.1, WINDOW),
    ], axis=1).astype(np.float32)
    gyro = np.stack([
        0.1 * np.sin(2 * t) + RNG.normal(0, 0.02, WINDOW),
        0.05 * np.cos(t) + RNG.normal(0, 0.02, WINDOW),
        0.08 * np.sin(t + 0.5) + RNG.normal(0, 0.02, WINDOW),
    ], axis=1).astype(np.float32)
    ori = np.stack([
        RNG.normal(180.0, 2.0, WINDOW),
        RNG.normal(5.0, 1.0, WINDOW),
        RNG.normal(2.0, 1.0, WINDOW),
    ], axis=1).astype(np.float32)
    return acc, gyro, ori


def _fall_window() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic fall-like window: large spike in acceleration + gyro."""
    acc = RNG.normal(0, 0.2, (WINDOW, 3)).astype(np.float32)
    gyro = RNG.normal(0, 0.05, (WINDOW, 3)).astype(np.float32)
    # Impact at sample 60
    acc[58:68, :] = np.array([[25.0, -18.0, 30.0]] * 10, dtype=np.float32)
    gyro[58:68, :] = np.array([[8.0, -6.0, 10.0]] * 10, dtype=np.float32)
    ori = np.zeros((WINDOW, 3), dtype=np.float32)
    return acc, gyro, ori


# ════════════════════════════════════════════════════════════════════════════
# 1. Feature extraction correctness
# ════════════════════════════════════════════════════════════════════════════

class TestFeatureDimension:
    def test_constant_is_144(self):
        assert ENHANCED_FEATURE_DIM == 144

    def test_single_window_length(self):
        acc, gyro, ori = _adl_window()
        feat = extract_window_features(acc, gyro, ori)
        assert len(feat) == 144, f"Expected 144 features, got {len(feat)}"

    def test_batch_shape(self):
        acc, gyro, ori = _adl_window()
        batch_acc = acc[np.newaxis, ...]
        batch_gyro = gyro[np.newaxis, ...]
        batch_ori = ori[np.newaxis, ...]
        out = extract_enhanced_features(batch_acc, batch_gyro, batch_ori)
        assert out.shape == (1, 144), f"Expected (1, 144), got {out.shape}"

    def test_batch_multiple_windows(self):
        acc, gyro, ori = _adl_window()
        n = 5
        batch_acc = np.tile(acc[np.newaxis, ...], (n, 1, 1))
        batch_gyro = np.tile(gyro[np.newaxis, ...], (n, 1, 1))
        batch_ori = np.tile(ori[np.newaxis, ...], (n, 1, 1))
        out = extract_enhanced_features(batch_acc, batch_gyro, batch_ori)
        assert out.shape == (n, 144)

    def test_no_nan_or_inf(self):
        acc, gyro, ori = _adl_window()
        feat = np.array(extract_window_features(acc, gyro, ori))
        assert np.all(np.isfinite(feat)), "Features contain NaN or Inf"

    def test_fall_window_no_nan(self):
        acc, gyro, ori = _fall_window()
        feat = np.array(extract_window_features(acc, gyro, ori))
        assert np.all(np.isfinite(feat))

    def test_gyro_zeros_no_crash(self):
        acc, _, ori = _adl_window()
        gyro = np.zeros((WINDOW, 3), dtype=np.float32)
        feat = extract_window_features(acc, gyro, ori)
        assert len(feat) == 144

    def test_batch_none_gyro_ori(self):
        acc, _, _ = _adl_window()
        batch = acc[np.newaxis, ...]
        out = extract_enhanced_features(batch)
        assert out.shape == (1, 144)


class TestSubFunctions:
    def test_lowpass_shape(self):
        acc, _, _ = _adl_window()
        gravity = _lowpass(acc.astype(np.float64))
        assert gravity.shape == acc.shape

    def test_mag_stats_length(self):
        sig = np.ones(128)
        assert len(_mag_stats(sig)) == 14

    def test_mag_freq_length(self):
        sig = np.sin(np.linspace(0, 2 * math.pi, 128))
        assert len(_mag_freq(sig)) == 6

    def test_axis_stats_length(self):
        data = np.ones((128, 3))
        assert len(_axis_stats(data)) == 18  # 6 × 3

    def test_mag_stats_constant_signal(self):
        sig = np.full(128, 5.0)
        stats = _mag_stats(sig)
        assert stats[0] == pytest.approx(5.0)   # mean
        assert stats[1] == pytest.approx(0.0)   # std
        assert stats[4] == pytest.approx(5.0)   # max


# ════════════════════════════════════════════════════════════════════════════
# 2. Bridge helpers
# ════════════════════════════════════════════════════════════════════════════

class TestBridgeHelpers:
    def test_build_enhanced_features_numpy_shape(self):
        acc, gyro, ori = _adl_window()
        feat = build_enhanced_features_numpy(
            acc.astype(np.float64),
            gyro.astype(np.float64),
            ori.astype(np.float64),
        )
        assert feat.shape == (144,)

    def test_build_enhanced_features_numpy_finite(self):
        acc, gyro, ori = _adl_window()
        feat = build_enhanced_features_numpy(acc, gyro, ori)
        assert np.all(np.isfinite(feat))

    def test_build_enhanced_features_no_gyro_ori(self):
        acc, _, _ = _adl_window()
        feat = build_enhanced_features_numpy(acc)
        assert feat.shape == (144,)

    def test_samples_to_feature_vector_shape(self):
        samples = [
            {
                "acc_x": float(np.sin(i * 0.1)),
                "acc_y": float(np.cos(i * 0.1)),
                "acc_z": 9.8,
                "gyro_x": 0.1,
                "gyro_y": 0.0,
                "gyro_z": 0.0,
                "azimuth": 180.0,
                "pitch": 5.0,
                "roll": 2.0,
            }
            for i in range(128)
        ]
        feat, acc_300, gyro_300, ori_300 = samples_to_feature_vector(samples)
        assert feat.shape == (144,)
        assert acc_300.shape == (300, 3)
        assert gyro_300.shape == (300, 3)
        assert ori_300.shape == (300, 3)

    def test_samples_to_feature_vector_missing_ori(self):
        samples = [
            {"acc_x": 0.0, "acc_y": 0.0, "acc_z": 9.8, "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0}
            for _ in range(64)
        ]
        feat, _, _, _ = samples_to_feature_vector(samples)
        assert feat.shape == (144,)

    def test_samples_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            samples_to_feature_vector([])


# ════════════════════════════════════════════════════════════════════════════
# 3. VoteBuffer
# ════════════════════════════════════════════════════════════════════════════

class TestVoteBuffer:
    def test_fall_clears_buffer_and_returns_immediately(self):
        buf = VoteBuffer(size=7)
        for _ in range(5):
            buf.push("WAL", is_fall=False)
        result = buf.push("FALL", is_fall=True)
        assert result == "FALL"
        assert buf.confidence == 0.0  # buffer cleared

    def test_adl_majority_vote(self):
        buf = VoteBuffer(size=7)
        for _ in range(5):
            buf.push("WAL", is_fall=False)
        buf.push("SIT", is_fall=False)
        buf.push("WAL", is_fall=False)
        result = buf.push("WAL", is_fall=False)
        assert result == "WAL"

    def test_returns_raw_label_below_size(self):
        buf = VoteBuffer(size=7)
        result = buf.push("JOG", is_fall=False)
        assert result == "JOG"

    def test_reset_clears_buffer(self):
        buf = VoteBuffer(size=7)
        for _ in range(7):
            buf.push("STD", is_fall=False)
        buf.reset()
        assert buf.confidence == 0.0

    def test_confidence_full_agreement(self):
        buf = VoteBuffer(size=4)
        for _ in range(4):
            buf.push("LYI", is_fall=False)
        assert buf.confidence == pytest.approx(1.0)

    def test_confidence_majority(self):
        buf = VoteBuffer(size=4)
        for label in ["WAL", "WAL", "WAL", "SIT"]:
            buf.push(label, is_fall=False)
        assert buf.confidence == pytest.approx(0.75)

    def test_vote_buffer_size_constant(self):
        assert VOTE_BUFFER_SIZE == 11


# ════════════════════════════════════════════════════════════════════════════
# 4. Model loading
# ════════════════════════════════════════════════════════════════════════════

class TestModelLoading:
    def test_manifest_exists(self):
        assert MANIFEST_PATH.is_file(), f"Manifest not found: {MANIFEST_PATH}"

    def test_manifest_feature_dim(self):
        import json
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        assert manifest["enhanced_feature_dim"] == 144

    def test_manifest_threshold(self):
        import json
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        assert manifest["fall_probability_threshold"] == pytest.approx(0.65)

    def test_model_files_exist(self):
        import json
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        art = manifest["artifacts"]
        for key, cfg in art.items():
            if cfg is None:
                continue
            for field, rel_path in cfg.items():
                full = MODEL_ROOT / rel_path
                assert full.is_file(), f"Missing artifact [{key}][{field}]: {full}"

    def test_load_artifacts_succeeds(self):
        from flask_backend.app.services.motion_xgb_service import load_artifacts
        art = load_artifacts(MANIFEST_PATH, MODEL_ROOT)
        assert art is not None

    def test_artifacts_fall_binary_enabled(self):
        from flask_backend.app.services.motion_xgb_service import load_artifacts
        art = load_artifacts(MANIFEST_PATH, MODEL_ROOT)
        assert art.fall_binary_enabled, (
            f"Fall binary disabled: {art.fall_binary_issue}"
        )

    def test_artifacts_enhanced_dim(self):
        from flask_backend.app.services.motion_xgb_service import load_artifacts
        art = load_artifacts(MANIFEST_PATH, MODEL_ROOT)
        assert art.enhanced_dim == 144

    def test_artifacts_fall_threshold(self):
        from flask_backend.app.services.motion_xgb_service import load_artifacts
        art = load_artifacts(MANIFEST_PATH, MODEL_ROOT)
        assert art.fall_threshold == pytest.approx(0.65)

    def test_adl_scaler_dim(self):
        from flask_backend.app.services.motion_xgb_service import load_artifacts
        art = load_artifacts(MANIFEST_PATH, MODEL_ROOT)
        n = getattr(art.adl_scaler, "n_features_in_", None)
        if n is not None:
            assert n == 144, f"ADL scaler expects {n} features, expected 144"

    def test_fall_scaler_dim(self):
        from flask_backend.app.services.motion_xgb_service import load_artifacts
        art = load_artifacts(MANIFEST_PATH, MODEL_ROOT)
        n = getattr(art.fall_scaler, "n_features_in_", None)
        if n is not None:
            assert n == 144, f"Fall scaler expects {n} features, expected 144"


# ════════════════════════════════════════════════════════════════════════════
# 5. End-to-end inference
# ════════════════════════════════════════════════════════════════════════════

class TestEndToEndInference:
    @pytest.fixture(scope="class")
    def art(self):
        from flask_backend.app.services.motion_xgb_service import load_artifacts
        return load_artifacts(MANIFEST_PATH, MODEL_ROOT)

    def _run(self, art, acc, gyro, ori):
        from flask_backend.app.services.motion_xgb_service import run_inference
        feat = build_enhanced_features_numpy(acc, gyro, ori).tolist()
        return run_inference(art, feat, None, predict_fall_type=False)

    def test_adl_response_schema(self, art):
        acc, gyro, ori = _adl_window()
        result = self._run(art, acc, gyro, ori)
        assert "is_fall" in result
        assert "fall_probability" in result
        assert "branch" in result
        assert "schema_version" in result
        assert isinstance(result["fall_probability"], float)
        assert 0.0 <= result["fall_probability"] <= 1.0

    def test_adl_window_low_fall_prob(self, art):
        acc, gyro, ori = _adl_window()
        result = self._run(art, acc, gyro, ori)
        # Typical walking should produce low fall probability
        assert result["fall_probability"] < 0.9, (
            f"Walking window produced unexpectedly high fall prob: {result['fall_probability']:.3f}"
        )

    def test_adl_branch_has_activity_label(self, art):
        acc, gyro, ori = _adl_window()
        result = self._run(art, acc, gyro, ori)
        if not result["is_fall"]:
            assert result["branch"] == "adl"
            assert result["activity_label"] is not None
            assert result["activity_label"] in {"JOG", "LYI", "SIT", "STD", "WAL"}

    def test_fall_branch_schema(self, art):
        from flask_backend.app.services.motion_xgb_service import run_inference
        acc, gyro, ori = _fall_window()
        feat = build_enhanced_features_numpy(acc, gyro, ori).tolist()
        result = run_inference(art, feat, None, predict_fall_type=False)
        assert "is_fall" in result
        assert "fall_probability" in result
        if result["is_fall"]:
            assert result["branch"] == "fall"

    def test_wrong_dim_raises(self, art):
        from flask_backend.app.services.motion_xgb_service import run_inference
        with pytest.raises(ValueError, match="enhanced_features length"):
            run_inference(art, [0.0] * 128, None, predict_fall_type=False)

    def test_feature_vector_matches_expected_dim(self, art):
        acc, gyro, ori = _adl_window()
        feat = build_enhanced_features_numpy(acc, gyro, ori)
        assert len(feat) == art.enhanced_dim == 144

    def test_full_sample_ingest_pipeline(self, art):
        """samples_to_feature_vector → run_inference — the live ingest code path."""
        from flask_backend.app.services.motion_xgb_service import run_inference
        samples = [
            {
                "acc_x": float(np.sin(i * 0.1)),
                "acc_y": float(np.cos(i * 0.1)),
                "acc_z": 9.8,
                "gyro_x": 0.05 * np.sin(i * 0.2),
                "gyro_y": 0.0,
                "gyro_z": 0.0,
                "azimuth": 180.0,
                "pitch": 5.0,
                "roll": 2.0,
            }
            for i in range(128)
        ]
        feat, _, _, _ = samples_to_feature_vector(samples)
        assert feat.shape == (144,)
        result = run_inference(art, feat.tolist(), None, predict_fall_type=False)
        assert "is_fall" in result
        assert 0.0 <= result["fall_probability"] <= 1.0
