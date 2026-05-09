"""Dimension contracts vs inference_manifest and frozen scalers."""

from __future__ import annotations

import joblib
from pathlib import Path

import numpy as np
import pytest

from baseline_fall.enhanced_features import extract_enhanced_features
from baseline_falltype.config import WINDOW_SAMPLES
from baseline_falltype.feature_extractors import CompleteFallFeatureExtractor


def test_deployed_enhanced_dim_matches_manifest(repo_root: Path, inference_manifest: dict) -> None:
    """Fall + ADL inference scalers must agree with manifest (same vector for both tasks)."""
    model_dir = repo_root / "flask_backend" / "models"
    d = int(inference_manifest["enhanced_feature_dim"])
    art = inference_manifest["artifacts"]
    fall_p = model_dir / art["fall_binary"]["scaler_path"]
    adl_p = model_dir / art["adl"]["scaler_path"]
    if not fall_p.is_file() or not adl_p.is_file():
        pytest.skip("baseline fall/adl scalers not present (run train_mobiact_baselines.py)")
    sf = joblib.load(fall_p)
    sa = joblib.load(adl_p)
    nf = getattr(sf, "n_features_in_", None)
    na = getattr(sa, "n_features_in_", None)
    assert nf == d, f"Fall scaler wants {nf}, manifest says {d}"
    assert na == d, f"ADL scaler wants {na}, manifest says {d}"


def test_baseline_fall_extractor_is_128_dim() -> None:
    """Reference enhanced extractor (training notebook with full fusion) yields 128-D per window."""
    rng = np.random.default_rng(42)
    acc = rng.standard_normal((1, 300, 3))
    gyro = rng.standard_normal((1, 300, 3))
    ori = rng.standard_normal((1, 300, 3))
    X = extract_enhanced_features(acc, gyro, ori)
    assert X.shape == (1, 128)


def test_fall_type_manifest_matches_saved_scaler(repo_root: Path, inference_manifest: dict) -> None:
    scaler_path = repo_root / "flask_backend" / "models" / "baseline_falltype" / "scaler.pkl"
    if not scaler_path.is_file():
        pytest.skip("fall-type scaler not present")
    scaler = joblib.load(scaler_path)
    n = getattr(scaler, "n_features_in_", None)
    expected = int(inference_manifest["fall_type_raw_dim"])
    assert n == expected, f"Manifest fall_type_raw_dim {expected} != scaler n_features_in_ {n}"


def test_repo_fall_type_extractor_is_263_dim(repo_root: Path, inference_manifest: dict) -> None:
    """263-D raw vector matches inference_manifest fall_type_raw_dim / scaler.pkl."""
    rng = np.random.default_rng(43)
    acc = rng.standard_normal((1, WINDOW_SAMPLES, 3))
    gyro = rng.standard_normal((1, WINDOW_SAMPLES, 3))
    ori = rng.standard_normal((1, WINDOW_SAMPLES, 3))
    ext = CompleteFallFeatureExtractor(fs=50.0)
    X = ext.extract_batch(acc, gyro, ori, desc="test")
    raw_dim = int(inference_manifest["fall_type_raw_dim"])
    assert X.shape == (1, raw_dim)
