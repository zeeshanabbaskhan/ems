"""Direct tests on deployed fall-binary and fall-type joblibs (no ADL required)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from baseline_fall.enhanced_features import extract_enhanced_features
from baseline_falltype.feature_extractors import extract_fall_type_raw_vector


def test_fall_detection_binary_116d_forward(repo_root: Path, inference_manifest: dict) -> None:
    """RobustScaler + XGBoost fall classifier accept 116-D enhanced features."""
    model_dir = repo_root / "flask_backend" / "models"
    art = inference_manifest["artifacts"]["fall_binary"]
    scaler = joblib.load(model_dir / art["scaler_path"])
    model = joblib.load(model_dir / art["model_path"])
    d = int(inference_manifest["enhanced_feature_dim"])
    nf = getattr(scaler, "n_features_in_", None)
    assert nf == d, f"Fall scaler expects {nf}, manifest {d}"

    rng = np.random.default_rng(7)
    acc = rng.standard_normal((1, 300, 3))
    gyro = rng.standard_normal((1, 300, 3))
    ori = rng.standard_normal((1, 300, 3))
    X116 = extract_enhanced_features(acc, gyro, ori)
    assert X116.shape == (1, d)

    xs = scaler.transform(X116)
    proba = model.predict_proba(xs)
    assert proba.shape == (1, 2)
    p_fall = float(proba[0, 1])
    assert 0.0 <= p_fall <= 1.0


def test_fall_type_263d_pipeline_forward(repo_root: Path, inference_manifest: dict) -> None:
    """263-D raw vector → StandardScaler → MI columns → XGBoost 4-class."""
    model_dir = repo_root / "flask_backend" / "models"
    ft_art = inference_manifest["artifacts"].get("fall_type")
    if ft_art is None:
        pytest.skip("fall_type artifacts omitted — optional in inference_manifest.json")
    scaler = joblib.load(model_dir / ft_art["scaler_path"])
    model = joblib.load(model_dir / ft_art["model_path"])
    indices = np.asarray(joblib.load(model_dir / ft_art["feature_indices_path"]))
    encoder = joblib.load(model_dir / ft_art["label_encoder_path"])

    raw_dim = int(inference_manifest["fall_type_raw_dim"])
    assert getattr(scaler, "n_features_in_", None) == raw_dim

    rng = np.random.default_rng(11)
    acc = rng.standard_normal((300, 3))
    vec = extract_fall_type_raw_vector(acc, None, None)
    assert vec.shape == (raw_dim,)

    xs = scaler.transform(vec.reshape(1, -1))
    xsel = xs[:, indices]
    pred = model.predict(xsel)
    code = str(encoder.inverse_transform(pred)[0])
    classes = [str(c) for c in encoder.classes_]
    assert code in classes


def test_full_motion_inference_when_adl_present(repo_root: Path, inference_manifest: dict) -> None:
    """End-to-end `run_inference` only if baseline_adl artifacts exist."""
    model_dir = repo_root / "flask_backend" / "models"
    adl = inference_manifest["artifacts"]["adl"]
    adl_model = model_dir / adl["model_path"]
    if not adl_model.is_file():
        pytest.skip("baseline_adl models missing — run scripts/baseline_fall/train_mobiact_baselines.py")

    from flask_backend.app.services.motion_xgb_service import load_artifacts, run_inference

    art = load_artifacts(model_dir / "inference_manifest.json", model_dir)
    z = [0.0] * art.enhanced_dim
    ft = [0.0] * art.fall_type_dim
    out = run_inference(art, z, ft, predict_fall_type=True)
    assert out["branch"] in ("adl", "fall")
    assert "is_fall" in out
    if out["branch"] == "fall" and out.get("fall_type_skipped_reason") is None:
        assert out["fall_type_code"] is not None
