"""Load real joblib artifacts; validate fall-type (MI + classifier) alignment."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from flask_backend.app.services.motion_xgb_service import (
    InferenceArtifacts,
    _fall_type_vector_from_windows,
    load_artifacts,
    run_inference,
)


@pytest.fixture(scope="session")
def artifacts(repo_root: Path) -> InferenceArtifacts:
    mp = repo_root / "flask_backend" / "models" / "inference_manifest.json"
    md = repo_root / "flask_backend" / "models"
    if not mp.is_file():
        pytest.skip("flask_backend/models/inference_manifest.json not found")
    try:
        return load_artifacts(mp, md)
    except FileNotFoundError as e:
        pytest.skip(str(e))


def test_fall_type_vector_from_acc_window_is_263(artifacts: InferenceArtifacts) -> None:
    acc = np.random.default_rng(0).standard_normal((300, 3))
    v = _fall_type_vector_from_windows(acc, None, None)
    assert v.shape == (artifacts.fall_type_dim,)


def test_manifest_dims_match_scalers(artifacts: InferenceArtifacts, inference_manifest: dict) -> None:
    assert artifacts.enhanced_dim == int(inference_manifest["enhanced_feature_dim"])
    assert artifacts.fall_type_dim == int(inference_manifest["fall_type_raw_dim"])


def test_fall_type_feature_indices_in_bounds(artifacts: InferenceArtifacts) -> None:
    if not artifacts.fall_type_enabled:
        pytest.skip("fall_type stack not loaded — optional when manifest fall_type is null")
    idx = artifacts.fall_type_indices
    assert idx.ndim == 1
    assert np.all(idx >= 0)
    assert np.all(idx < artifacts.fall_type_dim), "MI indices must index scaled raw fall-type vector columns"


def test_fall_type_inference_matrix_shape(artifacts: InferenceArtifacts) -> None:
    """Scaled (1,350) -> column subset must match estimator input."""
    if not artifacts.fall_type_enabled:
        pytest.skip("fall_type stack not loaded — optional when manifest fall_type is null")
    ft = np.zeros((1, artifacts.fall_type_dim), dtype=np.float64)
    xs = artifacts.fall_type_scaler.transform(ft)
    xsel = xs[:, artifacts.fall_type_indices]
    pred = artifacts.fall_type_model.predict(xsel)
    assert pred.shape == (1,)


def test_run_inference_rejects_bad_lengths(artifacts: InferenceArtifacts) -> None:
    with pytest.raises(ValueError, match="enhanced_features length"):
        run_inference(artifacts, [0.0] * 10, None, predict_fall_type=False)

    with pytest.raises(ValueError, match="fall_type_features length"):
        run_inference(
            artifacts,
            [0.0] * artifacts.enhanced_dim,
            [0.0] * 10,
            predict_fall_type=True,
        )


def test_run_inference_adl_branch_returns_label(artifacts: InferenceArtifacts) -> None:
    """Zeros -> typically low fall prob; expect ADL branch with some label string."""
    z = [0.0] * artifacts.enhanced_dim
    out = run_inference(artifacts, z, None, predict_fall_type=True)
    assert "branch" in out
    if out["branch"] == "adl":
        assert out["activity_label"] is not None
        assert out["is_fall"] is False
    else:
        assert out["is_fall"] is True


def test_run_inference_full_vectors_fall_type_when_fall(artifacts: InferenceArtifacts) -> None:
    """Zeros + full fall-type vector; if binary fall branch, type label must be a known class."""
    z = [0.0] * artifacts.enhanced_dim
    ft = [0.0] * artifacts.fall_type_dim
    out = run_inference(artifacts, z, ft, predict_fall_type=True)
    assert out["branch"] in ("adl", "fall")
    if out["branch"] == "fall" and out.get("fall_type_skipped_reason") is None:
        assert out["fall_type_code"] is not None
        classes = [str(c) for c in artifacts.fall_type_encoder.classes_]
        assert str(out["fall_type_code"]) in classes


def test_fall_type_estimator_input_width(artifacts: InferenceArtifacts) -> None:
    """Updated fall-type model must accept MI-selected columns (regression guard)."""
    if not artifacts.fall_type_enabled:
        pytest.skip("fall_type stack not loaded — optional when manifest fall_type is null")
    ft = np.zeros((1, artifacts.fall_type_dim), dtype=np.float64)
    xs = artifacts.fall_type_scaler.transform(ft)
    xsel = xs[:, artifacts.fall_type_indices]
    m = artifacts.fall_type_model
    nfi = getattr(m, "n_features_in_", None)
    if nfi is not None:
        assert int(nfi) == xsel.shape[1], (
            f"Classifier expects {nfi} features after MI selection; got {xsel.shape[1]}. "
            "Re-export selected_features.pkl or best_fall_classifier.pkl together."
        )
