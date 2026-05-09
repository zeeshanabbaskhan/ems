"""Load frozen models and run fall → ADL | fall-type inference (canonical copy under scripts/).

Used by the FastAPI backend and any offline verification. Depends on `baseline_falltype`
for server-side 263-D fall-type feature extraction when windows are provided.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def _ensure_scripts_on_path() -> None:
    raw = os.environ.get("REPO_ROOT")
    if raw:
        repo = Path(raw).expanduser().resolve()
    else:
        repo = Path(__file__).resolve().parents[2]
    s = str(repo / "scripts")
    if s not in sys.path:
        sys.path.insert(0, s)


def _fall_type_vector_from_windows(
    acc_window: np.ndarray,
    gyro_window: np.ndarray | None,
    ori_window: np.ndarray | None,
) -> np.ndarray:
    _ensure_scripts_on_path()
    from baseline_falltype.feature_extractors import extract_fall_type_raw_vector

    return extract_fall_type_raw_vector(acc_window, gyro_window, ori_window)


@dataclass(frozen=True)
class InferenceArtifacts:
    manifest: dict[str, Any]
    fall_model: Any | None
    fall_scaler: Any | None
    adl_model: Any
    adl_scaler: Any
    adl_encoder: Any
    fall_type_model: Any | None
    fall_type_scaler: Any | None
    fall_type_indices: np.ndarray
    fall_type_encoder: Any | None
    enhanced_dim: int
    fall_type_dim: int
    fall_threshold: float
    fall_binary_enabled: bool
    fall_binary_issue: str | None
    fall_type_enabled: bool


def load_artifacts(manifest_path: Path, models_dir: Path) -> InferenceArtifacts:
    manifest_path = manifest_path.resolve()
    models_dir = models_dir.resolve()
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    enhanced_dim = int(manifest["enhanced_feature_dim"])
    fall_type_dim = int(manifest["fall_type_raw_dim"])
    threshold = float(manifest.get("fall_probability_threshold", 0.5))
    art = manifest["artifacts"]

    def p(rel: str) -> Path:
        out = (models_dir / rel).resolve()
        if not out.is_file():
            raise FileNotFoundError(str(out))
        return out

    adl_model = joblib.load(p(art["adl"]["model_path"]))
    adl_scaler = joblib.load(p(art["adl"]["scaler_path"]))
    adl_encoder = joblib.load(p(art["adl"]["label_encoder_path"]))

    ft_cfg = art.get("fall_type")
    fall_type_model: Any | None = None
    fall_type_scaler: Any | None = None
    fall_type_indices = np.array([], dtype=int)
    fall_type_encoder: Any | None = None
    fall_type_enabled = False
    if ft_cfg:
        fall_type_model = joblib.load(p(ft_cfg["model_path"]))
        fall_type_scaler = joblib.load(p(ft_cfg["scaler_path"]))
        fall_type_indices = np.asarray(joblib.load(p(ft_cfg["feature_indices_path"])), dtype=int)
        fall_type_encoder = joblib.load(p(ft_cfg["label_encoder_path"]))
        fall_type_enabled = True

    na = getattr(adl_scaler, "n_features_in_", None)
    if na is not None and int(na) != enhanced_dim:
        raise ValueError(f"ADL scaler wants {na}, manifest {enhanced_dim}")
    if fall_type_enabled and fall_type_scaler is not None:
        ft_n = getattr(fall_type_scaler, "n_features_in_", None)
        if ft_n is not None and int(ft_n) != fall_type_dim:
            raise ValueError(f"Fall-type scaler wants {ft_n}, manifest {fall_type_dim}")

    fall_model: Any | None = None
    fall_scaler: Any | None = None
    fall_binary_enabled = True
    fall_binary_issue: str | None = None
    try:
        fall_model = joblib.load(p(art["fall_binary"]["model_path"]))
        fall_scaler = joblib.load(p(art["fall_binary"]["scaler_path"]))
        nf = getattr(fall_scaler, "n_features_in_", None)
        if nf is not None and int(nf) != enhanced_dim:
            fall_binary_enabled = False
            fall_binary_issue = f"fall_scaler_dim_mismatch:{nf}!=manifest:{enhanced_dim}"
            fall_model = None
            fall_scaler = None
    except Exception as exc:
        fall_binary_enabled = False
        fall_binary_issue = f"fall_binary_unavailable:{exc}"

    return InferenceArtifacts(
        manifest=manifest,
        fall_model=fall_model,
        fall_scaler=fall_scaler,
        adl_model=adl_model,
        adl_scaler=adl_scaler,
        adl_encoder=adl_encoder,
        fall_type_model=fall_type_model,
        fall_type_scaler=fall_type_scaler,
        fall_type_indices=fall_type_indices,
        fall_type_encoder=fall_type_encoder,
        enhanced_dim=enhanced_dim,
        fall_type_dim=fall_type_dim,
        fall_threshold=threshold,
        fall_binary_enabled=fall_binary_enabled,
        fall_binary_issue=fall_binary_issue,
        fall_type_enabled=fall_type_enabled,
    )


def run_inference(
    art: InferenceArtifacts,
    enhanced_features: list[float],
    fall_type_features: list[float] | None,
    *,
    predict_fall_type: bool,
    acc_window: list[list[float]] | None = None,
    gyro_window: list[list[float]] | None = None,
    ori_window: list[list[float]] | None = None,
) -> dict[str, Any]:
    def _predict_proba_safely(model: Any, values: np.ndarray) -> np.ndarray:
        # LightGBM/XGBoost/sklearn can warn when model was fit with feature names but runtime
        # sends plain ndarray. This inference path intentionally sends arrays.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but XGBClassifier was fitted with feature names",
                category=UserWarning,
            )
            return model.predict_proba(values)

    x = np.asarray(enhanced_features, dtype=np.float64).reshape(1, -1)
    if x.shape[1] != art.enhanced_dim:
        raise ValueError(f"enhanced_features length {x.shape[1]} != {art.enhanced_dim}")

    if fall_type_features is not None:
        ft_chk = np.asarray(fall_type_features, dtype=np.float64).reshape(1, -1)
        if ft_chk.shape[1] != art.fall_type_dim:
            raise ValueError(f"fall_type_features length {ft_chk.shape[1]} != {art.fall_type_dim}")

    if not art.fall_binary_enabled or art.fall_model is None or art.fall_scaler is None:
        xa = art.adl_scaler.transform(x)
        cid = int(art.adl_model.predict(xa)[0])
        label = str(art.adl_encoder.inverse_transform(np.array([cid]))[0])
        return {
            "is_fall": False,
            "fall_probability": 0.0,
            "fall_threshold": art.fall_threshold,
            "schema_version": str(art.manifest.get("schema_version", "1.0")),
            "branch": "adl",
            "activity_class_index": cid,
            "activity_label": label,
            "fall_type_code": None,
            "fall_type_label": None,
            "fall_type_class_index": None,
            "fall_type_skipped_reason": art.fall_binary_issue or "fall_binary_disabled",
        }

    xf = art.fall_scaler.transform(x)
    p_fall = float(_predict_proba_safely(art.fall_model, xf)[0, 1])
    is_fall = p_fall >= art.fall_threshold

    out: dict[str, Any] = {
        "is_fall": is_fall,
        "fall_probability": p_fall,
        "fall_threshold": art.fall_threshold,
        "schema_version": str(art.manifest.get("schema_version", "1.0")),
    }

    if not is_fall:
        xa = art.adl_scaler.transform(x)
        cid = int(art.adl_model.predict(xa)[0])
        label = str(art.adl_encoder.inverse_transform(np.array([cid]))[0])
        out["branch"] = "adl"
        out["activity_class_index"] = cid
        out["activity_label"] = label
        out["fall_type_code"] = None
        out["fall_type_label"] = None
        out["fall_type_class_index"] = None
        out["fall_type_skipped_reason"] = None
        return out

    out["branch"] = "fall"
    out["activity_class_index"] = None
    out["activity_label"] = None

    if not art.fall_type_enabled or art.fall_type_model is None:
        out["fall_type_code"] = None
        out["fall_type_label"] = None
        out["fall_type_class_index"] = None
        out["fall_type_skipped_reason"] = "fall_type_not_configured"
        return out

    ft_source: list[float] | None = fall_type_features
    if ft_source is None and acc_window is not None:
        acc = np.asarray(acc_window, dtype=np.float64)
        gyro = np.asarray(gyro_window, dtype=np.float64) if gyro_window is not None else None
        ori = np.asarray(ori_window, dtype=np.float64) if ori_window is not None else None
        ft_source = _fall_type_vector_from_windows(acc, gyro, ori).tolist()

    if not predict_fall_type or ft_source is None:
        out["fall_type_code"] = None
        out["fall_type_label"] = None
        out["fall_type_class_index"] = None
        out["fall_type_skipped_reason"] = (
            "predict_fall_type_disabled"
            if not predict_fall_type
            else "fall_type_features_missing_no_acc_window"
        )
        return out

    ft = np.asarray(ft_source, dtype=np.float64).reshape(1, -1)
    if ft.shape[1] != art.fall_type_dim:
        raise ValueError(f"fall_type_features length {ft.shape[1]} != {art.fall_type_dim}")

    xs = art.fall_type_scaler.transform(ft)
    xsel = xs[:, art.fall_type_indices]
    pred = art.fall_type_model.predict(xsel)
    fi = int(pred[0])
    code = str(art.fall_type_encoder.inverse_transform(np.array([fi]))[0])
    out["fall_type_class_index"] = fi
    out["fall_type_code"] = code
    out["fall_type_label"] = code
    out["fall_type_skipped_reason"] = None
    return out


__all__ = ["InferenceArtifacts", "load_artifacts", "run_inference"]
