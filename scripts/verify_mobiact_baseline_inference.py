#!/usr/bin/env python3
"""Offline check for Colab-exported XGBoost + RobustScaler artifacts under ``models/baseline_adl&fall``.

Reads MobiAct ``*_annotated.csv`` files (per-label subfolders), builds one 128-sample window from a
stable segment, runs the same 128-D extraction as the FastAPI stack, then prints fall vs ADL + ADL code.

Also runs two synthetic windows (quiet standing vs sharp jerk) so you can sanity-check without CSVs.

Usage (from ``ems/``):

  python scripts/verify_mobiact_baseline_inference.py
  python scripts/verify_mobiact_baseline_inference.py --csv-root "E:/MERN/StepSafe_AI/ems/MobiAct_Dataset_v2.0/Annotated Data"
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("REPO_ROOT", str(_REPO))

from flask_backend.app.detector_state import build_detection_payload  # noqa: E402
from flask_backend.app.motion_enhanced_features import extract_enhanced_features  # noqa: E402

_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from inference.motion_pipeline import InferenceArtifacts, load_artifacts, run_inference  # noqa: E402

COL_ACC = ["acc_x", "acc_y", "acc_z"]
COL_GYRO = ["gyro_x", "gyro_y", "gyro_z"]
COL_ORI = ["azimuth", "pitch", "roll"]
COL_LABEL = "label"
WINDOW = 128
FS = 50.0
# MobiAct fall codes — matches Colab ``FALL_LABELS`` for CSV-vs-binary sanity stats.
FALL_LABELS = frozenset({"FOL", "FKL", "BSC", "SDL"})


def _repo_models_dir() -> Path:
    return (_REPO / "flask_backend" / "models").resolve()


def _window_to_ingest_samples(acc: np.ndarray, gyro: np.ndarray) -> list[dict[str, Any]]:
    """Same keys as ``POST /api/v1/ingest/live`` samples → ``detector_state.simple_signal_metrics``."""
    rows: list[dict[str, Any]] = []
    for i in range(acc.shape[0]):
        rows.append(
            {
                "acc_x": float(acc[i, 0]),
                "acc_y": float(acc[i, 1]),
                "acc_z": float(acc[i, 2]),
                "gyro_x": float(gyro[i, 0]),
                "gyro_y": float(gyro[i, 1]),
                "gyro_z": float(gyro[i, 2]),
            }
        )
    return rows


def _risk_from_window(
    art: InferenceArtifacts,
    acc: np.ndarray,
    gyro: np.ndarray,
    raw: dict[str, Any],
) -> dict[str, Any]:
    """Live-ingest equivalent risk: ``build_detection_payload`` (severity + score 0–1)."""
    samples = _window_to_ingest_samples(acc, gyro)
    act = str(raw["activity_label"]) if raw.get("branch") == "adl" and raw.get("activity_label") else None
    return build_detection_payload(
        samples=samples,
        fall_probability=float(raw["fall_probability"]),
        inferred_activity=act,
        ml_ok=True,
        threshold=float(art.fall_threshold),
    )


def _features_from_arrays(acc: np.ndarray, gyro: np.ndarray, ori: np.ndarray) -> np.ndarray:
    x = extract_enhanced_features(
        acc[np.newaxis, ...],
        gyro[np.newaxis, ...],
        ori[np.newaxis, ...],
    )[0]
    return np.asarray(x, dtype=np.float64)


def _stable_window_from_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
    """First contiguous block of rows with the same normalized label, length >= WINDOW."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    req = COL_ACC + COL_GYRO + COL_ORI + [COL_LABEL]
    if any(c not in df.columns for c in req):
        return None
    df[COL_LABEL] = df[COL_LABEL].astype(str).str.strip().str.upper()
    labels = df[COL_LABEL].values
    start = 0
    while start < len(labels):
        lab = labels[start]
        end = start
        while end < len(labels) and labels[end] == lab:
            end += 1
        seg_len = end - start
        if seg_len >= WINDOW:
            sl = slice(start, start + WINDOW)
            acc = df.loc[sl, COL_ACC].values.astype(np.float64)
            gyro = df.loc[sl, COL_GYRO].values.astype(np.float64)
            ori = df.loc[sl, COL_ORI].values.astype(np.float64)
            return acc, gyro, ori, str(lab)
        start = end
    return None


def _iter_sample_csvs(root: Path, limit: int) -> list[Path]:
    out: list[Path] = []
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*_annotated.csv")):
        out.append(p)
        if len(out) >= limit:
            break
    return out


def _synthetic_standing() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    t = np.arange(WINDOW, dtype=np.float64) / FS
    acc = np.column_stack(
        [
            rng.normal(0.2, 0.15, WINDOW),
            rng.normal(-9.7, 0.2, WINDOW),
            rng.normal(0.0, 0.15, WINDOW),
        ]
    )
    gyro = rng.normal(0.0, 0.05, size=(WINDOW, 3))
    ori = np.column_stack(
        [
            np.full(WINDOW, 120.0 + 2.0 * np.sin(2 * np.pi * 0.2 * t)),
            np.full(WINDOW, 40.0),
            np.full(WINDOW, -50.0),
        ]
    )
    return acc, gyro, ori


def _synthetic_fall_jerk() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    acc = rng.normal(0.0, 1.0, size=(WINDOW, 3))
    peak = slice(WINDOW // 2 - 5, WINDOW // 2 + 5)
    acc[peak, 0] += 18.0
    acc[peak, 1] -= 12.0
    acc[peak, 2] += 8.0
    gyro = rng.normal(0.0, 0.3, size=(WINDOW, 3))
    gyro[peak, :] += 5.0
    ori = np.column_stack(
        [
            np.linspace(0, 90, WINDOW),
            np.linspace(40, 80, WINDOW),
            np.linspace(-10, -70, WINDOW),
        ]
    )
    return acc, gyro, ori


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify MobiAct baseline XGBoost inference artifacts.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=_repo_models_dir() / "inference_manifest.json",
        help="Path to inference_manifest.json",
    )
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=_repo_models_dir(),
        help="Directory containing artifact pickle files",
    )
    ap.add_argument(
        "--csv-root",
        type=Path,
        default=_REPO / "MobiAct_Dataset_v2.0" / "Annotated Data",
        help="Root with label subfolders of *_annotated.csv",
    )
    ap.add_argument("--max-csv", type=int, default=8, help="Max CSV files to probe")
    ap.add_argument(
        "--summary",
        action="store_true",
        help="CSV section: print aggregate stats only (binary vs label + ADL match rate).",
    )
    args = ap.parse_args()

    need = [
        args.models_dir / "baseline_adl&fall" / "fall_xgboost_model.pkl",
        args.models_dir / "baseline_adl&fall" / "fall_scaler.pkl",
        args.models_dir / "baseline_adl&fall" / "adl_xgboost_model.pkl",
        args.models_dir / "baseline_adl&fall" / "adl_scaler.pkl",
        args.models_dir / "baseline_adl&fall" / "adl_label_encoder.pkl",
    ]
    missing = [p for p in need if not p.is_file()]
    if missing:
        print("Missing artifact files (copy from Colab export into models/baseline_adl&fall/):")
        for p in missing:
            print(f"  - {p}")
        print("\nContinuing with synthetic-only demo — loading will fail below.\n")

    try:
        art: InferenceArtifacts = load_artifacts(args.manifest.resolve(), args.models_dir.resolve())
    except Exception as exc:
        print(f"load_artifacts failed: {exc}")
        return 1

    print(
        f"Loaded inference: schema={art.manifest.get('schema_version')} "
        f"fall_type_enabled={art.fall_type_enabled} threshold={art.fall_threshold}\n"
    )

    def run_case(name: str, acc: np.ndarray, gyro: np.ndarray, ori: np.ndarray, true_lab: str | None) -> None:
        feat = _features_from_arrays(acc, gyro, ori).tolist()
        raw = run_inference(
            art,
            feat,
            None,
            predict_fall_type=False,
            acc_window=None,
            gyro_window=None,
            ori_window=None,
        )
        tag = f"true_csv_label={true_lab}" if true_lab else "synthetic"
        print(f"[{name}] {tag}")
        print(
            f"  branch={raw['branch']}  is_fall={raw['is_fall']}  "
            f"p_fall={raw['fall_probability']:.4f}  thr={raw['fall_threshold']}"
        )
        if raw.get("branch") == "adl":
            print(f"  activity_label={raw.get('activity_label')}")
        else:
            print(
                f"  fall_type={raw.get('fall_type_label')} "
                f"(skipped={raw.get('fall_type_skipped_reason')})"
            )
        print()

    run_case("synthetic_standing", *_synthetic_standing(), None)
    run_case("synthetic_jerk", *_synthetic_fall_jerk(), None)

    # Risk score line (matches Flutter “Risk %” = detection.score × 100 from ingest).
    for syn_name, acc, gyro, ori in (
        ("synthetic_standing", *_synthetic_standing()),
        ("synthetic_jerk", *_synthetic_fall_jerk()),
    ):
        feat = _features_from_arrays(acc, gyro, ori).tolist()
        rw = run_inference(
            art,
            feat,
            None,
            predict_fall_type=False,
            acc_window=None,
            gyro_window=None,
            ori_window=None,
        )
        det = _risk_from_window(art, acc, gyro, rw)
        print(
            f"[risk:{syn_name}] score={det['score']:.4f} ({det['score'] * 100:.1f}%)  "
            f"severity={det['severity']}  p_fall={rw['fall_probability']:.4f}  "
            f"peak_acc_g={det['peak_acc_g']:.2f}"
        )
    print()

    csv_root = args.csv_root.resolve()
    paths = _iter_sample_csvs(csv_root, args.max_csv)
    if not paths:
        print(f"No CSVs found under {csv_root} — skipped real-data smoke tests.")
        return 0

    header = f"CSV tests from {csv_root} — first {len(paths)} *_annotated.csv (glob order)"
    if args.summary:
        print(header + " — SUMMARY\n")
        tp = fp = tn = fn = 0
        adl_correct = 0
        adl_total = 0
        skipped = 0
        risk_scores: list[float] = []
        risk_adl: list[float] = []
        risk_fall_lab: list[float] = []
        sev_counts: dict[str, int] = {}
        for path in paths:
            block = _stable_window_from_csv(path)
            if block is None:
                skipped += 1
                continue
            acc, gyro, ori, lab = block
            feat = _features_from_arrays(acc, gyro, ori).tolist()
            raw = run_inference(
                art,
                feat,
                None,
                predict_fall_type=False,
                acc_window=None,
                gyro_window=None,
                ori_window=None,
            )
            det = _risk_from_window(art, acc, gyro, raw)
            sc = float(det["score"])
            risk_scores.append(sc)
            sev = str(det["severity"])
            sev_counts[sev] = sev_counts.get(sev, 0) + 1
            true_fall = lab in FALL_LABELS
            if true_fall:
                risk_fall_lab.append(sc)
            else:
                risk_adl.append(sc)
            pred_fall = bool(raw["is_fall"])
            if true_fall and pred_fall:
                tp += 1
            elif true_fall and not pred_fall:
                fn += 1
            elif not true_fall and pred_fall:
                fp += 1
            else:
                tn += 1
            if not true_fall:
                adl_total += 1
                if raw["branch"] == "adl" and str(raw.get("activity_label")) == lab:
                    adl_correct += 1
        n = tp + fp + tn + fn
        print(f"  Segments evaluated : {n}  (skipped no stable {WINDOW}-row block: {skipped})")
        print("  Binary fall vs CSV label (fall codes FOL/FKL/BSC/SDL):")
        print(f"    TP (label fall, pred fall) : {tp}")
        print(f"    FN (label fall, pred ADL)  : {fn}")
        print(f"    FP (label ADL, pred fall) : {fp}")
        print(f"    TN (label ADL, pred ADL)   : {tn}")
        if n:
            print(f"    Accuracy (4-cell)         : {(tp + tn) / n:.4f}")
        if adl_total:
            print(f"  ADL activity exact match    : {adl_correct}/{adl_total} = {adl_correct / adl_total:.4f}")
        print()
        print("  Risk score (ingest parity: detector_state.build_detection_payload):")
        print(
            "    score = min(1, max(p_fall, peak_acc_g/5*0.3 + p_fall*0.7)); "
            "severity from p_fall vs threshold + impact gates"
        )
        if risk_scores:
            print(f"    All windows    : min={min(risk_scores):.4f}  max={max(risk_scores):.4f}  "
                  f"mean={statistics.mean(risk_scores):.4f}  median={statistics.median(risk_scores):.4f}")
        if risk_adl:
            print(f"    CSV label ADL  : mean={statistics.mean(risk_adl):.4f}  "
                  f"median={statistics.median(risk_adl):.4f}  (n={len(risk_adl)})")
        if risk_fall_lab:
            print(f"    CSV label fall : mean={statistics.mean(risk_fall_lab):.4f}  "
                  f"median={statistics.median(risk_fall_lab):.4f}  (n={len(risk_fall_lab)})")
        print(f"    Severity counts: {dict(sorted(sev_counts.items()))}")
        print(
            "\n  Note: one window is taken from the first long same-label segment per file; "
            "this is a smoke test, not full sliding-window test-set accuracy.\n"
        )
        return 0

    print(header + ":\n")
    for path in paths:
        block = _stable_window_from_csv(path)
        if block is None:
            print(f"[skip] {path.name}: no stable segment of length >= {WINDOW}")
            continue
        acc, gyro, ori, lab = block
        try:
            rel = path.relative_to(csv_root)
            tag = str(rel)
        except ValueError:
            tag = path.name
        run_case(tag, acc, gyro, ori, lab)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
