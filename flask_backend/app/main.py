"""FastAPI: SisFall / elderly monitoring — full REST + ML inference."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import sklearn
import xgboost
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from flask_backend.app.database import init_schema, seed_default_admin
from flask_backend.app.elder_credential_allocator import ELDER_CREDENTIAL_RULES_VERSION
from flask_backend.app.monitoring_routes import router as monitoring_router, set_inference_runtime
from flask_backend.app.services.motion_xgb_service import InferenceArtifacts, load_artifacts
from flask_backend.app.settings import inference_manifest_path, model_root


def _versions() -> dict[str, str]:
    return {
        "numpy": __import__("numpy").__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


_state: dict[str, InferenceArtifacts | str | None] = {"art": None, "load_error": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    init_schema()
    seed_default_admin()
    try:
        _state["art"] = load_artifacts(inference_manifest_path(), model_root())
        _state["load_error"] = None
    except Exception as exc:
        _state["art"] = None
        _state["load_error"] = str(exc)
    set_inference_runtime(_state)
    yield


app = FastAPI(title="SisFall Elder Monitoring", version="2.0.0", lifespan=lifespan)
app.include_router(monitoring_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    res = await call_next(request)
    res.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
    return res


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "inference_ready": _state.get("art") is not None,
        "load_error": _state.get("load_error"),
        "versions": _versions(),
        "product": "SisFall_dataset_monitoring",
        "credential_rules_version": ELDER_CREDENTIAL_RULES_VERSION,
    }


@app.get("/api/v1/inference/status")
def inference_status():
    art = _state.get("art")
    if art is None:
        from fastapi import HTTPException

        raise HTTPException(503, detail=_state.get("load_error", "not loaded"))
    return {
        "loaded": True,
        "schema_version": art.manifest.get("schema_version"),
        "enhanced_feature_dim": art.enhanced_dim,
        "fall_type_raw_dim": art.fall_type_dim,
        "fall_type_enabled": getattr(art, "fall_type_enabled", False),
        "fall_threshold": art.fall_threshold,
        "model_root": str(model_root()),
    }
