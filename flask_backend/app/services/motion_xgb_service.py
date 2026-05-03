"""Re-export; canonical implementation: ``scripts/inference/motion_pipeline.py``."""

from __future__ import annotations

import sys

from flask_backend.app.settings import scripts_dir

_scripts = scripts_dir()
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from inference.motion_pipeline import (
    InferenceArtifacts,
    _fall_type_vector_from_windows,
    load_artifacts,
    run_inference,
)

__all__ = [
    "InferenceArtifacts",
    "load_artifacts",
    "run_inference",
    "_fall_type_vector_from_windows",
]
