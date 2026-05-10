"""Motion inference request/response."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MotionInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enhanced_features: list[float] | None = Field(
        default=None,
        description="144-D orientation-invariant feature vector (v2). Optional when acc_window is provided — "
        "the server always rebuilds features from raw windows for training parity.",
    )
    fall_type_features: list[float] | None = Field(
        default=None,
        description="Length = fall_type_raw_dim (263) when requesting 4-class fall type.",
    )
    predict_fall_type: bool = Field(default=True)
    acc_window: list[list[float]] | None = Field(
        default=None,
        description="Optional accelerometer (m/s²): 128×3 for training-parity 144-D server features, or 300×3 for "
        "fall-type 263-D extraction when fall_type artifacts are enabled.",
    )
    gyro_window: list[list[float]] | None = Field(
        default=None,
        description="Optional gyroscope (rad/s). Same row count as acc_window when present; zeros if null.",
    )
    ori_window: list[list[float]] | None = Field(
        default=None,
        description="Optional orientation degrees (azimuth, pitch, roll). Same row count as acc_window; zeros if null.",
    )

    @field_validator("acc_window", "gyro_window", "ori_window")
    @classmethod
    def _validate_sensor_window(
        cls,
        v: list[list[float]] | None,
    ) -> list[list[float]] | None:
        if v is None:
            return None
        n = len(v)
        if n not in (128, 300):
            raise ValueError("sensor window must have 128 or 300 rows")
        for i, row in enumerate(v):
            if len(row) != 3:
                raise ValueError(f"sensor row {i} must have 3 columns (x,y,z)")
        return v

    @model_validator(mode="after")
    def _require_input_source(self) -> MotionInferenceRequest:
        if self.acc_window is None and not self.enhanced_features:
            raise ValueError("Provide either acc_window (recommended) or enhanced_features (144-D).")
        return self

    @model_validator(mode="after")
    def _sensor_windows_same_length(self) -> MotionInferenceRequest:
        lengths: list[int] = []
        for w in (self.acc_window, self.gyro_window, self.ori_window):
            if w is not None:
                lengths.append(len(w))
        if len(set(lengths)) > 1:
            raise ValueError("acc_window, gyro_window, ori_window must have the same row count when provided")
        return self


class MotionInferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_fall: bool
    fall_probability: float
    fall_threshold: float
    branch: str
    activity_label: str | None = None
    activity_class_index: int | None = None
    fall_type_code: str | None = None
    fall_type_label: str | None = None
    fall_type_class_index: int | None = None
    fall_type_skipped_reason: str | None = None
    schema_version: str
