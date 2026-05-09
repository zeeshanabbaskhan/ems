"""Motion inference request/response."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MotionInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enhanced_features: list[float] = Field(
        ...,
        description="Length = inference_manifest enhanced_feature_dim (128 after retraining with MobiAct feature set).",
    )
    fall_type_features: list[float] | None = Field(
        default=None,
        description="Length = fall_type_raw_dim (263) when requesting 4-class fall type.",
    )
    predict_fall_type: bool = Field(default=True)
    acc_window: list[list[float]] | None = Field(
        default=None,
        description="Optional 300×3 accelerometer window (m/s²). If fall is predicted and fall_type_features "
        "are omitted, the server builds 263-D fall-type features from acc/gyro/ori windows.",
    )
    gyro_window: list[list[float]] | None = Field(
        default=None,
        description="Optional 300×3 gyroscope (rad/s). Defaults to zeros if null.",
    )
    ori_window: list[list[float]] | None = Field(
        default=None,
        description="Optional 300×3 orientation (azimuth, pitch, roll). Defaults to zeros if null.",
    )

    @field_validator("acc_window", "gyro_window", "ori_window")
    @classmethod
    def _validate_sensor_window(
        cls,
        v: list[list[float]] | None,
    ) -> list[list[float]] | None:
        if v is None:
            return None
        if len(v) != 300:
            raise ValueError("sensor window must have exactly 300 rows")
        for i, row in enumerate(v):
            if len(row) != 3:
                raise ValueError(f"sensor row {i} must have 3 columns (x,y,z)")
        return v


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
