"""Deliberately trivial baseline model used to exercise the pipeline system."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from anomx._shared import ensure_dataframe
from anomx.components.base import (
    ComponentCapabilities,
    DataStructure,
    ModelSignature,
    NormalityModel,
    Predictive,
)
from anomx.data.base.characteristics import Modality


class ConstantBaselineModel(NormalityModel, Predictive):
    """Predict the per-feature training mean for every observation.

    The model exists to validate orchestration end to end: it trains instantly,
    predicts deterministically, and produces residual-style `model_score`
    output that any scorer can consume.
    """

    component_key = "constant_baseline"
    component_icon = "Activity"
    signature = ModelSignature(
        structures={DataStructure.TABULAR, DataStructure.TEMPORAL_SEQUENCE},
        modalities={Modality.NUMERIC},
    )
    component_name = "Constant Baseline"
    component_default_config = {
        "feature_columns": [],
    }
    component_config_schema = {
        "feature_columns": {"type": "array"},
    }
    capabilities: ClassVar[ComponentCapabilities] = ComponentCapabilities(
        supports_sequences=True,
        supports_streaming=True,
        streaming_inference=True,
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.feature_columns: list[str] = []
        self.baseline: dict[str, float] = {}

    def fit(self, data: Any) -> None:
        frame = ensure_dataframe(data)
        configured_columns = self.config.get("feature_columns")
        self.feature_columns = (
            [str(column) for column in configured_columns]
            if configured_columns
            else frame.select_dtypes(include=["number"]).columns.tolist()
        )
        self.baseline = {column: float(frame[column].astype(float).mean()) for column in self.feature_columns}

    def predict(self, data: Any) -> pd.DataFrame:
        if not self.baseline:
            raise RuntimeError("Model must be fit before predict() is called.")

        frame = ensure_dataframe(data)
        result = frame.copy()
        residuals = pd.DataFrame(index=frame.index)
        for column, expected_value in self.baseline.items():
            result[f"expected__{column}"] = expected_value
            residuals[column] = (frame[column].astype(float) - expected_value).abs()
        result["model_score"] = residuals.mean(axis=1).round(6)
        return result

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(
                {
                    "baseline": self.baseline,
                    "config": self.config,
                    "feature_columns": self.feature_columns,
                },
                handle,
            )

    def load(self, path: str) -> ConstantBaselineModel:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        self.baseline = {str(column): float(value) for column, value in payload["baseline"].items()}
        self.config = dict(payload["config"])
        self.feature_columns = list(payload["feature_columns"])
        return self
