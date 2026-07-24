"""Forecasting-based anomaly model."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from anomx._shared import ensure_dataframe
from anomx.components.base import DataStructure, ModelSignature, NormalityModel, Predictive
from anomx.data.base.characteristics import Modality


class RollingWindowForecastModel(NormalityModel, Predictive):
    """Estimate expected values from a rolling window and score residuals."""

    component_key = "rolling_window_forecast"
    component_icon = "TrendUp01"
    signature = ModelSignature(
        structures={DataStructure.TEMPORAL_SEQUENCE, DataStructure.SEQUENCE},
        modalities={Modality.NUMERIC},
    )
    component_name = "Rolling Window Forecast"
    component_default_config = {
        "feature_columns": [],
        "min_periods": 3,
        "window": 24,
    }
    component_config_schema = {
        "feature_columns": {"type": "array"},
        "min_periods": {"type": "integer"},
        "window": {"type": "integer"},
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.feature_columns: list[str] = []
        self.window = int(self.config.get("window", 24))
        self.min_periods = int(self.config.get("min_periods", 3))

    def fit(self, data: Any) -> None:
        frame = ensure_dataframe(data)
        self.feature_columns = self._resolve_feature_columns(frame)

    def predict(self, data: Any) -> pd.DataFrame:
        if not self.feature_columns:
            raise RuntimeError("Model must be fit before predict() is called.")

        frame = ensure_dataframe(data)
        values = frame[self.feature_columns].astype(float)
        min_periods = min(max(1, self.min_periods), self.window)
        rolling = values.shift(1).rolling(window=self.window, min_periods=min_periods).mean()
        fallback = values.shift(1).expanding(min_periods=1).mean()
        baseline = rolling.combine_first(fallback).bfill().fillna(0.0)
        residuals = (values - baseline).abs()

        result = frame.copy()
        for column in self.feature_columns:
            result[f"forecast__{column}"] = baseline[column].round(6)
        result["model_score"] = residuals.mean(axis=1).round(6)
        return result

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(
                {
                    "config": self.config,
                    "feature_columns": self.feature_columns,
                    "min_periods": self.min_periods,
                    "window": self.window,
                },
                handle,
            )

    def load(self, path: str) -> RollingWindowForecastModel:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        self.config = dict(payload["config"])
        self.feature_columns = list(payload["feature_columns"])
        self.min_periods = int(payload["min_periods"])
        self.window = int(payload["window"])
        return self

    def _resolve_feature_columns(self, frame: pd.DataFrame) -> list[str]:
        configured_columns = self.config.get("feature_columns")
        if configured_columns:
            return [str(column) for column in configured_columns]
        return frame.select_dtypes(include=["number"]).columns.tolist()
