"""Darts-backed forecasting model (requires the `darts` extra)."""

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


class DartsNaiveSeasonalModel(NormalityModel, Predictive):
    """One-step-ahead naive seasonal forecasts backed by `darts`.

    Darts is the reference ecosystem for the time-series side of anomx; this
    component demonstrates the integration pattern with the smallest possible
    darts model. Residuals against the forecast land in `model_score`.
    """

    component_key = "darts_naive_seasonal"
    component_icon = "LineChartUp01"
    signature = ModelSignature(
        structures={DataStructure.TEMPORAL_SEQUENCE},
        modalities={Modality.NUMERIC},
        requirements={"regular_or_known_timestamps"},
    )
    component_name = "Darts Naive Seasonal"
    component_default_config = {
        "feature_columns": [],
        "seasonal_period": 1,
    }
    component_config_schema = {
        "feature_columns": {"type": "array"},
        "seasonal_period": {"type": "integer"},
    }
    capabilities: ClassVar[ComponentCapabilities] = ComponentCapabilities(
        supports_unordered_samples=False,
        supports_sequences=True,
        supports_irregular_sampling=False,
        requires_regular_sampling=True,
    )
    preferred_view: ClassVar[str] = "timeseries"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.seasonal_period = max(1, int(self.config.get("seasonal_period", 1)))
        self.feature_columns: list[str] = []

    @staticmethod
    def _require_darts() -> Any:
        try:
            from darts.models import NaiveSeasonal
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("The Darts Naive Seasonal model requires `pip install anomx[darts]`.") from exc
        return NaiveSeasonal

    def fit(self, data: Any) -> None:
        self._require_darts()
        frame = ensure_dataframe(data)
        configured_columns = self.config.get("feature_columns")
        self.feature_columns = (
            [str(column) for column in configured_columns]
            if configured_columns
            else frame.select_dtypes(include=["number"]).columns.tolist()
        )

    def predict(self, data: Any) -> pd.DataFrame:
        if not self.feature_columns:
            raise RuntimeError("Model must be fit before predict() is called.")

        self._require_darts()
        frame = ensure_dataframe(data)
        values = frame[self.feature_columns].astype(float)
        # A NaiveSeasonal(K) one-step-ahead historical forecast is the value K steps back.
        forecast = values.shift(self.seasonal_period).bfill().fillna(0.0)
        residuals = (values - forecast).abs()

        result = frame.copy()
        for column in self.feature_columns:
            result[f"forecast__{column}"] = forecast[column].round(6)
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
                    "seasonal_period": self.seasonal_period,
                },
                handle,
            )

    def load(self, path: str) -> DartsNaiveSeasonalModel:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        self.config = dict(payload["config"])
        self.feature_columns = list(payload["feature_columns"])
        self.seasonal_period = int(payload["seasonal_period"])
        return self
