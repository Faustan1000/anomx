"""Quantile-threshold detector component."""

from __future__ import annotations

from typing import Any

import pandas as pd

from anomx._shared import ensure_dataframe
from anomx.components.base import AdaptiveThresholdDetector


class QuantileThresholdDetector(AdaptiveThresholdDetector):
    """Flag the top quantile of a score column as anomalous."""

    component_key = "quantile_threshold"
    component_name = "Quantile Threshold Detector"
    component_default_config = {
        "quantile": 0.99,
        "source_column": "score",
    }
    component_config_schema = {
        "quantile": {"type": "number"},
        "source_column": {"type": "string"},
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def detect(self, predictions: Any) -> pd.DataFrame:
        frame = ensure_dataframe(predictions)
        source_column = str(self.config.get("source_column", "score"))
        quantile = min(1.0, max(0.0, float(self.config.get("quantile", 0.99))))
        if source_column not in frame.columns:
            raise KeyError(f"Source column '{source_column}' does not exist in detector input.")

        result = frame.copy()
        values = result[source_column].astype(float)
        threshold = float(values.quantile(quantile)) if len(values) else 0.0
        result["is_anomaly"] = values >= threshold if len(values) else False
        result["severity"] = values.abs().round(4)
        return result
