"""Threshold-based detector component."""

from __future__ import annotations

from typing import Any

import pandas as pd

from anomx._shared import ensure_dataframe
from anomx.components.base import StaticThresholdDetector


class ThresholdDetector(StaticThresholdDetector):
    """Flag anomalies whenever a score column crosses a fixed threshold."""

    component_key = "threshold"
    component_name = "Threshold Detector"
    component_default_config = {
        "source_column": "zscore",
        "threshold": 1.5,
    }
    component_config_schema = {
        "source_column": {"type": "string"},
        "threshold": {"type": "number"},
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def detect(self, predictions: Any) -> pd.DataFrame:
        frame = ensure_dataframe(predictions)
        source_column = str(self.config.get("source_column", "zscore"))
        threshold = float(self.config.get("threshold", 1.5))
        if source_column not in frame.columns:
            raise KeyError(f"Source column '{source_column}' does not exist in detector input.")

        result = frame.copy()
        result["is_anomaly"] = result[source_column].astype(float).abs() >= threshold
        result["severity"] = result[source_column].astype(float).abs().round(4)
        return result
