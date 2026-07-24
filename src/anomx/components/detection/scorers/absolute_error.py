"""Deliberately simple absolute-error scorer."""

from __future__ import annotations

from typing import Any

import pandas as pd

from anomx._shared import ensure_dataframe
from anomx.components.base import ResidualScorer


class AbsoluteErrorScorer(ResidualScorer):
    """Pass the absolute model score through as the anomaly score."""

    component_key = "absolute_error"
    component_name = "Absolute Error Scorer"
    component_default_config = {
        "source_column": "model_score",
    }
    component_config_schema = {
        "source_column": {"type": "string"},
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def score(self, data: Any) -> pd.DataFrame:
        frame = ensure_dataframe(data)
        source_column = str(self.config.get("source_column", "model_score"))
        if source_column not in frame.columns:
            raise KeyError(f"Source column '{source_column}' does not exist in scorer input.")

        result = frame.copy()
        result["score"] = result[source_column].astype(float).abs().round(6)
        return result
