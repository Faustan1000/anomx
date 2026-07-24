"""Z-score scorer component."""

from __future__ import annotations

from typing import Any

import pandas as pd

from anomx._shared import ensure_dataframe
from anomx.components.base import ResidualScorer


class ZScoreScorer(ResidualScorer):
    """Normalize a model score column with a z-score."""

    component_key = "zscore"
    component_name = "Z-Score Scorer"
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
        values = result[source_column].astype(float)
        std = values.std(ddof=0)
        if std == 0:
            result["zscore"] = 0.0
        else:
            result["zscore"] = ((values - values.mean()) / std).round(6)
        return result
