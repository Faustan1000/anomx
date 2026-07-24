"""Representation-based anomaly model."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest

from anomx._shared import ensure_dataframe
from anomx.components.base import (
    BoundaryEstimating,
    DataStructure,
    ModelSignature,
    NormalityModel,
    Representational,
)
from anomx.data.base.characteristics import Modality


class IsolationForestModel(NormalityModel, BoundaryEstimating, Representational):
    """Train an isolation forest on numeric columns and expose anomaly scores."""

    component_key = "isolation_forest"
    component_icon = "Dataflow03"
    signature = ModelSignature(
        structures={DataStructure.TABULAR},
        modalities={Modality.NUMERIC},
    )
    component_name = "Isolation Forest"
    component_default_config = {
        "contamination": 0.15,
        "feature_columns": [],
        "n_estimators": 100,
        "random_state": 42,
    }
    component_config_schema = {
        "contamination": {"type": "number"},
        "feature_columns": {"type": "array"},
        "n_estimators": {"type": "integer"},
        "random_state": {"type": "integer"},
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.model: IsolationForest | None = None
        self.feature_columns: list[str] = []

    def fit(self, data: Any) -> None:
        frame = ensure_dataframe(data)
        self.feature_columns = self._resolve_feature_columns(frame)
        self.model = IsolationForest(
            contamination=float(self.config.get("contamination", 0.15)),
            n_estimators=int(self.config.get("n_estimators", 100)),
            random_state=int(self.config.get("random_state", 42)),
            n_jobs=1,
        )
        self.model.fit(frame[self.feature_columns])

    def predict(self, data: Any) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model must be fit before predict() is called.")

        frame = ensure_dataframe(data)
        features = frame[self.feature_columns]
        result = frame.copy()
        result["model_score"] = (-self.model.score_samples(features)).round(6)
        result["model_prediction"] = self.model.predict(features)
        return result

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model must be fit before save() is called.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump({"model": self.model, "feature_columns": self.feature_columns}, handle)

    def load(self, path: str) -> IsolationForestModel:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        self.model = payload["model"]
        self.feature_columns = list(payload["feature_columns"])
        return self

    def _resolve_feature_columns(self, frame: pd.DataFrame) -> list[str]:
        configured_columns = self.config.get("feature_columns")
        if configured_columns:
            return [str(column) for column in configured_columns]
        return frame.select_dtypes(include=["number"]).columns.tolist()
