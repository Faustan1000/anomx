"""Reconstruction-based anomaly model."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from anomx._shared import ensure_dataframe
from anomx.components.base import DataStructure, ModelSignature, NormalityModel, Reconstructive
from anomx.data.base.characteristics import Modality


class PcaReconstructionModel(NormalityModel, Reconstructive):
    """Project numeric features into a low-rank space and score reconstruction error."""

    component_key = "pca_reconstruction"
    component_icon = "Cube01"
    signature = ModelSignature(
        structures={DataStructure.TABULAR, DataStructure.TEMPORAL_SEQUENCE},
        modalities={Modality.NUMERIC},
    )
    component_name = "PCA Reconstruction"
    component_default_config = {
        "feature_columns": [],
        "n_components": 2,
    }
    component_config_schema = {
        "feature_columns": {"type": "array"},
        "n_components": {"type": "integer"},
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.feature_columns: list[str] = []
        self.scaler: StandardScaler | None = None
        self.model: PCA | None = None

    def fit(self, data: Any) -> None:
        frame = ensure_dataframe(data)
        self.feature_columns = self._resolve_feature_columns(frame)
        features = frame[self.feature_columns].astype(float)
        requested_components = int(self.config.get("n_components", 2))
        n_components = max(1, min(requested_components, features.shape[1]))
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(features)
        self.model = PCA(n_components=n_components, random_state=42)
        self.model.fit(scaled)

    def predict(self, data: Any) -> pd.DataFrame:
        if self.scaler is None or self.model is None:
            raise RuntimeError("Model must be fit before predict() is called.")

        frame = ensure_dataframe(data)
        features = frame[self.feature_columns].astype(float)
        scaled = self.scaler.transform(features)
        projected = self.model.transform(scaled)
        reconstructed_scaled = self.model.inverse_transform(projected)
        reconstructed = pd.DataFrame(
            self.scaler.inverse_transform(reconstructed_scaled),
            columns=self.feature_columns,
            index=frame.index,
        )
        errors = ((features - reconstructed) ** 2).mean(axis=1)

        result = frame.copy()
        for column in self.feature_columns:
            result[f"reconstruction__{column}"] = reconstructed[column].round(6)
        result["model_score"] = errors.round(6)
        return result

    def save(self, path: str) -> None:
        if self.scaler is None or self.model is None:
            raise RuntimeError("Model must be fit before save() is called.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(
                {
                    "config": self.config,
                    "feature_columns": self.feature_columns,
                    "model": self.model,
                    "scaler": self.scaler,
                },
                handle,
            )

    def load(self, path: str) -> PcaReconstructionModel:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        self.config = dict(payload["config"])
        self.feature_columns = list(payload["feature_columns"])
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        return self

    def _resolve_feature_columns(self, frame: pd.DataFrame) -> list[str]:
        configured_columns = self.config.get("feature_columns")
        if configured_columns:
            return [str(column) for column in configured_columns]
        return frame.select_dtypes(include=["number"]).columns.tolist()
