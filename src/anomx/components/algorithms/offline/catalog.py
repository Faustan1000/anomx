"""Static implementation catalog for the default offline pipeline."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from anomx._shared import normalise_component_key
from anomx.components.detection.detectors import BaseDetector, ThresholdDetector
from anomx.components.detection.scorers import BaseScorer, ZScoreScorer
from anomx.components.models import (
    BaseAnomalyModel,
    IsolationForestModel,
    PcaReconstructionModel,
    RollingWindowForecastModel,
)
from anomx.data.connectors import BaseConnector, LocalFSConnector

ComponentKind = Literal["connector", "model", "scorer", "detector"]
ComponentClass: TypeAlias = type[Any]

CONNECTOR_IMPLEMENTATIONS: dict[str, type[BaseConnector]] = {
    "local_fs": LocalFSConnector,
}

MODEL_IMPLEMENTATIONS: dict[str, type[BaseAnomalyModel]] = {
    "isolation_forest": IsolationForestModel,
    "pca_reconstruction": PcaReconstructionModel,
    "rolling_window_forecast": RollingWindowForecastModel,
}

SCORER_IMPLEMENTATIONS: dict[str, type[BaseScorer]] = {
    "zscore": ZScoreScorer,
}

DETECTOR_IMPLEMENTATIONS: dict[str, type[BaseDetector]] = {
    "threshold": ThresholdDetector,
}

IMPLEMENTATION_CATALOGS: dict[ComponentKind, dict[str, ComponentClass]] = {
    "connector": CONNECTOR_IMPLEMENTATIONS,
    "model": MODEL_IMPLEMENTATIONS,
    "scorer": SCORER_IMPLEMENTATIONS,
    "detector": DETECTOR_IMPLEMENTATIONS,
}

COMPONENT_IMPLEMENTATIONS = IMPLEMENTATION_CATALOGS


def resolve_implementation(name: str, kind: ComponentKind) -> ComponentClass:
    """Resolve one configured implementation from the static catalog."""
    normalized_name = normalise_component_key(name)
    catalog = IMPLEMENTATION_CATALOGS[kind]
    implementation = catalog.get(normalized_name)
    if implementation is not None:
        return implementation
    available = ", ".join(sorted(catalog)) or "none"
    raise KeyError(f"Unknown {kind} implementation '{name}'. Available: {available}.")
