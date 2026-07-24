"""Shared data contracts flowing between pipeline stages.

```
ObservationSet → Model → InferenceOutput → Scorer → AnomalyScores
             → Calibrator → AnomalyScores → Detector → AnomalyEvents
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from anomx._shared import dataframe_to_records, ensure_dataframe


class ScoreLevel(str, Enum):
    """Alignment level of an anomaly score."""

    SAMPLE = "sample"
    TIMESTEP = "timestep"
    WINDOW = "window"
    SEQUENCE = "sequence"
    FEATURE = "feature"
    ENTITY = "entity"
    NODE = "node"
    EDGE = "edge"
    GRAPH = "graph"


class InferenceKind(str, Enum):
    """What kind of output a normality model produces."""

    FORECAST = "forecast"
    RECONSTRUCTION = "reconstruction"
    EMBEDDING = "embedding"
    DENSITY = "density"
    DISTANCE = "distance"
    CLASSIFICATION = "classification"


@dataclass(slots=True)
class InferenceOutput:
    """Model output aligned with the observations it was computed for."""

    kind: InferenceKind
    frame: pd.DataFrame
    output_columns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frame = ensure_dataframe(self.frame)

    def to_records(self) -> list[dict[str, Any]]:
        return dataframe_to_records(self.frame)


@dataclass(slots=True)
class AnomalyScores:
    """Continuous anomaly scores with explicit alignment information."""

    frame: pd.DataFrame
    score_column: str = "score"
    score_level: ScoreLevel = ScoreLevel.SAMPLE
    is_calibrated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frame = ensure_dataframe(self.frame)
        if self.score_column not in self.frame.columns:
            raise KeyError(f"Score column '{self.score_column}' does not exist in the score table.")

    def values(self) -> pd.Series:
        return self.frame[self.score_column].astype(float)

    def to_records(self) -> list[dict[str, Any]]:
        return dataframe_to_records(self.frame)


@dataclass(slots=True)
class AnomalyEvent:
    """One decided anomaly, possibly spanning an interval."""

    start_time: Any = None
    end_time: Any = None
    peak_score: float = 0.0
    severity: float = 0.0
    score_level: ScoreLevel = ScoreLevel.SAMPLE
    affected_features: list[str] = field(default_factory=list)
    affected_entities: list[Any] = field(default_factory=list)
    observation_ids: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": str(self.start_time) if self.start_time is not None else None,
            "end_time": str(self.end_time) if self.end_time is not None else None,
            "peak_score": float(self.peak_score),
            "severity": float(self.severity),
            "score_level": self.score_level.value,
            "affected_features": list(self.affected_features),
            "affected_entities": [str(entity) for entity in self.affected_entities],
            "observation_ids": [str(observation_id) for observation_id in self.observation_ids],
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class AnomalyEvents:
    """Detector decision output: events plus the labeled score table."""

    events: list[AnomalyEvent] = field(default_factory=list)
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    label_column: str = "is_anomaly"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def event_count(self) -> int:
        return len(self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_count": self.event_count,
            "events": [event.to_dict() for event in self.events],
            "label_column": self.label_column,
            "metadata": dict(self.metadata),
        }
