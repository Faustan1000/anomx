"""Dataset adapters producing the canonical observation representation.

Adapters are the only place where storage formats (records, time-series
batches, files, platform stores) are known. Everything downstream operates on
:class:`ObservationSet`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from anomx._shared import ensure_dataframe
from anomx.data.base.observation_set import ObservationSet
from anomx.data.sequences import TimeSeriesBatch


class DatasetAdapter(ABC):
    """Read data from a concrete source and produce an observation set."""

    @abstractmethod
    def load(self) -> ObservationSet:
        """Materialize the source into the canonical representation."""


class RecordsAdapter(DatasetAdapter):
    """Adapt in-memory records (e.g. independent JSON samples) into observations."""

    def __init__(
        self,
        records: list[dict[str, Any]] | pd.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        entity_column: str | None = None,
        timestamp_column: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.records = records
        self.feature_columns = list(feature_columns or [])
        self.entity_column = entity_column
        self.timestamp_column = timestamp_column
        self.metadata = dict(metadata or {})

    def load(self) -> ObservationSet:
        return ObservationSet(
            observations=ensure_dataframe(self.records),
            feature_columns=list(self.feature_columns),
            entity_column=self.entity_column,
            timestamp_column=self.timestamp_column,
            metadata=dict(self.metadata),
        )


class TimeSeriesBatchAdapter(DatasetAdapter):
    """Adapt a :class:`TimeSeriesBatch` into a temporal observation set."""

    def __init__(self, batch: TimeSeriesBatch, *, metadata: dict[str, Any] | None = None) -> None:
        self.batch = batch
        self.metadata = dict(metadata or {})

    def load(self) -> ObservationSet:
        frame = self.batch.sorted_frame()
        value_columns = self.batch.value_columns or frame.select_dtypes(include=["number"]).columns.tolist()
        return ObservationSet(
            observations=frame,
            feature_columns=[str(column) for column in value_columns],
            timestamp_column=self.batch.time_column,
            metadata=dict(self.metadata),
        )
