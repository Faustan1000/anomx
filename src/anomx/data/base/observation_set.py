"""Canonical observation container shared by every anomx pipeline.

An :class:`ObservationSet` is a collection of observations plus optional
structural information. Independent samples, sequences, time series, panel
data, and graph-backed observations are all expressed through the same
container; algorithms consume standardized views instead of raw storage
formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from anomx._shared import ensure_dataframe
from anomx.data.base.characteristics import (
    DataCharacteristics,
    Grouping,
    Modality,
    ObservationUnit,
    Ordering,
    Structure,
    TemporalSemantics,
    Topology,
)
from anomx.data.base.relations import Hierarchy, RelationSet
from anomx.data.base.views import (
    GraphView,
    ObservationWindow,
    SequenceView,
    TabularView,
    TimeSeriesView,
    WindowView,
)

REGULAR_SAMPLING_RELATIVE_TOLERANCE = 0.01


@dataclass(slots=True)
class ObservationSet:
    """A table of observations together with optional structural overlays."""

    observations: pd.DataFrame
    feature_columns: list[str] = field(default_factory=list)
    target_columns: list[str] = field(default_factory=list)
    entity_column: str | None = None
    sequence_column: str | None = None
    timestamp_column: str | None = None
    position_column: str | None = None
    relations: RelationSet | None = None
    hierarchy: Hierarchy | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observations = ensure_dataframe(self.observations)
        if not self.feature_columns:
            excluded = {self.entity_column, self.sequence_column, self.timestamp_column, self.position_column}
            self.feature_columns = [
                str(column)
                for column in self.observations.select_dtypes(include=["number"]).columns
                if column not in excluded
            ]
        for column in (self.entity_column, self.sequence_column, self.timestamp_column, self.position_column):
            if column is not None and column not in self.observations.columns:
                raise KeyError(f"Structural column '{column}' does not exist in the observation table.")

    @property
    def observation_count(self) -> int:
        return int(len(self.observations))

    def characteristics(self) -> DataCharacteristics:
        """Derive the orthogonal structural properties of this set."""
        return DataCharacteristics(
            ordering=Ordering.ORDERED if self.timestamp_column or self.position_column else Ordering.UNORDERED,
            temporal_semantics=self._temporal_semantics(),
            grouping=self._grouping(),
            topology=Topology.GRAPH_EDGES if self.relations is not None else Topology.NONE,
            structure=Structure.HIERARCHICAL if self.hierarchy is not None else Structure.FLAT,
            modality=self._modality(),
            observation_unit=ObservationUnit.TIMESTEP if self.timestamp_column else ObservationUnit.SAMPLE,
            feature_count=len(self.feature_columns),
            observation_count=self.observation_count,
            has_missing_values=bool(self.observations[self.feature_columns].isna().any().any()) if self.feature_columns else False,
        )

    def as_tabular(self) -> TabularView:
        """Materialize an unordered `n_samples × n_features` matrix."""
        return TabularView(source=self, matrix=self.observations[self.feature_columns].copy())

    def as_sequences(self, *, group_by: str | None = None) -> SequenceView:
        """Materialize per-sequence ordered frames."""
        group_column = group_by or self.sequence_column or self.entity_column
        ordered = self._ordered_observations()
        if group_column is None:
            return SequenceView(source=self, sequences={"__all__": ordered})
        return SequenceView(
            source=self,
            sequences={key: frame.reset_index(drop=True) for key, frame in ordered.groupby(group_column, sort=False)},
        )

    def as_windows(self, *, window_size: int, stride: int = 1, group_by: str | None = None) -> WindowView:
        """Materialize fixed-size sliding windows over ordered observations."""
        if window_size <= 0:
            raise ValueError("Window size must be greater than zero.")
        if stride <= 0:
            raise ValueError("Window stride must be greater than zero.")

        windows: list[ObservationWindow] = []
        for group, frame in self.as_sequences(group_by=group_by).sequences.items():
            for start_position in range(0, max(0, len(frame) - window_size + 1), stride):
                windows.append(
                    ObservationWindow(
                        window_id=len(windows),
                        group=group,
                        start_position=start_position,
                        end_position=start_position + window_size,
                        frame=frame.iloc[start_position:start_position + window_size].reset_index(drop=True),
                    )
                )
        return WindowView(source=self, windows=windows, window_size=window_size, stride=stride)

    def as_timeseries(self) -> TimeSeriesView:
        """Materialize a time-indexed frame for temporal models."""
        if self.timestamp_column is None:
            raise ValueError("Time-series views require a timestamp column.")
        ordered = self._ordered_observations()
        return TimeSeriesView(
            source=self,
            frame=ordered[[self.timestamp_column, *self.feature_columns]].copy(),
            time_column=self.timestamp_column,
            value_columns=list(self.feature_columns),
        )

    def as_graph(self) -> GraphView:
        """Materialize node features with the relational overlay."""
        if self.relations is None:
            raise ValueError("Graph views require a relation set.")
        return GraphView(source=self, nodes=self.observations.copy(), relations=self.relations)

    def _ordered_observations(self) -> pd.DataFrame:
        order_column = self.timestamp_column or self.position_column
        if order_column is None:
            return self.observations.copy()
        return self.observations.sort_values(order_column).reset_index(drop=True)

    def _temporal_semantics(self) -> TemporalSemantics:
        if self.timestamp_column is None:
            return TemporalSemantics.NONE
        timestamps = pd.to_datetime(self.observations[self.timestamp_column], errors="coerce").dropna().sort_values()
        if len(timestamps) < 3:
            return TemporalSemantics.EVENT_TIME
        deltas = timestamps.diff().dropna().dt.total_seconds()
        median_delta = float(deltas.median())
        if median_delta <= 0:
            return TemporalSemantics.EVENT_TIME
        if bool(((deltas - median_delta).abs() <= median_delta * REGULAR_SAMPLING_RELATIVE_TOLERANCE).all()):
            return TemporalSemantics.REGULAR_TIME
        return TemporalSemantics.IRREGULAR_TIME

    def _grouping(self) -> Grouping:
        if self.sequence_column is not None and self.observations[self.sequence_column].nunique() > 1:
            return Grouping.MULTIPLE_SEQUENCES
        if self.entity_column is not None and self.observations[self.entity_column].nunique() > 1:
            return Grouping.MULTIPLE_ENTITIES
        return Grouping.SINGLE_POPULATION

    def _modality(self) -> Modality:
        if not self.feature_columns:
            return Modality.MIXED
        numeric_columns = set(self.observations[self.feature_columns].select_dtypes(include=["number"]).columns)
        if numeric_columns == set(self.feature_columns):
            return Modality.NUMERIC
        if not numeric_columns:
            return Modality.CATEGORICAL
        return Modality.MIXED
