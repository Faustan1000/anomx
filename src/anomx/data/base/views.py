"""Standardized materialized views over an :class:`ObservationSet`.

The source observation set stays canonical; algorithms request the
materialization that matches their capabilities:

```
ObservationSet
    ├── TabularView      n_samples × n_features matrix
    ├── SequenceView     per-sequence ordered frames
    ├── WindowView       fixed-size sliding windows
    ├── TimeSeriesView   time-indexed frame (darts-compatible)
    └── GraphView        node features plus relational overlay
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from anomx.data.base.relations import RelationSet

if TYPE_CHECKING:
    from anomx.data.base.observation_set import ObservationSet


@dataclass(slots=True)
class BaseObservationView:
    """Common surface for all materialized observation views."""

    source: ObservationSet
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TabularView(BaseObservationView):
    """Unordered `n_samples × n_features` matrix materialization."""

    matrix: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_numpy(self) -> Any:
        return self.matrix.to_numpy(dtype=float)


@dataclass(slots=True)
class SequenceView(BaseObservationView):
    """Ordered frames grouped by sequence (or entity) identifier."""

    sequences: dict[Any, pd.DataFrame] = field(default_factory=dict)

    @property
    def sequence_count(self) -> int:
        return len(self.sequences)

    def lengths(self) -> dict[Any, int]:
        return {key: int(len(frame)) for key, frame in self.sequences.items()}


@dataclass(slots=True)
class ObservationWindow:
    """One fixed-size window sliced out of an ordered observation frame."""

    window_id: int
    group: Any
    start_position: int
    end_position: int
    frame: pd.DataFrame


@dataclass(slots=True)
class WindowView(BaseObservationView):
    """Sliding fixed-size windows over ordered observations."""

    windows: list[ObservationWindow] = field(default_factory=list)
    window_size: int = 0
    stride: int = 1

    @property
    def window_count(self) -> int:
        return len(self.windows)


@dataclass(slots=True)
class TimeSeriesView(BaseObservationView):
    """Time-indexed materialization for temporal models."""

    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    time_column: str = "timestamp"
    value_columns: list[str] = field(default_factory=list)

    def to_darts(self) -> Any:
        """Convert into a `darts.TimeSeries` (requires the `darts` extra)."""
        try:
            from darts import TimeSeries
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Converting to darts requires `pip install anomx[darts]`.") from exc
        return TimeSeries.from_dataframe(self.frame, time_col=self.time_column, value_cols=self.value_columns or None)


@dataclass(slots=True)
class GraphView(BaseObservationView):
    """Node feature matrix combined with the relational overlay."""

    nodes: pd.DataFrame = field(default_factory=pd.DataFrame)
    relations: RelationSet | None = None

    @property
    def node_count(self) -> int:
        return int(len(self.nodes))
