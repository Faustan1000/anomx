"""Orthogonal descriptive properties of an observation set.

Instead of hard-coding mutually exclusive dataset types (samples, sequences,
time series, graphs, hierarchies), an :class:`ObservationSet` is described by
independent structural dimensions. Algorithms declare requirements against
these dimensions through :class:`anomx.components.base.ComponentCapabilities`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Ordering(str, Enum):
    UNORDERED = "unordered"
    ORDERED = "ordered"


class TemporalSemantics(str, Enum):
    NONE = "none"
    EVENT_TIME = "event_time"
    REGULAR_TIME = "regular_time"
    IRREGULAR_TIME = "irregular_time"


class Grouping(str, Enum):
    SINGLE_POPULATION = "single_population"
    MULTIPLE_ENTITIES = "multiple_entities"
    MULTIPLE_SEQUENCES = "multiple_sequences"


class Topology(str, Enum):
    NONE = "none"
    GRAPH_EDGES = "graph_edges"
    SPATIAL_ADJACENCY = "spatial_adjacency"


class Structure(str, Enum):
    FLAT = "flat"
    NESTED = "nested"
    HIERARCHICAL = "hierarchical"


class Modality(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    MIXED = "mixed"


class ObservationUnit(str, Enum):
    SAMPLE = "sample"
    TIMESTEP = "timestep"
    WINDOW = "window"
    SEQUENCE = "sequence"
    NODE = "node"
    EDGE = "edge"
    GRAPH = "graph"


@dataclass(frozen=True, slots=True)
class DataCharacteristics:
    """Derived structural description of an observation set."""

    ordering: Ordering = Ordering.UNORDERED
    temporal_semantics: TemporalSemantics = TemporalSemantics.NONE
    grouping: Grouping = Grouping.SINGLE_POPULATION
    topology: Topology = Topology.NONE
    structure: Structure = Structure.FLAT
    modality: Modality = Modality.NUMERIC
    observation_unit: ObservationUnit = ObservationUnit.SAMPLE
    feature_count: int = 0
    observation_count: int = 0
    has_missing_values: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_temporal(self) -> bool:
        return self.temporal_semantics is not TemporalSemantics.NONE

    @property
    def is_ordered(self) -> bool:
        return self.ordering is Ordering.ORDERED

    @property
    def is_multivariate(self) -> bool:
        return self.feature_count > 1

    @property
    def has_topology(self) -> bool:
        return self.topology is not Topology.NONE

    @property
    def has_hierarchy(self) -> bool:
        return self.structure is Structure.HIERARCHICAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordering": self.ordering.value,
            "temporal_semantics": self.temporal_semantics.value,
            "grouping": self.grouping.value,
            "topology": self.topology.value,
            "structure": self.structure.value,
            "modality": self.modality.value,
            "observation_unit": self.observation_unit.value,
            "feature_count": self.feature_count,
            "observation_count": self.observation_count,
            "has_missing_values": self.has_missing_values,
            "metadata": dict(self.metadata),
        }
