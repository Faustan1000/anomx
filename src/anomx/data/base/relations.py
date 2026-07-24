"""Optional relational overlays for observation sets.

Graphs and hierarchies are not separate top-level dataset types. They are
additional structural dimensions that can be attached to any
:class:`ObservationSet` and consumed by views or models that support them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from anomx._shared import ensure_dataframe


@dataclass(slots=True)
class RelationSet:
    """Edge list connecting observations or entities."""

    edges: pd.DataFrame
    source_column: str = "source"
    target_column: str = "target"
    weight_column: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.edges = ensure_dataframe(self.edges)
        for column in (self.source_column, self.target_column):
            if column not in self.edges.columns:
                raise KeyError(f"Relation column '{column}' does not exist in the edge table.")

    @property
    def edge_count(self) -> int:
        return int(len(self.edges))

    def neighbors(self, node: Any) -> list[Any]:
        outgoing = self.edges.loc[self.edges[self.source_column] == node, self.target_column]
        incoming = self.edges.loc[self.edges[self.target_column] == node, self.source_column]
        return sorted(set(outgoing.tolist()) | set(incoming.tolist()), key=str)


@dataclass(slots=True)
class Hierarchy:
    """Parent-child relationships between entities."""

    links: pd.DataFrame
    child_column: str = "child"
    parent_column: str = "parent"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.links = ensure_dataframe(self.links)
        for column in (self.child_column, self.parent_column):
            if column not in self.links.columns:
                raise KeyError(f"Hierarchy column '{column}' does not exist in the link table.")

    def parent_of(self, child: Any) -> Any | None:
        matches = self.links.loc[self.links[self.child_column] == child, self.parent_column]
        return matches.iloc[0] if len(matches) else None

    def children_of(self, parent: Any) -> list[Any]:
        matches = self.links.loc[self.links[self.parent_column] == parent, self.child_column]
        return matches.tolist()

    def roots(self) -> list[Any]:
        parents = set(self.links[self.parent_column].tolist())
        children = set(self.links[self.child_column].tolist())
        return sorted(parents - children, key=str)
