"""Data signatures declared by components.

A signature states which data structures, modalities, and additional
requirements an implementation supports. Orchestrators match signatures against
observation-set characteristics before building pipelines, and platforms can
render them as capability badges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from anomx.data.base.characteristics import Modality


class DataStructure(str, Enum):
    TABULAR = "tabular"
    SEQUENCE = "sequence"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    PANEL = "panel"
    GRAPH = "graph"
    HIERARCHY = "hierarchy"


@dataclass(frozen=True, slots=True)
class ModelSignature:
    """Structural contract an implementation supports.

    ```python
    signature = ModelSignature(
        structures={DataStructure.TEMPORAL_SEQUENCE},
        modalities={Modality.NUMERIC},
        requirements={"regular_or_known_timestamps"},
    )
    ```
    """

    structures: frozenset[DataStructure] = field(default_factory=frozenset)
    modalities: frozenset[Modality] = field(default_factory=frozenset)
    requirements: frozenset[str] = field(default_factory=frozenset)

    def __init__(
        self,
        *,
        structures: set[DataStructure] | frozenset[DataStructure] | None = None,
        modalities: set[Modality] | frozenset[Modality] | None = None,
        requirements: set[str] | frozenset[str] | None = None,
    ) -> None:
        object.__setattr__(self, "structures", frozenset(structures or ()))
        object.__setattr__(self, "modalities", frozenset(modalities or ()))
        object.__setattr__(self, "requirements", frozenset(requirements or ()))

    def supports_structure(self, structure: DataStructure) -> bool:
        return not self.structures or structure in self.structures

    def supports_modality(self, modality: Modality) -> bool:
        return not self.modalities or modality in self.modalities

    def to_dict(self) -> dict[str, Any]:
        return {
            "structures": sorted(structure.value for structure in self.structures),
            "modalities": sorted(modality.value for modality in self.modalities),
            "requirements": sorted(self.requirements),
        }
