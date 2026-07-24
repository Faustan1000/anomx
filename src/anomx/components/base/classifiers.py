"""Classifier base contract.

Classifiers assign detected anomalies to categories — either configured target
groups or clusters discovered from projections (PCA / t-SNE) when the target
groups are unknown. Human feedback flows back into them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from anomx.components.base.capabilities import ComponentCapabilities
from anomx.components.base.component import BaseComponent


class Classifier(BaseComponent, ABC):
    """Assign detected anomalies to categories."""

    component_type = "classifier"
    capabilities: ClassVar[ComponentCapabilities] = ComponentCapabilities(requires_training=False, requires_labels=False)

    @abstractmethod
    def classify(self, events: Any, *, feedback: Any | None = None) -> Any:
        """Return events enriched with category assignments."""
