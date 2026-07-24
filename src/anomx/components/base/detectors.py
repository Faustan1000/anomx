"""Detector taxonomy.

Detectors convert continuous scores into decisions or anomaly events:

```
Detector
├── StaticThresholdDetector     fixed thresholds
├── AdaptiveThresholdDetector   data-driven thresholds (quantiles, baselines)
├── StatisticalDetector         hypothesis tests, control charts
├── ChangePointDetector         distribution shift detection
└── EventAggregationDetector    grouping decisions into interval events
```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from anomx.components.base.capabilities import ComponentCapabilities
from anomx.components.base.component import BaseComponent


class Detector(BaseComponent, ABC):
    """Convert continuous scores into decisions or anomaly events."""

    component_type = "detector"
    detector_family: ClassVar[str] = ""
    capabilities: ClassVar[ComponentCapabilities] = ComponentCapabilities(requires_training=False)

    @abstractmethod
    def detect(self, predictions: Any) -> Any:
        """Return predictions with anomaly labels."""


class StaticThresholdDetector(Detector, ABC):
    """Decide with a fixed, configured threshold."""

    detector_family = "static_threshold"


class AdaptiveThresholdDetector(Detector, ABC):
    """Decide with a threshold derived from the observed score distribution."""

    detector_family = "adaptive_threshold"


class StatisticalDetector(Detector, ABC):
    """Decide with statistical tests or control-chart rules."""

    detector_family = "statistical"


class ChangePointDetector(Detector, ABC):
    """Decide by detecting distribution shifts in the score sequence."""

    detector_family = "change_point"


class EventAggregationDetector(Detector, ABC):
    """Aggregate raw decisions into interval events with severity."""

    detector_family = "event_aggregation"


# Kept for callers that predate the taxonomy rework.
BaseDetector = Detector
