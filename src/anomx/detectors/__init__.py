"""Darts-native score-to-decision foundations and detectors."""

from darts.ad.detectors import QuantileDetector, ThresholdDetector  # type: ignore[import-untyped]
from darts.ad.detectors.detectors import Detector as DartsDetector  # type: ignore[import-untyped]


class Detector(DartsDetector):  # type: ignore[misc]
    """Abstract Darts detector foundation for custom Anomx decision rules."""


class QuantileThresholdDetector(QuantileDetector):  # type: ignore[misc]
    """Learn an upper anomaly-score quantile from a normal training series."""

    def __init__(self, quantile: float = 0.99) -> None:
        super().__init__(high_quantile=quantile)


__all__ = ["Detector", "QuantileDetector", "QuantileThresholdDetector", "ThresholdDetector"]
