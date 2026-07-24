"""Detector components."""

from anomx.components.base import BaseDetector, Detector
from anomx.components.detection.detectors.quantile import QuantileThresholdDetector
from anomx.components.detection.detectors.threshold import ThresholdDetector

__all__ = [
    "BaseDetector",
    "Detector",
    "QuantileThresholdDetector",
    "ThresholdDetector",
]
