"""Darts-native anomaly scoring foundations and reusable scorers."""

from darts.ad.scorers import (  # type: ignore[import-untyped]
    DifferenceScorer,
    KMeansScorer,
    NormScorer,
    WassersteinScorer,
)
from darts.ad.scorers.scorers import AnomalyScorer  # type: ignore[import-untyped]


class Scorer(AnomalyScorer):  # type: ignore[misc]
    """Abstract Darts anomaly scorer foundation for custom Anomx scorers."""


class AbsoluteErrorScorer(NormScorer):  # type: ignore[misc]
    """Absolute prediction error per component, returned as a Darts TimeSeries."""

    def __init__(self) -> None:
        super().__init__(ord=1, component_wise=True)


__all__ = [
    "AbsoluteErrorScorer",
    "DifferenceScorer",
    "KMeansScorer",
    "NormScorer",
    "Scorer",
    "WassersteinScorer",
]
