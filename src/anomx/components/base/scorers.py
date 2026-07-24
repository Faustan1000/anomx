"""Scorer taxonomy.

Scorers convert model output into continuous anomaly scores. The family bases
express *where the score comes from*; concrete implementations subclass the
family that matches their mechanism:

```
Scorer
├── ResidualScorer          forecast / prediction errors
├── ReconstructionScorer    reconstruction errors
├── RepresentationScorer    latent-space distances
├── LikelihoodScorer        negative log likelihoods
├── BoundaryScorer          signed boundary distances
├── DirectDataScorer        scores computed on the raw data itself
└── CompositeScorer         combinations of child scorers
```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from anomx.components.base.capabilities import ComponentCapabilities
from anomx.components.base.component import BaseComponent


class Scorer(BaseComponent, ABC):
    """Convert model outputs into standardized anomaly scores."""

    component_type = "scorer"
    scorer_family: ClassVar[str] = ""
    capabilities: ClassVar[ComponentCapabilities] = ComponentCapabilities(requires_training=False)

    @abstractmethod
    def score(self, data: Any) -> Any:
        """Return scored prediction records."""


class ResidualScorer(Scorer, ABC):
    """Score anomalies from prediction or forecast residuals."""

    scorer_family = "residual"


class ReconstructionScorer(Scorer, ABC):
    """Score anomalies from reconstruction errors."""

    scorer_family = "reconstruction"


class RepresentationScorer(Scorer, ABC):
    """Score anomalies from latent-space distances or densities."""

    scorer_family = "representation"


class LikelihoodScorer(Scorer, ABC):
    """Score anomalies from estimated likelihoods."""

    scorer_family = "likelihood"


class BoundaryScorer(Scorer, ABC):
    """Score anomalies from distances to a learned normality boundary."""

    scorer_family = "boundary"


class DirectDataScorer(Scorer, ABC):
    """Score anomalies directly on the observations without a model output."""

    scorer_family = "direct_data"


class CompositeScorer(Scorer):
    """Combine the scores of several child scorers into one score column."""

    scorer_family = "composite"
    component_status = "disabled"

    def __init__(self, scorers: list[Scorer] | None = None, config: dict[str, Any] | None = None) -> None:
        self.scorers = list(scorers or [])
        self.config = config or {}

    def score(self, data: Any) -> Any:
        scored = data
        score_columns: list[str] = []
        for child_index, child_scorer in enumerate(self.scorers):
            scored = child_scorer.score(scored)
            if "score" in getattr(scored, "columns", []):
                child_column = f"score__{child_index}"
                scored[child_column] = scored["score"]
                score_columns.append(child_column)
        if score_columns:
            scored["score"] = scored[score_columns].mean(axis=1)
        return scored


# Kept for callers that predate the taxonomy rework.
BaseScorer = Scorer
