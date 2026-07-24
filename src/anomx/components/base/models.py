"""Normality model base contract.

A :class:`NormalityModel` learns what "normal" looks like. What it can do with
that knowledge is expressed through capability mixins
(:class:`~anomx.components.base.capabilities.Predictive`,
:class:`Reconstructive`, :class:`Representational`, :class:`Distributional`,
:class:`BoundaryEstimating`), and what data it accepts through its
:class:`~anomx.components.base.signature.ModelSignature`:

```python
class TemporalFusionTransformer(NormalityModel, Predictive, Representational):
    signature = ModelSignature(
        structures={DataStructure.TEMPORAL_SEQUENCE},
        modalities={Modality.NUMERIC},
        requirements={"regular_or_known_timestamps"},
    )
```

Trainable implementations live in the PyTorch domain: torch / lightning
backends are the expected default for learned models (install the `torch`
extra), while lightweight statistical baselines remain plain Python. A model
never emits anomaly scores itself — scorers do that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import pandas as pd

from anomx.components.base.capabilities import ComponentCapabilities
from anomx.components.base.component import BaseComponent


@dataclass(frozen=True)
class Forecast:
    """Forecast values with optional model metadata."""

    values: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Return forecast values as a DataFrame."""
        return cast(pd.DataFrame, self.values.copy())


class NormalityModel(BaseComponent, ABC):
    """Base class for trainable normality models."""

    component_type = "model"
    capabilities: ClassVar[ComponentCapabilities] = ComponentCapabilities()
    preferred_view: ClassVar[str] = "tabular"

    @abstractmethod
    def fit(self, data: Any) -> None:
        """Train the model using input data."""

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Generate model outputs for input data."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model state."""

    @abstractmethod
    def load(self, path: str) -> NormalityModel:
        """Load model state and return the model instance."""

    def partial_fit(self, batch: Any) -> None:
        """Incrementally update the model with one streaming batch."""
        raise NotImplementedError(f"{type(self).__name__} does not support incremental fitting.")

    def score_batch(self, batch: Any) -> Any:
        """Run streaming inference on one batch."""
        return self.predict(batch)


# Kept for callers that predate the taxonomy rework.
BaseAnomalyModel = NormalityModel
