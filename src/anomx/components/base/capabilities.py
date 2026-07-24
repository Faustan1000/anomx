"""Capability declarations for pipeline components.

Two complementary mechanisms live here:

- :class:`Capability` mixins express *what a normality model can do*
  (predict, reconstruct, embed, estimate densities or boundaries). They are
  composed onto implementations, e.g.
  ``class TemporalFusionTransformer(NormalityModel, Predictive, Representational)``.
- :class:`ComponentCapabilities` expresses *what data a component handles*,
  validated against observation-set characteristics before execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from anomx.data.base.characteristics import DataCharacteristics, TemporalSemantics


class Capability:
    """Marker base for model capability mixins."""

    capability_key: ClassVar[str] = ""

    @classmethod
    def get_capability_key(cls) -> str:
        return cls.capability_key


class Predictive(Capability):
    """The model can forecast expected future values."""

    capability_key = "predictive"

    def forecast(self, data: Any, horizon: int) -> Any:
        """Forecast `horizon` future steps."""
        raise NotImplementedError(f"{type(self).__name__} does not implement forecasting.")


class Reconstructive(Capability):
    """The model can reconstruct its input from a compressed representation."""

    capability_key = "reconstructive"

    def reconstruct(self, data: Any) -> Any:
        """Return the reconstruction of the input."""
        raise NotImplementedError(f"{type(self).__name__} does not implement reconstruction.")


class Representational(Capability):
    """The model can embed observations into a latent representation."""

    capability_key = "representational"

    def embed(self, data: Any) -> Any:
        """Return latent embeddings for the input."""
        raise NotImplementedError(f"{type(self).__name__} does not implement embeddings.")


class Distributional(Capability):
    """The model can estimate densities or likelihoods."""

    capability_key = "distributional"

    def log_likelihood(self, data: Any) -> Any:
        """Return per-observation log likelihoods."""
        raise NotImplementedError(f"{type(self).__name__} does not implement likelihoods.")


class BoundaryEstimating(Capability):
    """The model can estimate a decision boundary around normal data."""

    capability_key = "boundary_estimating"

    def decision_function(self, data: Any) -> Any:
        """Return signed distances to the normality boundary."""
        raise NotImplementedError(f"{type(self).__name__} does not implement decision functions.")


def collect_capability_keys(component_class: type) -> list[str]:
    """Collect the capability keys mixed into a component class."""
    keys: list[str] = []
    for base in component_class.__mro__:
        if issubclass(base, Capability) and base is not Capability:
            key = getattr(base, "capability_key", "")
            if key and key not in keys:
                keys.append(key)
    return keys


@dataclass(frozen=True, slots=True)
class ComponentCapabilities:
    """What a component supports and what it requires from its input."""

    supports_unordered_samples: bool = True
    supports_sequences: bool = False
    supports_streaming: bool = False
    supports_multivariate: bool = True
    supports_variable_length: bool = True
    supports_irregular_sampling: bool = True
    supports_missing_values: bool = False
    supports_graphs: bool = False
    supports_hierarchies: bool = False

    requires_regular_sampling: bool = False
    requires_fixed_length: bool = False
    requires_training: bool = True
    requires_labels: bool = False

    batch_fit: bool = True
    incremental_fit: bool = False
    batch_inference: bool = True
    streaming_inference: bool = False

    def validate_against(self, characteristics: DataCharacteristics) -> list[str]:
        """Return human-readable problems, empty when the pipeline is valid."""
        problems: list[str] = []
        if not characteristics.is_ordered and not self.supports_unordered_samples:
            problems.append("The component does not support unordered samples.")
        if characteristics.is_ordered and not (self.supports_sequences or self.supports_unordered_samples):
            problems.append("The component supports neither sequences nor sample views.")
        if characteristics.is_multivariate and not self.supports_multivariate:
            problems.append("The component does not support multivariate inputs.")
        if characteristics.has_missing_values and not self.supports_missing_values:
            problems.append("The component does not support missing values.")
        if characteristics.has_topology and not self.supports_graphs:
            problems.append("The component does not support graph topologies.")
        if characteristics.has_hierarchy and not self.supports_hierarchies:
            problems.append("The component does not support hierarchical structures.")
        if self.requires_regular_sampling and characteristics.temporal_semantics not in (
            TemporalSemantics.NONE,
            TemporalSemantics.REGULAR_TIME,
        ):
            problems.append("The component requires regularly sampled temporal data.")
        if characteristics.temporal_semantics is TemporalSemantics.IRREGULAR_TIME and not self.supports_irregular_sampling:
            problems.append("The component does not support irregularly sampled data.")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
