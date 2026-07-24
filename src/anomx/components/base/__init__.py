"""Component taxonomy: metadata base, capabilities, signatures, and family contracts."""

from anomx.components.base.capabilities import (
    BoundaryEstimating,
    Capability,
    ComponentCapabilities,
    Distributional,
    Predictive,
    Reconstructive,
    Representational,
    collect_capability_keys,
)
from anomx.components.base.classifiers import Classifier
from anomx.components.base.component import (
    BaseComponent,
    discover_component_payloads,
    humanize_component_name,
    import_component_modules,
    iter_component_classes,
    read_component_source_path,
)
from anomx.components.base.detectors import (
    AdaptiveThresholdDetector,
    BaseDetector,
    ChangePointDetector,
    Detector,
    EventAggregationDetector,
    StaticThresholdDetector,
    StatisticalDetector,
)
from anomx.components.base.models import BaseAnomalyModel, Forecast, NormalityModel
from anomx.components.base.scorers import (
    BaseScorer,
    BoundaryScorer,
    CompositeScorer,
    DirectDataScorer,
    LikelihoodScorer,
    ReconstructionScorer,
    RepresentationScorer,
    ResidualScorer,
    Scorer,
)
from anomx.components.base.signature import DataStructure, ModelSignature

__all__ = [
    "AdaptiveThresholdDetector",
    "BaseAnomalyModel",
    "BaseComponent",
    "BaseDetector",
    "BaseScorer",
    "BoundaryEstimating",
    "BoundaryScorer",
    "Capability",
    "ChangePointDetector",
    "Classifier",
    "ComponentCapabilities",
    "CompositeScorer",
    "DataStructure",
    "Detector",
    "DirectDataScorer",
    "Distributional",
    "EventAggregationDetector",
    "Forecast",
    "LikelihoodScorer",
    "ModelSignature",
    "NormalityModel",
    "Predictive",
    "Reconstructive",
    "ReconstructionScorer",
    "Representational",
    "RepresentationScorer",
    "ResidualScorer",
    "Scorer",
    "StaticThresholdDetector",
    "StatisticalDetector",
    "collect_capability_keys",
    "discover_component_payloads",
    "humanize_component_name",
    "import_component_modules",
    "iter_component_classes",
    "read_component_source_path",
]
