"""Anomx public API.

The public objects are loaded lazily so lightweight entry points such as the
``anomx`` command do not import the scientific Python stack before they can
render their first frame.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.2.40"

_COMPONENT_EXPORTS = {
    "AbsoluteErrorScorer",
    "AnomalyEvent",
    "AnomalyEvents",
    "AnomalyPipeline",
    "AnomalyScores",
    "BaseAlgorithm",
    "ConstantBaselineModel",
    "DartsNaiveSeasonalModel",
    "InferenceKind",
    "InferenceOutput",
    "IsolationForestModel",
    "PcaReconstructionModel",
    "PipelineOrchestrator",
    "PipelineValidationError",
    "QuantileThresholdDetector",
    "RollingWindowForecastModel",
    "ScoreLevel",
    "ThresholdDetector",
    "TorchAutoencoderModel",
    "ZScoreScorer",
}
_BASE_COMPONENT_EXPORTS = {
    "AdaptiveThresholdDetector",
    "BaseComponent",
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
    "ReconstructionScorer",
    "Reconstructive",
    "Representational",
    "RepresentationScorer",
    "ResidualScorer",
    "Scorer",
    "StaticThresholdDetector",
    "StatisticalDetector",
    "discover_component_payloads",
}
_DATA_EXPORTS = {
    "AnomxDataset",
    "BaseConnector",
    "DataCharacteristics",
    "DatasetAdapter",
    "GraphView",
    "Hierarchy",
    "LocalFSConnector",
    "ObservationSet",
    "RecordsAdapter",
    "RelationSet",
    "SequenceView",
    "TabularView",
    "TimeSeriesBatch",
    "TimeSeriesBatchAdapter",
    "TimeSeriesDataset",
    "TimeSeriesView",
    "WindowView",
}
_RUNNER_EXPORTS = {
    "JobDefinition",
    "JobNode",
    "JobNodeType",
    "JobRunResult",
    "JobRunner",
}

_LAZY_EXPORTS = {
    **{name: "anomx.components" for name in _COMPONENT_EXPORTS},
    **{name: "anomx.components.base" for name in _BASE_COMPONENT_EXPORTS},
    **{name: "anomx.data" for name in _DATA_EXPORTS},
    **{name: "anomx.runner" for name in _RUNNER_EXPORTS},
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Load a public API object on first access."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public objects in interactive discovery."""

    return sorted({*globals(), *__all__})
