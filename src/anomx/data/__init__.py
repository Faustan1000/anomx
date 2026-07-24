"""Data domain: canonical observation abstractions, containers, and anomx-connected datasets."""

from anomx.data.base import (
    BaseObservationView,
    DataCharacteristics,
    DatasetAdapter,
    GraphView,
    Grouping,
    Hierarchy,
    Modality,
    ObservationSet,
    ObservationUnit,
    ObservationWindow,
    Ordering,
    RecordsAdapter,
    RelationSet,
    SequenceView,
    Structure,
    TabularView,
    TemporalSemantics,
    TimeSeriesBatchAdapter,
    TimeSeriesView,
    Topology,
    WindowView,
)
from anomx.data.connectors import BaseConnector, LocalFSConnector
from anomx.data.datasets import ChannelMetadata, TimeSeriesDataset
from anomx.data.loaders import make_sine_anomaly_dataset
from anomx.data.remote import AnomxConnectionError, AnomxDataset, read_platform_connection
from anomx.data.sequences import TimeSeriesBatch

__all__ = [
    "AnomxConnectionError",
    "AnomxDataset",
    "BaseConnector",
    "BaseObservationView",
    "ChannelMetadata",
    "DataCharacteristics",
    "DatasetAdapter",
    "GraphView",
    "Grouping",
    "Hierarchy",
    "LocalFSConnector",
    "Modality",
    "ObservationSet",
    "ObservationUnit",
    "ObservationWindow",
    "Ordering",
    "RecordsAdapter",
    "RelationSet",
    "SequenceView",
    "Structure",
    "TabularView",
    "TemporalSemantics",
    "TimeSeriesBatch",
    "TimeSeriesBatchAdapter",
    "TimeSeriesDataset",
    "TimeSeriesView",
    "Topology",
    "WindowView",
    "make_sine_anomaly_dataset",
    "read_platform_connection",
]
