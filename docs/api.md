# API Outline

## Base taxonomy

- `anomx.components.base.BaseComponent`
- `anomx.components.base.NormalityModel`
- `anomx.components.base.Predictive` / `Reconstructive` / `Representational` / `Distributional` / `BoundaryEstimating`
- `anomx.components.base.ModelSignature` / `DataStructure`
- `anomx.components.base.Scorer` with `ResidualScorer`, `ReconstructionScorer`, `RepresentationScorer`, `LikelihoodScorer`, `BoundaryScorer`, `DirectDataScorer`, `CompositeScorer`
- `anomx.components.base.Detector` with `StaticThresholdDetector`, `AdaptiveThresholdDetector`, `StatisticalDetector`, `ChangePointDetector`, `EventAggregationDetector`
- `anomx.components.base.Classifier`

## Runner

- `anomx.runner.JobDefinition` / `JobNode` / `JobNodeType`
- `anomx.runner.JobRunner` / `JobRunResult`

## Observations

- `anomx.data.base.ObservationSet`
- `anomx.data.base.DataCharacteristics`
- `anomx.data.base.TabularView` / `SequenceView` / `WindowView` / `TimeSeriesView` / `GraphView`

## Components

- `anomx.components.AnomalyPipeline`
- `anomx.components.ConstantBaselineModel`, `RollingWindowForecastModel`, `PcaReconstructionModel`, `IsolationForestModel`, `DartsNaiveSeasonalModel`, `TorchAutoencoderModel`
- `anomx.components.AbsoluteErrorScorer`, `ZScoreScorer`
- `anomx.components.ThresholdDetector`, `QuantileThresholdDetector`
- `anomx.components.discover_component_payloads`

## Datasets

- `anomx.datasets.TimeSeriesDataset`
- `anomx.datasets.ChannelMetadata`
- `anomx.datasets.make_sine_anomaly_dataset`

## Integrations

- `anomx.integrations.DartsForecastingModel`
- `anomx.integrations.is_darts_available`
