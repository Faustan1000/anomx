"""Run a baseline anomaly pipeline on a synthetic dataset."""

from anomx import (
    AbsoluteErrorScorer,
    AnomalyPipeline,
    ConstantBaselineModel,
    ObservationSet,
    ThresholdDetector,
)
from anomx.data import make_sine_anomaly_dataset


def main() -> None:
    frame = make_sine_anomaly_dataset().to_dataframe().reset_index()
    observation_set = ObservationSet(observations=frame, timestamp_column=str(frame.columns[0]))
    pipeline = AnomalyPipeline(
        model=ConstantBaselineModel(),
        scorer=AbsoluteErrorScorer(),
        detector=ThresholdDetector({"source_column": "score", "threshold": 3.0}),
    )
    pipeline.fit(observation_set)
    events = pipeline.detect(observation_set)
    print(events.to_dict())


if __name__ == "__main__":
    main()
