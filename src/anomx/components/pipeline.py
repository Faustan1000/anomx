"""Composable anomaly pipeline over the canonical observation representation.

```
pipeline = AnomalyPipeline(
    model=ConstantBaselineModel(),
    scorer=AbsoluteErrorScorer(),
    detector=ThresholdDetector(),
)
pipeline.fit(observation_set)
events = pipeline.detect(observation_set)
```

The pipeline generalizes the anomaly-detection process, not the individual
algorithms: each component declares capabilities that are validated against
the observation set characteristics before anything is executed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from anomx.components.base import Detector, NormalityModel, Scorer
from anomx.components.contracts import AnomalyEvent, AnomalyEvents, ScoreLevel
from anomx.data.base.observation_set import ObservationSet


class PipelineValidationError(ValueError):
    """Raised when component capabilities do not match the observation set."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("Invalid pipeline: " + " ".join(self.problems))


class AnomalyPipeline:
    """Compose model, scorer, optional calibrator, and detector into one method."""

    def __init__(
        self,
        *,
        model: NormalityModel,
        scorer: Scorer,
        detector: Detector,
        calibrator: Any | None = None,
        score_level: ScoreLevel = ScoreLevel.SAMPLE,
    ) -> None:
        self.model = model
        self.scorer = scorer
        self.detector = detector
        self.calibrator = calibrator
        self.score_level = score_level

    def validate(self, observation_set: ObservationSet) -> list[str]:
        """Validate dataset characteristics against every component capability."""
        characteristics = observation_set.characteristics()
        problems: list[str] = []
        for component in (self.model, self.scorer, self.detector):
            capabilities = getattr(component, "capabilities", None)
            if capabilities is None:
                continue
            component_name = type(component).__name__
            problems.extend(f"{component_name}: {problem}" for problem in capabilities.validate_against(characteristics))
        return problems

    def fit(self, observation_set: ObservationSet) -> None:
        problems = self.validate(observation_set)
        if problems:
            raise PipelineValidationError(problems)
        self.model.fit(self._materialize(observation_set))

    def detect(self, observation_set: ObservationSet) -> AnomalyEvents:
        frame = self._materialize(observation_set)
        predictions = self.model.predict(frame)
        scored = self.scorer.score(predictions)
        if self.calibrator is not None:
            scored = self.calibrator.calibrate(scored)
        labeled = self.detector.detect(scored)
        return self._build_events(observation_set, labeled)

    def _materialize(self, observation_set: ObservationSet) -> pd.DataFrame:
        preferred_view = getattr(self.model, "preferred_view", "tabular")
        if preferred_view == "timeseries" and observation_set.timestamp_column is not None:
            return observation_set.as_timeseries().frame
        return observation_set.as_tabular().matrix

    def _build_events(self, observation_set: ObservationSet, labeled: pd.DataFrame) -> AnomalyEvents:
        label_column = "is_anomaly"
        if label_column not in labeled.columns:
            return AnomalyEvents(frame=labeled, label_column=label_column)

        timestamp_column = observation_set.timestamp_column
        timestamps = (
            observation_set.observations[timestamp_column].reset_index(drop=True)
            if timestamp_column is not None and len(observation_set.observations) == len(labeled)
            else None
        )
        events: list[AnomalyEvent] = []
        current_positions: list[int] = []
        for position, is_anomaly in enumerate(labeled[label_column].astype(bool).tolist()):
            if is_anomaly:
                current_positions.append(position)
                continue
            if current_positions:
                events.append(self._build_event(labeled, current_positions, timestamps))
                current_positions = []
        if current_positions:
            events.append(self._build_event(labeled, current_positions, timestamps))
        return AnomalyEvents(events=events, frame=labeled, label_column=label_column)

    def _build_event(self, labeled: pd.DataFrame, positions: list[int], timestamps: pd.Series | None) -> AnomalyEvent:
        severities = (
            labeled.iloc[positions]["severity"].astype(float)
            if "severity" in labeled.columns
            else pd.Series([0.0] * len(positions))
        )
        return AnomalyEvent(
            start_time=timestamps.iloc[positions[0]] if timestamps is not None else positions[0],
            end_time=timestamps.iloc[positions[-1]] if timestamps is not None else positions[-1],
            peak_score=float(severities.max()) if len(severities) else 0.0,
            severity=float(severities.mean()) if len(severities) else 0.0,
            score_level=self.score_level,
            observation_ids=[int(position) for position in positions],
        )
