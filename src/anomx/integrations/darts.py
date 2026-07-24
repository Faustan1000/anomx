"""Darts integration adapters.

Install with `pip install "anomx[darts]"` to use this module with Darts models.
"""

from __future__ import annotations

from importlib import import_module, util
from typing import Protocol, cast

import pandas as pd

from anomx.components.base import Forecast
from anomx.data import TimeSeriesDataset


class _DartsPrediction(Protocol):
    def pd_dataframe(self) -> pd.DataFrame:
        """Return a pandas forecast frame."""


class _DartsModel(Protocol):
    def fit(self, series: object) -> object:
        """Fit a Darts model."""

    def predict(self, n: int) -> _DartsPrediction:
        """Predict a Darts forecast horizon."""


class _DartsTimeSeriesFactory(Protocol):
    def from_series(self, series: pd.Series) -> object:
        """Create a Darts TimeSeries from a pandas series."""


def is_darts_available() -> bool:
    """Return whether Darts can be imported."""
    return util.find_spec("darts") is not None


class DartsForecastingModel:
    """Adapter for any fitted or unfitted Darts forecasting model.

    The wrapped model must implement the usual Darts `fit(TimeSeries)` and
    `predict(n)` methods. This gives Anomx access to Darts' forecasting model
    ecosystem without making Darts a required dependency.
    """

    def __init__(self, model: _DartsModel, *, target_column: str | None = None) -> None:
        self.model = model
        self.target_column = target_column

    def fit(self, dataset: TimeSeriesDataset) -> DartsForecastingModel:
        """Fit the wrapped Darts model."""
        try:
            darts_module = import_module("darts")
        except ImportError as exc:
            msg = 'Darts is not installed. Install with `pip install "anomx[darts]"`.'
            raise ImportError(msg) from exc
        time_series = cast(_DartsTimeSeriesFactory, darts_module.TimeSeries)

        target = dataset.target
        if isinstance(target, pd.DataFrame):
            if self.target_column is None:
                msg = "target_column is required for multivariate Darts adapters."
                raise ValueError(msg)
            target = target[self.target_column]

        series = time_series.from_series(target)
        self.model.fit(series)
        self._target_name = target.name or self.target_column or "value"
        return self

    def predict(self, horizon: int) -> Forecast:
        """Predict with the wrapped Darts model and return an Anomx forecast."""
        prediction = self.model.predict(horizon)
        frame = prediction.pd_dataframe()
        return Forecast(
            values=frame,
            metadata={
                "model": self.model.__class__.__name__,
                "integration": "darts",
            },
        )
