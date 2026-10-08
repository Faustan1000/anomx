"""Common Darts model foundation."""

from darts.models.forecasting.forecasting_model import (  # type: ignore[import-untyped]
    GlobalForecastingModel,
)


class Model(GlobalForecastingModel):  # type: ignore[misc]  # Darts ships no typing marker
    """Abstract Anomx forecasting foundation inheriting the Darts contract.

    Custom Darts implementations supply ``fit``, ``predict``, their supported
    inputs and lag properties. Existing Darts models can also be used directly.
    """
