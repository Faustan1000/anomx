"""Synthetic datasets for examples and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomx.data.datasets import TimeSeriesDataset


def make_sine_anomaly_dataset(
    *,
    periods: int = 240,
    freq: str = "h",
    anomaly_indices: tuple[int, ...] = (80, 160, 200),
    anomaly_magnitude: float = 5.0,
    noise: float = 0.2,
    seed: int = 7,
) -> TimeSeriesDataset:
    """Create a synthetic univariate dataset with injected point anomalies."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=periods, freq=freq, tz="UTC")
    signal = np.sin(np.linspace(0, 8 * np.pi, periods))
    values = signal + rng.normal(0, noise, size=periods)
    labels = np.zeros(periods, dtype=int)

    for idx in anomaly_indices:
        if 0 <= idx < periods:
            values[idx] += anomaly_magnitude
            labels[idx] = 1

    frame = pd.DataFrame({"value": values, "is_anomaly": labels}, index=index)
    return TimeSeriesDataset.from_dataframe(
        frame,
        target_columns="value",
        name="synthetic_sine_anomalies",
        metadata={"source": "anomx.synthetic", "label_column": "is_anomaly"},
    )
