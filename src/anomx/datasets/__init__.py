"""Darts-native datasets and optional, platform-agnostic channel loading."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

import pandas as pd

try:
    from darts import TimeSeries  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError("Install `anomx[darts]` to use Anomx Dataset.") from exc


class ChannelClient(Protocol):
    """Loader contract shared by HTTP and custom data adapters."""

    def load_channels(
        self,
        channels: Sequence[str],
        *,
        start: str | None = None,
        end: str | None = None,
        frequency: str | None = None,
    ) -> object: ...


class Dataset(TimeSeries):  # type: ignore[misc]  # Darts has no py.typed marker
    """An Anomx time series that is directly usable by every Darts model.

    All Darts factories, splitting, transforms, slicing, and covariate APIs are
    available. ``from_channels`` additionally resolves Anomx object references
    or external references (for example DOOCS properties) through a host loader
    or an explicit :class:`~anomx.integrations.platform.PlatformClient`.
    """

    @classmethod
    def from_channels(
        cls,
        channels: Sequence[str],
        *,
        client: ChannelClient | None = None,
        loader: Callable[..., Any] | None = None,
        start: str | None = None,
        end: str | None = None,
        frequency: str | None = None,
        fill_missing_dates: bool = False,
    ) -> Dataset:
        """Resolve references into aligned, numeric, regularly sampled data.

        Loaders receive ``channels``, ``start``, ``end``, and ``frequency`` as
        keyword arguments and return a TimeSeries or wide pandas DataFrame.
        Frames use a DatetimeIndex or a ``timestamp`` column. Duplicate times,
        missing values and unaligned channels fail explicitly. This class does
        not fill or average values; the loader defines any preprocessing.
        """
        if isinstance(channels, str):
            raise TypeError("Pass channel references as a list, not a single string.")
        references = [str(reference).strip() for reference in channels]
        if not references or any(not reference for reference in references):
            raise ValueError("At least one nonempty channel reference is required.")
        if len(references) != len(set(references)):
            raise ValueError("Channel references must be unique.")
        if loader is not None and client is not None:
            raise ValueError("Provide either client or loader, not both.")
        if loader is None:
            if client is None:
                from anomx.integrations.platform import PlatformClient

                client = PlatformClient.from_env()
            loader = client.load_channels
        data = loader(channels=references, start=start, end=end, frequency=frequency)
        metadata = {"channel_references": references, "start": start, "end": end}
        if isinstance(data, TimeSeries):
            return cast(
                Dataset,
                cls.from_times_and_values(
                    data.time_index,
                    data.all_values(),
                    columns=data.components,
                    freq=data.freq,
                    static_covariates=data.static_covariates,
                    metadata={**(data.metadata or {}), **metadata},
                ),
            )
        if not isinstance(data, (pd.DataFrame, list, dict)):
            raise TypeError("Channel loaders must return a DataFrame, records, or TimeSeries.")
        frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.set_index("timestamp")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("Channel data needs a DatetimeIndex or timestamp column.")
        if frame.empty or frame.index.has_duplicates:
            raise ValueError("Channel data must contain samples with unique timestamps.")
        frame = frame.sort_index()
        # Darts uses timezone-naive time axes. Preserve the timezone contract in metadata.
        time_index = cast(pd.DatetimeIndex, frame.index)
        if time_index.tz is not None:
            frame.index = time_index.tz_convert("UTC").tz_localize(None)
            metadata["timezone"] = "UTC"
        if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
            raise TypeError("All channel values must be numeric.")
        import numpy as np

        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(
                "Channel data has missing or nonfinite values; align/fill it explicitly."
            )
        return cast(
            Dataset,
            cls.from_dataframe(
                frame,
                freq=frequency,
                fill_missing_dates=fill_missing_dates,
                metadata=metadata,
            ),
        )


AnomxDataset = Dataset

__all__ = ["AnomxDataset", "Dataset"]
