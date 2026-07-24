"""Core dataset abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd


@dataclass(frozen=True)
class ChannelMetadata:
    """Metadata describing one logical time-series channel."""

    name: str
    unit: str | None = None
    description: str | None = None
    tags: tuple[str, ...] = ()


@dataclass
class TimeSeriesDataset:
    """A typed container for one or more aligned time-series channels.

    The container keeps the public library independent from the Anomx platform's
    storage layer while preserving the metadata needed for auditable pipelines.
    """

    data: pd.DataFrame
    target_columns: tuple[str, ...]
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    channels: dict[str, ChannelMetadata] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data.index, pd.DatetimeIndex):
            msg = "TimeSeriesDataset.data must be indexed by pandas.DatetimeIndex."
            raise TypeError(msg)
        missing = set(self.target_columns).difference(self.data.columns)
        if missing:
            msg = f"Target columns are missing from data: {sorted(missing)}"
            raise ValueError(msg)
        self.data.sort_index(inplace=True)

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        *,
        target_columns: str | list[str] | tuple[str, ...],
        time_column: str | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TimeSeriesDataset:
        """Create a dataset from a DataFrame.

        Args:
            frame: Source frame.
            target_columns: Column or columns modeled by detectors and forecasters.
            time_column: Optional column to convert into the DatetimeIndex.
            name: Optional human-readable dataset name.
            metadata: Optional metadata carried through pipelines.
        """
        data = frame.copy()
        if time_column is not None:
            data[time_column] = pd.to_datetime(data[time_column], utc=True)
            data = data.set_index(time_column)
        if isinstance(target_columns, str):
            target_columns = (target_columns,)
        else:
            target_columns = tuple(target_columns)
        return cls(
            data=data,
            target_columns=target_columns,
            name=name,
            metadata=dict(metadata or {}),
        )

    @property
    def frequency(self) -> str | None:
        """Infer the dataset frequency when pandas can determine it."""
        index = cast(pd.DatetimeIndex, self.data.index)
        inferred = pd.infer_freq(index)
        return str(inferred) if inferred is not None else None

    @property
    def target(self) -> pd.Series | pd.DataFrame:
        """Return the modeled target column or columns."""
        if len(self.target_columns) == 1:
            return self.data[self.target_columns[0]]
        return cast(pd.DataFrame, self.data.loc[:, list(self.target_columns)])

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame."""
        return cast(pd.DataFrame, self.data.copy())

    def train_test_split(
        self,
        test_size: float | int,
    ) -> tuple[TimeSeriesDataset, TimeSeriesDataset]:
        """Split the dataset into chronological train and test portions."""
        if isinstance(test_size, float):
            if not 0 < test_size < 1:
                msg = "Float test_size must be between 0 and 1."
                raise ValueError(msg)
            split_at = int(len(self.data) * (1 - test_size))
        else:
            if test_size <= 0:
                msg = "Integer test_size must be positive."
                raise ValueError(msg)
            split_at = len(self.data) - test_size
        if split_at <= 0 or split_at >= len(self.data):
            msg = "test_size leaves an empty train or test split."
            raise ValueError(msg)
        return (
            self._replace_data(self.data.iloc[:split_at].copy()),
            self._replace_data(self.data.iloc[split_at:].copy()),
        )

    def resample(self, rule: str, *, method: str = "mean") -> TimeSeriesDataset:
        """Return a resampled dataset using a pandas aggregation method."""
        resampler = self.data.resample(rule)
        if not hasattr(resampler, method):
            msg = f"Unsupported resample method: {method}"
            raise ValueError(msg)
        resampled = getattr(resampler, method)()
        return self._replace_data(resampled)

    def _replace_data(self, data: pd.DataFrame) -> TimeSeriesDataset:
        return TimeSeriesDataset(
            data=data,
            target_columns=self.target_columns,
            name=self.name,
            metadata=dict(self.metadata),
            channels=dict(self.channels),
        )
