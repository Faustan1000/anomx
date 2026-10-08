import numpy as np
import pandas as pd
import pytest

pytest.importorskip("darts")

from darts import TimeSeries
from darts.ad.detectors.detectors import Detector as DartsDetector
from darts.ad.scorers.scorers import AnomalyScorer

from anomx import Dataset, WorkContext
from anomx.detectors import Detector, QuantileThresholdDetector
from anomx.scorers import AbsoluteErrorScorer, Scorer

CHANNEL_REFERENCE = "data_channel-12345678-1234-5678-1234-567812345678"

def test_dataset_and_anomaly_foundations_inherit_darts():
    assert issubclass(Dataset, TimeSeries)
    assert issubclass(Scorer, AnomalyScorer)
    assert issubclass(Detector, DartsDetector)
    assert isinstance(Dataset.from_values(np.arange(10.0)), Dataset)


def test_channel_dataset_preserves_references_and_utc_times():
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=5, freq="s", tz="Europe/Berlin"),
                "channel": np.arange(5.0),
            }
        )

    series = WorkContext(data_loader=load).dataset([CHANNEL_REFERENCE], frequency="s")
    assert isinstance(series, Dataset)
    assert calls[0]["channels"] == [CHANNEL_REFERENCE]
    assert series.metadata["channel_references"] == [CHANNEL_REFERENCE]
    assert series.start_time() == pd.Timestamp("2025-12-31T23:00:00")
    assert series.metadata["timezone"] == "UTC"


def test_channel_dataset_rejects_gaps_and_duplicate_timestamps():
    frame = pd.DataFrame(
        {"x": [1.0, np.nan, 3.0]}, index=pd.date_range("2026", periods=3, freq="s")
    )
    with pytest.raises(ValueError, match="nonfinite"):
        Dataset.from_channels(["DOOCS/DEVICE/PROPERTY"], loader=lambda **_: frame)
    frame = frame.fillna(2)
    frame.index = pd.DatetimeIndex([frame.index[0]] * 3)
    with pytest.raises(ValueError, match="unique timestamps"):
        Dataset.from_channels([CHANNEL_REFERENCE], loader=lambda **_: frame)


def test_scorer_detector_compose_using_native_time_series():
    expected = Dataset.from_values(np.zeros(20))
    actual = Dataset.from_values(np.r_[np.zeros(19), 10.0])
    scores = AbsoluteErrorScorer().score_from_prediction(actual, expected)
    detector = QuantileThresholdDetector(0.95).fit(scores[:10])
    flags = detector.detect(scores)
    assert flags.values()[-1, 0] == 1
    assert flags.values()[:-1].sum() == 0
