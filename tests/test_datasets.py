import pandas as pd
import pytest

from anomx.data import TimeSeriesDataset, make_sine_anomaly_dataset


def test_synthetic_dataset_has_datetime_index_and_targets():
    dataset = make_sine_anomaly_dataset(periods=48)

    assert isinstance(dataset.data.index, pd.DatetimeIndex)
    assert dataset.target_columns == ("value",)
    assert len(dataset.data) == 48


def test_train_test_split_is_chronological():
    dataset = make_sine_anomaly_dataset(periods=20)

    train, test = dataset.train_test_split(test_size=5)

    assert len(train.data) == 15
    assert len(test.data) == 5
    assert train.data.index.max() < test.data.index.min()


def test_dataset_requires_datetime_index():
    frame = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(TypeError, match="DatetimeIndex"):
        TimeSeriesDataset(data=frame, target_columns=("value",))
