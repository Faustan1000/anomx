import numpy as np
import pytest

pytest.importorskip("darts")
pytest.importorskip("torch")

import torch
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from anomx import Dataset, PyTorchModel


def test_training_forecast_and_onnx_round_trip(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from anomx import ONNXModel

    series = Dataset.from_values(np.sin(np.arange(64, dtype=np.float32) / 8))
    model = PyTorchModel(input_chunk_length=6, output_chunk_length=2, n_epochs=3)
    assert isinstance(model, GlobalForecastingModel)
    model.fit(series)
    prediction = model.predict(7)
    assert len(prediction) == 7
    assert prediction.start_time() == series.end_time() + 1
    assert len(model.training_history) == 3
    assert np.isfinite(model.training_history[-1]["loss"])
    artifact = model.export_onnx(tmp_path / "model.onnx")
    assert artifact.with_suffix(".json").is_file()
    window = series.values()[-6:][None].astype(np.float32)
    exported = ONNXModel(artifact).predict(window)
    np.testing.assert_allclose(exported[0], model.predict(2).values(), atol=1e-6)
    # Dynamic batch dimension is part of the exported inference contract.
    assert ONNXModel(artifact).predict(np.repeat(window, 3, axis=0)).shape == (3, 2, 1)


def test_custom_module_and_multi_series_training():
    series = Dataset.from_values(np.arange(20, dtype=np.float32))
    module = torch.nn.Sequential(
        torch.nn.Flatten(1), torch.nn.Linear(4, 1), torch.nn.Unflatten(1, (1, 1))
    )
    model = PyTorchModel(input_chunk_length=4, module=module, n_epochs=1).fit([series, series])
    assert len(model.predict(2, series=series)) == 2
    assert len(model.predict(2, series=[series, series])) == 2
    with pytest.raises(ValueError, match="prediction series"):
        model.predict(2)


def test_bad_data_and_wrong_network_shapes_fail_before_training():
    model = PyTorchModel(input_chunk_length=4, n_epochs=1)
    with pytest.raises(ValueError, match="Fit"):
        model.predict(1)
    with pytest.raises(ValueError, match="At least"):
        model.fit(Dataset.from_values(np.arange(3.0)))
    wrong = PyTorchModel(input_chunk_length=4, module=torch.nn.Linear(1, 1), n_epochs=1)
    with pytest.raises(ValueError, match="Module output"):
        wrong.fit(Dataset.from_values(np.arange(10.0)))
