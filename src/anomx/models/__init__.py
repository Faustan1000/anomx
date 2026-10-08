"""Darts forecasting foundations and the generic Anomx PyTorch model.

Install ``anomx[darts]`` for forecasting; ``anomx[ml]`` also installs PyTorch,
ONNX export and ONNX inference. Heavy dependencies are loaded on demand.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Model": ("anomx.models.base", "Model"),
    "PyTorchModel": ("anomx.models.torch", "PyTorchModel"),
    "ONNXModel": ("anomx.models.onnx", "ONNXModel"),
    **{
        name: ("darts.models", name)
        for name in (
            "NaiveMean",
            "NaiveSeasonal",
            "NaiveDrift",
            "LinearRegressionModel",
            "RandomForest",
            "ExponentialSmoothing",
            "ARIMA",
            "NBEATSModel",
            "NHiTSModel",
            "RNNModel",
            "BlockRNNModel",
            "TCNModel",
            "TransformerModel",
            "TFTModel",
            "DLinearModel",
            "NLinearModel",
            "TiDEModel",
        )
    },
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:  # noqa: ANN401 - lazy public exports
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    try:
        value = getattr(import_module(module), attribute)
    except ImportError as exc:
        raise ImportError("Install `anomx[ml]` to use the full model catalog.") from exc
    globals()[name] = value
    return value
