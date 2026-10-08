"""Inference from portable ONNX model artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ONNXModel:
    """Run a published graph without importing or unpickling Python model code."""

    def __init__(self, path: str | Path | bytes, *, providers: list[str] | None = None) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Install `anomx[onnx]` for ONNX inference.") from exc
        self.session = ort.InferenceSession(
            path if isinstance(path, bytes) else str(path),
            providers=providers or ["CPUExecutionProvider"],
        )

    def predict(self, observations: ArrayLike) -> np.ndarray:
        """Predict a batch of ``(batch, input_window, channels)`` observations."""
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Use run() with named tensors for multi-input models.")
        values = np.asarray(observations, dtype=np.float32)
        if values.ndim != 3 or not np.isfinite(values).all():
            raise ValueError("Expected finite (batch, input_window, channels) observations.")
        return cast(np.ndarray, np.asarray(self.session.run(None, {inputs[0].name: values})[0]))

    def run(self, inputs: Mapping[str, ArrayLike]) -> dict[str, Any]:
        """Run named ONNX tensors and return JSON-compatible named outputs.

        Supports numeric, boolean and string tensors, including models with
        multiple inputs/outputs. The runtime validates tensor dimensions.
        """
        descriptors = self.session.get_inputs()
        if set(inputs) != {item.name for item in descriptors}:
            raise ValueError("Provide exactly the model's named input tensors.")
        dtypes = {
            "float": np.float32,
            "double": np.float64,
            "float16": np.float16,
            "int64": np.int64,
            "int32": np.int32,
            "int16": np.int16,
            "int8": np.int8,
            "uint64": np.uint64,
            "uint32": np.uint32,
            "uint16": np.uint16,
            "uint8": np.uint8,
            "bool": np.bool_,
            "string": np.object_,
        }
        tensors = {}
        for item in descriptors:
            kind = item.type.removeprefix("tensor(").removesuffix(")")
            if kind not in dtypes:
                raise ValueError(f"Unsupported ONNX input type: {item.type}")
            tensor: NDArray[Any] = np.asarray(inputs[item.name], dtype=dtypes[kind])
            if kind != "string" and not np.isfinite(tensor).all():
                raise ValueError("Model inputs must contain finite values.")
            tensors[item.name] = tensor
        outputs = self.session.run(None, tensors)
        return {
            item.name: np.asarray(value).tolist()
            for item, value in zip(self.session.get_outputs(), outputs, strict=True)
        }
