"""A generic PyTorch network behind the Darts forecasting interface."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries  # type: ignore[import-untyped]
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from anomx.models.base import Model


class PyTorchModel(Model):
    """Train any ``torch.nn.Module`` on deterministic time-series windows.

    The module receives ``(batch, input_chunk_length, channels)`` and must
    return ``(batch, output_chunk_length, channels)``. Omit it for a tiny linear
    baseline. Training accepts one or many Darts TimeSeries; prediction rolls
    the network forward for an arbitrary horizon. No normalization is implicit.
    Use Darts transformers fitted on the training split when scaling is needed.

    The exported ONNX graph uses the identical tensor contract. The adjacent
    JSON manifest describes channel names, window sizes and training KPIs.
    """

    def __init__(
        self,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        module: nn.Module | None = None,
        n_epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        random_state: int = 42,
        device: str = "cpu",
        loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__(add_encoders=None)
        for name, value in {
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        self._input_length = input_chunk_length
        self._output_length = output_chunk_length
        self.module = copy.deepcopy(module)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device = device
        self.loss_fn = loss_fn or nn.MSELoss()
        self.training_history: list[dict[str, float]] = []
        self._components: list[str] = []

    @property
    def input_chunk_length(self) -> int:
        return self._input_length

    @property
    def output_chunk_length(self) -> int:
        return self._output_length

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def supports_sample_weight(self) -> bool:
        return False

    @property
    def supports_optimized_historical_forecasts(self) -> bool:
        return False

    @property
    def min_train_series_length(self) -> int:
        return self.input_chunk_length + self.output_chunk_length

    @property
    def min_train_samples(self) -> int:
        return 1

    @property
    def extreme_lags(self) -> tuple[int, int, None, None, None, None, int]:
        return -self.input_chunk_length, self.output_chunk_length - 1, None, None, None, None, 0

    @property
    def _target_window_lengths(self) -> tuple[int, int]:
        return self.input_chunk_length, self.output_chunk_length

    @property
    def _model_encoder_settings(self) -> tuple[int, int, bool, bool, None, None]:
        return self.input_chunk_length, self.output_chunk_length, False, False, None, None

    def _values(self, series: TimeSeries, *, training: bool) -> np.ndarray:
        if not isinstance(series, TimeSeries) or not series.is_deterministic:
            raise ValueError("PyTorchModel expects deterministic Darts TimeSeries.")
        required = self.min_train_series_length if training else self.input_chunk_length
        if len(series) < required:
            raise ValueError(f"At least {required} observations are required.")
        values = cast(np.ndarray, np.asarray(series.values(), dtype=np.float32))
        if not np.isfinite(values).all():
            raise ValueError("Training and prediction inputs must be finite; fill gaps explicitly.")
        return values

    def fit(
        self,
        series: TimeSeries | Sequence[TimeSeries],
        past_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        future_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        verbose: bool | None = None,
        *,
        on_epoch: Callable[[dict[str, float]], Any] | None = None,
    ) -> PyTorchModel:
        """Fit on chronological sliding windows and record mean loss per epoch."""
        if past_covariates is not None or future_covariates is not None:
            raise ValueError("Use a Darts catalog model for past/future covariates.")
        series_list = [series] if isinstance(series, TimeSeries) else list(series)
        if not series_list:
            raise ValueError("At least one training series is required.")
        arrays = [self._values(item, training=True) for item in series_list]
        components = list(map(str, series_list[0].components))
        if any(list(map(str, item.components)) != components for item in series_list):
            raise ValueError("All training series must have the same ordered component names.")
        self._fit_called = False
        self._components = components
        x, y = [], []
        for values in arrays:
            for position in range(len(values) - self.min_train_series_length + 1):
                split = position + self.input_chunk_length
                x.append(values[position:split])
                y.append(values[split : split + self.output_chunk_length])
        generator = torch.Generator().manual_seed(self.random_state)
        if self.module is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.random_state)
                self.module = nn.Sequential(
                    nn.Flatten(start_dim=1),
                    nn.Linear(
                        self.input_chunk_length * len(components),
                        self.output_chunk_length * len(components),
                    ),
                    nn.Unflatten(1, (self.output_chunk_length, len(components))),
                )
        self.module.to(self.device)
        loss_fn = self.loss_fn.to(self.device)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y))),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        self.training_history = []
        self.module.train()
        for epoch in range(self.n_epochs):
            total_loss, count = 0.0, 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                predictions = self.module(inputs)
                if predictions.shape != targets.shape:
                    raise ValueError(
                        f"Module output {tuple(predictions.shape)} "
                        f"must match {tuple(targets.shape)}."
                    )
                loss = loss_fn(predictions, targets)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise ValueError("Loss must be a finite scalar.")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(inputs)
                count += len(inputs)
            metric = {"epoch": float(epoch + 1), "loss": total_loss / count}
            self.training_history.append(metric)
            if on_epoch is not None:
                on_epoch(dict(metric))
        self.module.eval()
        super().fit(series)
        return self

    def predict(
        self,
        n: int,
        series: TimeSeries | Sequence[TimeSeries] | None = None,
        past_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        future_covariates: TimeSeries | Sequence[TimeSeries] | None = None,
        num_samples: int = 1,
        verbose: bool | None = None,
        **kwargs: object,
    ) -> TimeSeries | list[TimeSeries]:
        """Autoregressively forecast ``n`` steps, preserving components and index."""
        if not self._fit_called or self.module is None:
            raise ValueError("Fit the model before prediction.")
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0 or num_samples != 1:
            raise ValueError("n must be positive and num_samples must equal one.")
        if past_covariates is not None or future_covariates is not None:
            raise ValueError("This model does not accept covariates.")
        if kwargs.get("predict_likelihood_parameters"):
            raise ValueError("This deterministic model has no likelihood parameters.")
        source = series if series is not None else self.training_series
        if source is None:
            raise ValueError("Pass a prediction series after fitting multiple time series.")
        if not isinstance(source, TimeSeries):
            return [self.predict(n, series=item) for item in source]
        if list(map(str, source.components)) != self._components:
            raise ValueError("Prediction components must match the ordered training components.")
        history = self._values(source, training=False)
        chunks = []
        remaining = n
        self.module.eval()
        with torch.no_grad():
            while remaining:
                window = torch.from_numpy(history[-self.input_chunk_length :][None]).to(self.device)
                output = self.module(window).detach().cpu().numpy()[0]
                expected = (self.output_chunk_length, len(self._components))
                if output.shape != expected or not np.isfinite(output).all():
                    raise ValueError(f"Module must return finite predictions of shape {expected}.")
                chunk = output[:remaining]
                chunks.append(chunk)
                history = np.concatenate([history, chunk], axis=0)
                remaining -= len(chunk)
        index: pd.DatetimeIndex | pd.RangeIndex
        if source.has_datetime_index:
            index = pd.date_range(source.end_time() + source.freq, periods=n, freq=source.freq)
        else:
            start = int(source.end_time()) + int(source.freq)
            index = pd.RangeIndex(start, start + n * int(source.freq), int(source.freq))
        return TimeSeries.from_times_and_values(
            index,
            np.concatenate(chunks),
            columns=source.components,
            static_covariates=source.static_covariates,
            metadata=source.metadata,
        )

    def export_onnx(self, path: str | Path, *, opset_version: int = 17) -> Path:
        """Export a checked, self-contained ONNX graph and JSON input manifest."""
        if not self._fit_called or self.module is None:
            raise ValueError("Fit the model before exporting it.")
        try:
            import onnx
        except ImportError as exc:
            raise ImportError("Install `anomx[onnx]` to export models.") from exc
        destination = Path(path).expanduser().resolve()
        if destination.suffix.lower() != ".onnx":
            raise ValueError("ONNX destinations must end in .onnx.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        module = copy.deepcopy(self.module).cpu().eval()
        sample = torch.zeros(1, self.input_chunk_length, len(self._components))
        # The tracing exporter supports custom nn.Modules on torch >=2.2.
        # Darts also uses this exporter for its own neural model export.
        torch.onnx.export(
            module,
            (sample,),
            str(destination),
            input_names=["observations"],
            output_names=["predictions"],
            opset_version=opset_version,
            dynamic_axes={"observations": {0: "batch"}, "predictions": {0: "batch"}},
            dynamo=False,
            external_data=False,
        )
        onnx.checker.check_model(str(destination))
        manifest = {
            "format": "onnx",
            "schema_version": 1,
            "model": "anomx.PyTorchModel",
            "input_name": "observations",
            "output_name": "predictions",
            "dtype": "float32",
            "input_chunk_length": self.input_chunk_length,
            "output_chunk_length": self.output_chunk_length,
            "components": self._components,
            "training_history": self.training_history,
        }
        destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        return destination
