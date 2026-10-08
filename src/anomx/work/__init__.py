"""Portable interfaces between user-authored work and its execution host.

No platform, worker, or ML dependency is imported by this module. A platform
supplies callbacks; a standalone script can use the same context to collect
events locally. Compute submission and model publication require a host.
"""

from __future__ import annotations

import base64
import hashlib
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anomx.datasets import Dataset
    from anomx.models.onnx import ONNXModel

_active_context: ContextVar[WorkContext | None] = ContextVar("anomx_work_context", default=None)


class WorkContext:
    """Host-injected logging, findings, metrics, datasets, and compute services.

    Callback keys are ``log``, ``notify``, ``detection``, ``metric``,
    ``publish_model``, ``model_artifact``, and ``compute``. Callbacks receive keyword
    arguments matching the public methods. They execute synchronously; ``compute`` may
    return a submission reference rather than the computation's final result.
    """

    def __init__(
        self,
        callbacks: Mapping[str, Callable[..., object]] | None = None,
        data_loader: Callable[..., object] | None = None,
        *,
        inputs: Mapping[str, Any] | None = None,
        job_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.callbacks = dict(callbacks or {})
        self.data_loader = data_loader
        self.inputs = dict(inputs or {})
        self.job_id = job_id
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []

    @contextmanager
    def activate(self) -> Iterator[WorkContext]:
        """Bind this context to the current thread/async context temporarily."""
        token = _active_context.set(self)
        try:
            yield self
        finally:
            _active_context.reset(token)

    def _emit(self, kind: str, **payload: object) -> object:
        callback = self.callbacks.get(kind)
        if callback is not None:
            result = callback(**payload)
        elif kind in {"compute", "publish_model", "model_artifact"}:
            raise RuntimeError(f"This execution host does not provide `{kind}`.")
        else:
            result = None
        self.events.append(
            {
                "kind": kind,
                "timestamp": datetime.now(UTC).isoformat(),
                **payload,
            }
        )
        return result

    def log(self, message: str, level: str = "info", **metadata: object) -> object:
        """Append an inspectable job log entry."""
        if level not in {"debug", "info", "warning", "error", "critical"}:
            raise ValueError("Unsupported log level.")
        return self._emit("log", message=str(message), level=level, **metadata)

    def notify(self, title: str, message: str = "", severity: str = "info") -> object:
        """Request a notification from the host."""
        if severity not in {"info", "success", "warning", "error", "critical"}:
            raise ValueError("Unsupported notification severity.")
        return self._emit("notify", title=title, message=message, severity=severity)

    def detection(self, title: str, score: float | None = None, **metadata: object) -> object:
        """Publish an anomaly finding with optional score and source metadata."""
        if score is not None and not math.isfinite(float(score)):
            raise ValueError("Detection score must be finite.")
        return self._emit("detection", title=title, score=score, **metadata)

    def metric(self, name: str, value: float, step: int | None = None) -> object:
        """Record a named finite KPI, optionally at a training step."""
        if not str(name).strip() or not math.isfinite(float(value)):
            raise ValueError("Metrics require a name and a finite value.")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ValueError("Metric step must be a nonnegative integer.")
        return self._emit("metric", name=name, value=float(value), step=step)

    def publish_model(
        self,
        path: str | Path,
        name: str,
        metrics: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> object:
        """Publish a local ONNX file through the host's durable model storage.

        The host owns authorization, storage, versioning, and registry sync.
        This call returns only after the host has accepted the artifact.
        """
        artifact = Path(path).expanduser().resolve()
        if artifact.suffix.lower() != ".onnx" or not artifact.is_file():
            raise ValueError("Model publication requires an existing .onnx file.")
        if not name.strip():
            raise ValueError("A model name is required.")
        model_metrics = {key: float(value) for key, value in (metrics or {}).items()}
        if any(not math.isfinite(value) for value in model_metrics.values()):
            raise ValueError("Model metrics must be finite.")
        return self._emit(
            "publish_model",
            path=str(artifact),
            name=name,
            metrics=model_metrics,
            metadata=dict(metadata or {}),
        )

    def compute(
        self,
        code: str,
        inputs: Mapping[str, Any] | None = None,
        target: str | None = None,
    ) -> object:
        """Submit a compute block to the work item's selected compute target."""
        if not str(code).strip():
            raise ValueError("Compute code cannot be empty.")
        compile(code, "<anomx-compute>", "exec")
        return self._emit("compute", code=code, inputs=dict(inputs or {}), target=target)

    def load_model(self, reference: str) -> ONNXModel:
        """Load a stored ONNX model through the host's scoped artifact access.

        Training code and storage credentials never enter the inference session.
        The host returns the self-contained graph and its SHA-256 checksum.
        """
        from anomx.models.onnx import ONNXModel

        response = self._emit("model_artifact", reference=str(reference))
        if not isinstance(response, dict) or not isinstance(response.get("artifact_base64"), str):
            raise ValueError("The host returned an invalid model artifact.")
        encoded = response["artifact_base64"]
        if len(encoded) > 64 * 1024 * 1024 * 4 // 3 + 4:
            raise ValueError("Model artifacts must be at most 64 MiB.")
        payload = base64.b64decode(encoded, validate=True)
        if len(payload) > 64 * 1024 * 1024:
            raise ValueError("Model artifacts must be at most 64 MiB.")
        if hashlib.sha256(payload).hexdigest() != response.get("checksum_sha256"):
            raise ValueError("The model artifact checksum does not match.")
        return ONNXModel(payload)

    def dataset(
        self,
        channels: Sequence[str],
        *,
        start: str | None = None,
        end: str | None = None,
        frequency: str | None = None,
    ) -> Dataset:
        """Load channel references through the host into a Darts-backed Dataset."""
        from anomx.datasets import Dataset

        if self.data_loader is None:
            raise RuntimeError("This execution host does not provide a dataset loader.")
        return Dataset.from_channels(
            channels,
            loader=self.data_loader,
            start=start,
            end=end,
            frequency=frequency,
        )


def get_work_context() -> WorkContext:
    """Return the context activated by the current execution host."""
    context = _active_context.get()
    if context is None:
        raise RuntimeError("No work context is active. Use `with work.activate():`. ")
    return context


__all__ = ["WorkContext", "get_work_context"]
