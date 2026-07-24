"""Node execution shared by the synchronous runner and platform tasks.

Every node type reads its inputs from the JSON-encodable context and writes
its output back into it. The platform runs each node in its own task with the
context travelling through redis; :class:`~anomx.runner.runner.JobRunner`
walks the same executor in-process.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from anomx.runner.context import (
    read_context_flag,
    read_context_frame,
    write_context_flag,
    write_context_frame,
)
from anomx.runner.jobs import JobDefinition, JobNode, JobNodeType


class JobRunError(RuntimeError):
    """Raised when a job or node cannot be executed."""


@dataclass(slots=True)
class NodeExecution:
    """Outcome of executing one node."""

    node_id: str
    node_type: str
    status: str
    context: dict[str, Any]
    next_node_ids: list[str] = field(default_factory=list)


def run_python_body(body: str, context: dict[str, Any]) -> dict[str, Any]:
    """Execute user-authored function-body code against the context.

    The user only writes the body; it runs inside
    ``def __anomx_python_node__(context): ...`` and may mutate or return the
    context. Risk assessment of the body is the platform's responsibility
    before the job is allowed to run.
    """
    body_lines = [line for line in str(body or "").splitlines()]
    indented_body = "\n".join(f"    {line}" for line in body_lines) or "    pass"
    source = f"def __anomx_python_node__(context):\n{indented_body}\n    return context\n"
    namespace: dict[str, Any] = {}
    exec(compile(source, "<anomx-python-node>", "exec"), {"json": json, "math": math}, namespace)  # noqa: S102
    result = namespace["__anomx_python_node__"](context)
    return result if isinstance(result, dict) else context


class NodeExecutor:
    """Execute job nodes against a shared context.

    Components referenced by nodes are resolved from the `components` registry
    (key → instance) first, then from dotted import paths. Fitted models stay
    in `model_registry` and are referenced from the context via `model_ref`.
    """

    def __init__(
        self,
        *,
        components: dict[str, Any] | None = None,
        python_functions: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
        model_registry: dict[str, Any] | None = None,
    ) -> None:
        self.components = dict(components or {})
        self.python_functions = dict(python_functions or {})
        self.model_registry = dict(model_registry or {})

    def execute(self, definition: JobDefinition, node_id: str, context: dict[str, Any]) -> NodeExecution:
        node = definition.node(str(node_id))
        next_node_ids = list(node.next)
        status = "processed"

        if node.type is JobNodeType.START:
            self._run_start(node, context)
        elif node.type is JobNodeType.END:
            next_node_ids = []
        elif node.type is JobNodeType.IF_ELSE:
            next_node_ids = self._run_if_else(node, context, next_node_ids)
        elif node.type is JobNodeType.PYTHON:
            context = self._run_python(node, context)
        elif node.type is JobNodeType.MODEL_TRAINING:
            self._run_model_training(node, context)
        elif node.type is JobNodeType.MODEL_INFERENCE:
            self._run_model_inference(node, context)
        elif node.type is JobNodeType.SCORER:
            self._run_scorer(node, context)
        elif node.type is JobNodeType.DETECTOR:
            self._run_detector(node, context)
        elif node.type is JobNodeType.CLASSIFIER:
            status = self._run_classifier(node, context)

        return NodeExecution(
            node_id=node.id,
            node_type=node.type.value,
            status=status,
            context=context,
            next_node_ids=next_node_ids,
        )

    # --- Node handlers ---------------------------------------------------------------------------

    def _run_start(self, node: JobNode, context: dict[str, Any]) -> None:
        if read_context_frame(context, "data") is not None:
            return
        dataset_identifier = str(node.config.get("dataset") or "").strip()
        if not dataset_identifier:
            return
        from anomx.data.remote import AnomxDataset

        write_context_frame(context, "data", AnomxDataset.from_anomx(dataset_identifier).frame)

    def _run_if_else(self, node: JobNode, context: dict[str, Any], next_node_ids: list[str]) -> list[str]:
        branches = node.config.get("branches") if isinstance(node.config.get("branches"), dict) else {}
        condition_key = str(node.config.get("condition_key") or "should_train")
        condition_value = read_context_flag(context, condition_key)
        branch_target = branches.get("true" if condition_value else "false")
        if branch_target:
            return [str(branch_target)]
        return next_node_ids[:1]

    def _run_python(self, node: JobNode, context: dict[str, Any]) -> dict[str, Any]:
        body = str(node.config.get("body") or "")
        if body.strip():
            return run_python_body(body, context)
        function_name = str(node.config.get("function") or "")
        python_function = self.python_functions.get(function_name)
        if python_function is None:
            raise JobRunError(f"Python node `{node.id}` has no body and references unregistered function `{function_name}`.")
        context[f"python__{node.id}"] = python_function(context)
        return context

    def _run_model_training(self, node: JobNode, context: dict[str, Any]) -> None:
        data_frame = read_context_frame(context, "data")
        if data_frame is None:
            raise JobRunError(f"Model training node `{node.id}` requires `data` in the context.")
        component_reference, component = self._resolve_component(node)
        component.fit(data_frame)
        self.model_registry[component_reference] = component
        context["model_ref"] = component_reference
        write_context_flag(context, "trained", True)

    def _run_model_inference(self, node: JobNode, context: dict[str, Any]) -> None:
        data_frame = read_context_frame(context, "data")
        if data_frame is None:
            raise JobRunError(f"Model inference node `{node.id}` requires `data` in the context.")
        model_reference = str(context.get("model_ref") or "")
        model = self.model_registry.get(model_reference)
        if model is None:
            _, model = self._resolve_component(node)
        try:
            predictions = model.predict(data_frame)
        except RuntimeError:
            # An untrained model in an inference-only graph is fitted in place.
            model.fit(data_frame)
            predictions = model.predict(data_frame)
        write_context_frame(context, "predictions", predictions)

    def _run_scorer(self, node: JobNode, context: dict[str, Any]) -> None:
        input_frame = read_context_frame(context, "predictions")
        if input_frame is None:
            input_frame = read_context_frame(context, "data")
        if input_frame is None:
            raise JobRunError(f"Scorer node `{node.id}` requires `predictions` or `data` in the context.")
        _, component = self._resolve_component(node)
        write_context_frame(context, "scores", component.score(input_frame))

    def _run_detector(self, node: JobNode, context: dict[str, Any]) -> None:
        input_frame = read_context_frame(context, "scores")
        if input_frame is None:
            input_frame = read_context_frame(context, "predictions")
        if input_frame is None:
            raise JobRunError(f"Detector node `{node.id}` requires `scores` or `predictions` in the context.")
        _, component = self._resolve_component(node)
        write_context_frame(context, "decisions", component.detect(input_frame))

    def _run_classifier(self, node: JobNode, context: dict[str, Any]) -> str:
        decisions_frame = read_context_frame(context, "decisions")
        if decisions_frame is None:
            return "skipped"
        if not str(node.config.get("component") or "").strip():
            return "skipped"
        _, component = self._resolve_component(node)
        write_context_frame(context, "classes", component.classify(decisions_frame))
        return "processed"

    def _resolve_component(self, node: JobNode) -> tuple[str, Any]:
        component_reference = str(node.config.get("component") or "").strip()
        if not component_reference:
            raise JobRunError(f"Job node `{node.id}` requires a `component` reference in its config.")
        registered_component = self.components.get(component_reference)
        if registered_component is not None:
            return component_reference, registered_component

        module_path, _, class_name = component_reference.rpartition(".")
        if not module_path:
            raise JobRunError(f"Job node `{node.id}` references unknown component `{component_reference}`.")
        component_class = getattr(import_module(module_path), class_name)
        component_config = node.config.get("config") if isinstance(node.config.get("config"), dict) else {}
        component = component_class(component_config)
        self.components[component_reference] = component
        return component_reference, component
