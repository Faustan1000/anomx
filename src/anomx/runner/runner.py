"""Synchronous job runner.

Runs a :class:`JobDefinition` in-process with the same context-passing node
semantics the platform executes as one task per node — useful for local
development, tests, and batch experiments. Jobs can run as a whole
(:meth:`JobRunner.run`) or one node at a time (:meth:`JobRunner.run_node`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from anomx.runner.context import normalize_context
from anomx.runner.execution import JobRunError, NodeExecution, NodeExecutor
from anomx.runner.jobs import JobDefinition, JobDefinitionError

MAX_JOB_STEPS = 1000


@dataclass(slots=True)
class JobRunResult:
    """Outcome of one synchronous job execution."""

    status: str = "completed"
    visited_node_ids: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "visited_node_ids": list(self.visited_node_ids),
            "duration_ms": self.duration_ms,
            "context_keys": sorted(str(key) for key in self.context),
        }


class JobRunner:
    """Execute job definitions synchronously with a propagated context."""

    def __init__(
        self,
        *,
        components: dict[str, Any] | None = None,
        python_functions: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
    ) -> None:
        self.executor = NodeExecutor(components=components, python_functions=python_functions)

    def run(
        self,
        definition: JobDefinition | dict[str, Any],
        data: Any = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> JobRunResult:
        """Run a whole job from its start node to an end node."""
        job_definition = self._normalize_definition(definition)
        run_context = normalize_context(context, data)
        result = JobRunResult(context=run_context)
        started_at = time.perf_counter()

        pending_node_ids = [job_definition.start_node().id]
        while pending_node_ids:
            if len(result.visited_node_ids) >= MAX_JOB_STEPS:
                raise JobRunError(f"Job aborted after {MAX_JOB_STEPS} steps; the graph likely contains an unbounded loop.")
            node_id = pending_node_ids.pop(0)
            execution = self.executor.execute(job_definition, node_id, run_context)
            run_context = execution.context
            result.visited_node_ids.append(execution.node_id)
            pending_node_ids.extend(execution.next_node_ids)

        result.context = run_context
        result.duration_ms = max(0, int((time.perf_counter() - started_at) * 1000))
        return result

    def run_node(
        self,
        definition: JobDefinition | dict[str, Any],
        node_id: str,
        context: dict[str, Any] | None = None,
    ) -> NodeExecution:
        """Run a single node of a job — the platform task entry point."""
        return self.executor.execute(self._normalize_definition(definition), node_id, normalize_context(context))

    @staticmethod
    def _normalize_definition(definition: JobDefinition | dict[str, Any]) -> JobDefinition:
        job_definition = definition if isinstance(definition, JobDefinition) else JobDefinition.from_dict(definition)
        problems = job_definition.validate()
        if problems:
            raise JobDefinitionError("Invalid job definition: " + " ".join(problems))
        return job_definition
