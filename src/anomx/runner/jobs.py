"""JSON job definitions.

A job is a directed graph of nodes running from `start` to `end`. The same
JSON structure is edited visually on the platform's pipeline canvas, executed
synchronously by :class:`~anomx.runner.runner.JobRunner`, and executed
asynchronously on the platform where every node becomes its own task that
queues its `next` nodes.

```json
{
    "nodes": [
        {"id": "start", "type": "start", "last": [], "next": ["infer"], "config": {}, "_meta": {"x": 0, "y": 0}},
        {"id": "infer", "type": "model_inference", "last": ["start"], "next": ["score"], "config": {"component": "constant_baseline"}},
        {"id": "score", "type": "scorer", "last": ["infer"], "next": ["end"], "config": {"component": "absolute_error"}},
        {"id": "end", "type": "end", "last": ["score"], "next": [], "config": {}}
    ]
}
```

`_meta` carries editor-only information (canvas position and the like); the
runner ignores it but round-trips it untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobNodeType(str, Enum):
    START = "start"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    SCORER = "scorer"
    DETECTOR = "detector"
    CLASSIFIER = "classifier"
    IF_ELSE = "if_else"
    PYTHON = "python"
    END = "end"


@dataclass(slots=True)
class JobNode:
    """One processing step in a job graph."""

    id: str
    type: JobNodeType
    last: list[str] = field(default_factory=list)
    next: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JobNode:
        return cls(
            id=str(payload.get("id") or "").strip(),
            type=JobNodeType(str(payload.get("type") or "").strip()),
            last=[str(node_id) for node_id in payload.get("last") or []],
            next=[str(node_id) for node_id in payload.get("next") or []],
            config=dict(payload.get("config") or {}),
            meta=dict(payload.get("_meta") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "last": list(self.last),
            "next": list(self.next),
            "config": dict(self.config),
            "_meta": dict(self.meta),
        }


class JobDefinitionError(ValueError):
    """Raised when a job definition is structurally invalid."""


@dataclass(slots=True)
class JobDefinition:
    """A validated graph of job nodes."""

    nodes: list[JobNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JobDefinition:
        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, list):
            raise JobDefinitionError("Job definitions require a `nodes` list.")
        return cls(
            nodes=[JobNode.from_dict(raw_node) for raw_node in raw_nodes if isinstance(raw_node, dict)],
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "metadata": dict(self.metadata),
        }

    def node(self, node_id: str) -> JobNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise JobDefinitionError(f"Job node `{node_id}` does not exist.")

    def start_node(self) -> JobNode:
        start_nodes = [node for node in self.nodes if node.type is JobNodeType.START]
        if len(start_nodes) != 1:
            raise JobDefinitionError("Job definitions require exactly one start node.")
        return start_nodes[0]

    def next_nodes(self, node: JobNode) -> list[JobNode]:
        return [self.node(node_id) for node_id in node.next]

    def validate(self) -> list[str]:
        """Return structural problems, empty when the definition is runnable."""
        problems: list[str] = []
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            problems.append("Job node ids must be unique.")
        if not any(node.id for node in self.nodes):
            problems.append("Job definitions require at least one node with an id.")
        start_count = sum(1 for node in self.nodes if node.type is JobNodeType.START)
        if start_count != 1:
            problems.append("Job definitions require exactly one start node.")
        if not any(node.type is JobNodeType.END for node in self.nodes):
            problems.append("Job definitions require at least one end node.")
        known_ids = set(node_ids)
        for node in self.nodes:
            for reference in (*node.last, *node.next):
                if reference not in known_ids:
                    problems.append(f"Job node `{node.id}` references unknown node `{reference}`.")
            if node.type is JobNodeType.END and node.next:
                problems.append(f"End node `{node.id}` cannot have next nodes.")
            if node.type is not JobNodeType.END and not node.next:
                problems.append(f"Job node `{node.id}` requires at least one next node.")
        return problems
