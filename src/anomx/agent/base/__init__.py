"""Base classes with lazy imports so exception types work without loading tools."""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "AgentKind": "agents",
    "BaseAgent": "agents",
    "BaseException": "exceptions",
    "BaseTool": "tools",
    "QuestionOption": "interactions",
    "QuestionRequest": "interactions",
    "QuestionResponse": "interactions",
    "AsyncProcessState": "processes",
    "SubagentRuntimeState": "subagents",
}

__all__ = [
    "AgentKind",
    "BaseAgent",
    "BaseException",
    "BaseTool",
    "QuestionOption",
    "QuestionRequest",
    "QuestionResponse",
    "AsyncProcessState",
    "SubagentRuntimeState",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
