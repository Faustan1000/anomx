"""Public agent API with lazy interface/runtime imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "AgentKind": "anomx.agent.agents",
    "MainAgent": "anomx.agent.agents",
    "SubAgent": "anomx.agent.agents",
    "AnomxCliApp": "anomx.agent.app",
    "BaseAgent": "anomx.agent.base",
    "BaseTool": "anomx.agent.base",
    "AgentMode": "anomx.agent.helpers.mode",
    "AgentModePolicy": "anomx.agent.helpers.mode",
    "mode_policy": "anomx.agent.helpers.mode",
    "next_agent_mode": "anomx.agent.helpers.mode",
    "AI_PROVIDER_KEYS": "anomx.agent.store",
    "AI_PROVIDERS": "anomx.agent.store",
    "DEFAULT_CONFIG": "anomx.agent.store",
    "MODEL_METADATA": "anomx.agent.store",
    "AnomxHome": "anomx.agent.store",
    "ModelMetadata": "anomx.agent.store",
    "ProviderOption": "anomx.agent.store",
    "SessionRecord": "anomx.agent.store",
    "ThinkingIntensityOption": "anomx.agent.store",
    "model_context_window": "anomx.agent.store",
    "model_detail": "anomx.agent.store",
    "model_metadata": "anomx.agent.store",
    "resolve_anomx_home": "anomx.agent.store",
    "thinking_intensity_options": "anomx.agent.store",
    "thinking_intensity_supported": "anomx.agent.store",
    "AgentState": "anomx.agent.ui.models",
}

__all__ = [
    "AI_PROVIDERS",
    "AI_PROVIDER_KEYS",
    "DEFAULT_CONFIG",
    "MODEL_METADATA",
    "AgentMode",
    "AgentModePolicy",
    "AgentKind",
    "AgentState",
    "AnomxCliApp",
    "AnomxHome",
    "BaseAgent",
    "BaseTool",
    "MainAgent",
    "ModelMetadata",
    "ProviderOption",
    "SubAgent",
    "SessionRecord",
    "ThinkingIntensityOption",
    "model_context_window",
    "model_detail",
    "model_metadata",
    "mode_policy",
    "next_agent_mode",
    "resolve_anomx_home",
    "thinking_intensity_options",
    "thinking_intensity_supported",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
