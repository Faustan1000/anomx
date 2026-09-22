"""The two agent roles supported by the Anomx runtime."""

from anomx.agent.agents.main_agent import (
    CONNECTED_PLATFORM_AGENT_PROMPT,
    MAIN_AGENT_PROMPT,
    MainAgent,
)
from anomx.agent.agents.sub_agent import SUBAGENT_PROMPT, SubAgent
from anomx.agent.helpers.utils import AgentKind, AgentSpec, agent_spec, parse_agent_kind

__all__ = [
    "CONNECTED_PLATFORM_AGENT_PROMPT",
    "MAIN_AGENT_PROMPT",
    "SUBAGENT_PROMPT",
    "AgentKind",
    "AgentSpec",
    "MainAgent",
    "SubAgent",
    "agent_spec",
    "parse_agent_kind",
]
