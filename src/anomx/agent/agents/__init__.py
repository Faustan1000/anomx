"""Class-based agent definitions for the Anomx agent runtime."""

from anomx.agent.agents.main import (
    AUTO_AGENT_PROMPT,
    AUTOMATIC_AGENT_PROMPT,
    AUTONOMOUS_AGENT_PROMPT,
    CONNECTED_PLATFORM_AGENT_PROMPT,
    PLAN_AGENT_PROMPT,
    STANDARD_AGENT_PROMPT,
    AutoAgent,
    AutomaticAgent,
    AutonomousAgent,
    BuildAgent,
    PlanAgent,
    StandardAgent,
)
from anomx.agent.agents.sub_agents import ExploreAgent, GeneralAgent, PlatformAgent
from anomx.agent.agents.sub_agents.explore import EXPLORE_AGENT_PROMPT
from anomx.agent.agents.sub_agents.general import GENERAL_AGENT_PROMPT
from anomx.agent.agents.sub_agents.platform import PLATFORM_AGENT_PROMPT
from anomx.agent.helpers.utils import (
    AgentKind,
    AgentSpec,
    agent_spec,
    main_agent_kinds,
    next_main_agent_kind,
    parse_agent_kind,
)

__all__ = [
    "AUTO_AGENT_PROMPT",
    "AUTOMATIC_AGENT_PROMPT",
    "AUTONOMOUS_AGENT_PROMPT",
    "BUILD_AGENT_PROMPT",
    "CONNECTED_PLATFORM_AGENT_PROMPT",
    "EXPLORE_AGENT_PROMPT",
    "GENERAL_AGENT_PROMPT",
    "PLAN_AGENT_PROMPT",
    "PLATFORM_AGENT_PROMPT",
    "STANDARD_AGENT_PROMPT",
    "AgentKind",
    "AgentSpec",
    "AutoAgent",
    "AutomaticAgent",
    "AutonomousAgent",
    "BuildAgent",
    "ExploreAgent",
    "GeneralAgent",
    "PlanAgent",
    "PlatformAgent",
    "StandardAgent",
    "agent_spec",
    "main_agent_kinds",
    "next_main_agent_kind",
    "parse_agent_kind",
]
