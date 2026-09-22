"""Object-oriented agent primitives for Anomx."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from anomx.agent.base.tools import BaseTool


class AgentKind(StrEnum):
    """Supported Anomx agent kinds."""

    MAIN = "main"
    SUB = "sub"


@dataclass(frozen=True)
class BaseAgent:
    """Base class for class-based Anomx agents."""

    kind: AgentKind
    name: str
    system_prompt: str
    tools: tuple[BaseTool, ...]
    color: str = "accent"
    symbol: str = "Ω"
    can_spawn_subagents: bool = False
    can_ask_questions: bool = False
    can_use_plans: bool = False
    can_start_processes: bool = False
    can_use_web: bool = True

    @property
    def prompt(self) -> str:
        """Compatibility alias for older runtime code."""

        return self.system_prompt

    @property
    def prompt_hint(self) -> str:
        """Return compact prompt-bar text for this agent."""

        return f"{self.symbol}  {self.name} (shift+tab to cycle)"

    def tool_definitions(self) -> list[dict[str, object]]:
        """Return function definitions for this agent's assigned tools."""

        return [tool.definition() for tool in self.tools]

    def tool_for(self, name: str) -> BaseTool | None:
        """Return the assigned tool that handles a requested tool name."""

        for tool in self.tools:
            if tool.handles(name):
                return tool
        return None
