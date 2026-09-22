"""Shared utility helpers for the Anomx agent package."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from anomx.agent.base.agents import AgentKind, BaseAgent

AgentSpec = BaseAgent


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp suitable for JSONL events."""

    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def session_id_from_path(session_path: Path) -> str:
    """Extract a session identifier from a session transcript path."""

    stem = session_path.stem
    if stem.startswith("rollout-"):
        return stem.rsplit("-", 1)[-1]
    parts = stem.split("-", 2)
    if len(parts) >= 3:
        return parts[-1]
    return stem


def _new_agents() -> dict[AgentKind, BaseAgent]:
    from anomx.agent.agents.main_agent import MainAgent
    from anomx.agent.agents.sub_agent import SubAgent

    return {
        AgentKind.MAIN: MainAgent(),
        AgentKind.SUB: SubAgent(),
    }


def parse_agent_kind(value: object, default: AgentKind = AgentKind.MAIN) -> AgentKind:
    """Parse stored config/session values into an agent kind."""

    aliases = {
        "operator": AgentKind.MAIN,
        "build": AgentKind.MAIN,
        "standard": AgentKind.MAIN,
        "auto": AgentKind.MAIN,
        "automatic": AgentKind.MAIN,
        "autonomous": AgentKind.MAIN,
        "plan": AgentKind.MAIN,
        "planning": AgentKind.MAIN,
        "worker": AgentKind.SUB,
        "general": AgentKind.SUB,
        "explore": AgentKind.SUB,
        "platform": AgentKind.SUB,
    }
    normalized = (
        (value.value if isinstance(value, AgentKind) else str(value or ""))
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if normalized in aliases:
        return aliases[normalized]
    try:
        return AgentKind(normalized)
    except ValueError:
        return default


def agent_spec(kind: AgentKind | str | object) -> BaseAgent:
    """Return a fresh agent object for a kind."""

    return _new_agents()[parse_agent_kind(kind)]


__all__ = [
    "AgentKind",
    "AgentSpec",
    "BaseAgent",
    "agent_spec",
    "parse_agent_kind",
    "session_id_from_path",
    "utc_now_iso",
]
