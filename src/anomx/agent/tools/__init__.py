"""Concrete tool classes and tool-set factories for Anomx agents."""

from __future__ import annotations

from anomx.agent.base.tools import BaseTool
from anomx.agent.tools.anomx_platform import (
    GetAnomxDataChannelHistoryTool,
    GetAnomxObjectDetailsTool,
    GetBackgroundRunsTool,
    SearchAnomxDataChannelsTool,
    SearchAnomxObjectsTool,
)
from anomx.agent.tools.data_adapters import ManageDataAdaptersTool
from anomx.agent.tools.ask_question import AskQuestionTool
from anomx.agent.tools.check_command_status import CheckCommandStatusTool
from anomx.agent.tools.cli_command import CliCommandTool
from anomx.agent.tools.create_plan import CreatePlanTool
from anomx.agent.tools.end_process import EndProcessTool
from anomx.agent.tools.finish_anyways import FinishAnywaysTool
from anomx.agent.tools.get_subagent_info import GetSubagentInfoTool
from anomx.agent.tools.glob import GlobTool
from anomx.agent.tools.grep import GrepTool
from anomx.agent.tools.kill_command import KillCommandTool
from anomx.agent.tools.list_directory import ListDirectoryTool
from anomx.agent.tools.memorize import MemorizeTool
from anomx.agent.tools.output_response import OutputResponseTool
from anomx.agent.tools.prompt_subagent import PromptSubagentTool
from anomx.agent.tools.read_file import ReadFileTool
from anomx.agent.tools.remove_plan import RemovePlanTool
from anomx.agent.tools.remove_subagent import RemoveSubagentTool
from anomx.agent.tools.send_feedback import SendFeedbackTool
from anomx.agent.tools.start_process import StartProcessTool
from anomx.agent.tools.start_subagent import StartSubagentTool
from anomx.agent.tools.update_plan import UpdatePlanTool
from anomx.agent.tools.use_anomx_api import UseAnomxApiTool
from anomx.agent.tools.wait import WaitTool
from anomx.agent.tools.web_fetch import WebFetchTool
from anomx.agent.tools.web_search import WebSearchTool

MAIN_AGENT_STATEMENT_DESCRIPTION = "Persistent user-visible working message for this tool call."
SUBAGENT_STATEMENT_DESCRIPTION = "Persistent working message for this tool call."


def main_agent_tools() -> tuple[BaseTool, ...]:
    """Return every tool available to the main agent."""

    statement = MAIN_AGENT_STATEMENT_DESCRIPTION
    return (
        CliCommandTool(
            statement_description=statement,
            description="Run a CLI command for main-agent inspection or validation.",
            aliases=("run_cli_command",),
            main_agent=True,
        ),
        ReadFileTool(statement_description=statement),
        ListDirectoryTool(statement_description=statement),
        GlobTool(statement_description=statement),
        GrepTool(statement_description=statement),
        WebSearchTool(statement_description=statement),
        WebFetchTool(statement_description=statement),
        UseAnomxApiTool(statement_description=statement),
        GetAnomxObjectDetailsTool(),
        SearchAnomxObjectsTool(),
        SearchAnomxDataChannelsTool(),
        GetAnomxDataChannelHistoryTool(),
        ManageDataAdaptersTool(),
        StartProcessTool(statement_description=statement, main_agent=True),
        EndProcessTool(statement_description=statement),
        AskQuestionTool(statement_description=statement),
        OutputResponseTool(),
        SendFeedbackTool(statement_description=statement),
        MemorizeTool(statement_description=statement),
        CreatePlanTool(),
        UpdatePlanTool(),
        RemovePlanTool(statement_description=statement),
        FinishAnywaysTool(statement_description=statement),
        StartSubagentTool(statement_description=statement),
        PromptSubagentTool(statement_description=statement),
        RemoveSubagentTool(statement_description=statement),
        GetSubagentInfoTool(),
    )


def subagent_tools() -> tuple[BaseTool, ...]:
    """Return main-agent tools except direct-user, process, plan, and delegation tools."""

    statement = SUBAGENT_STATEMENT_DESCRIPTION
    return (
        CliCommandTool(
            statement_description=statement,
            aliases=("run_cli_command",),
        ),
        ReadFileTool(statement_description=statement),
        ListDirectoryTool(statement_description=statement),
        GlobTool(statement_description=statement),
        GrepTool(statement_description=statement),
        WebSearchTool(statement_description=statement),
        WebFetchTool(statement_description=statement),
        UseAnomxApiTool(statement_description=statement),
        GetAnomxObjectDetailsTool(),
        SearchAnomxObjectsTool(),
        SearchAnomxDataChannelsTool(),
        GetAnomxDataChannelHistoryTool(),
        ManageDataAdaptersTool(),
        SendFeedbackTool(statement_description=statement),
    )


def read_only_mode_tools() -> tuple[BaseTool, ...]:
    """Return tools exposed to either agent while Plan mode is active."""

    statement = MAIN_AGENT_STATEMENT_DESCRIPTION
    return (
        CliCommandTool(
            statement_description=statement,
            description="Run a read-only CLI command for planning and inspection.",
            access="read",
            aliases=("run_cli_command",),
            main_agent=True,
        ),
        ReadFileTool(statement_description=statement),
        ListDirectoryTool(statement_description=statement),
        GlobTool(statement_description=statement),
        GrepTool(statement_description=statement),
        WebSearchTool(statement_description=statement),
        WebFetchTool(statement_description=statement),
        GetAnomxObjectDetailsTool(),
        SearchAnomxObjectsTool(),
        SearchAnomxDataChannelsTool(),
        GetAnomxDataChannelHistoryTool(),
        ManageDataAdaptersTool(),
    )


def recommendation_mode_tools(*, main_agent: bool) -> tuple[BaseTool, ...]:
    """Return read tools plus recommendation creation for a connected platform."""

    statement = MAIN_AGENT_STATEMENT_DESCRIPTION if main_agent else SUBAGENT_STATEMENT_DESCRIPTION
    tools: list[BaseTool] = [
        CliCommandTool(
            statement_description=statement,
            description="Run a read-only CLI command for recommendation research.",
            access="read",
            aliases=("run_cli_command",),
            main_agent=main_agent,
        ),
        ReadFileTool(statement_description=statement),
        ListDirectoryTool(statement_description=statement),
        GlobTool(statement_description=statement),
        GrepTool(statement_description=statement),
        WebSearchTool(statement_description=statement),
        WebFetchTool(statement_description=statement),
        UseAnomxApiTool(statement_description=statement),
        GetAnomxObjectDetailsTool(),
        SearchAnomxObjectsTool(),
        SearchAnomxDataChannelsTool(),
        GetAnomxDataChannelHistoryTool(),
        ManageDataAdaptersTool(),
    ]
    tools.append(GetBackgroundRunsTool())
    if main_agent:
        tools.append(OutputResponseTool())
    return tuple(tools)


def command_control_tools() -> tuple[BaseTool, ...]:
    """Return tools for active long-running command calls."""

    return (CheckCommandStatusTool(), KillCommandTool())


def wait_tool(target_description: str) -> BaseTool:
    """Return a wait tool for the provided target description."""

    return WaitTool(target_description=target_description)


__all__ = [
    "BaseTool",
    "ManageDataAdaptersTool",
    "CheckCommandStatusTool",
    "CliCommandTool",
    "CreatePlanTool",
    "EndProcessTool",
    "FinishAnywaysTool",
    "GetSubagentInfoTool",
    "GetBackgroundRunsTool",
    "GetAnomxDataChannelHistoryTool",
    "GetAnomxObjectDetailsTool",
    "GlobTool",
    "GrepTool",
    "KillCommandTool",
    "ListDirectoryTool",
    "MemorizeTool",
    "OutputResponseTool",
    "PromptSubagentTool",
    "ReadFileTool",
    "RemovePlanTool",
    "RemoveSubagentTool",
    "SendFeedbackTool",
    "StartProcessTool",
    "StartSubagentTool",
    "SearchAnomxDataChannelsTool",
    "SearchAnomxObjectsTool",
    "UpdatePlanTool",
    "UseAnomxApiTool",
    "WaitTool",
    "WebFetchTool",
    "WebSearchTool",
    "command_control_tools",
    "main_agent_tools",
    "read_only_mode_tools",
    "recommendation_mode_tools",
    "subagent_tools",
    "wait_tool",
]
