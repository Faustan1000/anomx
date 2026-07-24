"""Object-oriented tool primitives for Anomx agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anomx.agent.exceptions import ToolExecutionError

JsonSchema = dict[str, Any]
ToolDefinition = dict[str, Any]


@dataclass(frozen=True)
class ToolExecutionContext:
    """Runtime resources available to a tool execution."""

    runtime: Any
    callbacks: Any
    session_path: Path | None = None

    def json_result(self, payload: dict[str, Any]) -> str:
        """Serialize a tool result payload."""

        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def workspace_path(self, raw_path: object) -> Path | str:
        """Resolve a path under the active trusted workspace."""

        raw = str(raw_path or "").strip()
        if not raw:
            return "Path is required."
        try:
            resolved = self.runtime.tool_manager.resolve_trusted_path(raw)
        except OSError as error:
            return str(error)
        if not self.path_inside_workspace(resolved):
            return f"Path is outside the trusted workspace: {raw}"
        return resolved

    def path_inside_workspace(self, path: Path) -> bool:
        """Return whether a resolved path stays inside the trusted workspace."""

        return self.runtime.tool_manager.path_inside_workspace(path)

    def positive_int(self, value: object, fallback: int) -> int:
        """Parse a positive integer value with a fallback."""

        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError:
                return fallback
        else:
            return fallback
        return parsed if parsed > 0 else fallback

    def emit_operator_statement(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        default_statement: str | None = None,
    ) -> None:
        """Publish the operator-facing statement for a tool call."""

        statement = str(arguments.get("statement", "")).strip()
        statement = statement or default_statement or default_tool_statement(tool_name)
        callbacks = self.callbacks
        if callbacks.command is not None:
            callbacks.command(statement, operator_tool_detail(tool_name, arguments), "")
            return
        callback = callbacks.tool_message or callbacks.status
        if callback is not None:
            callback(statement)


@dataclass(frozen=True)
class BaseTool:
    """Base class for model-callable tools.

    Tool objects own both the public schema exposed to models and the execution
    behavior for model tool calls. Runtime-specific resources are supplied
    through :class:`ToolExecutionContext`.
    """

    name: str
    description: str
    parameters: JsonSchema
    aliases: tuple[str, ...] = ()

    def definition(self) -> ToolDefinition:
        """Return the OpenAI/Anthropic-compatible function definition."""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def handles(self, name: str) -> bool:
        """Return whether this tool should execute a requested tool name."""

        normalized = name.strip()
        return normalized == self.name or normalized in self.aliases

    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> str:
        """Execute the tool and return a serialized model-facing result."""

        raise ToolExecutionError(f"{self.name} does not implement execute().")


def object_schema(
    properties: dict[str, JsonSchema],
    required: list[str],
) -> JsonSchema:
    """Build a strict object schema for a tool parameter payload."""

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def statement_property(description: str) -> JsonSchema:
    """Return the common statement property schema."""

    return {"type": "string", "description": description}


def operator_tool_detail(tool_name: str, arguments: dict[str, Any]) -> str:
    """Return a human-readable summary for a structured tool call."""

    parameters = {
        key: value
        for key, value in arguments.items()
        if key != "statement"
    }
    if not parameters:
        return f"Tool: {tool_name}\nParameters: none"
    return (
        f"Tool: {tool_name}\n"
        "Parameters:\n"
        f"{json.dumps(parameters, indent=2, ensure_ascii=False, default=str)}"
    )


def default_tool_statement(tool_name: str) -> str:
    """Return the default UI statement for a tool name."""

    return {
        "run_command": "Running command",
        "run_cli_command": "Running command",
        "create_plan": "Creating plan",
        "update_plan": "Updating plan",
        "start_process": "Starting process",
        "end_process": "Ending process",
        "check_command_status": "Checking command",
        "kill_command": "Killing command",
        "ask_question": "Asking question",
        "memorize": "Saving memory",
        "remove_plan": "Removing plan",
        "finish_anyways": "Finishing anyway",
        "output_response": "Preparing response",
        "send_feedback": "Sending platform feedback",
        "start_subagent": "Starting subagent",
        "prompt_subagent": "Prompting subagent",
        "remove_subagent": "Removing subagent",
        "get_subagent_info": "Checking subagent",
        "start_agent": "Starting subagent",
        "prompt_agent": "Prompting subagent",
        "remove_agent": "Removing subagent",
        "interrupt_agent": "Removing subagent",
        "check_agent": "Checking subagent",
        "web_search": "Searching web",
        "web_fetch": "Fetching web page",
        "websearch": "Searching web",
        "webfetch": "Fetching web page",
        "use_anomx_api": "Calling Anomx API",
        "read": "Reading file",
        "list": "Listing directory",
        "glob": "Finding files",
        "grep": "Searching files",
        "bash": "Running command",
    }.get(tool_name, "Working")
