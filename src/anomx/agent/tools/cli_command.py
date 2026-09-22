"""CLI command tool."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from anomx.agent.base.processes import AsyncProcessState
from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.helpers.tool_manager import CommandResult, CommandSafety
from anomx.agent.helpers.utils import utc_now_iso

CliCommandAccess = Literal["read", "write"]


@dataclass(frozen=True)
class CliCommandTool(BaseTool):
    """Run a CLI command through the runtime command policy engine."""

    access: CliCommandAccess = "write"

    def __init__(
        self,
        *,
        statement_description: str,
        name: str = "run_command",
        description: str | None = None,
        access: CliCommandAccess = "write",
        aliases: tuple[str, ...] = (),
        main_agent: bool = False,
    ) -> None:
        command_description = (
            "A read-only shell command inside the trusted workspace."
            if access == "read"
            else (
                "A single CLI command, for example 'ls -la'. Shell operators and "
                "redirection may be used when necessary; paths must resolve inside "
                "the trusted workspace root."
                if main_agent
                else "A single CLI command inside the trusted workspace."
            )
        )
        object.__setattr__(self, "access", access)
        super().__init__(
            name=name,
            description=description
            or (
                "Run a read-only CLI command inside the trusted workspace."
                if access == "read"
                else "Run a CLI command inside the trusted workspace."
            ),
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "command": {
                        "type": "string",
                        "description": command_description,
                    },
                },
                ["statement", "command"],
            ),
            aliases=aliases,
        )

    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> str:
        """Execute a CLI command using the configured access policy."""

        command = str(arguments.get("command", "")).strip()
        statement = str(arguments.get("statement", "")).strip()
        long_running_command: AsyncProcessState | None = None
        readonly_denial = self._readonly_command_denial(context, command, statement)
        if readonly_denial is not None:
            return readonly_denial

        def publish_long_running_command(process: subprocess.Popen[str]) -> str | None:
            nonlocal long_running_command
            if context.session_path is None or long_running_command is not None:
                return None
            process_id = uuid4().hex[:8]
            long_running_command = AsyncProcessState(
                process_id=process_id,
                command=command,
                statement=statement or "Running command",
                status="running",
                started_at=utc_now_iso(),
                process=process,
                source="command",
                owner_id=context.runtime.process_owner_id,
                owner_name=context.runtime.process_owner_name,
                session_path=context.session_path,
            )
            with context.runtime._process_lock:
                context.runtime._processes[long_running_command.process_id] = long_running_command
            context.runtime._publish_process_state(
                long_running_command,
                context.session_path,
                context.callbacks,
            )
            context.runtime._start_process_monitor(
                long_running_command,
                context.session_path,
                context.callbacks,
            )
            if context.callbacks.status is not None:
                context.callbacks.status("Waiting:60.0")
            return f"Command {process_id} is still running."

        if context.runtime.cancel_event.is_set():
            return context.json_result(
                {
                    "approved": False,
                    "output": "Command was interrupted.",
                }
            )

        local_sandbox_session = context.runtime.local_sandbox_session
        if local_sandbox_session is not None and local_sandbox_session.is_running:
            authorization = context.runtime.tool_manager._authorize_command(
                command,
                statement,
                context.callbacks.approval,
            )
            if isinstance(authorization, CommandResult):
                result = authorization
            else:
                policy = authorization
                sandbox_output = local_sandbox_session.exec_command(policy.canonical_command)
                context.runtime.tool_manager.current_dir = local_sandbox_session.current_dir
                sandbox_blocked = sandbox_output.startswith("Local sandbox blocked")
                result = CommandResult(
                    sandbox_output,
                    approved=not sandbox_blocked,
                    safety=CommandSafety.FORBIDDEN if sandbox_blocked else policy.safety,
                    command=policy.canonical_command,
                    reason=sandbox_output if sandbox_blocked else policy.reason,
                )
        else:
            sandbox_session = context.runtime.sandbox_session
            if sandbox_session is not None and sandbox_session.is_running:
                authorization = context.runtime.tool_manager._authorize_command(
                    command,
                    statement,
                    context.callbacks.approval,
                )
                if isinstance(authorization, CommandResult):
                    result = authorization
                else:
                    policy = authorization
                    sandbox_output = sandbox_session.exec_command(command)
                    result = CommandResult(
                        sandbox_output,
                        approved=True,
                        safety=policy.safety,
                        command=policy.canonical_command,
                        reason=policy.reason,
                    )
            else:
                result = context.runtime.tool_manager.run_command(
                    command,
                    statement or "Operator command",
                    context.callbacks.approval,
                    long_running_callback=publish_long_running_command,
                )

        tool_payload: dict[str, Any] = {
            "approved": result.approved,
            "output": result.output,
        }
        command_history_output = result.output
        if long_running_command is not None:
            wait_payload = self._wait_for_command_state(context, long_running_command)
            output = str(wait_payload.get("output") or result.output)
            tool_payload.update(wait_payload)
            tool_payload["approved"] = result.approved
            tool_payload["output"] = output
            command_history_output = output or str(wait_payload.get("status", ""))
            if context.callbacks.status is not None:
                context.callbacks.status("Thinking")
        if result.approved:
            statement_text = statement or "Running command"
            if context.callbacks.command is not None:
                context.callbacks.command(statement_text, command, command_history_output)
            elif context.callbacks.tool_message is not None:
                context.callbacks.tool_message(statement_text)
        context.runtime._emit_command_system_message(context.callbacks, result, statement)
        return context.json_result(tool_payload)

    def _readonly_command_denial(
        self,
        context: ToolExecutionContext,
        command: str,
        statement: str,
    ) -> str | None:
        if self.access != "read":
            return None
        policy = context.runtime.tool_manager.classify(
            command,
            include_session_allowances=False,
        )
        if policy.safety == CommandSafety.ALLOW:
            return None
        return context.json_result(
            {
                "approved": False,
                "output": (
                    "This tool is read-only. The command was denied because it is "
                    f"not classified as a read-only exploration command. Reason: {policy.reason}"
                ),
                "safety": CommandSafety.FORBIDDEN.value,
                "command": policy.canonical_command,
                "statement": statement,
            }
        )

    def _wait_for_command_state(
        self,
        context: ToolExecutionContext,
        process_state: AsyncProcessState,
    ) -> dict[str, object]:
        seconds = 60.0
        started_at = time.monotonic()
        if context.callbacks.status is not None:
            context.callbacks.status(f"Waiting:{seconds}")
        deadline = started_at + seconds
        while time.monotonic() < deadline:
            if context.runtime._turn_aborted():
                break
            current = context.runtime._command_state(process_state.process_id)
            if current is None or current.status != "running":
                break
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

        waited_seconds = min(seconds, max(0.0, time.monotonic() - started_at))
        current = context.runtime._command_state(process_state.process_id) or process_state
        payload = context.runtime._command_state_payload(current)
        context.runtime._append_command_event_snapshot(current)
        payload["waited_seconds"] = waited_seconds
        return payload
