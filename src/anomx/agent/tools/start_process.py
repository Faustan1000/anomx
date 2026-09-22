"""Start-process tool."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from anomx.agent.base.processes import AsyncProcessState
from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.helpers.utils import utc_now_iso


class StartProcessTool(BaseTool):
    def __init__(self, *, statement_description: str, main_agent: bool = False) -> None:
        super().__init__(
            name="start_process",
            description="Start a long-running async CLI process.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "command": {
                        "type": "string",
                        "description": (
                            "Long-running CLI command, for example 'npm run dev'. "
                            "It continues after the agent turn until ended."
                            if main_agent
                            else "Long-running CLI command."
                        ),
                    },
                },
                ["statement", "command"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        if not context.runtime.agent_spec.can_start_processes:
            return context.json_result({"error": "This agent kind cannot start async processes."})
        if context.session_path is None:
            return context.json_result({"error": "start_process requires a session."})

        if context.runtime.sandbox_session is not None:
            return context.json_result(
                {
                    "approved": False,
                    "started": False,
                    "output": "start_process is not supported in sandbox mode. "
                    "Use run_command instead.",
                }
            )

        command = str(arguments.get("command", "")).strip()
        statement = str(arguments.get("statement", "")).strip() or "Starting process"
        if not command:
            return context.json_result({"error": "start_process requires a command."})

        result = context.runtime.tool_manager.start_process(
            command,
            statement,
            context.callbacks.approval,
        )
        context.runtime._emit_command_system_message(context.callbacks, result, statement)
        if result.process is None:
            return context.json_result(
                {
                    "approved": result.approved,
                    "started": False,
                    "output": result.output,
                }
            )

        process_id = uuid4().hex[:8]
        process_state = AsyncProcessState(
            process_id=process_id,
            command=command,
            statement=statement,
            status="running",
            started_at=utc_now_iso(),
            process=result.process,
            owner_id=context.runtime.process_owner_id,
            owner_name=context.runtime.process_owner_name,
            session_path=context.session_path,
        )
        with context.runtime._process_lock:
            context.runtime._processes[process_id] = process_state

        context.runtime._publish_process_state(
            process_state,
            context.session_path,
            context.callbacks,
        )
        context.runtime._start_process_monitor(
            process_state,
            context.session_path,
            context.callbacks,
        )
        return context.json_result(
            {
                "approved": True,
                "started": True,
                "process_id": process_id,
                "status": process_state.status,
                "command": command,
            }
        )
