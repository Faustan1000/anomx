"""Start-subagent tool."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from anomx.agent.base.agents import AgentKind
from anomx.agent.base.subagents import SubagentRuntimeState
from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.helpers.utils import session_id_from_path, utc_now_iso

SUBAGENT_MAX_CONCURRENT = 5


class StartSubagentTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="start_subagent",
            description="Start an asynchronous subagent.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "name": {
                        "type": "string",
                        "description": "Short display name for the subagent.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Complete task prompt for the subagent.",
                    },
                },
                ["statement", "name", "prompt"],
            ),
            aliases=("start_agent",),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        if not context.runtime.agent_spec.can_spawn_subagents:
            return context.json_result({"error": "Only the main agent can start subagents."})
        if context.session_path is None:
            return context.json_result({"error": "start_subagent requires a session."})

        kind = AgentKind.SUB

        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return context.json_result({"error": "start_subagent requires a prompt."})

        with context.runtime._subagent_lock:
            visible_agents = [
                agent for agent in context.runtime._subagents.values() if agent.status != "removed"
            ]
            if len(visible_agents) >= SUBAGENT_MAX_CONCURRENT:
                return context.json_result(
                    {
                        "error": "At most five subagents can be active at the same time.",
                        "limit": SUBAGENT_MAX_CONCURRENT,
                    }
                )

        agent_id = uuid4().hex[:8]
        name = context.runtime._subagent_display_name(
            str(arguments.get("name", "")).strip(),
            kind,
            agent_id,
        )
        statement = str(arguments.get("statement", "")).strip() or f"Starting {name}"
        state = SubagentRuntimeState(
            agent_id=agent_id,
            kind=kind,
            name=name,
            prompt=prompt,
            status="running",
            statement=statement,
            started_at=utc_now_iso(),
        )
        local_sandbox_session = context.runtime.local_sandbox_session
        child_runtime = context.runtime.__class__(
            context.runtime.home,
            context.runtime.cwd,
            context.runtime.session_allowed_commands,
            context.runtime.session_rejected_commands,
            context.runtime.tool_manager.mode,
            agent_kind=kind,
            cancel_event=state.cancel_event,
            workspace_root=context.runtime.workspace_root,
            process_owner_id=agent_id,
            process_owner_name=name,
            local_sandbox_enabled=local_sandbox_session is not None,
            local_sandbox_home=(
                local_sandbox_session.home if local_sandbox_session is not None else None
            ),
            local_sandbox_allow_subprocess=(
                local_sandbox_session.config.allow_subprocess
                if local_sandbox_session is not None
                else False
            ),
            additional_instructions=context.runtime.additional_instructions,
        )
        state.runtime = child_runtime
        child_runtime._parent_session_id = session_id_from_path(context.session_path)
        with context.runtime._subagent_lock:
            context.runtime._subagents[agent_id] = state

        payload = context.runtime._publish_subagent_state(
            state,
            context.session_path,
            message=statement,
        )
        callback = getattr(context.callbacks, "subagent", None)
        if payload is not None and callback is not None:
            callback(payload)
        context.runtime._start_subagent_worker(
            state,
            prompt,
            context.session_path,
            context.callbacks,
        )
        return context.json_result(
            {
                "started": True,
                "agent_id": agent_id,
                "agent_kind": kind.value,
                "name": name,
                "status": state.status,
            }
        )
