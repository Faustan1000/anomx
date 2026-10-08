"""Focus an existing object through the platform presentation adapter."""

from __future__ import annotations

import re
from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class FocusObjectTool(BaseTool):
    """Select the object to display alongside the ongoing conversation."""

    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="focus_object",
            description=(
                "Open an existing Anomx object in a large, prominent panel beside this chat "
                "so the user can see the full object and work with you. "
                "Pass its actual canonical object_reference returned by platform tools. "
                "After a successful focus, the object is already visible: do not repeat it "
                "in produce_output object or objects items unless explicitly requested. "
                "Finish with concise text and any necessary references instead. "
                "The new object replaces the previous focus and remains available when the "
                "chat is reopened. This only changes the presentation: it does not modify, "
                "save or publish the object, and does not finish the turn. Platform only."
            ),
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "object_reference": {
                        "type": "string",
                        "description": "Existing object reference: <model_reference>-<uuid>.",
                    },
                },
                ["statement", "object_reference"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        callback = getattr(context.callbacks, "focus_object", None)
        if not context.runtime.can_output_response() or callback is None:
            return context.json_result(
                {"ok": False, "error": "focus_object is only available in platform runs."}
            )
        reference = arguments.get("object_reference")
        if not isinstance(reference, str) or not re.fullmatch(
            r"[a-zA-Z0-9_]+-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", reference.strip()
        ):
            return context.json_result(
                {
                    "ok": False,
                    "error": "Provide a canonical object_reference returned by a platform tool.",
                }
            )
        try:
            focused = callback(reference.strip())
        except ValueError as error:
            return context.json_result({"ok": False, "error": str(error)})
        context.emit_operator_statement(self.name, arguments)
        if context.session_path is not None:
            context.runtime.home.append_session_event(
                context.session_path, "focused_object", focused
            )
        return context.json_result({"ok": True, "focused_object": focused})
