"""Model-callable feedback delivery for connected Anomx Platform sessions."""

from __future__ import annotations

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.helpers.anomx_api import call_anomx_api, connection_from_home


class SendFeedbackTool(BaseTool):
    """Send concrete agent-experience feedback to the connected platform."""

    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="send_feedback",
            description=(
                "Send anonymous feedback about the Anomx Platform when platform behavior, "
                "tools, permissions, or missing context made the user's task harder. Use this "
                "for concrete improvement signals such as an unexpected tool result, information "
                "that should have been available earlier, or a recurring permission failure. "
                "Do not use it as a substitute for completing or explaining the user's task."
            ),
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "sentiment": {
                        "type": "string",
                        "enum": ["good", "bad"],
                        "description": (
                            "Whether the platform experience helped or hindered the task."
                        ),
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short reusable category labels, or an empty list.",
                    },
                    "comment": {
                        "type": "string",
                        "description": "A concise, actionable explanation of the experience.",
                    },
                    "context": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": (
                            "Non-secret structured context such as a tool or command name."
                        ),
                    },
                },
                ["statement", "sentiment", "categories", "comment", "context"],
            ),
        )

    def execute(self, arguments: dict[str, object], context: ToolExecutionContext) -> str:
        connection = connection_from_home(context.runtime.home)
        if connection is None:
            return context.json_result({"error": "Feedback requires a connected Anomx Platform."})

        sentiment = str(arguments.get("sentiment") or "").strip().lower()
        if sentiment not in {"good", "bad"}:
            return context.json_result({"error": "sentiment must be good or bad."})
        raw_categories = arguments.get("categories")
        categories = [
            str(category).strip()
            for category in raw_categories
            if str(category).strip()
        ][:20] if isinstance(raw_categories, list) else []
        raw_context = arguments.get("context")
        feedback_context = raw_context if isinstance(raw_context, dict) else {}
        chat_id = str(getattr(context.runtime, "platform_chat_id", "") or "").strip()
        context.emit_operator_statement(
            "send_feedback",
            arguments,
            default_statement="Sending platform feedback",
        )
        result = call_anomx_api(
            connection,
            method="POST",
            path="/agents/feedback",
            body={
                "chat": chat_id,
                "sentiment": sentiment,
                "source": "agent",
                "categories": categories,
                "comment": str(arguments.get("comment") or "").strip(),
                "context": feedback_context,
            },
            output_name="agent-feedback",
        )
        if not result.get("ok"):
            return context.json_result({
                "error": "The platform did not accept the feedback.",
                "status_code": result.get("status_code", 0),
            })
        return context.json_result({"sent": True})
