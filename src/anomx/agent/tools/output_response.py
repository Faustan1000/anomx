"""Platform rich-output response tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema


class OutputResponseTool(BaseTool):
    """Emit platform-renderable outputs from an in-platform agent run."""

    def __init__(self) -> None:
        super().__init__(
            name="output_response",
            description=(
                "Output things beyond plain text to the user in the Anomx Platform UI. "
                "Use this to show concrete objects, object displays, object databases, "
                "object lists, or object forms. Set end_turn to true when the agent should "
                "finish after these outputs. If the user asks for a specific platform object, "
                "list, form, or database-style result, always use this tool at the very end."
            ),
            parameters=object_schema(
                {
                    "outputs": {
                        "type": "array",
                        "description": (
                            "Outputs to render. Supported type values are text, object_card, "
                            "full_object, objects_grid, objects_list, and object_form."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "text",
                                        "object_card",
                                        "full_object",
                                        "objects_grid",
                                        "objects_list",
                                        "object_form",
                                    ],
                                },
                                "text": {"type": "string"},
                                "object_reference": {"type": "string"},
                                "model_reference": {"type": "string"},
                                "endpoint": {"type": "string"},
                                "query": {
                                    "type": "object",
                                    "additionalProperties": True,
                                },
                                "data": {
                                    "type": "object",
                                    "additionalProperties": True,
                                },
                                "title": {"type": "string"},
                            },
                            "required": ["type"],
                            "additionalProperties": True,
                        },
                    },
                    "end_turn": {
                        "type": "boolean",
                        "description": (
                            "Whether the agent should finish after rendering these outputs."
                        ),
                    },
                },
                ["outputs", "end_turn"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        if not context.runtime.can_output_response():
            return context.json_result(
                {
                    "error": (
                        "output_response is only available inside connected Anomx Platform "
                        "agent runs."
                    ),
                    "ok": False,
                }
            )

        outputs = arguments.get("outputs")
        if not isinstance(outputs, list):
            return context.json_result({"error": "outputs must be a list.", "ok": False})

        normalized_outputs = [
            output
            for output in outputs
            if isinstance(output, dict) and isinstance(output.get("type"), str)
        ]
        end_turn = arguments.get("end_turn") is True
        payload = {
            "end_turn": end_turn,
            "outputs": normalized_outputs,
        }
        callback = getattr(context.callbacks, "output_response", None)
        if callback is not None:
            callback(payload)
        return context.json_result(
            {
                "end_turn": end_turn,
                "ok": True,
                "output_count": len(normalized_outputs),
            }
        )
