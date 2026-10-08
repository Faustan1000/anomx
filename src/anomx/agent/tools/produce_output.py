"""Ordered, structured final output for the platform presentation adapter."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlparse

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema

logger = logging.getLogger(__name__)


class ProduceOutputTool(BaseTool):
    """Deliver a final response through a platform-provided callback."""

    def __init__(self) -> None:
        super().__init__(
            name="produce_output",
            description=(
                "Publish the complete final answer in the platform and finish the turn. "
                "Send items as a native JSON array, not a JSON-encoded string. "
                'Example: {"items":[{"kind":"text","content":"Done."},'
                '{"kind":"reference","content":{"url":"https://example.org",'
                '"title":"Source"}}]}. Each item has exactly kind and content.\n'
                "Kinds and their content:\n"
                "- text: a Markdown string.\n"
                "- object: {object_reference: string}, displayed inline.\n"
                "- objects: an array of object-reference strings, displayed as cards.\n"
                "- database: {model_reference: string, query?: object, search?: string, "
                "view?: 'list'|'grid', title?: string}, a live authorized list of objects. "
                "Use documented model references and API filters.\n"
                "- proposition: {prompt: string, label: string, icon: string, description: "
                "string}. An optional follow-up button. prompt is a self-contained hidden "
                "instruction executed only when clicked; label says what the click does; "
                "icon is an Untitled UI PascalCase name; description is a user-facing "
                "tooltip explaining scope and benefit in at most 300 characters. The "
                "platform also saves it as a recommendation. Offer at most one, only for "
                "a concrete next step justified by the user's request and your findings. "
                "Most answers need none. Never offer generic help or work the user declined.\n"
                "- reference: {url: string, title?: string} for a website or "
                "{object_reference: string, title?: string} for a platform source.\n"
                "Include real references for sources you used. Body items keep their order; "
                "the proposition follows the body and references appear last. Do not "
                "duplicate an object already shown by focus_object unless explicitly asked. "
                "Include all final text here; do not repeat it before or after this call. "
                "If validation fails, correct the indicated field and resend the complete "
                "answer including its sources. This tool is unavailable in the CLI."
            ),
            parameters=object_schema(
                {
                    "items": {
                        "type": "array",
                        "description": (
                            "Ordered output items as a native array. Do not stringify this array."
                        ),
                        "minItems": 1,
                        "maxItems": 50,
                        "items": object_schema(
                            {
                                "kind": {
                                    "type": "string",
                                    "enum": [
                                        "text",
                                        "object",
                                        "objects",
                                        "database",
                                        "proposition",
                                        "reference",
                                    ],
                                },
                                "content": {
                                    "description": (
                                        "Content for this kind; see the tool instructions."
                                    ),
                                    "anyOf": [
                                        {"type": "string"},
                                        {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "maxItems": 50,
                                        },
                                        {
                                            "type": "object",
                                            "properties": {
                                                "object_reference": {"type": "string"},
                                                "model_reference": {"type": "string"},
                                                "url": {"type": "string"},
                                                "title": {"type": "string"},
                                                "query": {
                                                    "type": "object",
                                                    "additionalProperties": True,
                                                },
                                                "search": {"type": "string"},
                                                "view": {
                                                    "type": "string",
                                                    "enum": ["list", "grid"],
                                                },
                                                "prompt": {"type": "string"},
                                                "label": {"type": "string"},
                                                "icon": {"type": "string"},
                                                "description": {
                                                    "type": "string",
                                                    "maxLength": 300,
                                                    "description": (
                                                        "Short user-facing explanation of the "
                                                        "follow-up action for its hover tooltip."
                                                    ),
                                                },
                                            },
                                            "additionalProperties": False,
                                        },
                                    ],
                                },
                            },
                            ["kind", "content"],
                        ),
                    },
                },
                ["items"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        callback = getattr(context.callbacks, "output_response", None)
        if not context.runtime.can_output_response() or callback is None:
            return context.json_result(
                {"ok": False, "error": "produce_output is only available in platform runs."}
            )
        items = arguments.get("items")
        # Some providers stringify structured parameters. Decode only valid JSON;
        # never guess missing quotes or drop parts of the user's final answer.
        if isinstance(items, str):
            try:
                items = json.loads(items)
            except json.JSONDecodeError as error:
                return self.invalid_arguments(
                    context,
                    "items is a string containing invalid JSON "
                    f"({error.msg}, line {error.lineno}, column {error.colno}). "
                    "Pass items as a native JSON array instead of a string.",
                )
            except RecursionError:
                return self.invalid_arguments(context, "items contains excessively nested JSON.")
        if not isinstance(items, list):
            received = "missing or null" if items is None else type(items).__name__
            return self.invalid_arguments(
                context, f"items must be a JSON array; received {received}."
            )
        if not 1 <= len(items) <= 50:
            return self.invalid_arguments(
                context, f"items must contain 1 to 50 items; received {len(items)}."
            )
        for index, item in enumerate(items):
            error = self.validate_item(item)
            if error:
                return self.invalid_arguments(context, f"items[{index}]: {error}")
        if sum(item["kind"] == "proposition" for item in items) > 1:
            return self.invalid_arguments(context, "items may contain at most one proposition.")
        # Stable partition: never change the order within the body or references.
        ordered = [item for item in items if item["kind"] not in ("proposition", "reference")]
        ordered.extend(item for item in items if item["kind"] == "proposition")
        ordered.extend(item for item in items if item["kind"] == "reference")
        callback({"items": ordered, "end_turn": True})
        context.runtime.produced_output = "\n\n".join(
            item["content"] for item in ordered if item["kind"] == "text"
        )
        if context.session_path is not None:
            context.runtime.home.append_session_event(
                context.session_path, "produced_output", {"items": ordered}
            )
        return context.json_result({"ok": True, "end_turn": True, "item_count": len(items)})

    @staticmethod
    def invalid_arguments(context: ToolExecutionContext, error: str) -> str:
        """Explain a validation failure without publishing a partial answer."""
        logger.warning(
            "produce_output_validation_failed session=%s error_code=invalid_tool_arguments",
            context.session_path.name if context.session_path else "unsaved",
        )
        return context.json_result({
            "ok": False,
            "error_code": "invalid_tool_arguments",
            "error": error,
            "hint": (
                "Resend the complete answer with items as an array of {kind, content} objects. "
                "Keep all intended text, references and follow-up items; correct only the "
                "reported problem. No output has been published."
            ),
            "example": {"items": [{"kind": "text", "content": "Your complete answer."}]},
        })

    @staticmethod
    def validate_item(item: object) -> str | None:
        """Validate payloads before invoking the persistence adapter."""
        if not isinstance(item, dict):
            return "Each item must be an object with kind and content."
        missing = {"kind", "content"} - set(item)
        extra = set(item) - {"kind", "content"}
        if missing:
            return (
                f"Missing fields: {', '.join(sorted(missing))}. Each item needs kind and content."
            )
        if extra:
            return (
                f"Remove unsupported fields: {', '.join(sorted(extra))}. "
                "Keep only kind and content."
            )
        kind, content = item["kind"], item["content"]
        if not isinstance(kind, str):
            return "kind must be a string."
        if kind == "text":
            return (
                None
                if isinstance(content, str) and content.strip()
                else "text requires nonempty Markdown."
            )
        if kind == "objects":
            return (
                None
                if isinstance(content, list)
                and 1 <= len(content) <= 50
                and all(isinstance(ref, str) and ref.strip() for ref in content)
                else "objects requires 1 to 50 object reference strings."
            )
        if not isinstance(content, dict):
            return f"{kind} requires an object as content."
        allowed_fields = {
            "object": {"object_reference"},
            "database": {"model_reference", "query", "search", "view", "title"},
            "proposition": {"prompt", "label", "icon", "description"},
            "reference": {"object_reference", "url", "title"},
        }
        if kind not in allowed_fields or set(content) - allowed_fields[kind]:
            return "Unsupported kind or content fields."
        if "title" in content and not isinstance(content["title"], str):
            return "title must be a string."
        if kind == "object":
            return (
                None
                if isinstance(content.get("object_reference"), str)
                and content["object_reference"].strip()
                else "object_reference is required."
            )
        if kind == "database":
            if (
                not isinstance(content.get("model_reference"), str)
                or not content["model_reference"].strip()
            ):
                return "database requires model_reference."
            if not isinstance(content.get("query", {}), dict) or content.get(
                "view", "list"
            ) not in ("list", "grid"):
                return "Invalid database query or view."
            if not isinstance(content.get("search", ""), str):
                return "database search must be a string."
            for value in content.get("query", {}).values():
                values = value if isinstance(value, list) else [value]
                if any(
                    entry is not None and not isinstance(entry, (str, int, float, bool))
                    for entry in values
                ):
                    return "database query values must be scalars or lists of scalars."
            return None
        if kind == "proposition":
            if not all(
                isinstance(content.get(field), str) and content[field].strip()
                for field in ("prompt", "label", "icon", "description")
            ):
                return "proposition requires nonempty prompt, label, icon, and description strings."
            if len(content["description"]) > 300:
                return "proposition description must be at most 300 characters."
            return None
        if kind == "reference":
            ref, url = content.get("object_reference"), content.get("url")
            if isinstance(ref, str) and ref.strip() and not url:
                return None
            if isinstance(url, str) and not ref:
                try:
                    parsed = urlparse(url)
                except ValueError:
                    return "Invalid reference URL."
                if (
                    parsed.scheme in ("https", "http")
                    and parsed.hostname
                    and not parsed.username
                    and not parsed.password
                ):
                    return None
            return "reference requires either object_reference or an HTTP(S) URL."
        return "Unsupported kind."
