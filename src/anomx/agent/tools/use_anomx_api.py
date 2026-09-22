"""Anomx Platform API tool."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.helpers.anomx_api import AnomxApiError, call_anomx_api, connection_from_home


class UseAnomxApiTool(BaseTool):
    """Call the connected Anomx Platform API and store the raw response."""

    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="use_anomx_api",
            description=(
                "Call the connected Anomx Platform REST API. Reads return a bounded parsed "
                "response directly. The full response is written to ~/.anomx/responses as JSON. "
                "Persistent writes are enforced by the active agent mode and approval policy."
            ),
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                        "description": "HTTP method.",
                    },
                    "path": {
                        "type": "string",
                        "description": "API path relative to the connection, e.g. /channels.",
                    },
                    "query": {
                        "type": "object",
                        "description": "Optional query parameters.",
                        "additionalProperties": True,
                    },
                    "body": {
                        "type": "object",
                        "description": "Optional JSON request body for non-GET requests.",
                        "additionalProperties": True,
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional extra HTTP headers.",
                        "additionalProperties": {"type": "string"},
                    },
                    "output_name": {
                        "type": "string",
                        "description": "Optional response filename stem.",
                    },
                },
                ["statement", "method", "path"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        connection = connection_from_home(context.runtime.home)
        if connection is None:
            return context.json_result(
                {
                    "error": "No Anomx Platform connection is configured.",
                    "connected": False,
                }
            )
        method = str(arguments.get("method") or "GET").strip().upper()
        path = str(arguments.get("path") or "").strip()
        for field in ("query", "body", "headers"):
            if arguments.get(field) is not None and not isinstance(arguments[field], dict):
                return context.json_result(
                    {
                        "ok": False,
                        "error": f"{field} must be a JSON object, not encoded JSON or text.",
                    }
                )
        if method in {"PATCH", "PUT"} and not arguments.get("body"):
            return context.json_result(
                {"ok": False, "error": f"{method} requires a non-empty JSON body."}
            )
        authorization_path = path
        if path.startswith(("http://", "https://")):
            authorization_path = urlparse(path).path
            api_base_path = urlparse(connection.base_url).path.rstrip("/")
            if api_base_path:
                if authorization_path == api_base_path:
                    authorization_path = "/"
                elif authorization_path.startswith(f"{api_base_path}/"):
                    authorization_path = authorization_path.removeprefix(api_base_path)
                else:
                    authorization_path = f"/__outside_api__{authorization_path}"
        authorization = context.runtime.tool_manager.authorize_platform_api_request(
            method,
            authorization_path,
            str(arguments.get("statement") or "Calling Anomx API"),
            context.callbacks.approval,
        )
        if authorization is not None:
            context.runtime._emit_command_system_message(
                context.callbacks,
                authorization,
                str(arguments.get("statement") or "Calling Anomx API"),
            )
            return context.json_result(
                {
                    "approved": False,
                    "blocked_by_mode": authorization.blocked_by_mode,
                    "error": authorization.output,
                    "method": method,
                    "path": path,
                }
            )
        try:
            result = call_anomx_api(
                connection,
                method=method,
                path=path,
                query=_object_or_none(arguments.get("query")),
                body=arguments.get("body") if isinstance(arguments.get("body"), dict) else None,
                headers=_string_object_or_none(arguments.get("headers")),
                output_name=str(arguments.get("output_name") or ""),
            )
        except AnomxApiError as error:
            return context.json_result({"error": str(error), "connected": True})
        return context.json_result(result)


def _object_or_none(value: object) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _string_object_or_none(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): str(item) for key, item in value.items()}
