"""Focused read tools for a connected Anomx Platform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema
from anomx.agent.helpers.anomx_api import AnomxApiError, call_anomx_api, connection_from_home


def _clamped_limit(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(10, min(100, parsed))


def _read_call_payload(result: dict[str, object]) -> object:
    response_path = str(result.get("response_path") or "").strip()
    if not response_path:
        return {}
    try:
        return json.loads(Path(response_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _request_metadata(result: dict[str, object]) -> dict[str, object]:
    """Exclude the response preview when a focused tool already returns the payload."""

    return {
        key: value for key, value in result.items() if key not in {"response", "response_truncated"}
    }


def _object_search_summary(item: dict[str, Any]) -> dict[str, Any]:
    """Keep exact lookup keys and bounded discovery text, leaving details on disk."""

    summary = {
        key: item[key]
        for key in ("id", "object_reference", "model_reference", "identifier", "external_ref")
        if isinstance(item.get(key), str)
    }
    for key in (
        "name",
        "title",
        "description",
        "kind",
        "status",
        "unit",
        "dtype_hint",
        "shape_kind",
        "value_type",
        "history_status",
        "job_type",
        "created_at",
        "updated_at",
    ):
        value = item.get(key)
        if not isinstance(value, (str, bool, int, float)):
            continue
        maximum = 1_000 if key == "description" else 256
        summary[key] = value[:maximum] if isinstance(value, str) else value
        if isinstance(value, str) and len(value) > maximum:
            summary[f"{key}_truncated"] = True
    metadata = item.get("_anomx")
    if isinstance(metadata, dict):
        summary["_anomx"] = {
            key: metadata[key]
            for key in ("object_reference", "model_reference", "endpoint", "rest_endpoint")
            if isinstance(metadata.get(key), str)
        }
        for key in ("object_reference", "model_reference"):
            if key in summary["_anomx"]:
                summary[key] = summary["_anomx"][key]
    return summary


def _get_payload(
    context: ToolExecutionContext,
    *,
    path: str,
    query: dict[str, object] | None = None,
) -> tuple[dict[str, object], object] | str:
    connection = connection_from_home(context.runtime.home)
    if connection is None:
        return context.json_result(
            {"connected": False, "error": "No Anomx Platform connection is configured."}
        )
    try:
        result = call_anomx_api(connection, method="GET", path=path, query=query)
    except AnomxApiError as error:
        return context.json_result({"ok": False, "connected": True, "error": str(error)})
    if not result.get("ok") or not result.get("parsed_as_json"):
        status_code = result.get("status_code", 0)
        error = str(result.get("error") or f"Unexpected API response (HTTP {status_code}).")
        if status_code == 404:
            error += (
                " Check the endpoint or object reference. Changing search terms does not "
                "repair a missing collection endpoint."
            )
        return context.json_result(
            {
                "ok": False,
                "connected": True,
                "error": error,
                "request": result,
            }
        )
    return _request_metadata(result), _read_call_payload(result)


class GetBackgroundRunsTool(BaseTool):
    """Inspect the authenticated creator's past unattended runs, with pagination."""

    def __init__(self) -> None:
        super().__init__(
            name="get_background_runs",
            description=(
                "Get the creator's past background runs, including results and status. "
                "Read one page once per run; retain that check after context compression. "
                "Use page for older runs only when needed. Results are bounded summaries; "
                "use /agents/chats/<id> for specific details."
            ),
            parameters=object_schema({"page": {"type": "integer", "minimum": 1}}, []),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(
            self.name, arguments, default_statement="Loading recent background runs"
        )
        page = context.positive_int(arguments.get("page"), 1)
        size = 20
        response = _get_payload(
            context,
            path="/agents/planned-prompts/background-runs",
            query={"page": page, "size": size},
        )
        if isinstance(response, str):
            return response
        request, payload = response
        if isinstance(payload, list):
            # Older servers return every chat, including recursive compression metadata.
            total = len(payload)
            items = payload[(page - 1) * size : page * size]
        elif isinstance(payload, dict):
            items = payload.get("items", payload.get("results", []))
            total = payload.get("total", payload.get("count"))
        else:
            items, total = None, None
        if not isinstance(items, list):
            return context.json_result(
                {"ok": False, "error": "Invalid background-runs response.", "request": request}
            )
        runs = []
        for item in items[:size]:
            if not isinstance(item, dict):
                continue
            run = {
                key: str(item[key])[:255]
                for key in ("id", "name", "status", "created_at", "latest_run_status")
                if item.get(key) is not None
            }
            result = str(item.get("last_agent_message") or "")
            run["last_agent_message"] = result[:2_000]
            run["result_truncated"] = bool(item.get("result_truncated")) or len(result) > 2_000
            runs.append(run)
        has_more = page * size < total if isinstance(total, int) else len(items) >= size
        return context.json_result(
            {
                "ok": True,
                "runs": runs,
                "page": page,
                "size": size,
                "total": total,
                "has_more": has_more,
                "next_page": page + 1 if has_more else None,
                "request": request,
            }
        )


class GetAnomxObjectDetailsTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="get_anomx_object_details",
            description=(
                "Get the complete serialized details of an Anomx object by object reference."
            ),
            parameters=object_schema(
                {
                    "object_reference": {
                        "type": "string",
                        "description": "Canonical Anomx object reference.",
                    }
                },
                ["object_reference"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(
            self.name, arguments, default_statement="Loading Anomx object"
        )
        object_reference = str(arguments.get("object_reference") or "").strip()
        response = _get_payload(context, path=f"/objects/{object_reference}")
        if isinstance(response, str):
            return response
        result, payload = response
        return context.json_result(
            {"object_reference": object_reference, "object": payload, "request": result}
        )


class SearchAnomxObjectsTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="search_anomx_objects",
            aliases=("serach_anomx_objects",),
            description=(
                "Search Anomx objects through the unified objects endpoint. Results are compact "
                "summaries with complete canonical object references and identifiers. Use "
                "get_anomx_object_details for full details or read request.response_path for "
                "the saved full response. limit is clamped to 10-100."
            ),
            parameters=object_schema(
                {
                    "query": {"type": "string", "description": "Object name or search terms."},
                    "limit": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 100,
                        "description": "Maximum results (default 10).",
                    },
                },
                ["query"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(
            self.name, arguments, default_statement="Searching Anomx objects"
        )
        query = str(arguments.get("query") or "").strip()
        limit = _clamped_limit(arguments.get("limit"), 10)
        response = _get_payload(context, path="/objects", query={"query": query, "limit": limit})
        if isinstance(response, str):
            return response
        result, payload = response
        items = payload.get("items") if isinstance(payload, dict) else payload
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            return context.json_result(
                {"ok": False, "error": "Invalid object search response.", "request": result}
            )
        summaries = [_object_search_summary(item) for item in items]
        results = {**payload, "items": summaries} if isinstance(payload, dict) else summaries
        return context.json_result(
            {
                "query": query,
                "limit": limit,
                "results": results,
                "request": result,
                "result_mode": "summaries",
            }
        )


class SearchAnomxDataChannelsTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="search_anomx_data_channels",
            description=(
                "Search channels using the platform search bar: empty query lists roots; names "
                "and path prefixes browse directories; * matches wildcard segments; a full address "
                "resolves a channel. Returns compact matches with references, completion hints, "
                "pagination and discovery status. Full responses are saved in "
                "request.response_path. "
                "A typed suggestion is not proof a channel exists. If loading, wait briefly then "
                "repeat with refresh=false. No matches in a partial search does not prove absence."
            ),
            parameters=object_schema(
                {
                    "query": {
                        "type": "string",
                        "description": (
                            "Name, path prefix, wildcard pattern, full address, or empty for roots."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 100,
                        "description": "Maximum concrete channels and hints (default 10).",
                    },
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Live channel result page (default 1).",
                    },
                    "refresh": {
                        "type": "boolean",
                        "description": (
                            "Request live discovery (default true); false polls cached results "
                            "without queueing more work."
                        ),
                    },
                },
                ["query"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(
            self.name, arguments, default_statement="Searching Anomx data channels"
        )
        query = str(arguments.get("query") or "").strip()
        limit = _clamped_limit(arguments.get("limit"), 10)
        page = context.positive_int(arguments.get("page"), 1)
        response = _get_payload(
            context,
            path="/channels/search",
            query={
                "query": query,
                "limit": limit,
                "page": page,
                "refresh": "false" if arguments.get("refresh") is False else "true",
            },
        )
        if not isinstance(response, str):
            request, payload = response
            if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
                return context.json_result(
                    {"ok": False, "error": "Invalid channel-search response.", "request": request}
                )
            hits = []
            for hit in payload["items"][:limit]:
                if not isinstance(hit, dict):
                    continue
                channel = hit.get("channel")
                summary = _object_search_summary(channel) if isinstance(channel, dict) else None
                hits.append(
                    {
                        **{
                            key: hit[key]
                            for key in ("value", "completion", "hit_type", "source")
                            if isinstance(hit.get(key), str)
                        },
                        "channel": summary,
                        "object_reference": summary.get("object_reference") if summary else None,
                        "availability": "catalog"
                        if summary
                        else "unverified"
                        if hit.get("source") == "typed"
                        else "discovered",
                    }
                )
            return context.json_result(
                {
                    "ok": True,
                    "result_mode": "summaries",
                    "query": query,
                    "matches": hits,
                    **{
                        key: payload[key]
                        for key in (
                            "page",
                            "limit",
                            "total",
                            "has_more",
                            "next_page",
                            "loading",
                            "discovery_complete",
                            "catalog_limited",
                            "checked_at",
                        )
                        if key in payload
                    },
                    "request": request,
                }
            )
        if json.loads(response).get("request", {}).get("status_code") != 404:
            return response
        # Older platforms expose the legacy discovery endpoints. Keep bounded
        # summaries while package and platform upgrades roll out independently.
        channel_response = _get_payload(
            context,
            path="/channels/live-search",
            query={"query": query, "limit": limit, "page": page},
        )
        if isinstance(channel_response, str):
            return channel_response
        hints_response = _get_payload(
            context, path="/channels/live-hints", query={"query": query, "limit": limit}
        )
        channel_request, channel_payload = channel_response
        if not isinstance(channel_payload, dict) or not isinstance(
            channel_payload.get("items"), list
        ):
            return context.json_result(
                {
                    "ok": False,
                    "error": "Invalid live-search response: expected items.",
                    "request": channel_request,
                }
            )
        if isinstance(hints_response, str):
            hints_error = json.loads(hints_response)
            hints_request = hints_error.get("request", {})
            hints_payload = []
        else:
            hints_request, hints_payload = hints_response
            hints_error = None
            if not isinstance(hints_payload, list):
                hints_error = {"error": "Invalid live-hints response: expected a list."}
        hints = (
            [item for item in hints_payload if isinstance(item, dict) and not item.get("channel")]
            if isinstance(hints_payload, list)
            else []
        )
        return context.json_result(
            {
                "ok": hints_error is None,
                "partial": hints_error is not None,
                **({"error": hints_error["error"]} if hints_error else {}),
                "result_mode": "summaries",
                "channels": [
                    _object_search_summary(item)
                    for item in channel_payload["items"][:limit]
                    if isinstance(item, dict)
                ],
                "hints": [
                    {
                        key: item[key]
                        for key in ("value", "query", "source")
                        if isinstance(item.get(key), str)
                    }
                    for item in hints[:limit]
                ],
                "discovery_complete": False,
                "limit": limit,
                "page": page,
                "pagination": {
                    key: channel_payload[key]
                    for key in ("has_more", "next_offset", "offset", "total")
                    if key in channel_payload
                },
                "query": query,
                "requests": [channel_request, hints_request],
            }
        )


class GetAnomxDataChannelHistoryTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="get_anomx_data_channel_history",
            description=(
                "Get historical samples for a concrete Anomx data channel object reference."
            ),
            parameters=object_schema(
                {
                    "object_reference": {
                        "type": "string",
                        "description": "Canonical data_channel object reference.",
                    },
                    "range": {
                        "type": "string",
                        "description": "Relative history window such as 15m, 1h, or 7d.",
                    },
                    "max_data_points": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum returned data points (default 100).",
                    },
                },
                ["object_reference", "range"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(
            self.name, arguments, default_statement="Loading Anomx channel history"
        )
        object_reference = str(arguments.get("object_reference") or "").strip()
        if not object_reference.startswith("data_channel-"):
            return context.json_result({"error": "object_reference must identify a data_channel."})
        history_range = str(arguments.get("range") or "1h").strip()
        max_points = max(1, min(100, context.positive_int(arguments.get("max_data_points"), 100)))
        response = _get_payload(
            context,
            path=f"/channels/{object_reference}/history",
            query={"range": history_range, "max_points": max_points},
        )
        if isinstance(response, str):
            return response
        result, payload = response
        return context.json_result(
            {
                "history": payload,
                "max_data_points": max_points,
                "object_reference": object_reference,
                "range": history_range,
                "request": result,
            }
        )
