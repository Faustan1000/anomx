"""Web-fetch tool."""

from __future__ import annotations

import html
import re
import urllib.error
import urllib.request
from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property

WEB_FETCH_MAX_CHARS = 2_000_000


class WebFetchTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="web_fetch",
            description="Fetch a web page by URL.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "url": {"type": "string", "description": "HTTP or HTTPS URL."},
                },
                ["statement", "url"],
            ),
            aliases=("webfetch",),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        url = str(arguments.get("url", "")).strip()
        if not url:
            return context.json_result({"error": "web_fetch requires a url."})
        if not url.startswith(("http://", "https://")):
            return context.json_result({"error": "Only http and https URLs are supported."})

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AnomxAgent/0.1"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read(WEB_FETCH_MAX_CHARS + 1)
                content_type = str(response.headers.get("content-type", ""))
                status = int(getattr(response, "status", 200))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as error:
            return context.json_result({"error": str(error), "url": url})

        text = raw[:WEB_FETCH_MAX_CHARS].decode("utf-8", errors="replace")
        return context.json_result(
            {
                "url": url,
                "status": status,
                "content_type": content_type,
                "truncated": len(raw) > WEB_FETCH_MAX_CHARS,
                "content": plain_web_text(text),
            }
        )


def plain_web_text(text: str) -> str:
    """Return plain text from a small HTML fragment."""

    without_scripts = re.sub(
        r"<(script|style)\b.*?</\1>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    without_tags = re.sub(r"<[^>]+>", " ", without_scripts)
    return " ".join(html.unescape(without_tags).split())
