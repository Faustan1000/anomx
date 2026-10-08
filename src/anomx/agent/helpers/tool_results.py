"""Lossless result artifacts with bounded, deterministic context previews."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

MAX_INLINE_RESULT_CHARACTERS = 16_000
RESULT_PREVIEW_CHARACTERS = 4_000


def result_preview(value: object, maximum: int = RESULT_PREVIEW_CHARACTERS) -> object:
    """Describe large values without treating an excerpt as a complete result."""

    def visit(item: object, budget: int, depth: int = 0) -> object:
        if len(json.dumps(item, ensure_ascii=False)) <= budget:
            return item
        if isinstance(item, str):
            return {
                "excerpt": item[: max(0, min(512, budget // 2))],
                "characters": len(item),
                "truncated": True,
            }
        if isinstance(item, (dict, list)):
            if depth >= 4 or budget < 256:
                return {"type": type(item).__name__, "count": len(item), "truncated": True}
            if isinstance(item, list):
                count = min(len(item), 5)
                return {
                    "items": [
                        visit(child, (budget - 100) // max(1, count), depth + 1)
                        for child in item[:count]
                    ],
                    "count": len(item),
                    "truncated": True,
                }
            keys = sorted(
                item,
                key=lambda key: (
                    key
                    not in {
                        "ok",
                        "error",
                        "status",
                        "status_code",
                        "approved",
                        "exit_code",
                        "total",
                        "count",
                        "has_more",
                        "next_page",
                        "next_offset",
                        "response_path",
                        "path",
                    },
                    key,
                ),
            )[:20]
            return {
                "fields": {
                    key: visit(item[key], (budget - 200) // max(1, len(keys)), depth + 1)
                    for key in keys
                },
                "field_count": len(item),
                "truncated": True,
            }
        return item

    preview = visit(value, maximum)
    if len(json.dumps(preview, ensure_ascii=False)) > maximum:
        return {"type": type(value).__name__, "truncated": True}
    return preview


def store_tool_result(
    content: str,
    directory: Path,
    *,
    tool: str = "",
    maximum_preview: int = RESULT_PREVIEW_CHARACTERS,
) -> str:
    """Store the complete result before returning a small, explicitly partial view.

    Content-addressed files deduplicate retries without overwriting prior evidence.
    The file retains the exact UTF-8 payload, including numeric precision.
    """

    try:
        value = json.loads(content)
        extension = "json"
    except ValueError:
        value, extension = content, "txt"
    raw = content.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"tool-result-{digest}.{extension}"
    if not path.exists():
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=".tool-result-", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(raw)
            # Readers can only see complete artifacts, including after a failed write.
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    payload = {
        "result_mode": "file",
        "tool": tool,
        "result_path": str(path),
        "result_format": extension,
        "result_bytes": len(raw),
        "result_sha256": digest,
        "preview": result_preview(value, maximum_preview),
        "preview_truncated": True,
        "read_hint": (
            "Read result_path in bounded pages, or query the JSON file with a command. "
            "The preview is incomplete; omitted items are not absent."
        ),
    }
    if isinstance(value, dict):
        for key in ("ok", "status", "status_code", "approved", "exit_code", "error"):
            if key in value and isinstance(value[key], (bool, int, float, str, type(None))):
                payload[key] = value[key][:512] if isinstance(value[key], str) else value[key]
    return json.dumps(payload, ensure_ascii=False)
