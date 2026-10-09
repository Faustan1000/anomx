"""Helpers to write the notes file the VS Code extension would write (``.anomx/notes.json``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def note_entry(
    note_id: str,
    file: str | None,
    line: int | None,
    text: str,
    *,
    kind: str = "note",
    status: str = "open",
    end_line: int | None = None,
    context_file: str | None = None,
    extra_messages: tuple[str, ...] = (),
) -> dict[str, Any]:
    """One note as the extension stores it (``line`` is 1-based)."""

    messages = [{"id": f"m-{note_id}", "role": "user", "text": text, "at": "2026-10-09T08:00:00Z"}]
    for index, extra in enumerate(extra_messages, start=2):
        messages.append(
            {"id": f"m-{note_id}-{index}", "role": "user", "text": extra, "at": "2026-10-09T09:00Z"}
        )
    return {
        "id": note_id,
        "kind": kind,
        "file": file,
        "line": line,
        "endLine": end_line if end_line is not None else line,
        "anchorText": "",
        "contextFile": context_file,
        "status": status,
        "createdAt": "2026-10-09T08:00:00Z",
        "messages": messages,
    }


def write_notes(workspace: Path, *notes: dict[str, Any]) -> Path:
    path = workspace / ".anomx" / "notes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "notes": list(notes)}), encoding="utf-8")
    return path


def read_agent_file(workspace: Path) -> dict[str, Any]:
    path = workspace / ".anomx" / "agent.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return data.get("notes", {})
