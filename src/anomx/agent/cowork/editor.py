"""Read what the user is looking at in VS Code.

The ``anomx-editor-context`` extension (``cowork/vscode``) writes the active file, cursor,
selection and visible lines to ``<anomx home>/editor-context.json``. This module reads that
file and relates it to Margo notes, so the agent can tell which note the user is working on.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anomx.agent.cowork.margo import MargoNote

CONTEXT_FILE_NAME = "editor-context.json"
SUPPORTED_SCHEMA = 1


@dataclass(frozen=True)
class EditorContext:
    """Editor state as written by the extension (lines and characters are 1-based)."""

    file: Path | None
    workspace_folders: tuple[Path, ...]
    line: int
    character: int
    selection_start_line: int
    selection_end_line: int
    visible_start_line: int
    visible_end_line: int
    is_dirty: bool
    language_id: str
    active_line_text: str
    updated_at: datetime

    def age_seconds(self, now: datetime | None = None) -> float:
        return ((now or datetime.now(UTC)) - self.updated_at).total_seconds()


def context_path(anomx_home: Path) -> Path:
    return anomx_home / CONTEXT_FILE_NAME


def read_editor_context(
    path: Path, *, max_age_seconds: float | None = None, now: datetime | None = None
) -> EditorContext | None:
    """Parse the context file; ``None`` if missing, unreadable, foreign schema or too old.

    The file is only rewritten when something changes, so an old timestamp can still be a
    correct (idle) state. ``max_age_seconds`` is therefore opt-in.
    """

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SUPPORTED_SCHEMA:
        return None
    context = _parse(data)
    if context is None:
        return None
    if max_age_seconds is not None and context.age_seconds(now) > max_age_seconds:
        return None
    return context


def _parse(data: dict[str, Any]) -> EditorContext | None:
    updated = _parse_time(data.get("updatedAt"))
    if updated is None:
        return None
    active_file = data.get("activeFile")
    cursor = data.get("cursor") if isinstance(data.get("cursor"), dict) else {}
    selection = data.get("selection") if isinstance(data.get("selection"), dict) else {}
    visible = data.get("visibleRange") if isinstance(data.get("visibleRange"), dict) else {}
    folders = data.get("workspaceFolders")
    line = _int(cursor.get("line"))
    return EditorContext(
        file=Path(active_file) if isinstance(active_file, str) and active_file else None,
        workspace_folders=tuple(Path(f) for f in folders if isinstance(f, str))
        if isinstance(folders, list)
        else (),
        line=line,
        character=_int(cursor.get("character")),
        selection_start_line=_int(selection.get("startLine"), line),
        selection_end_line=_int(selection.get("endLine"), line),
        visible_start_line=_int(visible.get("startLine"), line),
        visible_end_line=_int(visible.get("endLine"), line),
        is_dirty=bool(data.get("isDirty")),
        language_id=str(data.get("languageId") or ""),
        active_line_text=str(data.get("activeLineText") or ""),
        updated_at=updated,
    )


def _int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _same_file(note_file: Path, context_file: Path | None) -> bool:
    if context_file is None:
        return False
    try:
        return note_file.resolve() == context_file.resolve()
    except OSError:
        return False


def note_near_cursor(
    context: EditorContext,
    notes: Iterable[MargoNote],
    workspace: Path,
    *,
    radius: int = 0,
) -> MargoNote | None:
    """The note on (or within ``radius`` lines of) the cursor line, nearest first."""

    best: tuple[int, MargoNote] | None = None
    for note in notes:
        if not _same_file(workspace / note.file, context.file):
            continue
        distance = abs(note.line_number - context.line)
        if distance <= radius and (best is None or distance < best[0]):
            best = (distance, note)
    return best[1] if best else None


def notes_in_view(
    context: EditorContext, notes: Iterable[MargoNote], workspace: Path
) -> list[MargoNote]:
    """Notes of the active file that lie inside the visible lines, top to bottom."""

    visible = [
        note
        for note in notes
        if _same_file(workspace / note.file, context.file)
        and context.visible_start_line <= note.line_number <= context.visible_end_line
    ]
    return sorted(visible, key=lambda note: note.line_number)
