"""Read the notes written by the ``anomx-editor-context`` VS Code extension.

The extension owns the comments the user leaves in the editor and keeps them in one plain
JSON file below the workspace folder::

    <workspace folder>/.anomx/notes.json

The format is documented in ``cowork/vscode/README.md``. Two kinds of notes exist:

``note``
    a comment on code lines; the agent changes the code in its own worktree.
``question``
    a general question without a line ("where is the GUI built?"); the agent only reads the
    repository and answers.

The workspace is whatever folder VS Code has open. That is often a parent folder holding
several git repositories, so the file is looked up from the current directory upwards
(``find_notes_workspace``) and every note is resolved to its own repository by the git layer.
This module only *reads* the file and tolerates unknown or missing fields; the answers of
the agent go to a separate file (``replies.py``), so each file has exactly one writer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NOTES_DIR = ".anomx"
NOTES_FILE = Path(NOTES_DIR) / "notes.json"
SUPPORTED_SCHEMA = 1

KIND_NOTE = "note"
KIND_QUESTION = "question"
TITLE_CHARS = 80


def find_notes_workspace(start: Path) -> Path | None:
    """Nearest folder at or above ``start`` that holds a notes file."""

    start = start.resolve()
    for folder in (start, *start.parents):
        if (folder / NOTES_FILE).is_file():
            return folder
    return None


def normalize_note_path(path: str) -> str:
    """Workspace-relative POSIX path without a leading ``./``."""

    cleaned = path.replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


@dataclass(frozen=True)
class NoteMessage:
    """One thing the user wrote in a note's thread."""

    id: str
    text: str
    at: str = ""


@dataclass(frozen=True)
class Note:
    """One note or question (``line`` is 1-based like editors show it)."""

    id: str
    kind: str
    file: str
    line: int
    end_line: int
    messages: tuple[NoteMessage, ...]
    anchor_text: str = ""
    context_file: str = ""
    status: str = "open"
    created_at: str = ""

    @property
    def is_question(self) -> bool:
        return self.kind == KIND_QUESTION

    @property
    def is_open(self) -> bool:
        return self.status != "resolved"

    @property
    def request(self) -> str:
        """What the user first asked."""

        return self.messages[0].text.strip() if self.messages else ""

    @property
    def title(self) -> str:
        """First line of the request, shortened."""

        first = next((line.strip() for line in self.request.splitlines() if line.strip()), "")
        return first if len(first) <= TITLE_CHARS else first[: TITLE_CHARS - 1] + "…"

    @property
    def text(self) -> str:
        """The request without its title line."""

        lines = self.request.splitlines()
        return "\n".join(lines[1:]).strip() if len(lines) > 1 else ""


@dataclass(frozen=True)
class NoteContext:
    """The code around a note, taken from the working tree it points into."""

    file_exists: bool
    first_line: int
    anchor_text: str
    snippet: str


def read_notes(workspace: Path, *, include_resolved: bool = False) -> list[Note]:
    """Return the notes of the workspace, questions last, ordered by file and line.

    A missing or malformed file and broken entries are skipped silently: the extension may be
    rewriting the file at the moment it is read, and a broken note must not block the others.
    """

    try:
        data = json.loads((workspace / NOTES_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = data.get("notes") if isinstance(data, dict) else None
    if not isinstance(entries, list) or data.get("schema") != SUPPORTED_SCHEMA:
        return []
    notes = [note for entry in entries if (note := _parse_note(entry)) is not None]
    if not include_resolved:
        notes = [note for note in notes if note.is_open]
    notes.sort(key=lambda note: (note.is_question, note.file, note.line, note.created_at, note.id))
    return notes


def _parse_note(entry: Any) -> Note | None:
    if not isinstance(entry, dict):
        return None
    note_id = entry.get("id")
    if not isinstance(note_id, str) or not note_id:
        return None
    kind = KIND_QUESTION if entry.get("kind") == KIND_QUESTION else KIND_NOTE
    messages = tuple(
        message for raw in _list(entry.get("messages")) if (message := _parse_message(raw))
    )
    if not messages:
        return None
    if kind == KIND_QUESTION:
        return Note(
            note_id,
            kind,
            "",
            0,
            0,
            messages,
            context_file=_text(entry.get("contextFile")),
            status=_text(entry.get("status")) or "open",
            created_at=_text(entry.get("createdAt")),
        )
    file_path = _text(entry.get("file"))
    line = _line(entry.get("line"))
    if not file_path or line is None:
        return None
    end_line = _line(entry.get("endLine"))
    return Note(
        note_id,
        kind,
        normalize_note_path(file_path),
        line,
        max(end_line or line, line),
        messages,
        anchor_text=_text(entry.get("anchorText")),
        status=_text(entry.get("status")) or "open",
        created_at=_text(entry.get("createdAt")),
    )


def _parse_message(raw: Any) -> NoteMessage | None:
    if not isinstance(raw, dict) or raw.get("role", "user") != "user":
        return None
    message_id, text = raw.get("id"), raw.get("text")
    if not isinstance(message_id, str) or not message_id or not isinstance(text, str):
        return None
    if not text.strip():
        return None
    return NoteMessage(message_id, text, _text(raw.get("at")))


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _line(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else None


def note_context(workspace: Path, note: Note, *, radius: int = 6) -> NoteContext:
    """Read the lines around a note, numbered for use in an agent prompt.

    A note is anchored by line number, so the text of the anchored line is kept as
    ``anchor_text``; callers can compare it later to notice that the code moved.
    """

    target = workspace / note.file
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return NoteContext(False, note.line, "", "")
    index = min(max(note.line - 1, 0), max(len(lines) - 1, 0))
    start = max(0, index - radius)
    end = min(len(lines), max(index, note.end_line - 1) + radius + 1)
    numbered = "\n".join(f"{number + 1:>5}  {lines[number]}" for number in range(start, end))
    anchor = lines[index] if lines else ""
    return NoteContext(True, start + 1, anchor, numbered)
