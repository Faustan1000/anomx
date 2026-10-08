"""Read the note vaults of the Margo VS Code extension (``aikaisi.margo``).

Margo keeps notes as plain JSON in the workspace::

    .margo/local.notes.json   private vault (Margo gitignores it)
    .notes.json               team vault (shared in the repository)

Both files map a workspace-relative file path to a list of notes. The keys carry a
leading ``./`` (checked against a real vault)::

    {"./anomx/src/auth.ts": [{"id": "1791449926580-x93ibwz6i", "line": 2,
        "title": "...", "text": "", "timestamp": 1791449926580, "author": "...",
        "priority": "medium", "category": "note", "status": "open", "vault": "local"}]}

The workspace is whatever folder VS Code has open. That is often a parent folder holding
several git repositories, so the vault is looked up from the current directory upwards
(``find_margo_workspace``) and every note is resolved to its own repository by the git
layer. The format is Margo's, not ours: this module only *reads* it, tolerates unknown or
missing fields and never writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCAL_VAULT = Path(".margo") / "local.notes.json"
TEAM_VAULT = Path(".notes.json")

# Margo builds its comment ranges straight from ``note.line``, i.e. the stored value is
# 0-based (the first line of a file is 0). Display and prompts use 1-based numbers.
MARGO_LINE_BASE = 0


def find_margo_workspace(start: Path) -> Path | None:
    """Nearest folder at or above ``start`` that holds a Margo vault file."""

    start = start.resolve()
    for folder in (start, *start.parents):
        if (folder / LOCAL_VAULT).is_file() or (folder / TEAM_VAULT).is_file():
            return folder
    return None


def normalize_note_path(path: str) -> str:
    """Workspace-relative POSIX path without Margo's leading ``./``."""

    cleaned = path.replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


@dataclass(frozen=True)
class MargoNote:
    """One Margo note, as stored (``line`` keeps Margo's own numbering)."""

    id: str
    file: str
    line: int
    title: str
    text: str
    status: str = "open"
    category: str = "note"
    priority: str = "medium"
    author: str = ""
    vault: str = "local"
    timestamp: int = 0

    @property
    def line_number(self) -> int:
        """1-based line number as shown in editors."""

        return self.line + 1 - MARGO_LINE_BASE

    @property
    def is_open(self) -> bool:
        return self.status != "resolved"

    @property
    def request(self) -> str:
        """The instruction the user wrote: title and body joined."""

        return f"{self.title}\n\n{self.text}".strip() if self.text.strip() else self.title.strip()


@dataclass(frozen=True)
class NoteContext:
    """The code around a note, taken from the working tree it points into."""

    file_exists: bool
    first_line: int
    anchor_text: str
    snippet: str


def read_margo_notes(workspace: Path, *, include_resolved: bool = False) -> list[MargoNote]:
    """Return the notes of both vaults, ordered by file and line.

    Missing or malformed files and entries are skipped silently: Margo may be rewriting a
    file at the moment it is read, and a broken note must not block the others.
    """

    notes: list[MargoNote] = []
    for relative, vault in ((LOCAL_VAULT, "local"), (TEAM_VAULT, "team")):
        notes.extend(_read_vault(workspace / relative, vault))
    if not include_resolved:
        notes = [note for note in notes if note.is_open]
    notes.sort(key=lambda note: (note.file, note.line, note.id))
    return notes


def _read_vault(path: Path, default_vault: str) -> list[MargoNote]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    notes: list[MargoNote] = []
    for file_path, entries in data.items():
        if not isinstance(file_path, str) or not isinstance(entries, list):
            continue
        for entry in entries:
            note = _parse_note(file_path, entry, default_vault)
            if note is not None:
                notes.append(note)
    return notes


def _parse_note(file_path: str, entry: Any, default_vault: str) -> MargoNote | None:
    if not isinstance(entry, dict):
        return None
    note_id = entry.get("id")
    line = entry.get("line")
    if not isinstance(note_id, str) or not note_id or isinstance(line, bool):
        return None
    if not isinstance(line, int) or line < 0:
        return None
    timestamp = entry.get("timestamp")
    if not isinstance(timestamp, int) or isinstance(timestamp, bool):
        timestamp = 0
    return MargoNote(
        id=note_id,
        file=normalize_note_path(file_path),
        line=line,
        title=str(entry.get("title") or ""),
        text=str(entry.get("text") or ""),
        status=str(entry.get("status") or "open"),
        category=str(entry.get("category") or "note"),
        priority=str(entry.get("priority") or "medium"),
        author=str(entry.get("author") or ""),
        vault=str(entry.get("vault") or default_vault),
        timestamp=timestamp,
    )


def note_context(workspace: Path, note: MargoNote, *, radius: int = 6) -> NoteContext:
    """Read the lines around a note, numbered for use in an agent prompt.

    Margo anchors notes by line number only, so the text of the anchored line is kept as
    ``anchor_text``; callers can compare it later to notice that the code moved.
    """

    target = workspace / note.file
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return NoteContext(False, note.line_number, "", "")
    index = min(max(note.line_number - 1, 0), max(len(lines) - 1, 0))
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    numbered = "\n".join(f"{number + 1:>5}  {lines[number]}" for number in range(start, end))
    anchor = lines[index] if lines else ""
    return NoteContext(True, start + 1, anchor, numbered)
