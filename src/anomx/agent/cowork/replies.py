"""What anomx tells the editor about each note: status, progress and the agent's answers.

The counterpart of ``notes.py``: the VS Code extension writes ``.anomx/notes.json``, anomx
writes ``.anomx/agent.json`` next to it (format in ``cowork/vscode/README.md``). Because each
file has one writer there are no write races; the extension watches this file and shows
the answers in the comment thread.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from anomx.agent.cowork.notes import NOTES_DIR
from anomx.agent.helpers.utils import utc_now_iso

REPLIES_FILE = Path(NOTES_DIR) / "agent.json"
SCHEMA = 1

IDLE = "idle"
WORKING = "working"
READY = "ready"
FAILED = "failed"

MAX_LOCATIONS = 12
_LOCATION = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.\w{1,8}):(\d+)(?:[-–](\d+))?")


@dataclass(frozen=True)
class Location:
    """A place in the code an answer refers to (path relative to the workspace folder)."""

    file: str
    line: int
    end_line: int | None = None

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"file": self.file, "line": self.line}
        if self.end_line is not None:
            data["endLine"] = self.end_line
        return data


def find_locations(text: str, repo_root: Path, workspace: Path) -> list[Location]:
    """``path:line`` references in an answer that exist in the repository.

    Paths are accepted relative to the repository or to the workspace folder; the result is
    relative to the workspace folder so the editor can open it. Invented paths and line
    numbers beyond the end of a file are dropped, so every link the user sees works.
    """

    workspace = workspace.resolve()
    found: list[Location] = []
    seen: set[tuple[str, int]] = set()
    for match in _LOCATION.finditer(text):
        raw, first, last = match.group(1), int(match.group(2)), match.group(3)
        target = _resolve(raw, repo_root, workspace)
        if target is None or first < 1:
            continue
        try:
            relative = target.relative_to(workspace).as_posix()
            length = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
        except (OSError, ValueError):
            continue
        if first > max(length, 1) or (relative, first) in seen:
            continue
        seen.add((relative, first))
        end = int(last) if last is not None and int(last) > first else None
        found.append(Location(relative, first, min(end, length) if end is not None else None))
        if len(found) >= MAX_LOCATIONS:
            break
    return found


def _resolve(raw: str, repo_root: Path, workspace: Path) -> Path | None:
    candidate = Path(raw)
    options = [candidate] if candidate.is_absolute() else [repo_root / raw, workspace / raw]
    for option in options:
        try:
            resolved = option.resolve()
            resolved.relative_to(workspace)
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            return resolved
    return None


class AgentReplies:
    """Read-modify-write access to ``agent.json`` (the only writer is this process)."""

    def __init__(self, workspace: Path) -> None:
        self.path = workspace / REPLIES_FILE
        self._lock = threading.RLock()

    def get(self, note_id: str) -> dict[str, Any]:
        with self._lock:
            entry = self._load().get(note_id)
            return entry if isinstance(entry, dict) else {}

    def status(self, note_id: str) -> str:
        return str(self.get(note_id).get("status") or IDLE)

    def picked_up(self, note_id: str) -> set[str]:
        raw = self.get(note_id).get("pickedUp")
        return {item for item in raw if isinstance(item, str)} if isinstance(raw, list) else set()

    def session(self, note_id: str) -> str:
        return str(self.get(note_id).get("session") or "")

    def update(self, note_id: str, **fields: Any) -> None:
        """Set top-level fields of a note's entry (``status``, ``detail``, ``session`` ...)."""

        with self._lock:
            notes = self._load()
            entry = notes.setdefault(note_id, {})
            if all(entry.get(key) == value for key, value in fields.items()) and entry:
                return
            entry.update(fields)
            entry["updatedAt"] = utc_now_iso()
            self._write(notes)

    def pick_up(self, note_id: str, message_id: str) -> None:
        """Record that a user message is being handled and the agent is now working."""

        with self._lock:
            notes = self._load()
            entry = notes.setdefault(note_id, {})
            picked = [item for item in entry.get("pickedUp", []) if isinstance(item, str)]
            if message_id not in picked:
                picked.append(message_id)
            entry.update(pickedUp=picked, status=WORKING, detail="Starting…")
            entry["updatedAt"] = utc_now_iso()
            self._write(notes)

    def add_answer(
        self,
        note_id: str,
        text: str,
        *,
        after_message: str = "",
        locations: list[Location] | None = None,
        failed: bool = False,
    ) -> None:
        """Append the agent's answer and mark the note ready (or failed)."""

        with self._lock:
            notes = self._load()
            entry = notes.setdefault(note_id, {})
            messages = [item for item in entry.get("messages", []) if isinstance(item, dict)]
            if text.strip():
                messages.append(
                    {
                        "id": f"a-{uuid4().hex[:12]}",
                        "role": "agent",
                        "text": text.strip(),
                        "at": utc_now_iso(),
                        "afterMessage": after_message or None,
                        "locations": [item.to_json() for item in locations or []],
                    }
                )
            entry.update(
                messages=messages,
                status=FAILED if failed else READY,
                detail=text.strip()[:300] if failed else "",
            )
            entry["updatedAt"] = utc_now_iso()
            self._write(notes)

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        notes = data.get("notes") if isinstance(data, dict) else None
        return notes if isinstance(notes, dict) else {}

    def _write(self, notes: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump({"schema": SCHEMA, "notes": notes}, handle, indent=2)
            handle.write("\n")
        tmp.chmod(0o600)
        tmp.replace(self.path)
