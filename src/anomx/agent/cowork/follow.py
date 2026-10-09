"""Follow the note the user is working on in the editor.

The editor-context extension writes the cursor position; a cursor inside a note's lines
(for example after clicking the note in the editor's sidebar) is what
:class:`CoworkFollower` turns into "the user moved to note X" events: debounced (so arrowing
past a note does not switch chats) and sticky (the active note stays until the user selects
another one).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anomx.agent.cowork.editor import note_near_cursor, read_editor_context
from anomx.agent.cowork.notes import Note, read_notes
from anomx.agent.cowork.workspace import CoworkWorkspace

DEFAULT_STABLE_POLLS = 2


@dataclass(frozen=True)
class Activation:
    """The user has moved to a (new) note."""

    note: Note
    editor_dirty: bool


class CoworkFollower:
    """Polls the editor-context file and reports when the active note changes."""

    def __init__(
        self,
        panel: CoworkWorkspace,
        context_file: Path,
        *,
        stable_polls: int = DEFAULT_STABLE_POLLS,
    ) -> None:
        self.panel = panel
        self.context_file = context_file
        self.stable_polls = max(1, stable_polls)
        self.active_note_id: str | None = None
        self._pending: tuple[str, int] | None = None

    def reset(self) -> None:
        """Forget the active note so the next click on it triggers again."""

        self.active_note_id = None
        self._pending = None

    def poll(self) -> Activation | None:
        context = read_editor_context(self.context_file)
        if context is None or context.file is None:
            self._pending = None
            return None
        workspace = self.panel.notes_workspace
        note = note_near_cursor(context, read_notes(workspace), workspace)
        if note is None or note.id == self.active_note_id:
            self._pending = None
            return None
        seen = self._pending[1] + 1 if self._pending and self._pending[0] == note.id else 1
        if seen < self.stable_polls:
            self._pending = (note.id, seen)
            return None
        self._pending = None
        self.active_note_id = note.id
        return Activation(note, context.is_dirty)
