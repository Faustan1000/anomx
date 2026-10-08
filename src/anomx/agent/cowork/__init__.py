"""Cowork mode: turn editor notes into small, isolated agent tasks.

A note written in VS Code (via the Margo extension) becomes one task. Each task gets its
own git worktree and branch, so several agents can work next to the user without
touching each other's files; the user merges a branch back only after approving it.
"""

from __future__ import annotations

from anomx.agent.cowork.git import CoworkGit, FollowResult, GitError, MergeResult, TaskWorkspace
from anomx.agent.cowork.margo import (
    MARGO_LINE_BASE,
    MargoNote,
    NoteContext,
    note_context,
    read_margo_notes,
)
from anomx.agent.cowork.tasks import CoworkTask, TaskStatus, TaskStore

__all__ = [
    "MARGO_LINE_BASE",
    "CoworkGit",
    "CoworkTask",
    "FollowResult",
    "GitError",
    "MargoNote",
    "MergeResult",
    "NoteContext",
    "TaskStatus",
    "TaskStore",
    "TaskWorkspace",
    "note_context",
    "read_margo_notes",
]
