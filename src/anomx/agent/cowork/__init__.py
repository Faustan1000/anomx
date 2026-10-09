"""Cowork mode: turn editor notes into small, isolated agent tasks.

A note written in VS Code (a comment made with the bundled extension) becomes one task.
Each task gets its own git
worktree and branch, so several agents can work next to the user without
touching each other's files; the user merges a branch back only after approving it.
"""

from __future__ import annotations

from anomx.agent.cowork.editor import (
    EditorContext,
    context_path,
    note_near_cursor,
    notes_in_view,
    read_editor_context,
)
from anomx.agent.cowork.extension import build_vsix, install_command
from anomx.agent.cowork.follow import Activation, CoworkFollower
from anomx.agent.cowork.git import (
    CoworkGit,
    FollowResult,
    GitError,
    MergeResult,
    TaskWorkspace,
    find_repo_root,
)
from anomx.agent.cowork.notes import (
    Note,
    NoteContext,
    NoteMessage,
    find_notes_workspace,
    normalize_note_path,
    note_context,
    read_notes,
)
from anomx.agent.cowork.questions import QuestionService
from anomx.agent.cowork.replies import AgentReplies, Location, find_locations
from anomx.agent.cowork.runner import (
    ComparisonFile,
    CoworkService,
    RunResult,
    build_task_prompt,
)
from anomx.agent.cowork.tasks import CoworkTask, TaskStatus, TaskStore
from anomx.agent.cowork.workspace import CoworkAction, CoworkRow, CoworkWorkspace

__all__ = [
    "CoworkGit",
    "Activation",
    "ComparisonFile",
    "CoworkAction",
    "CoworkFollower",
    "CoworkRow",
    "CoworkService",
    "CoworkTask",
    "CoworkWorkspace",
    "EditorContext",
    "FollowResult",
    "GitError",
    "Note",
    "NoteMessage",
    "AgentReplies",
    "Location",
    "QuestionService",
    "find_locations",
    "MergeResult",
    "NoteContext",
    "RunResult",
    "TaskStatus",
    "TaskStore",
    "TaskWorkspace",
    "build_task_prompt",
    "build_vsix",
    "context_path",
    "find_notes_workspace",
    "find_repo_root",
    "note_context",
    "install_command",
    "normalize_note_path",
    "note_near_cursor",
    "notes_in_view",
    "read_editor_context",
    "read_notes",
]
