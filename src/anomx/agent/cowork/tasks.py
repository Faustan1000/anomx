"""Persistent state of cowork tasks (one per note)."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any

from anomx.agent.helpers.utils import utc_now_iso


class TaskStatus(StrEnum):
    """Lifecycle of a task: ``queued`` -> ``running`` -> ``ready`` -> ``merged``."""

    QUEUED = "queued"  # branch and worktree exist, no agent has run yet
    RUNNING = "running"  # an agent is working in the worktree
    READY = "ready"  # the agent finished; the user reviews the branch
    FAILED = "failed"  # the agent stopped without a usable result
    MERGING = "merging"  # staged in the user's tree, waiting for their commit
    MERGED = "merged"  # approved and merged into the base branch
    DISCARDED = "discarded"  # thrown away by the user


@dataclass
class CoworkTask:
    """What anomx remembers about the agent working on one note."""

    note_id: str
    file: str
    line: int
    title: str
    request: str
    branch: str
    worktree: str
    base_branch: str
    base_commit: str
    end_line: int = 0
    anchor_text: str = ""
    anchor_kind: str = "source"
    """``source`` for a committed file, ``generated`` when the note sits on a file git does
    not track (build output, ignored or new files) that the agent cannot see."""
    status: TaskStatus = TaskStatus.QUEUED
    summary: str = ""
    session_path: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @property
    def is_active(self) -> bool:
        """Still owns a branch and worktree."""

        return self.status in {
            TaskStatus.QUEUED,
            TaskStatus.RUNNING,
            TaskStatus.READY,
            TaskStatus.FAILED,
            TaskStatus.MERGING,
        }


class TaskStore:
    """JSON file with all tasks of one workspace, keyed by note id.

    The editor's note files are never modified; everything anomx adds lives here.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    def list(self) -> list[CoworkTask]:
        with self._lock:
            return sorted(self._load().values(), key=lambda task: task.created_at)

    def get(self, note_id: str) -> CoworkTask | None:
        with self._lock:
            return self._load().get(note_id)

    def save(self, task: CoworkTask) -> CoworkTask:
        """Insert or replace a task and stamp ``updated_at``."""

        with self._lock:
            tasks = self._load()
            task.updated_at = utc_now_iso()
            tasks[task.note_id] = task
            self._write(tasks)
            return task

    def update(self, note_id: str, **changes: Any) -> CoworkTask | None:
        with self._lock:
            task = self.get(note_id)
            if task is None:
                return None
            for key, value in changes.items():
                setattr(task, key, TaskStatus(value) if key == "status" else value)
            return self.save(task)

    def remove(self, note_id: str) -> None:
        with self._lock:
            tasks = self._load()
            if tasks.pop(note_id, None) is not None:
                self._write(tasks)

    def _load(self) -> dict[str, CoworkTask]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        raw_tasks = data.get("tasks") if isinstance(data, dict) else None
        if not isinstance(raw_tasks, dict):
            return {}
        known = {item.name for item in fields(CoworkTask)}
        tasks: dict[str, CoworkTask] = {}
        for note_id, raw in raw_tasks.items():
            if not isinstance(raw, dict):
                continue
            try:
                values = {key: value for key, value in raw.items() if key in known}
                values["status"] = TaskStatus(values.get("status", TaskStatus.QUEUED))
                tasks[str(note_id)] = CoworkTask(**values)
            except (TypeError, ValueError):
                continue
        return tasks

    def _write(self, tasks: dict[str, CoworkTask]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "tasks": {note_id: _task_to_json(task) for note_id, task in tasks.items()},
        }
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self.path)


def _task_to_json(task: CoworkTask) -> dict[str, Any]:
    data = asdict(task)
    data["status"] = task.status.value
    return data
