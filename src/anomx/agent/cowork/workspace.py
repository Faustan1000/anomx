"""View model of the cowork panel: notes, their tasks and the actions that apply.

All decisions the terminal panel needs live here (and are tested without a terminal): which
notes exist, what state each task is in, which actions are offered, and how agent runs are
started in the background. The curses layer only draws rows and forwards the user's choice.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from anomx.agent.cowork.agent import CoworkRuntime
from anomx.agent.cowork.git import FollowResult, GitError, MergeResult, find_repo_root
from anomx.agent.cowork.margo import MargoNote, find_margo_workspace, read_margo_notes
from anomx.agent.cowork.runner import ComparisonFile, CoworkService
from anomx.agent.cowork.tasks import CoworkTask, TaskStatus
from anomx.agent.runtime import RuntimeCallbacks
from anomx.agent.store import AnomxHome, SessionRecord

NEW = "new"
ORPHAN = "orphan"
NO_REPO = "no-repo"
MAX_PROGRESS_CHARS = 90


@dataclass(frozen=True)
class CoworkRow:
    """One line of the panel: a note and, when it has one, its task."""

    note_id: str
    title: str
    file: str
    line: int
    status: str
    detail: str
    note: MargoNote | None = None
    task: CoworkTask | None = None
    repo_root: Path | None = None
    running: bool = False

    @property
    def label(self) -> str:
        return f"{self.file}:{self.line}  {self.title}"


@dataclass(frozen=True)
class CoworkAction:
    value: str
    label: str
    detail: str = ""


class CoworkWorkspace:
    """Notes of one VS Code workspace together with their cowork tasks."""

    def __init__(self, home: AnomxHome, margo_workspace: Path) -> None:
        self.home = home
        self.margo_workspace = margo_workspace
        self._services: dict[Path, CoworkService] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._progress: dict[str, str] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()

    @classmethod
    def locate(cls, home: AnomxHome, start: Path) -> CoworkWorkspace | None:
        """Workspace for the Margo vault found at or above ``start`` (``None`` if none)."""

        workspace = find_margo_workspace(start)
        return cls(home, workspace) if workspace is not None else None

    # -- rows -------------------------------------------------------------------------

    def rows(self) -> list[CoworkRow]:
        """Open notes first (by file and line), then tasks whose note is gone."""

        rows: list[CoworkRow] = []
        seen: set[str] = set()
        for note in read_margo_notes(self.margo_workspace):
            seen.add(note.id)
            repo = find_repo_root(self.margo_workspace / note.file)
            service = self._service(repo) if repo is not None else None
            task = service.tasks.get(note.id) if service is not None else None
            if service is not None and task is not None:
                service.refresh_merge_state(note.id)  # the user may have committed or aborted
                task = service.tasks.get(note.id)
            rows.append(
                self._row(note.id, note.title, note.file, note.line_number, note, task, repo)
            )
        for repo, service in list(self._services.items()):
            for task in service.tasks.list():
                if task.note_id in seen or not task.is_active:
                    continue
                service.refresh_merge_state(task.note_id)
                task = service.tasks.get(task.note_id) or task
                if not task.is_active:
                    continue
                rows.append(
                    self._row(task.note_id, task.title, task.file, task.line, None, task, repo)
                )
        return rows

    def _row(
        self,
        note_id: str,
        title: str,
        file: str,
        line: int,
        note: MargoNote | None,
        task: CoworkTask | None,
        repo: Path | None,
    ) -> CoworkRow:
        running = self.is_running(note_id)
        if repo is None:
            status, detail = NO_REPO, "The file is not inside a git repository."
        elif task is None or not task.is_active and task.status is not TaskStatus.MERGED:
            status, detail = NEW, ""
        elif note is None:
            status, detail = ORPHAN, "The note was resolved or deleted; the task is still open."
        else:
            status, detail = task.status.value, ""
        if task is not None and status not in {NEW, NO_REPO}:
            detail = self._detail(note_id, task, running)
        if running:
            status = TaskStatus.RUNNING.value
        return CoworkRow(note_id, title, file, line, status, detail, note, task, repo, running)

    def _detail(self, note_id: str, task: CoworkTask, running: bool) -> str:
        with self._lock:
            progress = self._progress.get(note_id, "")
            error = self._errors.get(note_id, "")
        if running:
            return progress or "Starting…"
        if task.status is TaskStatus.MERGING:
            return "Staged in your repository: review in Source Control, then commit there."
        return error or " ".join(task.summary.split())[:MAX_PROGRESS_CHARS * 2]

    # -- actions ----------------------------------------------------------------------

    def actions(self, row: CoworkRow) -> list[CoworkAction]:
        if row.status in {NO_REPO, TaskStatus.MERGED.value} or row.repo_root is None:
            return []
        base = row.task.base_branch if row.task else ""
        if row.running:
            return [CoworkAction("stop", "Stop agent")]
        if row.status == TaskStatus.MERGING.value:
            return [
                CoworkAction("diff", "Show changes"),
                CoworkAction(
                    "abort_merge",
                    "Abort the pending merge",
                    "Unstages the agent's changes again",
                ),
            ]
        has_chat = bool(row.task and row.task.session_path)
        if row.status in {NEW, TaskStatus.QUEUED.value}:
            actions = [
                CoworkAction(
                    "start",
                    "Start agent",
                    "Makes the change in its own branch; your files stay untouched until you merge",
                ),
                CoworkAction(
                    "chat" if has_chat else "discuss",
                    "Open chat" if has_chat else "Discuss first",
                    "The agent only talks until you ask for a change",
                ),
            ]
            return actions
        changes = row.task is not None and self._changed(row)
        actions = [CoworkAction("chat", "Open chat", "Talk to the agent about this note")]
        if changes:
            actions.append(CoworkAction("diff", "Show changes"))
            actions.append(
                CoworkAction(
                    "compare",
                    "Compare in VS Code",
                    "Side by side: original on the left, the agent's version on the right",
                )
            )
            actions.append(CoworkAction("follow", "Show in editor", "Switches your working tree"))
            if row.status == TaskStatus.READY.value:
                actions.append(
                    CoworkAction(
                        "merge",
                        f"Merge into {base} (review in Source Control)",
                        "Stages the changes in your repository; you commit them in VS Code",
                    )
                )
                actions.append(
                    CoworkAction(
                        "merge_now",
                        f"Merge into {base} now",
                        "Commits the merge immediately",
                    )
                )
        if row.status in {TaskStatus.READY.value, TaskStatus.FAILED.value, ORPHAN}:
            if row.status != ORPHAN:
                actions.append(CoworkAction("start", "Run again"))
            actions.append(CoworkAction("discard", "Discard task"))
        return actions

    def _changed(self, row: CoworkRow) -> bool:
        assert row.task is not None and row.repo_root is not None
        service = self._service(row.repo_root)
        return service.git.commits_ahead(row.task.branch, row.task.base_commit) > 0

    # -- commands ---------------------------------------------------------------------

    def start(self, row: CoworkRow) -> str | None:
        """Prepare (if needed) and run the agent in a background thread.

        Returns an error text when preparation fails, ``None`` once the agent is running.
        """

        if row.repo_root is None:
            return "The file is not inside a git repository."
        service = self._service(row.repo_root)
        try:
            if row.task is None or not row.task.is_active:  # merged or discarded: start fresh
                if row.note is None:
                    return "The note no longer exists."
                service.prepare(row.note, self.margo_workspace)
        except GitError as error:
            return str(error)
        with self._lock:
            self._errors.pop(row.note_id, None)
            self._progress[row.note_id] = "Starting…"
            thread = threading.Thread(
                target=self._run, args=(service, row.note_id), daemon=True, name="anomx-cowork"
            )
            self._threads[row.note_id] = thread
        thread.start()
        return None

    def _run(self, service: CoworkService, note_id: str) -> None:
        def progress(text: str) -> None:
            cleaned = " ".join(text.split())
            if cleaned:
                with self._lock:
                    self._progress[note_id] = cleaned[:MAX_PROGRESS_CHARS]

        callbacks = RuntimeCallbacks(
            status=progress,
            tool_message=progress,
            command=lambda statement, _command, _output: progress(statement),
        )
        try:
            service.run(note_id, callbacks)
        except Exception as error:  # noqa: BLE001 - shown in the panel instead of crashing it
            with self._lock:
                self._errors[note_id] = f"{type(error).__name__}: {error}"
            service.tasks.update(note_id, status=TaskStatus.FAILED, summary=str(error))

    def stop(self, row: CoworkRow) -> bool:
        return row.repo_root is not None and self._service(row.repo_root).abort(row.note_id)

    def is_running(self, note_id: str) -> bool:
        with self._lock:
            thread = self._threads.get(note_id)
        return thread is not None and thread.is_alive()

    def wait(self, note_id: str, timeout: float = 30.0) -> None:
        """Block until the run of a note ends (used by tests and shutdown)."""

        with self._lock:
            thread = self._threads.get(note_id)
        if thread is not None:
            thread.join(timeout)

    def open_chat(
        self, row: CoworkRow, *, discuss: bool
    ) -> tuple[SessionRecord | None, str | None]:
        """Chat session of a note (created on first use) or an error text."""

        if row.repo_root is None:
            return None, "The file is not inside a git repository."
        if self.is_running(row.note_id):
            return None, "The agent is working; stop it or wait before opening the chat."
        service = self._service(row.repo_root)
        try:
            if row.note is not None:
                _task, record = service.open_chat(row.note, self.margo_workspace, discuss=discuss)
                return record, None
            task = service.tasks.get(row.note_id)
            existing = self.home.session_record(Path(task.session_path)) if task else None
            return (existing, None) if existing else (None, "This task has no chat yet.")
        except GitError as error:
            return None, str(error)

    def awaiting_reply(self, session_path: Path) -> bool:
        """The chat is one of ours and its latest message is still unanswered."""

        self.rows()
        return any(
            service.note_for_session(session_path) is not None
            and service.awaiting_reply(session_path)
            for service in list(self._services.values())
        )

    def runtime_for_session(self, session_path: Path) -> CoworkRuntime | None:
        """Runtime for a session that is the chat of one of our tasks, else ``None``."""

        self.rows()  # makes sure the services of all notes' repositories are loaded
        for service in list(self._services.values()):
            note_id = service.note_for_session(session_path)
            if note_id is not None:
                return service.runtime_for(note_id)
        return None

    def comparison_files(self, row: CoworkRow) -> list[ComparisonFile]:
        if row.task is None or row.repo_root is None:
            return []
        return self._service(row.repo_root).comparison_files(row.note_id)

    def diff_lines(self, row: CoworkRow) -> list[str]:
        if row.task is None or row.repo_root is None:
            return []
        return self._service(row.repo_root).diff(row.note_id).splitlines()

    def follow(self, row: CoworkRow) -> FollowResult:
        if row.repo_root is None:
            return FollowResult(False, "No repository.")
        return self._service(row.repo_root).follow(row.note_id)

    def merge(self, row: CoworkRow, *, review: bool = True) -> MergeResult:
        if row.repo_root is None:
            return MergeResult(False, reason="No repository.")
        return self._service(row.repo_root).merge(row.note_id, review=review)

    def abort_merge(self, row: CoworkRow) -> None:
        if row.repo_root is not None:
            self._service(row.repo_root).abort_pending_merge(row.note_id)

    def discard(self, row: CoworkRow) -> None:
        if row.repo_root is not None:
            self._service(row.repo_root).discard(row.note_id)

    def following(self) -> CoworkRow | None:
        """The task whose branch the user's working tree currently shows, if any."""

        for row in self.rows():
            if row.task is None or row.repo_root is None or not row.task.is_active:
                continue
            git = self._service(row.repo_root).git
            if git.current_branch() is None and git.head() == git.tip(row.task.branch):
                return row
        return None

    def sync_tree(self, row: CoworkRow, *, editor_dirty: bool) -> str:
        """Make the user's working tree show the state of ``row``'s task.

        Returns a short notice for the user, or an empty string when nothing needs saying.
        A tree that is already right, has uncommitted changes or belongs to an editor with
        unsaved text is never touched.
        """

        if row.repo_root is None:
            return ""
        git = self._service(row.repo_root).git
        task = row.task
        shows_task = task is not None and task.is_active
        if shows_task and task is not None and self._has_changes(row):
            if git.current_branch() is None and git.head() == git.tip(task.branch):
                return ""
            if editor_dirty:
                return "Unsaved changes in the editor, so your working tree was not switched."
            result = git.follow(task.branch)
            return "" if result.ok else result.reason
        following = self.following()
        if following is None or following.repo_root != row.repo_root:
            return ""
        if editor_dirty:
            return "Unsaved changes in the editor, so your working tree was not switched back."
        back = self.return_to_base(following)
        return "" if back.ok else back.reason

    def _has_changes(self, row: CoworkRow) -> bool:
        return row.task is not None and row.repo_root is not None and self._changed(row)

    def return_to_base(self, row: CoworkRow) -> FollowResult:
        if row.repo_root is None:
            return FollowResult(False, "No repository.")
        return self._service(row.repo_root).return_to_base(row.note_id)

    # -- internals --------------------------------------------------------------------

    def _service(self, repo_root: Path) -> CoworkService:
        with self._lock:
            service = self._services.get(repo_root)
            if service is None:
                service = CoworkService(self.home, repo_root)
                self._services[repo_root] = service
            return service
