"""Answer general questions about a repository ("where is the GUI built?").

A question is a note without a line. There is nothing to change, so no branch or worktree is
made: a read-only agent (no ``edit_file``) searches the repository the user works in, answers,
and the places it names become clickable locations in the editor. The conversation is an
ordinary anomx session, so follow-up questions in the same thread build on what was found.
"""

from __future__ import annotations

import threading
from pathlib import Path

from anomx.agent.base.agents import AgentKind
from anomx.agent.cowork.agent import CoworkRuntime
from anomx.agent.cowork.git import find_repo_root
from anomx.agent.cowork.notes import Note
from anomx.agent.cowork.replies import AgentReplies, find_locations
from anomx.agent.exceptions import BackendFailure
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.runtime import RuntimeCallbacks
from anomx.agent.store import AnomxHome


def question_repo(note: Note, workspace: Path) -> Path | None:
    """Repository a question is about.

    The file the user had open when asking decides; otherwise the workspace folder itself,
    or the only git repository directly below it.
    """

    if note.context_file:
        repo = find_repo_root(Path(note.context_file))
        if repo is not None:
            return repo
    repo = find_repo_root(workspace)
    if repo is not None:
        return repo
    try:
        children = sorted(path for path in workspace.iterdir() if (path / ".git").exists())
    except OSError:
        return None
    return children[0].resolve() if len(children) == 1 else None


def build_question_prompt(note: Note, repo_root: Path) -> str:
    lines = [note.request, ""]
    if note.context_file:
        try:
            shown = Path(note.context_file).resolve().relative_to(repo_root)
        except ValueError:
            shown = Path(note.context_file)
        lines.append(f"(The user had {shown} open in the editor when asking.)")
    return "\n".join(lines).strip()


class QuestionService:
    """Runs question threads of one workspace; every answer is written to ``agent.json``."""

    def __init__(self, home: AnomxHome, workspace: Path, replies: AgentReplies) -> None:
        self.home = home
        self.workspace = workspace.resolve()
        self.replies = replies
        self._running: dict[str, CoworkRuntime] = {}
        self._lock = threading.Lock()

    def answer(
        self,
        note: Note,
        message_id: str,
        text: str,
        callbacks: RuntimeCallbacks | None = None,
    ) -> None:
        """Answer one user message of a question thread. Blocks until the agent is done."""

        repo = question_repo(note, self.workspace)
        if repo is None:
            self.replies.add_answer(
                note.id,
                "I cannot tell which repository this question is about. Open a file of the "
                "repository in VS Code and ask again.",
                after_message=message_id,
                failed=True,
            )
            return
        runtime = CoworkRuntime(self.home, repo, question=True, mode=AgentMode.STANDARD)
        try:
            session_path = self._session(note, repo, message_id, text)
            with self._lock:
                self._running[note.id] = runtime
            response = runtime.backend_response(session_path, callbacks or RuntimeCallbacks())
        except Exception as exc:  # noqa: BLE001 - the thread must end in a recorded state
            self.replies.add_answer(
                note.id, f"{type(exc).__name__}: {exc}", after_message=message_id, failed=True
            )
            return
        finally:
            with self._lock:
                self._running.pop(note.id, None)
        failed = isinstance(response, BackendFailure)
        aborted = runtime.turn_aborted()
        answer = str(response)
        if aborted and not failed:
            answer, failed = "Stopped before the agent finished.", True
        self.replies.add_answer(
            note.id,
            answer,
            after_message=message_id,
            locations=[] if failed else find_locations(answer, repo, self.workspace),
            failed=failed,
        )

    def abort(self, note_id: str) -> bool:
        with self._lock:
            runtime = self._running.get(note_id)
        if runtime is None:
            return False
        runtime.abort_current_turn()
        return True

    def is_running(self, note_id: str) -> bool:
        with self._lock:
            return note_id in self._running

    def _session(self, note: Note, repo: Path, message_id: str, text: str) -> Path:
        """The thread's conversation with ``text`` appended (created on the first message)."""

        existing = self.replies.session(note.id)
        if existing and Path(existing).exists():
            self.home.append_session_event(Path(existing), "user_message", {"message": text})
            return Path(existing)
        config = self.home.load_config()
        session = self.home.create_session(
            repo,
            str(config.get("provider", "")),
            str(config.get("model", "")),
            agent_kind=AgentKind.MAIN,
        )
        self.home.update_session_title(session.path, note.title or "Question")
        first = build_question_prompt(note, repo)
        self.home.append_session_event(session.path, "user_message", {"message": first})
        if message_id != note.messages[0].id:  # the earlier conversation was lost
            self.home.append_session_event(session.path, "user_message", {"message": text})
        self.replies.update(note.id, session=str(session.path))
        return session.path
