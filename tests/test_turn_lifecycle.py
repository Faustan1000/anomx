"""Aborting and requeueing a turn must never leave two workers on one transcript."""

from __future__ import annotations

import queue
import threading
import time

import anomx.agent.app as app_module
from anomx.agent.app import AnomxCliApp
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.store import AnomxHome
from anomx.agent.ui.models import ActiveSessionTurn, RuntimeUiEvent


class _Runtime:
    def __init__(self, fail_abort: bool = False):
        self.aborted = False
        self.shut_down = False
        self.fail_abort = fail_abort

    def abort_current_turn(self, _session_path=None):
        if self.fail_abort:
            raise RuntimeError("runtime not reachable")
        self.aborted = True

    def shutdown(self, _session_path=None):
        self.shut_down = True


def _setup(tmp_path, worker_target):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=repo)
    worker = threading.Thread(target=worker_target, daemon=True)
    runtime = _Runtime()
    turn = ActiveSessionTurn(
        session=session,
        runtime=runtime,
        events=queue.SimpleQueue(),
        result={},
        turn_id="turn-1",
        started_at=time.monotonic(),
        worker=worker,
        mode=AgentMode.AUTOMATIC,
    )
    worker.start()
    app._active_session_turns[app._session_turn_key(session)] = turn
    app._drain_session_turn_events = lambda *args, **kwargs: None
    return app, session, turn, runtime


def _payloads(app, session, kind):
    return [
        event["payload"]
        for event in app.home.read_session_events(session.path)
        if event["type"] == "event_msg" and event["payload"].get("type") == kind
    ]


def test_turn_state_reflects_lifecycle(tmp_path):
    app, _session, turn, _runtime = _setup(tmp_path, lambda: None)

    assert turn.state == "running"
    turn.aborting = True
    assert turn.state == "aborting"
    turn.completed = True
    assert turn.state == "done"


def test_requeue_is_refused_while_the_old_worker_is_still_running(tmp_path, monkeypatch):
    release = threading.Event()
    app, session, turn, runtime = _setup(tmp_path, lambda: release.wait(10))
    monkeypatch.setattr(app_module, "REQUEUE_STOP_GRACE_SECONDS", 0.05)
    started: list[object] = []
    monkeypatch.setattr(app, "_start_session_turn", lambda *a, **k: started.append((a, k)))

    try:
        requeued = app._interrupt_and_requeue_session_turn(None, turn, session, "new message")
    finally:
        release.set()

    assert requeued is False
    assert runtime.aborted is True
    assert started == []  # no second worker on the same transcript
    assert turn.state == "aborting" and not turn.completed
    assert app._active_session_turns  # still tracked so its events are drained
    assert _payloads(app, session, "user_message") == []


def test_requeue_starts_successor_with_the_same_turn_id(tmp_path, monkeypatch):
    app, session, turn, _runtime = _setup(tmp_path, lambda: None)
    started: list[dict[str, object]] = []
    monkeypatch.setattr(app, "_start_session_turn", lambda *a, **k: started.append(k))

    assert app._interrupt_and_requeue_session_turn(None, turn, session, "next") is True

    assert started == [{"turn_id": "turn-1"}]
    assert turn.completed
    messages = _payloads(app, session, "user_message")
    assert [m["message"] for m in messages] == ["next"]
    assert messages[0]["intermediate"] is True


def test_abort_failure_is_reported_in_the_session_and_callbacks_are_still_released(tmp_path):
    app, session, turn, runtime = _setup(tmp_path, lambda: None)
    runtime.fail_abort = True
    answer: queue.SimpleQueue = queue.SimpleQueue()
    turn.events.put(RuntimeUiEvent("approval", approval_response=answer))

    app._abort_turn(turn, shutdown=True)

    assert turn.aborting is True
    assert runtime.shut_down is True
    assert answer.get_nowait() == "reject"
    notices = [m["message"] for m in _payloads(app, session, "system_message")]
    assert any("Could not abort the running agent cleanly" in text for text in notices)
