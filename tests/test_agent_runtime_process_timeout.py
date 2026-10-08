
import pytest

from anomx.agent.runtime_process import RuntimeProcessClient, RuntimeProcessError


def test_control_call_times_out_instead_of_blocking_the_ui_thread():
    client = RuntimeProcessClient.__new__(RuntimeProcessClient)
    client._pending = {}
    client._callbacks = {}
    client._pending_lock = __import__("threading").Lock()
    client._ensure_started = lambda: None
    client._send = lambda payload: None  # child never answers

    with pytest.raises(RuntimeProcessError, match="did not answer"):
        client._call("abort_current_turn", None, _timeout=0.05)
    assert client._pending == {}


def test_reap_idle_runtimes_closes_only_finished_runtimes_without_live_children(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from anomx.agent.app import AnomxCliApp
    from anomx.agent.store import AnomxHome

    class FakeRuntime:
        def __init__(self, live):
            self.live = live
            self.closed = False

        def has_live_children(self):
            return self.live

        def close(self):
            self.closed = True

    monkeypatch.setattr("anomx.agent.app.RuntimeProcessClient", FakeRuntime)
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=tmp_path, use_color=False)
    foreground = FakeRuntime(False)
    idle = FakeRuntime(False)
    background = FakeRuntime(True)
    in_turn = FakeRuntime(False)
    app.runtime = foreground
    app._runtime_processes = [foreground, idle, background, in_turn]
    app._active_session_turns = {"s": SimpleNamespace(runtime=in_turn)}
    app._next_runtime_reap = 0.0

    app._reap_idle_runtimes()

    assert idle.closed
    assert not (foreground.closed or background.closed or in_turn.closed)
    assert app._runtime_processes == [foreground, background, in_turn]


def _stale_activity_app(tmp_path, monkeypatch, live):
    from anomx.agent.app import AnomxCliApp
    from anomx.agent.store import AnomxHome

    class FakeRuntime:
        def __init__(self, live):
            self.live = live
            self.cleaned = []

        def has_live_children(self):
            return self.live

        def cleanup_session_runtime_state(self, path):
            self.cleaned.append(path)

    monkeypatch.setattr("anomx.agent.app.RuntimeProcessClient", FakeRuntime)
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=tmp_path, use_color=False)
    app.runtime = FakeRuntime(live)
    app._runtime_processes = [app.runtime]
    monkeypatch.setattr(app, "_has_running_session_activity", lambda _path: True)
    return app


def test_reconcile_ends_transcript_activity_no_runtime_owns(tmp_path, monkeypatch):
    app = _stale_activity_app(tmp_path, monkeypatch, live=False)
    app._reconcile_stale_activity(tmp_path / "s.jsonl")
    assert app.runtime.cleaned == [tmp_path / "s.jsonl"]


def test_reconcile_keeps_activity_a_runtime_still_owns(tmp_path, monkeypatch):
    app = _stale_activity_app(tmp_path, monkeypatch, live=True)
    app._reconcile_stale_activity(tmp_path / "s.jsonl")
    assert app.runtime.cleaned == []


def _bare_runtime():
    from anomx.agent.runtime import AgentRuntime

    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime._unfulfilled_action_nudges = 0
    return runtime


def test_announced_action_without_tool_call_is_nudged():
    runtime = _bare_runtime()
    for text in (
        "Das passt.\n\nLet me read the relevant lines.",
        "Ich habe die Lage geprüft.\nLass mich die Datei lesen.",
        "I need to fix the strict-mode errors. Let me read topology.ts.",
        "ok\n</｜DSML｜parameter>\n</｜DSML｜invoke>",
    ):
        assert runtime._unfulfilled_action_prompt(text) is not None, text


def test_finished_or_questioning_text_is_not_nudged():
    runtime = _bare_runtime()
    for text in (
        "Alle Tests laufen durch. Die Änderung ist fertig.",
        "Let me know if you want more detail.\n\nDone: the fix is in app.py.",
        "Ich habe alles umgesetzt.\n\nSoll ich fortfahren?",
    ):
        assert runtime._unfulfilled_action_prompt(text) is None, text


def test_unfulfilled_action_nudges_are_bounded():
    from anomx.agent.runtime import MAX_UNFULFILLED_ACTION_NUDGES

    runtime = _bare_runtime()
    runtime._unfulfilled_action_nudges = MAX_UNFULFILLED_ACTION_NUDGES
    assert runtime._unfulfilled_action_prompt("Let me read the file.") is None
