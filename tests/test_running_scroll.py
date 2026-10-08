"""Scrolling while an answer is pending: the pinned message can be left and re-entered."""

from __future__ import annotations

import curses
import queue
import time

import pytest

import anomx.agent.ui as ui_module
from anomx.agent.app import AnomxCliApp
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.store import AnomxHome
from anomx.agent.ui.models import SessionViewportState, SessionViewState


def viewport(max_scroll: int = 0) -> SessionViewportState:
    return SessionViewportState(0, 0, 20, 100, max_scroll=max_scroll)


def pinned(scroll: int = 0) -> SessionViewState:
    return SessionViewState(scroll=scroll, pinned_anchor=61)


def test_scrolling_up_with_nothing_left_below_the_message_releases_the_pin():
    view = pinned()

    AnomxCliApp._apply_running_scroll(view, 5, viewport(max_scroll=0))

    assert view.anchor_released and view.pinned_anchor is None and view.scroll == 5


def test_scrolling_up_inside_a_long_answer_keeps_the_message_pinned():
    view = pinned(scroll=3)

    AnomxCliApp._apply_running_scroll(view, 5, viewport(max_scroll=10))

    assert view.scroll == 8 and view.pinned_anchor == 61 and not view.anchor_released


def test_scrolling_up_at_the_top_of_a_long_answer_releases_the_pin():
    view = pinned(scroll=10)

    AnomxCliApp._apply_running_scroll(view, 1, viewport(max_scroll=10))

    assert view.anchor_released and view.scroll == 1


def test_scrolling_down_never_releases_the_pin():
    view = pinned(scroll=4)

    AnomxCliApp._apply_running_scroll(view, -1, viewport(max_scroll=10))

    assert view.scroll == 3 and view.pinned_anchor == 61 and not view.anchor_released


def test_without_a_viewport_scrolling_up_releases_the_pin():
    view = pinned()

    AnomxCliApp._apply_running_scroll(view, 1, None)

    assert view.anchor_released


def test_scrolling_back_to_the_bottom_pins_the_message_again():
    view = SessionViewState(scroll=3, anchor_released=True)

    AnomxCliApp._apply_running_scroll(view, -1, viewport())
    assert view.scroll == 2 and view.anchor_released

    AnomxCliApp._apply_running_scroll(view, -5, viewport())
    assert view.scroll == 0 and not view.anchor_released


def test_jump_to_bottom_clears_the_released_state():
    view = SessionViewState(scroll=7, anchor_released=True)

    view.jump_to_bottom()

    assert not view.anchor_released and view.scroll == 0


# -- the session loop ----------------------------------------------------------------


class _Worker:
    def is_alive(self) -> bool:
        return True

    def join(self, timeout=None) -> None:
        pass


class _Window:
    def __init__(self, keys):
        self._keys = list(keys)

    def get_wch(self):
        if not self._keys:
            raise curses.error("no input")
        return self._keys.pop(0)

    def nodelay(self, flag):
        pass

    def getmaxyx(self):
        return 30, 100


def test_the_running_view_leaves_and_re_enters_the_pinned_message(tmp_path, monkeypatch):
    class Stop(Exception):
        pass

    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    for index in range(30):
        home.append_session_event(session.path, "user_message", {"message": f"Q{index}"})
        home.append_session_event(session.path, "agent_message", {"message": f"A{index}"})
    home.append_session_event(session.path, "user_message", {"message": "LATEST"})
    app = AnomxCliApp(home=home, cwd=repo, use_color=False)
    turn = ui_module.ActiveSessionTurn(
        session=session,
        runtime=app.runtime,
        events=queue.SimpleQueue(),
        result={},
        turn_id="t",
        started_at=time.monotonic(),
        worker=_Worker(),
        mode=AgentMode.STANDARD,
    )
    app._active_session_turns[app._session_turn_key(session)] = turn
    monkeypatch.setattr(app, "_active_turn_anchor_line", lambda *a, **k: 61)
    frames: list[tuple[int | None, int, bool]] = []

    def fake_draw(_stdscr, _session, _messages, _text, _cursor, scroll, *_a, **kwargs):
        frames.append((kwargs.get("anchor_line"), scroll, bool(kwargs.get("sticky_anchor"))))
        if len(frames) > 8:
            raise Stop
        anchored = kwargs.get("anchor_line") is not None
        return SessionViewportState(0, 0 if anchored else scroll, 20, 100, max_scroll=0)

    monkeypatch.setattr(app, "_draw_session", fake_draw)
    up, down = curses.KEY_PPAGE, curses.KEY_NPAGE
    keys = [up, up, down, down, down]

    with pytest.raises(Stop):
        app._run_session(_Window(keys), session)

    assert frames[0] == (61, 0, True)  # pinned while the answer is pending
    assert frames[1] == (None, 5, False)  # one PageUp: earlier chat reachable
    assert frames[2] == (None, 10, False)
    assert frames[3] == (None, 5, False)
    assert (61, 0, True) in frames[4:]  # back at the bottom: pinned again
