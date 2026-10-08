"""Idle screens must wake up periodically instead of blocking on the next key."""

from __future__ import annotations

import curses

from anomx.agent.app import IDLE_POLL_MILLISECONDS, AnomxCliApp
from anomx.agent.store import AnomxHome


class _TimeoutWindow:
    def __init__(self, keys):
        self._keys = list(keys)
        self.timeouts: list[int] = []

    def timeout(self, milliseconds):
        self.timeouts.append(milliseconds)

    def nodelay(self, flag):
        pass

    def get_wch(self):
        if not self._keys:
            raise curses.error("no input")
        return self._keys.pop(0)


def _app(tmp_path):
    return AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=tmp_path)


def test_idle_read_returns_none_on_timeout_and_restores_blocking(tmp_path):
    window = _TimeoutWindow([])

    assert _app(tmp_path)._read_prompt_key_idle(window) is None
    assert window.timeouts == [IDLE_POLL_MILLISECONDS, -1]


def test_idle_read_returns_pressed_key(tmp_path):
    window = _TimeoutWindow(["a"])

    assert _app(tmp_path)._read_prompt_key_idle(window) == "a"
    assert window.timeouts[-1] == -1


def test_idle_read_falls_back_to_blocking_read_without_timeout_support(tmp_path):
    class Plain:
        def get_wch(self):
            return "x"

        def nodelay(self, flag):
            pass

    assert _app(tmp_path)._read_prompt_key_idle(Plain()) == "x"
