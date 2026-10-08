"""Validation and unsaved-change protection in the configuration screens."""

from __future__ import annotations

import curses
import time

import pytest

from anomx.agent.app import AnomxCliApp
from anomx.agent.store import AnomxHome
from anomx.agent.ui.views.config import validate_sandbox_limit


@pytest.mark.parametrize(
    ("kind", "value"),
    [("cpu", "2"), ("cpu", "1.5"), ("ram", "4g"), ("ram", "2048m"), ("ram", "8GB"), ("hd", "10g")],
)
def test_valid_sandbox_limits_are_accepted(kind, value):
    assert validate_sandbox_limit(kind, value) is None


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("cpu", "0"),
        ("cpu", "abc"),
        ("cpu", "-1"),
        ("cpu", "2g"),
        ("ram", "-1"),
        ("ram", "abc"),
        ("ram", "4 GB"),
        ("hd", "0g"),
    ],
)
def test_invalid_sandbox_limits_are_rejected_with_a_message(kind, value):
    assert validate_sandbox_limit(kind, value)


class _Keys:
    def __init__(self, keys):
        self._keys = list(keys)
        self._paused_until = 0.0
        self._nonblocking = False

    def get_wch(self):
        # ``None`` marks a pause longer than the escape-sequence window: no input is
        # pending for a moment, as between two separate key presses.
        if self._keys and self._keys[0] is None:
            self._keys.pop(0)
            self._paused_until = time.monotonic() + 0.1
        if time.monotonic() < self._paused_until:
            if self._nonblocking:
                raise curses.error("no input pending")
            time.sleep(self._paused_until - time.monotonic())  # a blocking read waits
        if not self._keys:
            raise curses.error("no input pending")
        return self._keys.pop(0)

    def nodelay(self, flag):
        self._nonblocking = bool(flag)

    def getmaxyx(self):
        return 24, 80


def _editor(tmp_path, monkeypatch, keys, confirm_answers):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=tmp_path)
    monkeypatch.setattr(app, "_draw_overlay", lambda *a, **k: None)
    monkeypatch.setattr(app, "_message", lambda *a, **k: None)
    answers = list(confirm_answers)
    asked: list[str] = []

    def fake_menu(_stdscr, title, *_a, **_k):
        asked.append(title)
        return answers.pop(0)

    monkeypatch.setattr(app, "_menu", fake_menu)
    app._run_manage_instructions_panel(_Keys(keys))
    return app, asked


def test_escape_without_changes_leaves_without_asking(tmp_path, monkeypatch):
    _app, asked = _editor(tmp_path, monkeypatch, ["\x1b"], [])

    assert asked == []


def test_escape_with_changes_asks_and_keep_editing_stays(tmp_path, monkeypatch):
    # type "x", Esc -> asked -> keep editing, Esc -> asked -> discard
    app, asked = _editor(tmp_path, monkeypatch, ["x", "\x1b", None, "\x1b"], ["keep", "discard"])

    assert asked == ["Unsaved Changes", "Unsaved Changes"]
    assert not (app.home.instructions_dir / "instruction.md").exists()


def test_ctrl_s_saves_the_edited_text(tmp_path, monkeypatch):
    app, asked = _editor(tmp_path, monkeypatch, ["h", "i", "\x13"], [])

    assert asked == []
    assert (app.home.instructions_dir / "instruction.md").read_text(encoding="utf-8") == "hi"
