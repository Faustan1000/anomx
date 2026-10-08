"""The shared popup loops behind session/project menus and text popovers."""

from __future__ import annotations

import curses

from anomx.agent.app import AnomxCliApp
from anomx.agent.store import AnomxHome
from anomx.agent.ui.models import MenuChoice


class _Keys:
    def __init__(self, keys):
        self._keys = list(keys)

    def get_wch(self):
        if not self._keys:
            raise curses.error("no more input")
        return self._keys.pop(0)

    def nodelay(self, flag):
        pass


def _app(tmp_path):
    return AnomxCliApp(home=AnomxHome(tmp_path / "home"), cwd=tmp_path)


def _input(app, keys, **kwargs):
    frames: list[tuple[str, int]] = []
    defaults = {"mask": False, "optional": True, "default": ""}
    result = app._run_single_line_input(
        _Keys(keys), lambda value, cursor: frames.append((value, cursor)), **{**defaults, **kwargs}
    )
    return result, frames


def test_single_line_input_edits_and_returns_value(tmp_path):
    result, frames = _input(_app(tmp_path), ["a", "b", curses.KEY_LEFT, "X", "\n"])

    assert result == "aXb"
    assert frames[-1] == ("aXb", 2)


def test_single_line_input_masks_display_but_returns_real_value(tmp_path):
    result, frames = _input(_app(tmp_path), ["s", "3", "\n"], mask=True)

    assert result == "s3"
    assert frames[-1][0] == "**"


def test_single_line_input_escape_cancels_unless_required(tmp_path):
    app = _app(tmp_path)

    assert _input(app, ["x", "\x1b"], optional=True)[0] is None
    assert _input(app, ["x", "\x1b"], optional=False)[0] == ""


def test_single_line_input_requires_value_when_not_optional(tmp_path):
    result, _ = _input(_app(tmp_path), ["\n", "k", "\n"], optional=False)

    assert result == "k"


def _menu(app, keys, choices, **kwargs):
    panels = []
    result = app._run_bottom_menu(
        _Keys(keys), "Title", "Sub", choices, draw=panels.append, **{"searchable": False, **kwargs}
    )
    return result, panels


CHOICES = (
    MenuChoice("Alpha", "a"),
    MenuChoice("Beta", "b"),
    MenuChoice("Gamma", "g"),
)


def test_bottom_menu_moves_selection_and_returns_value(tmp_path):
    result, panels = _menu(_app(tmp_path), [curses.KEY_DOWN, curses.KEY_DOWN, "\n"], CHOICES)

    assert result == "g"
    assert [panel.selected for panel in panels] == [0, 1, 2]


def test_bottom_menu_selection_is_clamped(tmp_path):
    result, _ = _menu(_app(tmp_path), [curses.KEY_UP, curses.KEY_UP, "\n"], CHOICES)

    assert result == "a"


def test_bottom_menu_escape_returns_none(tmp_path):
    assert _menu(_app(tmp_path), ["\x1b"], CHOICES)[0] is None


def test_bottom_menu_search_filters_choices(tmp_path):
    result, panels = _menu(_app(tmp_path), ["g", "\n"], CHOICES, searchable=True)

    assert result == "g"
    assert [choice.value for choice in panels[-1].choices] == ["g"]


class _IdleKeys:
    """Window with timeout support: ``None`` in the key list is a read that times out."""

    def __init__(self, keys):
        self._keys = list(keys)

    def timeout(self, milliseconds):
        pass

    def nodelay(self, flag):
        pass

    def get_wch(self):
        key = self._keys.pop(0) if self._keys else None
        if key is None:
            raise curses.error("timeout")
        return key


def test_refreshing_menu_keeps_the_selection_by_value_when_rows_appear(tmp_path):
    app = _app(tmp_path)
    frames: list[tuple[str, ...]] = []
    snapshots = [
        (MenuChoice("Alpha", "a"), MenuChoice("Beta", "b")),
        (MenuChoice("Alpha", "a"), MenuChoice("Beta", "b")),
        (MenuChoice("New", "n"), MenuChoice("Alpha", "a"), MenuChoice("Beta", "b")),
        (MenuChoice("New", "n"), MenuChoice("Alpha", "a"), MenuChoice("Beta", "b")),
    ]
    calls = iter(snapshots)

    def refresh():
        return "subtitle", next(calls)

    app._draw_overlay = lambda _w, **kw: frames.append(
        (kw["choices"][kw["selected"]].value, str(len(kw["choices"])))
    )

    result = app._run_overlay_menu(
        _IdleKeys([curses.KEY_DOWN, None, "\n"]), "Cowork", refresh=refresh
    )

    # Selection moved to Beta, a row appeared above it, and Enter still returns Beta.
    assert result == "b"
    assert frames[-1] == ("b", "3")


def test_refreshing_menu_idles_without_returning_and_ends_on_escape(tmp_path):
    app = _app(tmp_path)
    app._draw_overlay = lambda *a, **k: None
    refreshes: list[int] = []

    def refresh():
        refreshes.append(1)
        return "s", (MenuChoice("Only", "o"),)

    assert app._run_overlay_menu(_IdleKeys([None, None, "\x1b"]), "T", refresh=refresh) is None
    assert len(refreshes) >= 3  # refreshed on every wake-up


def test_menu_without_refresh_is_unchanged_for_empty_choices(tmp_path):
    assert _app(tmp_path)._run_overlay_menu(_Keys([]), "T", choices=()) is None
