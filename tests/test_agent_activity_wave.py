import curses

import pytest

from anomx.agent.app import AnomxCliApp
from anomx.agent.store import AnomxHome
from anomx.agent.ui.constants import (
    ACTIVITY_FRAME_SECONDS,
    ACTIVITY_WAVE_PAUSE_FRAMES,
    ACTIVITY_WAVE_TRAVEL_FRAMES,
    ACTIVITY_WAVE_WIDTH,
)


class Window:
    def __init__(self):
        self.writes = []

    def erase(self):
        self.writes.clear()

    def getmaxyx(self):
        return 40, 100

    def addnstr(self, y, x, text, n, attr=0):
        self.writes.append((y, x, text[:n], attr))

    def refresh(self):
        pass


@pytest.fixture
def chat(tmp_path):
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="blablador", model="coding")
    for kind, payload in (
        ("user_message", {"message": "Inspect channels"}),
        ("work_message", {"message": "Reading channels", "role": "tool"}),
        ("agent_message", {"message": "I found the channels.", "intermediate": True}),
        ("work_message", {"message": "Reading metadata", "role": "tool"}),
        ("work_message", {"message": "Counting channels", "role": "tool"}),
    ):
        home.append_session_event(session.path, kind, {
            **payload, **({"turn_id": "turn-1"} if kind != "user_message" else {}),
        })
    app = AnomxCliApp(home=home, cwd=tmp_path, use_color=False)
    app._colors = {"light": curses.A_DIM, "work_box": curses.A_REVERSE}
    return home, session, app


@pytest.mark.parametrize("mode", ["collapsed", "expanded", "extended", "details"])
def test_only_latest_visible_tool_label_gets_the_wave(chat, monkeypatch, mode):
    home, session, app = chat
    if mode in {"expanded", "details"}:
        app._toggle_work_turn("turn-1:1")
    if mode == "extended":
        app.work_visualization = "extended"
    if mode == "details":
        latest_tool = app._read_message_lines(session.path)[-2]
        app._toggle_work_line(latest_tool.expansion_key)
    waves = []
    monkeypatch.setattr(app, "_draw_activity_wave", lambda *args: waves.append(args[3]))
    messages = app._read_message_lines(session.path)
    app._draw_session(Window(), session, messages, "", 0, 0, working_text="Thinking")
    expected = "Counting channels"
    assert waves == [expected]
    assert not any(line.activity_wave for line in messages)

    # The latest row owns the wave until a newer tool replaces it.
    home.append_session_event(session.path, "work_message", {
        "message": "Inspecting results", "role": "tool", "turn_id": "turn-1",
    })
    waves.clear()
    messages = app._read_message_lines(session.path)
    app._draw_session(Window(), session, messages, "", 0, 0, working_text="Thinking")
    expected = "Inspecting results"
    assert waves == [expected]

    # The same cached transcript becomes still as soon as execution stops.
    waves.clear()
    app._draw_session(Window(), session, messages, "", 0, 0)
    assert waves == []


def test_messages_do_not_move_the_wave_to_an_older_request(chat):
    home, session, app = chat
    home.append_session_event(session.path, "agent_message", {
        "message": "The count is nearly ready.", "intermediate": True, "turn_id": "turn-1",
    })
    messages = app._messages_with_working_status(app._read_message_lines(session.path), "Thinking")
    assert [line.text for line in messages if line.activity_wave] == ["Counting channels"]
    home.append_session_event(session.path, "user_message", {"message": "Inspect another source"})
    messages = app._messages_with_working_status(app._read_message_lines(session.path), "Thinking")
    assert not any(line.activity_wave for line in messages)
    assert messages[-2].text == "Thinking"


@pytest.mark.parametrize("attr", [curses.A_DIM, curses.A_NORMAL, curses.A_REVERSE])
def test_wave_moves_pauses_repeats_and_preserves_terminal_colors(chat, attr):
    _, _, app = chat
    window = Window()
    label = "Inspecting channel metadata"
    cycle = ACTIVITY_WAVE_TRAVEL_FRAMES + ACTIVITY_WAVE_PAUSE_FRAMES
    positions = []
    for frame in range(cycle * 2):
        window.writes.clear()
        app._draw_activity_wave(window, 3, 4, label + " · collapse", 60, frame, attr)
        positions.append(list(window.writes))
        for _, x, text, wave_attr in window.writes:
            assert 0 < len(text) <= ACTIVITY_WAVE_WIDTH
            assert text == label[x - 4:x - 4 + len(text)]
            assert wave_attr == attr ^ curses.A_DIM
            assert not wave_attr & curses.A_BOLD
    travel = [writes[0][1] for writes in positions[:ACTIVITY_WAVE_TRAVEL_FRAMES] if writes]
    assert travel == sorted(travel)
    assert travel[0] < travel[-1]
    assert all(not writes for writes in positions[ACTIVITY_WAVE_TRAVEL_FRAMES:cycle])
    assert positions[:cycle] == positions[cycle:]


@pytest.mark.parametrize("width", [0, 1, 8])
def test_wave_stays_inside_the_visible_label(chat, width):
    _, _, app = chat
    window = Window()
    for frame in range(ACTIVITY_WAVE_TRAVEL_FRAMES):
        app._draw_activity_wave(window, 3, 4, "Inspecting channel metadata", width, frame, 0)
    assert all(x >= 4 and x + len(text) <= 4 + width for _, x, text, _ in window.writes)


def test_color_wave_has_soft_edges_and_a_stable_pause(chat):
    _, _, app = chat
    window = Window()
    palette = tuple(index << 8 for index in range(12, 25))
    app._activity_wave_palettes[False] = palette
    label = "a" * 24
    app._draw_activity_wave(window, 3, 4, label, 60, 10, curses.A_DIM)
    shades = [palette.index(attr) for _, _, _, attr in window.writes]
    assert len(set(shades)) >= 5
    assert shades[:12] == sorted(shades[:12])
    assert shades[12:] == sorted(shades[12:], reverse=True)
    assert shades[0] == shades[-1] == 0
    assert shades[11] == shades[12] == len(palette) - 1
    window.writes.clear()
    app._draw_activity_wave(window, 3, 4, label, 60, ACTIVITY_WAVE_TRAVEL_FRAMES, curses.A_DIM)
    assert "".join(text for _, _, text, _ in window.writes) == label
    assert {attr for _, _, _, attr in window.writes} == {palette[0]}
    assert ACTIVITY_WAVE_TRAVEL_FRAMES * ACTIVITY_FRAME_SECONDS <= 0.8
    cycle = (ACTIVITY_WAVE_TRAVEL_FRAMES + ACTIVITY_WAVE_PAUSE_FRAMES) * ACTIVITY_FRAME_SECONDS
    assert cycle <= 1.4


def test_gradient_preserves_wide_glyphs_and_combining_marks(chat):
    _, _, app = chat
    app._activity_wave_palettes[False] = (12 << 8, 13 << 8)
    window = Window()
    app._draw_activity_wave(window, 3, 4, "測定 e\u0301", 6, 10, curses.A_DIM)
    assert [(x, text) for _, x, text, _ in window.writes] == [
        (4, "測"), (6, "定"), (8, " "), (9, "e\u0301"),
    ]


@pytest.mark.parametrize("light_theme", [False, True])
def test_gradient_palette_adapts_to_the_terminal_theme(chat, monkeypatch, light_theme):
    from anomx.agent.ui.components import messages

    _, _, app = chat
    foreground, background = ((30, 30, 30), (245, 245, 245)) if light_theme else (
        (230, 230, 230), (20, 20, 20),
    )
    pairs = []
    monkeypatch.setattr(messages, "read_terminal_theme", lambda: (foreground, background))
    monkeypatch.setattr(curses, "COLORS", 256, raising=False)
    monkeypatch.setattr(curses, "COLOR_PAIRS", 256, raising=False)
    monkeypatch.setattr(curses, "init_pair", lambda pair, fg, bg: pairs.append((pair, fg, bg)))
    monkeypatch.setattr(curses, "color_pair", lambda pair: pair << 8)
    app._configure_activity_wave_colors(-1, -1)
    normal = pairs[:len(app._activity_wave_palettes[False])]
    reverse = pairs[len(normal):]
    assert len({fg for _, fg, _ in normal}) >= 5
    assert all(bg == -1 for _, _, bg in normal)
    assert all(fg == -1 for _, fg, _ in reverse)
    assert normal[0][1] > normal[-1][1] if light_theme else normal[0][1] < normal[-1][1]
    assert all(pair >= 12 for pair, _, _ in pairs)
