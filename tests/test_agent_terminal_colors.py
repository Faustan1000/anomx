from anomx.agent.helpers import terminal_colors
from anomx.agent.helpers.terminal_colors import read_terminal_theme, terminal_color_index


def test_terminal_color_query_handles_partial_replies_and_preserves_input(monkeypatch):
    chunks = iter([
        b"hi\x1b]10;rgb:eeee/",
        b"aaaa/8888\x1b\\",
        b"\x1b]11;rgb:10/20/30\x07!",
    ])
    written = []
    restored = []
    monkeypatch.setattr(terminal_colors.os, "isatty", lambda fd: True)
    monkeypatch.setattr(terminal_colors.os, "write", lambda fd, data: written.append(data))
    monkeypatch.setattr(terminal_colors.os, "read", lambda fd, size: next(chunks))
    monkeypatch.setattr(terminal_colors.select, "select", lambda *args: ([0], [], []))
    monkeypatch.setattr(terminal_colors.curses, "ungetch", restored.append)
    assert read_terminal_theme() == ((238, 170, 136), (16, 32, 48))
    assert written == [b"\x1b]10;?\x1b\\\x1b]11;?\x1b\\"]
    assert bytes(reversed(restored)) == b"hi!"


def test_unsupported_terminal_color_query_times_out(monkeypatch):
    monkeypatch.delenv("COLORFGBG", raising=False)
    monkeypatch.setattr(terminal_colors.os, "isatty", lambda fd: True)
    monkeypatch.setattr(terminal_colors.os, "write", lambda fd, data: len(data))
    monkeypatch.setattr(terminal_colors.select, "select", lambda *args: ([], [], []))
    assert read_terminal_theme() is None


def test_noninteractive_color_detection_uses_colorfgbg_without_querying(monkeypatch):
    monkeypatch.setattr(terminal_colors.os, "isatty", lambda fd: False)
    monkeypatch.setenv("COLORFGBG", "0;15")
    monkeypatch.setattr(terminal_colors.curses, "color_content", lambda index: (
        (0, 0, 0) if index == 0 else (1000, 1000, 1000)
    ))
    assert read_terminal_theme() == ((0, 0, 0), (255, 255, 255))


def test_terminal_color_mapping_uses_gray_cube_or_direct_rgb():
    assert terminal_color_index((128, 128, 128), 256) == 244
    assert terminal_color_index((255, 0, 0), 256) == 196
    assert terminal_color_index((10, 20, 30), 1 << 24) == 0x0A141E
