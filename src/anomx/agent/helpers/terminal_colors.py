"""Read terminal theme colors and map RGB shades to its existing palette."""

from __future__ import annotations

import curses
import os
import re
import select
import time
from contextlib import suppress

Rgb = tuple[int, int, int]
COLOR_REPLY = re.compile(
    rb"\x1b\](10|11);rgb:([\da-fA-F]{1,4})/([\da-fA-F]{1,4})/([\da-fA-F]{1,4})"
    rb"(?:\x07|\x1b\\)"
)


def read_terminal_theme(timeout: float = 0.12) -> tuple[Rgb, Rgb] | None:
    """Query foreground/background once, preserving any typed input for curses.

    Called after curses enables raw input. OSC 10/11 queries do not change colors:
    https://invisible-island.net/xterm/ctlseqs/ctlseqs.html
    """
    response = b""
    colors: dict[int, Rgb] = {}
    if os.isatty(0) and os.isatty(1):
        try:
            os.write(1, b"\x1b]10;?\x1b\\\x1b]11;?\x1b\\")
            deadline = time.monotonic() + timeout
            while len(colors) < 2:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([0], [], [], remaining)[0]:
                    break
                chunk = os.read(0, 1024)
                if not chunk:
                    break
                response += chunk
                for match in COLOR_REPLY.finditer(response):
                    channels = tuple(
                        round(int(value, 16) * 255 / (16 ** len(value) - 1))
                        for value in match.groups()[1:]
                    )
                    colors[int(match[1])] = (channels[0], channels[1], channels[2])
        except (OSError, ValueError):
            pass
        finally:
            for byte in reversed(COLOR_REPLY.sub(b"", response)):
                with suppress(curses.error):
                    curses.ungetch(byte)
    if 10 in colors and 11 in colors:
        return colors[10], colors[11]
    # Some terminals advertise the theme through COLORFGBG instead of OSC.
    try:
        indices = os.environ["COLORFGBG"].split(";")
        foreground = curses.color_content(int(indices[0]))
        background = curses.color_content(int(indices[-1]))
        return (
            (round(foreground[0] * 255 / 1000), round(foreground[1] * 255 / 1000),
             round(foreground[2] * 255 / 1000)),
            (round(background[0] * 255 / 1000), round(background[1] * 255 / 1000),
             round(background[2] * 255 / 1000)),
        )
    except (KeyError, ValueError, curses.error):
        return None


def blend_color(background: Rgb, foreground: Rgb, amount: float) -> Rgb:
    """Interpolate a foreground shade against its actual terminal background."""
    return (
        round(background[0] + (foreground[0] - background[0]) * amount),
        round(background[1] + (foreground[1] - background[1]) * amount),
        round(background[2] + (foreground[2] - background[2]) * amount),
    )


def terminal_color_index(color: Rgb, color_count: int) -> int:
    """Use direct RGB when supported, otherwise the nearest xterm-256 shade."""
    if color_count >= 1 << 24:
        return (color[0] << 16) | (color[1] << 8) | color[2]
    levels = (0, 95, 135, 175, 215, 255)
    cube = tuple(min(range(6), key=lambda index: abs(levels[index] - c)) for c in color)
    cube_color = tuple(levels[index] for index in cube)
    gray = max(0, min(23, round((sum(color) / 3 - 8) / 10)))
    gray_value = 8 + 10 * gray
    cube_distance = sum((a - b) ** 2 for a, b in zip(color, cube_color, strict=True))
    gray_distance = sum((c - gray_value) ** 2 for c in color)
    if gray_distance < cube_distance:
        return 232 + gray
    return 16 + 36 * cube[0] + 6 * cube[1] + cube[2]
