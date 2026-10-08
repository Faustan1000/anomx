"""Matrix-style startup loading view."""

from __future__ import annotations

import curses
import math
import queue
import random
import threading
import time
from contextlib import suppress

from anomx.agent.helpers.platform_client import (
    heartbeat_platform_connection,
)
from anomx.agent.helpers.terminal import cell_width
from anomx.agent.ui.constants import (
    STARTUP_ANOMX_GLYPH,
    STARTUP_BRAND_REVEAL_SECONDS,
    STARTUP_COLUMN_WIDTH,
    STARTUP_FRAME_SECONDS,
    STARTUP_LINE_REVEAL_SECONDS,
    STARTUP_LOADING_SECONDS,
    STARTUP_MATRIX_ALPHABET,
    STARTUP_OVERLAY_DELAY_SECONDS,
    STARTUP_PHASE_SECONDS,
    STARTUP_REVEAL_SECONDS,
    STARTUP_WIPE_SECONDS,
)
from anomx.agent.ui.models import (
    CursesWindow,
    StartupPreparation,
)


class MatrixLoadingViewMixin:
    """Matrix-style startup loading view."""

    def _run_startup_loading(self, stdscr: CursesWindow) -> bool:
        """Show the startup matrix while the optional platform link warms up."""

        connection = self.home.platform_connection()
        results: queue.SimpleQueue[bool] = queue.SimpleQueue()
        worker: threading.Thread | None = None
        preparation_results: queue.SimpleQueue[StartupPreparation | None] = queue.SimpleQueue()
        preparation_worker: threading.Thread | None = None
        preparation_ready = not self._prepare_startup_during_loading
        self._startup_preparation = None

        frame = 0
        connected = False
        with suppress(curses.error):
            stdscr.nodelay(True)
        try:
            started_at = time.monotonic()
            now = started_at
            while (now - started_at) < STARTUP_LOADING_SECONDS or not preparation_ready:
                elapsed = now - started_at
                activity = self._startup_loading_activity(elapsed)
                if activity == "Connecting" and worker is None and connection is not None:

                    def run_heartbeat() -> None:
                        try:
                            results.put(heartbeat_platform_connection(self.home))
                        except Exception:
                            results.put(False)

                    worker = threading.Thread(target=run_heartbeat, daemon=True)
                    worker.start()
                if (
                    activity == "Screening"
                    and preparation_worker is None
                    and self._prepare_startup_during_loading
                ):

                    def run_startup_preparation() -> None:
                        try:
                            preparation_results.put(self._prepare_startup_state())
                        except Exception:
                            preparation_results.put(None)

                    preparation_worker = threading.Thread(
                        target=run_startup_preparation,
                        daemon=True,
                    )
                    preparation_worker.start()
                with suppress(queue.Empty):
                    connected = results.get_nowait() or connected
                if not preparation_ready:
                    with suppress(queue.Empty):
                        self._startup_preparation = preparation_results.get_nowait()
                        preparation_ready = True
                self._draw_startup_loading(
                    stdscr,
                    frame,
                    elapsed=now - started_at,
                    activity_text=activity,
                )
                frame += 1
                with suppress(curses.error):
                    stdscr.get_wch()
                time.sleep(STARTUP_FRAME_SECONDS)
                now = time.monotonic()
        finally:
            with suppress(curses.error):
                stdscr.nodelay(False)
        with suppress(queue.Empty):
            connected = results.get_nowait() or connected
        if not preparation_ready:
            with suppress(queue.Empty):
                self._startup_preparation = preparation_results.get_nowait()
                preparation_ready = True
        frame = self._run_startup_wipe(stdscr, frame)
        if worker is not None and connected:
            worker.join(timeout=0)
        if preparation_worker is not None and preparation_ready:
            preparation_worker.join(timeout=0)
        return connected

    def _startup_loading_activity(self, elapsed: float) -> str:
        if elapsed < STARTUP_PHASE_SECONDS:
            return "Booting"
        if elapsed < STARTUP_PHASE_SECONDS * 2:
            return "Connecting"
        return "Screening"

    def _draw_startup_loading(
        self,
        stdscr: CursesWindow,
        frame: int,
        *,
        elapsed: float | None = None,
        visible_rows: int | None = None,
        removal_progress: float = 0.0,
        show_overlays: bool | None = None,
        line_progress: float | None = None,
        brand_progress: float | None = None,
        activity_text: str = "",
    ) -> None:
        """Render the fullscreen alphanumeric startup matrix."""

        with suppress(curses.error):
            curses.curs_set(0)
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height <= 0 or width <= 0:
            return

        column_count = math.ceil(width / STARTUP_COLUMN_WIDTH)
        column_heights = self._startup_column_heights(
            column_count,
            height,
            elapsed,
            visible_rows,
        )
        removal_progress = min(1.0, max(0.0, removal_progress))
        line_reveal, brand_reveal = self._startup_overlay_progress(
            elapsed,
            line_progress,
            brand_progress,
        )
        overlays_visible = (
            all(column_height >= height for column_height in column_heights)
            and removal_progress == 0.0
            if show_overlays is None
            else show_overlays
        )

        rng = random.Random((frame + 1) * 104_729 + height * 8_191 + width * 193)
        matrix_attr = self._attr("matrix_dim")
        background_attr = self._attr("background")
        for y in range(height):
            line = self._startup_matrix_line(
                rng,
                y,
                width,
                column_heights,
                removal_progress,
            )
            row_attr = matrix_attr if line.strip() else background_attr
            with suppress(curses.error):
                stdscr.addnstr(y, 0, line, width, row_attr)

        if overlays_visible:
            self._draw_startup_function(
                stdscr,
                height,
                width,
                frame,
                reveal_progress=line_reveal,
                removal_progress=removal_progress,
            )
            self._draw_startup_brand(
                stdscr,
                height,
                width,
                frame,
                reveal_progress=brand_reveal,
                removal_progress=removal_progress,
            )
        if activity_text:
            x = max(0, (width - len(activity_text)) // 2)
            y = max(0, height - 2)
            self._add(stdscr, y, x, activity_text, cell_width(activity_text), self._attr("accent"))
        stdscr.refresh()

    def _run_startup_wipe(self, stdscr: CursesWindow, frame: int) -> int:
        height, _ = stdscr.getmaxyx()
        if height <= 0:
            return frame
        started_at = time.monotonic()
        now = started_at
        deadline = started_at + STARTUP_WIPE_SECONDS
        while now < deadline:
            progress = min(1.0, max(0.0, (now - started_at) / STARTUP_WIPE_SECONDS))
            self._draw_startup_loading(
                stdscr,
                frame,
                visible_rows=height,
                removal_progress=progress,
                show_overlays=True,
                line_progress=1.0,
                brand_progress=1.0,
            )
            frame += 1
            time.sleep(STARTUP_FRAME_SECONDS)
            now = time.monotonic()
        self._draw_startup_loading(
            stdscr,
            frame,
            visible_rows=height,
            removal_progress=1.0,
            show_overlays=True,
            line_progress=1.0,
            brand_progress=1.0,
        )
        return frame + 1

    def _startup_overlay_progress(
        self,
        elapsed: float | None,
        line_progress: float | None,
        brand_progress: float | None,
    ) -> tuple[float, float]:
        if line_progress is not None and brand_progress is not None:
            return (
                min(1.0, max(0.0, line_progress)),
                min(1.0, max(0.0, brand_progress)),
            )
        if elapsed is None:
            return (
                1.0 if line_progress is None else min(1.0, max(0.0, line_progress)),
                1.0 if brand_progress is None else min(1.0, max(0.0, brand_progress)),
            )
        overlay_elapsed = elapsed - STARTUP_REVEAL_SECONDS - STARTUP_OVERLAY_DELAY_SECONDS
        computed_line = min(1.0, max(0.0, overlay_elapsed / STARTUP_LINE_REVEAL_SECONDS))
        computed_brand = min(1.0, max(0.0, overlay_elapsed / STARTUP_BRAND_REVEAL_SECONDS))
        return (
            computed_line if line_progress is None else min(1.0, max(0.0, line_progress)),
            computed_brand if brand_progress is None else min(1.0, max(0.0, brand_progress)),
        )

    def _startup_column_heights(
        self,
        column_count: int,
        height: int,
        elapsed: float | None,
        visible_rows: int | None,
    ) -> tuple[int, ...]:
        if visible_rows is not None:
            visible_height = max(0, min(height, visible_rows))
            return tuple(visible_height for _ in range(column_count))
        if elapsed is None or elapsed >= STARTUP_REVEAL_SECONDS:
            return tuple(height for _ in range(column_count))
        return tuple(
            self._startup_column_height(column, height, elapsed) for column in range(column_count)
        )

    def _startup_column_height(self, column: int, height: int, elapsed: float) -> int:
        rng = random.Random((column + 1) * 7_919)
        start_delay = rng.uniform(0.0, STARTUP_REVEAL_SECONDS * 0.45)
        duration = rng.uniform(STARTUP_REVEAL_SECONDS * 0.35, STARTUP_REVEAL_SECONDS * 0.78)
        progress = min(1.0, max(0.0, (elapsed - start_delay) / duration))
        eased_progress = 1.0 - ((1.0 - progress) ** 2)
        return max(0, min(height, round(height * eased_progress)))

    def _startup_matrix_line(
        self,
        rng: random.Random,
        y: int,
        width: int,
        column_heights: tuple[int, ...],
        removal_progress: float,
    ) -> str:
        parts: list[str] = []
        for x in range(width):
            column = x // STARTUP_COLUMN_WIDTH
            column_height = column_heights[min(column, len(column_heights) - 1)]
            visible = y < column_height and not self._startup_cell_removed(
                x,
                y,
                removal_progress,
            )
            if visible:
                parts.append(rng.choice(STARTUP_MATRIX_ALPHABET))
            else:
                parts.append(" ")
        return "".join(parts)

    def _startup_cell_removed(
        self,
        x: int,
        y: int,
        removal_progress: float,
    ) -> bool:
        if removal_progress <= 0:
            return False
        if removal_progress >= 1:
            return True
        threshold = int(removal_progress * 10_000)
        return self._startup_cell_rank(x, y, 0xA11CE) < threshold

    def _startup_cell_rank(self, x: int, y: int, salt: int) -> int:
        value = ((x + 0x9E37_79B9) * 0x85EB_CA6B) & 0xFFFF_FFFF
        value ^= ((y + 0xC2B2_AE35) * 0x27D4_EB2D) & 0xFFFF_FFFF
        value ^= salt & 0xFFFF_FFFF
        value ^= value >> 16
        value = (value * 0x7FEB_352D) & 0xFFFF_FFFF
        value ^= value >> 15
        value = (value * 0x846C_A68B) & 0xFFFF_FFFF
        value ^= value >> 16
        return value % 10_000

    def _draw_startup_function(
        self,
        stdscr: CursesWindow,
        height: int,
        width: int,
        frame: int,
        *,
        reveal_progress: float,
        removal_progress: float,
    ) -> None:
        reveal_progress = min(1.0, max(0.0, reveal_progress))
        if height < 3 or width < 2 or reveal_progress <= 0:
            return
        attr = self._attr("matrix_function")
        char_rng = random.Random((frame // 3 + 1) * 65_537 + width * 97 + height)
        previous_y: int | None = None
        max_x = min(width - 1, math.floor((width - 1) * reveal_progress))
        for x in range(max_x + 1):
            y = self._startup_function_y(x, width, height, frame)
            ys: tuple[int, ...]
            if previous_y is None:
                ys = (y,)
            else:
                start = min(previous_y, y)
                stop = max(previous_y, y)
                ys = tuple(range(start, stop + 1))
            for point_y in ys:
                if self._startup_cell_removed(x, point_y, removal_progress):
                    continue
                self._add(
                    stdscr,
                    point_y,
                    x,
                    char_rng.choice(STARTUP_MATRIX_ALPHABET),
                    1,
                    attr,
                )
            previous_y = y

    def _startup_function_y(self, x: int, width: int, height: int, frame: int) -> int:
        t = x / max(1, width - 1)
        slow_time = frame * 0.032
        wave = (
            math.sin((t * math.tau * 1.17) + (slow_time * 0.83)) * 0.38
            + math.sin((t * math.tau * 2.71) - (slow_time * 0.47) + 1.9) * 0.25
            + math.sin((t * math.tau * 4.63) + (slow_time * 0.29) + 0.7) * 0.17
            + math.sin((t * math.tau * 0.53) - (slow_time * 0.19) + 2.6) * 0.20
        )
        center = (height - 1) / 2 + math.sin(slow_time * 0.41) * max(1.0, height * 0.08)
        amplitude = max(1.0, (height - 4) * 0.36)
        return max(0, min(height - 1, round(center - wave * amplitude)))

    def _draw_startup_brand(
        self,
        stdscr: CursesWindow,
        height: int,
        width: int,
        frame: int,
        *,
        reveal_progress: float,
        removal_progress: float,
    ) -> None:
        reveal_progress = min(1.0, max(0.0, reveal_progress))
        if reveal_progress <= 0:
            return
        scale_x = 2 if width >= 90 else 1
        glyph_width = len(STARTUP_ANOMX_GLYPH[0]) * scale_x
        dot_gap = 2 * scale_x
        dot_width = scale_x
        brand_width = glyph_width + dot_gap + dot_width
        y = 2 if height >= 14 else 0
        x = max(0, (width - brand_width) // 2)
        attr = self._attr("matrix_brand")
        rng = random.Random((frame + 1) * 131_071 + width * 17 + height * 31)
        for row_index, row in enumerate(STARTUP_ANOMX_GLYPH):
            draw_y = y + row_index
            if draw_y >= height:
                return
            draw_x = x
            for marker in row:
                if marker == "#":
                    for offset in range(scale_x):
                        cell_x = draw_x + offset
                        if self._startup_brand_cell_hidden(
                            cell_x,
                            draw_y,
                            reveal_progress,
                            removal_progress,
                        ):
                            continue
                        self._add(
                            stdscr,
                            draw_y,
                            cell_x,
                            rng.choice(STARTUP_MATRIX_ALPHABET),
                            1,
                            attr,
                        )
                draw_x += scale_x
        dot_y = y + len(STARTUP_ANOMX_GLYPH) - 1
        dot_x = x + glyph_width + dot_gap
        if dot_y < height:
            for offset in range(dot_width):
                cell_x = dot_x + offset
                if cell_x >= width or self._startup_brand_cell_hidden(
                    cell_x,
                    dot_y,
                    reveal_progress,
                    removal_progress,
                ):
                    continue
                self._add(
                    stdscr,
                    dot_y,
                    cell_x,
                    rng.choice(STARTUP_MATRIX_ALPHABET),
                    1,
                    self._attr("brand_dot"),
                )

    def _startup_brand_cell_hidden(
        self,
        x: int,
        y: int,
        reveal_progress: float,
        removal_progress: float,
    ) -> bool:
        if self._startup_cell_removed(x, y, removal_progress):
            return True
        threshold = int(reveal_progress * 10_000)
        return self._startup_cell_rank(x, y, 0xB4A3D) >= threshold
