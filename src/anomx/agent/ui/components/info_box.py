"""Shared shell, header, and information-box drawing primitives."""

from __future__ import annotations

import curses
from contextlib import suppress

from anomx import __version__
from anomx.agent.helpers.state import (
    PlanStep,
)
from anomx.agent.ui.constants import (
    AGENT_DESCRIPTOR,
    BRAND_DOT,
    BRAND_NAME,
)
from anomx.agent.ui.models import (
    CursesWindow,
    MenuChoice,
    SessionMouseAction,
)


class InfoBoxComponentMixin:
    """Shared shell, header, and information-box drawing primitives."""

    def _draw_shell(
        self,
        stdscr: CursesWindow,
        title: str,
        subtitle: str | tuple[str, ...] = "",
        plan_steps: tuple[PlanStep, ...] = (),
        header_meta: str = "",
        plan_expanded: bool = False,
        title_suffix: str = "",
    ) -> tuple[int, int]:
        with suppress(curses.error):
            curses.curs_set(0)
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        self._draw_header_box(
            stdscr,
            title,
            subtitle,
            plan_steps,
            header_meta,
            plan_expanded,
            title_suffix,
        )
        return height, width

    def _draw_overlay(
        self,
        stdscr: CursesWindow,
        title: str,
        subtitle: str = "",
        body_lines: tuple[str, ...] = (),
        post_body_lines: tuple[str, ...] = (),
        choices: tuple[MenuChoice, ...] = (),
        selected: int = 0,
        input_value: str = "",
        input_cursor: int = 0,
        editor_text: str = "",
        editor_cursor: int = -1,
        editor_width_override: int = 0,
        footer: str = "",
        show_input_cursor: bool = False,
    ) -> tuple[int, int, int, int]:
        """Draw a centered full-screen overlay.

        Returns (panel_top, panel_left, content_width, content_height)."""
        height, width = stdscr.getmaxyx()
        cursor_attr = self._attr("cursor")
        muted_attr = self._attr("muted")

        stdscr.erase()
        for y in range(height):
            self._add(stdscr, y, 0, " " * max(1, width), width, self._attr("background"))

        content_left = 4 if width >= 16 else 0
        content_width = max(1, width - (content_left * 2))
        subtitle_line_count = len(self._header_subtitle_lines(subtitle))
        self._draw_header_box(stdscr, title, subtitle)
        footer_y = max(0, height - 1)
        header_bottom = self._header_bottom((), subtitle_line_count, False)
        body_top = min(max(0, footer_y - 1), header_bottom + 2)
        body_bottom = max(body_top, footer_y - 1)
        available_height = max(1, body_bottom - body_top + 1)
        max_content_height = max(1, available_height - 3)
        body_count = len(body_lines)
        if editor_text and editor_cursor >= 0:
            editor_width = editor_width_override or content_width
            disp_lines = self._work_box_content_lines(editor_text, editor_width)
            editor_lines = len(disp_lines)
            desired_content_height = max(4, min(editor_lines, max_content_height))
        elif choices:
            desired_content_height = body_count + len(choices) + len(post_body_lines)
            desired_content_height = max(4, min(desired_content_height, max_content_height))
        elif body_lines:
            desired_content_height = max(2, min(body_count, max_content_height))
        elif input_value or input_cursor or show_input_cursor:
            desired_content_height = 1
        else:
            desired_content_height = max(4, max_content_height)
        block_height = min(available_height, desired_content_height)
        panel_top = body_top + max(0, (available_height - block_height) // 2)

        content_y = panel_top
        content_height = max(1, min(desired_content_height, body_bottom - content_y + 1))

        if editor_text and editor_cursor >= 0:
            visible_height = max(1, content_height)
            cursor_display_line = self._cursor_display_position(
                editor_text,
                editor_cursor,
                editor_width,
            )
            scroll_offset = max(0, cursor_display_line - visible_height + 1)
            for i in range(visible_height):
                doc_line = scroll_offset + i
                if doc_line < len(disp_lines):
                    self._add(
                        stdscr,
                        content_y + i,
                        content_left,
                        disp_lines[doc_line],
                        editor_width,
                    )
                else:
                    self._add(stdscr, content_y + i, content_left, " " * editor_width, editor_width)
            cursor_visible = cursor_display_line - scroll_offset
            if 0 <= cursor_visible < visible_height:
                cursor_col = self._cursor_column_in_display_line(
                    editor_text,
                    editor_cursor,
                    editor_width,
                )
                editor_line = disp_lines[cursor_display_line]
                c = editor_line[cursor_col] if cursor_col < len(editor_line) else " "
                self._add(
                    stdscr,
                    content_y + cursor_visible,
                    content_left + cursor_col,
                    c,
                    1,
                    cursor_attr,
                )
        elif choices:
            y = content_y
            for line in body_lines[:content_height]:
                self._add(stdscr, y, content_left, line, content_width, self._attr("light"))
                y += 1
            remaining_height = max(1, content_height - (y - content_y))
            longest_label = max(
                (len(choice.label) for choice in choices if choice.selectable),
                default=0,
            )
            label_width = min(
                38,
                max(14, content_width // 3),
                max(28, longest_label),
            )
            detail_x = content_left + label_width + 3
            detail_width = max(1, content_width - label_width - 3)
            selected = max(0, min(selected, len(choices) - 1))
            show_counts = len(choices) > remaining_height and remaining_height >= 3
            visible_rows = max(1, remaining_height - 2 if show_counts else remaining_height)
            max_offset = max(0, len(choices) - visible_rows)
            offset = min(max(0, selected - visible_rows + 1), max_offset)
            if show_counts:
                self._add(
                    stdscr,
                    y,
                    content_left,
                    f"↑ {offset} more above",
                    content_width,
                    self._attr("light"),
                )
                y += 1
            for choice_index in range(offset, min(len(choices), offset + visible_rows)):
                choice = choices[choice_index]
                if not choice.selectable:
                    if choice.label:
                        self._add(
                            stdscr,
                            y,
                            content_left + 2,
                            choice.label,
                            content_width - 2,
                            muted_attr,
                        )
                    y += 1
                    continue
                selected_row = choice_index == selected
                attr = self._attr("accent") if selected_row else curses.A_NORMAL
                marker = "›" if selected_row else " "
                self._add(stdscr, y, content_left, marker, 1, attr)
                self._add(stdscr, y, content_left + 2, choice.label, label_width, attr)
                separator_attr = self._attr("accent") if selected_row else muted_attr
                self._add(stdscr, y, content_left + label_width + 1, "│", 1, separator_attr)
                detail = input_value if selected_row and show_input_cursor else choice.detail
                detail_attr = attr if selected_row else muted_attr
                self._add(stdscr, y, detail_x, detail, detail_width, detail_attr)
                if selected_row and show_input_cursor:
                    self._paint_visual_cursor(stdscr, y, detail_x, detail, input_cursor)
                y += 1
            if show_counts:
                more_below = max(0, len(choices) - (offset + visible_rows))
                self._add(
                    stdscr,
                    y,
                    content_left,
                    f"↓ {more_below} more below",
                    content_width,
                    self._attr("light"),
                )
                y += 1
            for line in post_body_lines:
                if y > content_y + content_height - 1:
                    break
                if line:
                    attr = self._attr("danger") if line.startswith("!") else self._attr("light")
                    self._add(stdscr, y, content_left, line, content_width, attr)
                y += 1
        elif body_lines:
            for i, line in enumerate(body_lines[:content_height]):
                self._add(stdscr, content_y + i, content_left, line, content_width)
        elif input_value is not None:
            display_value = input_value
            self._add(
                stdscr,
                content_y,
                content_left,
                display_value,
                content_width,
                curses.A_NORMAL,
            )
            self._paint_visual_cursor(stdscr, content_y, content_left, display_value, input_cursor)

        self._add(
            stdscr,
            footer_y,
            0,
            " " * max(1, width),
            width,
            self._attr("selected"),
        )
        if footer:
            self._add(stdscr, footer_y, content_left, footer, content_width, self._attr("selected"))

        stdscr.refresh()
        return panel_top, content_left, content_width, content_height

    def _clear_row(self, stdscr: CursesWindow, y: int) -> None:
        _, width = stdscr.getmaxyx()
        self._add(stdscr, y, 0, " " * max(1, width), width, self._attr("background"))

    def _draw_header_box(
        self,
        stdscr: CursesWindow,
        title: str,
        subtitle: str | tuple[str, ...] = "",
        plan_steps: tuple[PlanStep, ...] = (),
        header_meta: str = "",
        plan_expanded: bool = False,
        title_suffix: str = "",
    ) -> None:
        _, width = stdscr.getmaxyx()
        right_text = self._header_right_text(header_meta)
        top = 1
        subtitle_lines = self._header_subtitle_lines(subtitle)
        bottom = self._header_bottom(plan_steps, len(subtitle_lines), plan_expanded)
        horizontal = "─" * max(1, width - 6)
        self._add(stdscr, top, 2, f"╭{horizontal}╮", width - 4, self._attr("accent"))
        for y in range(top + 1, bottom):
            self._add(stdscr, y, 2, "│", 1, self._attr("accent"))
            self._add(stdscr, y, max(2, width - 3), "│", 1, self._attr("accent"))
        self._add(stdscr, bottom, 2, f"╰{horizontal}╯", width - 4, self._attr("accent"))

        brand = BRAND_NAME
        descriptor = AGENT_DESCRIPTOR
        right_text = self._fit_header_right_text(right_text, max(1, width - 8))
        right_x = max(4, width - len(right_text) - 5)
        brand_dot_x = 4 + len(brand)
        descriptor_x = brand_dot_x + len(BRAND_DOT) + 2
        self._add(stdscr, top + 1, 4, brand, width - 8, self._attr("accent"))
        self._add(
            stdscr,
            top + 1,
            brand_dot_x,
            BRAND_DOT,
            1,
            self._attr("brand_dot"),
        )
        self._add(
            stdscr,
            top + 1,
            descriptor_x,
            descriptor,
            max(1, width - descriptor_x - 4),
            self._attr("light"),
        )
        self._add(
            stdscr,
            top + 1,
            right_x,
            right_text,
            len(right_text),
            self._attr("light"),
        )
        self._draw_header_title(
            stdscr,
            top + 2,
            4,
            title,
            plan_steps,
            plan_expanded,
            width - 8,
            title_suffix,
        )
        for index, line in enumerate(subtitle_lines):
            self._add(
                stdscr,
                top + 3 + index,
                4,
                line,
                width - 8,
                self._attr("light"),
            )
        if plan_steps and plan_expanded:
            plan_start_y = top + 4 + len(subtitle_lines)
            for index, step in enumerate(plan_steps):
                y = plan_start_y + index
                checkbox = "☑" if step.is_done else "☐"
                title_text = self._strike_text(step.title) if step.is_done else step.title
                attr = self._attr("light") if step.is_done else self._attr("bold")
                self._add(
                    stdscr,
                    y,
                    4,
                    f"{checkbox} {title_text}",
                    width - 8,
                    attr,
                )

    def _header_right_text(self, header_meta: str = "") -> str:
        version = f"v{__version__}"
        meta = header_meta.strip()
        return f"{meta} · {version}" if meta else version

    def _fit_header_right_text(self, text: str, width: int) -> str:
        safe_width = max(1, width)
        if len(text) <= safe_width:
            return text
        if safe_width <= 1:
            return text[:safe_width]
        return f"…{text[-(safe_width - 1) :]}"

    def _header_subtitle_lines(self, subtitle: str | tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(subtitle, str):
            return (subtitle,) if subtitle else ()
        return tuple(line for line in subtitle if line)

    def _header_bottom(
        self,
        plan_steps: tuple[PlanStep, ...] = (),
        subtitle_line_count: int = 0,
        plan_expanded: bool = False,
    ) -> int:
        base_bottom = 5 + max(0, subtitle_line_count - 1)
        if not plan_steps or not plan_expanded:
            return base_bottom
        return base_bottom + 1 + len(plan_steps)

    def _session_body_top(
        self,
        plan_steps: tuple[PlanStep, ...] = (),
        subtitle_line_count: int = 1,
        plan_expanded: bool = False,
    ) -> int:
        return self._header_bottom(plan_steps, subtitle_line_count, plan_expanded) + 2

    def _paint_visual_cursor(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        cursor: int,
    ) -> None:
        """Overwrite one cell at (y, x + cursor) with the cursor attribute."""
        clamped = min(cursor, len(text))
        char = text[clamped] if clamped < len(text) else " "
        self._add(stdscr, y, x + clamped, char, 1, self._attr("cursor"))

    def _draw_header_title(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        title: str,
        plan_steps: tuple[PlanStep, ...],
        plan_expanded: bool,
        width: int,
        title_suffix: str = "",
    ) -> None:
        del plan_expanded
        title_text = self._header_title_text(title, plan_steps, width, title_suffix)
        self._add(stdscr, y, x, title_text, width, self._attr("bold"))
        if plan_steps and title_text:
            self._add_click_target(
                y,
                SessionMouseAction(
                    "toggle_plan",
                    0,
                    x_start=x,
                    x_end=x + min(len(title_text), width),
                ),
            )

    def _header_title_text(
        self,
        title: str,
        plan_steps: tuple[PlanStep, ...],
        width: int,
        title_suffix: str = "",
    ) -> str:
        text = title
        if plan_steps:
            text = f"{text} › {self._current_plan_step_title(plan_steps)}"
        suffix = title_suffix.strip()
        if suffix:
            text = f"{text} {suffix}"
        safe_width = max(1, width)
        if len(text) > safe_width:
            if safe_width <= 1:
                text = text[:safe_width]
            elif safe_width <= 4:
                text = f"{text[: safe_width - 1]}…"
            else:
                text = f"{text[: safe_width - 2].rstrip()} …"
        return text

    def _current_plan_step_title(self, plan_steps: tuple[PlanStep, ...]) -> str:
        for step in plan_steps:
            if not step.is_done:
                return step.title
        return plan_steps[-1].title if plan_steps else "Plan complete"

    def _strike_text(self, text: str) -> str:
        return "".join(
            f"{character}\u0336" if character != " " else character for character in text
        )
