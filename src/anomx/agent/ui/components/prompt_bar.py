"""Prompt bar rendering, prompt geometry, and prompt mouse targeting."""

from __future__ import annotations

import curses
from collections.abc import Mapping, Sequence
from contextlib import suppress

from anomx.agent.ui.constants import (
    RAW_MOUSE_RE,
    RUNNING_NOTICE,
)
from anomx.agent.ui.models import (
    CommandSpec,
    CursesWindow,
    MenuChoice,
    PromptLayout,
    PromptPasteSpan,
    SessionMouseAction,
    SessionTextSelection,
)


class PromptBarComponentMixin:
    """Prompt bar rendering, prompt geometry, and prompt mouse targeting."""

    def _draw_prompt_bar(
        self,
        stdscr: CursesWindow,
        input_text: str,
        cursor: int,
        notice: str = "",
        notice_role: str = "light",
        file_references: Mapping[str, str] | None = None,
        *,
        draw_top_rule: bool = True,
        hint_suffix: str = "",
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> None:
        display_text, pasted_display_spans = self._prompt_display_text_and_spans(
            input_text,
            pasted_spans,
        )
        display_cursor = self._prompt_display_cursor(input_text, cursor, pasted_spans)
        layout = self._prompt_layout(stdscr, input_text, pasted_spans=pasted_spans)
        _, width = stdscr.getmaxyx()
        panel_width = max(1, width - 4)
        clear_top = layout.top_line if draw_top_rule else layout.prompt_line
        for y in range(clear_top, layout.hint_line + 1):
            self._clear_row(stdscr, y)
        if draw_top_rule:
            self._add(
                stdscr,
                layout.top_line,
                2,
                "─" * panel_width,
                panel_width,
                self._attr("light"),
            )
        running_placeholder = "Send a message to steer anomx..."
        visible_text = display_text or (
            running_placeholder if notice == RUNNING_NOTICE else self._prompt_placeholder
        )
        attr = self._attr("bold") if input_text else self._attr("light")
        lines = self._prompt_lines(visible_text, layout.input_width)
        view_start = self._prompt_view_start(display_text, display_cursor, layout)
        visible_lines = lines[view_start : view_start + layout.prompt_height]
        for offset in range(layout.prompt_height):
            y = layout.prompt_line + offset
            marker = "›" if offset == 0 else " "
            self._add(stdscr, y, 4, marker, 1, self._attr("accent"))
            line = visible_lines[offset] if offset < len(visible_lines) else ""
            if input_text and (file_references or pasted_display_spans):
                line_start = (view_start + offset) * layout.input_width
                self._draw_prompt_text_line(
                    stdscr,
                    y,
                    layout.input_x,
                    line,
                    line_start,
                    layout.input_width,
                    attr,
                    file_references,
                    pasted_display_spans,
                )
            else:
                self._add(stdscr, y, layout.input_x, line, layout.input_width, attr)
        self._draw_prompt_cursor_cell(
            stdscr,
            layout,
            display_text,
            display_cursor,
            visible_lines,
            view_start,
        )
        self._add(
            stdscr,
            layout.bottom_line,
            2,
            "─" * panel_width,
            panel_width,
            self._attr("light"),
        )
        show_notice = bool(notice and notice != RUNNING_NOTICE)
        hint_text = notice if show_notice else self.agent_mode.prompt_hint
        hint_attr = notice_role if show_notice else self._mode_hint_attr_name()
        hint_width = layout.input_width + 2
        self._add(
            stdscr,
            layout.hint_line,
            4,
            hint_text,
            hint_width,
            self._attr(hint_attr),
        )
        if not show_notice and hint_suffix:
            suffix_x = 4 + min(len(hint_text), hint_width)
            suffix_width = max(0, hint_width - min(len(hint_text), hint_width))
            parts = [hint_suffix]
            # if self._sandbox_is_active():
            #     parts.append("sandbox")
            combined = " · ".join(parts)
            if suffix_width:
                self._add(
                    stdscr,
                    layout.hint_line,
                    suffix_x,
                    combined,
                    suffix_width,
                    self._attr("light"),
                )
        elif not show_notice and self._sandbox_is_active():
            suffix_x = 4 + min(len(hint_text), hint_width)
            suffix_width = max(0, hint_width - min(len(hint_text), hint_width))
            if suffix_width:
                self._add(
                    stdscr,
                    layout.hint_line,
                    suffix_x,
                    "",
                    suffix_width,
                    self._attr("light"),
                )
        with suppress(curses.error):
            curses.curs_set(0)

    def _draw_prompt_cursor_cell(
        self,
        stdscr: CursesWindow,
        layout: PromptLayout,
        input_text: str,
        cursor: int,
        visible_lines: list[str],
        view_start: int,
    ) -> None:
        cursor_line, cursor_column = self._prompt_cursor_position(
            input_text,
            cursor,
            layout.input_width,
        )
        visible_cursor_line = cursor_line - view_start
        if not 0 <= visible_cursor_line < layout.prompt_height:
            return
        cursor_y = layout.prompt_line + visible_cursor_line
        cursor_x = layout.input_x + cursor_column
        line = (
            visible_lines[visible_cursor_line] if visible_cursor_line < len(visible_lines) else ""
        )
        character = line[cursor_column] if cursor_column < len(line) and input_text else " "
        self._add(stdscr, cursor_y, cursor_x, character, 1, self._attr("cursor"))

    def _draw_prompt_text_line(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        line_start: int,
        width: int,
        base_attr: int,
        file_references: Mapping[str, str] | None = None,
        pasted_display_spans: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        if not text:
            return
        spans = self._merge_spans(
            [
                *(
                    self._file_reference_spans(text, line_start, file_references)
                    if file_references
                    else []
                ),
                *self._line_relative_spans(
                    pasted_display_spans or (),
                    line_start,
                    len(text),
                ),
            ],
        )
        cursor = 0
        for start, end in spans:
            if start > cursor:
                self._add(stdscr, y, x + cursor, text[cursor:start], width - cursor, base_attr)
            self._add(stdscr, y, x + start, text[start:end], width - start, self._attr("accent"))
            cursor = end
        if cursor < len(text):
            self._add(stdscr, y, x + cursor, text[cursor:], width - cursor, base_attr)

    def _line_relative_spans(
        self,
        spans: Sequence[tuple[int, int]],
        line_start: int,
        line_length: int,
    ) -> list[tuple[int, int]]:
        line_end = line_start + line_length
        relative: list[tuple[int, int]] = []
        for start, end in spans:
            if start >= line_end or end <= line_start:
                continue
            relative.append((max(0, start - line_start), min(line_length, end - line_start)))
        return relative

    def _file_reference_spans(
        self,
        text: str,
        line_start: int,
        file_references: Mapping[str, str],
    ) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        full_start = line_start
        full_end = line_start + len(text)
        for label in file_references:
            if not label:
                continue
            for match in self._file_reference_label_pattern(label).finditer(text):
                absolute_start = full_start + match.start()
                absolute_end = full_start + match.end()
                if absolute_start < full_end and absolute_end > full_start:
                    spans.append((match.start(), match.end()))
        return self._merge_spans(spans)

    def _merge_spans(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not spans:
            return []
        merged: list[tuple[int, int]] = []
        for start, end in sorted(spans):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                previous_start, previous_end = merged[-1]
                merged[-1] = (previous_start, max(previous_end, end))
        return merged

    def _prompt_layout(
        self,
        stdscr: CursesWindow,
        input_text: str = "",
        *,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> PromptLayout:
        height, width = stdscr.getmaxyx()
        input_width = max(1, width - 10)
        max_prompt_height = max(1, height // 4)
        display_text = self._prompt_display_text(input_text, pasted_spans)
        prompt_line_count = len(self._prompt_lines(display_text, input_width)) if input_text else 1
        prompt_height = max(1, min(max_prompt_height, prompt_line_count))
        bottom_line = max(0, height - 2)
        prompt_line = max(0, bottom_line - prompt_height)
        top_line = max(0, prompt_line - 1)
        return PromptLayout(
            top_line=top_line,
            prompt_line=prompt_line,
            bottom_line=bottom_line,
            hint_line=max(0, height - 1),
            input_x=6,
            input_width=input_width,
            prompt_height=prompt_height,
        )

    def _prompt_display_text(
        self,
        input_text: str,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> str:
        display_text, _spans = self._prompt_display_text_and_spans(input_text, pasted_spans)
        return display_text

    def _prompt_display_text_and_spans(
        self,
        input_text: str,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> tuple[str, list[tuple[int, int]]]:
        spans = self._normalized_prompt_paste_spans(input_text, pasted_spans)
        if not input_text or not spans:
            return input_text, []
        parts: list[str] = []
        display_spans: list[tuple[int, int]] = []
        cursor = 0
        for span in spans:
            if span.start > cursor:
                parts.append(input_text[cursor : span.start])
            marker = self._prompt_paste_marker(self._prompt_paste_character_count(span))
            display_start = sum(len(part) for part in parts)
            parts.append(marker)
            display_spans.append((display_start, display_start + len(marker)))
            cursor = span.end
        if cursor < len(input_text):
            parts.append(input_text[cursor:])
        return "".join(parts), display_spans

    def _prompt_paste_marker(self, length: int) -> str:
        noun = "character" if length == 1 else "characters"
        return f"[{max(0, length)}\u00a0pasted {noun}]"

    def _prompt_paste_character_count(self, span: PromptPasteSpan) -> int:
        return span.character_count if span.character_count is not None else span.end - span.start

    def _normalized_prompt_paste_spans(
        self,
        input_text: str,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> list[PromptPasteSpan]:
        if not input_text or not pasted_spans:
            return []
        normalized: list[PromptPasteSpan] = []
        text_length = len(input_text)
        for span in sorted(pasted_spans, key=lambda item: (item.start, item.end)):
            start = max(0, min(text_length, span.start))
            end = max(start, min(text_length, span.end))
            if start == end:
                continue
            if normalized and start <= normalized[-1].end:
                previous = normalized[-1]
                character_count = None
                if previous.character_count is not None or span.character_count is not None:
                    character_count = (
                        previous.character_count or previous.end - previous.start
                    ) + (span.character_count or end - start)
                normalized[-1] = PromptPasteSpan(
                    previous.start,
                    max(previous.end, end),
                    character_count,
                )
            else:
                normalized.append(PromptPasteSpan(start, end, span.character_count))
        return normalized

    def _prompt_display_cursor(
        self,
        input_text: str,
        cursor: int,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> int:
        if not input_text:
            return 0
        spans = self._normalized_prompt_paste_spans(input_text, pasted_spans)
        bounded_cursor = max(0, min(cursor, len(input_text)))
        if not spans:
            return bounded_cursor
        display_cursor = 0
        real_cursor = 0
        for span in spans:
            if bounded_cursor <= span.start:
                return display_cursor + (bounded_cursor - real_cursor)
            display_cursor += span.start - real_cursor
            marker_length = len(self._prompt_paste_marker(self._prompt_paste_character_count(span)))
            if bounded_cursor <= span.end:
                if bounded_cursor == span.start:
                    return display_cursor
                return display_cursor + marker_length
            display_cursor += marker_length
            real_cursor = span.end
        return display_cursor + (bounded_cursor - real_cursor)

    def _prompt_real_cursor(
        self,
        input_text: str,
        display_cursor: int,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> int:
        if not input_text:
            return 0
        spans = self._normalized_prompt_paste_spans(input_text, pasted_spans)
        bounded_display_cursor = max(0, display_cursor)
        if not spans:
            return min(len(input_text), bounded_display_cursor)
        visible_cursor = 0
        real_cursor = 0
        for span in spans:
            text_segment_length = span.start - real_cursor
            if bounded_display_cursor <= visible_cursor + text_segment_length:
                return min(len(input_text), real_cursor + (bounded_display_cursor - visible_cursor))
            visible_cursor += text_segment_length
            marker_length = len(self._prompt_paste_marker(self._prompt_paste_character_count(span)))
            if bounded_display_cursor <= visible_cursor + marker_length:
                return span.start if bounded_display_cursor == visible_cursor else span.end
            visible_cursor += marker_length
            real_cursor = span.end
        return min(len(input_text), real_cursor + (bounded_display_cursor - visible_cursor))

    def _prompt_lines(self, text: str, width: int) -> list[str]:
        if not text:
            return [""]
        safe_width = max(1, width)
        return [text[index : index + safe_width] for index in range(0, len(text), safe_width)]

    def _prompt_view_start(
        self,
        input_text: str,
        cursor: int,
        layout: PromptLayout,
    ) -> int:
        if not input_text:
            return 0
        cursor_line, _ = self._prompt_cursor_position(input_text, cursor, layout.input_width)
        line_count = len(self._prompt_lines(input_text, layout.input_width))
        max_start = max(0, line_count - layout.prompt_height)
        return max(0, min(max_start, cursor_line - layout.prompt_height + 1))

    def _prompt_cursor_position(self, input_text: str, cursor: int, width: int) -> tuple[int, int]:
        safe_width = max(1, width)
        bounded_cursor = max(0, min(cursor, len(input_text)))
        if bounded_cursor > 0 and bounded_cursor % safe_width == 0:
            return (bounded_cursor - 1) // safe_width, safe_width - 1
        return bounded_cursor // safe_width, bounded_cursor % safe_width

    def _move_prompt_cursor_row(
        self,
        stdscr: CursesWindow,
        input_text: str,
        cursor: int,
        direction: int,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> int:
        if not input_text:
            return cursor
        layout = self._prompt_layout(stdscr, input_text, pasted_spans=pasted_spans)
        display_text = self._prompt_display_text(input_text, pasted_spans)
        display_cursor = self._prompt_display_cursor(input_text, cursor, pasted_spans)
        moved_display_cursor = self._prompt_cursor_for_row_delta(
            display_text,
            display_cursor,
            layout.input_width,
            direction,
        )
        return self._prompt_real_cursor(input_text, moved_display_cursor, pasted_spans)

    def _prompt_cursor_for_row_delta(
        self,
        input_text: str,
        cursor: int,
        width: int,
        direction: int,
    ) -> int:
        if not input_text or direction == 0:
            return cursor
        safe_width = max(1, width)
        line_count = len(self._prompt_lines(input_text, safe_width))
        current_line, current_column = self._prompt_cursor_position(
            input_text,
            cursor,
            safe_width,
        )
        target_line = current_line + direction
        if target_line < 0 or target_line >= line_count:
            return cursor
        line_start = target_line * safe_width
        line_end = min(len(input_text), line_start + safe_width)
        target_column = min(current_column, max(0, line_end - line_start))
        return min(len(input_text), line_start + target_column)

    def _previous_prompt_word(self, input_text: str, cursor: int) -> int:
        index = max(0, min(cursor, len(input_text)))
        while index > 0 and not self._is_prompt_word_char(input_text[index - 1]):
            index -= 1
        while index > 0 and self._is_prompt_word_char(input_text[index - 1]):
            index -= 1
        return index

    def _next_prompt_word(self, input_text: str, cursor: int) -> int:
        length = len(input_text)
        index = max(0, min(cursor, length))
        while index < length and not self._is_prompt_word_char(input_text[index]):
            index += 1
        while index < length and self._is_prompt_word_char(input_text[index]):
            index += 1
        return index

    def _delete_previous_prompt_word(self, input_text: str, cursor: int) -> tuple[str, int]:
        word_start = self._previous_prompt_word(input_text, cursor)
        return input_text[:word_start] + input_text[cursor:], word_start

    def _is_prompt_word_char(self, char: str) -> bool:
        return char.isalnum() or char == "_"

    def _session_mouse_action(
        self,
        stdscr: CursesWindow,
        input_text: str,
        command_suggestions: list[CommandSpec],
        command_selected: int = 0,
        file_suggestions: list[MenuChoice] | None = None,
        file_selected: int = 0,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> SessionMouseAction | None:
        with suppress(curses.error):
            _, x, y, _, button_state = curses.getmouse()
            return self._session_mouse_action_from_event(
                stdscr,
                input_text,
                command_suggestions,
                command_selected,
                file_suggestions,
                file_selected,
                x,
                y,
                button_state,
                pasted_spans,
            )
        return None

    def _raw_mouse_action(
        self,
        key: str | int,
        stdscr: CursesWindow,
        input_text: str,
        command_suggestions: list[CommandSpec],
        command_selected: int = 0,
        file_suggestions: list[MenuChoice] | None = None,
        file_selected: int = 0,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> SessionMouseAction | None:
        event = self._raw_mouse_event(key)
        if event is None:
            return None
        x, y, button_state = event
        return self._session_mouse_action_from_event(
            stdscr,
            input_text,
            command_suggestions,
            command_selected,
            file_suggestions,
            file_selected,
            x,
            y,
            button_state,
            pasted_spans,
        )

    def _raw_mouse_event(self, key: str | int) -> tuple[int, int, int] | None:
        if not isinstance(key, str):
            return None
        match = RAW_MOUSE_RE.match(key)
        if match is None:
            return None
        button_code = int(match.group("button"))
        x = max(0, int(match.group("x")) - 1)
        y = max(0, int(match.group("y")) - 1)
        button_state = self._raw_mouse_button_state(button_code, match.group("suffix"))
        return x, y, button_state

    def _raw_mouse_button_state(self, button_code: int, suffix: str) -> int:
        if button_code == 64:
            return getattr(curses, "BUTTON4_PRESSED", 0)
        if button_code == 65:
            return getattr(curses, "BUTTON5_PRESSED", 0)
        if suffix == "m":
            return getattr(curses, "BUTTON1_RELEASED", 0)
        state = getattr(curses, "BUTTON1_PRESSED", 0)
        if button_code & 32:
            state |= getattr(curses, "REPORT_MOUSE_POSITION", 0)
        return state

    def _session_mouse_action_from_event(
        self,
        stdscr: CursesWindow,
        input_text: str,
        command_suggestions: list[CommandSpec],
        command_selected: int,
        file_suggestions: list[MenuChoice] | None,
        file_selected: int,
        x: int,
        y: int,
        button_state: int,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> SessionMouseAction | None:
        wheel_up = getattr(curses, "BUTTON4_PRESSED", 0)
        wheel_down = getattr(curses, "BUTTON5_PRESSED", 0)
        if wheel_up and button_state & wheel_up:
            activity_action = self._activity_scroll_action_at(y, 1)
            if activity_action is not None:
                return activity_action
            return SessionMouseAction("scroll", 1)
        if wheel_down and button_state & wheel_down:
            activity_action = self._activity_scroll_action_at(y, -1)
            if activity_action is not None:
                return activity_action
            return SessionMouseAction("scroll", -1)

        selection_action = self._session_selection_mouse_action(x, y, button_state)
        if selection_action is not None:
            return selection_action

        click_action = self._click_target_action_at(x, y, button_state)
        if click_action is not None:
            return click_action

        active_file_suggestions = file_suggestions or []
        if active_file_suggestions and self._is_left_click(button_state):
            panel = self._file_reference_bottom_panel(
                active_file_suggestions,
                file_selected,
            )
            if panel is not None:
                index = self._bottom_panel_mouse_choice_at(
                    stdscr,
                    panel,
                    y,
                    input_text,
                    pasted_spans,
                )
                if index is not None:
                    return SessionMouseAction("file_reference", index)

        if command_suggestions and self._is_left_click(button_state):
            panel = self._command_bottom_panel(
                command_suggestions,
                selected=command_selected,
            )
            if panel is not None:
                index = self._bottom_panel_mouse_choice_at(
                    stdscr,
                    panel,
                    y,
                    input_text,
                    pasted_spans,
                )
                if index is not None:
                    return SessionMouseAction("command", index)

        layout = self._prompt_layout(stdscr, input_text, pasted_spans=pasted_spans)
        clicked_prompt = layout.prompt_line <= y < layout.prompt_line + layout.prompt_height
        if clicked_prompt and self._is_left_click(button_state):
            display_text = self._prompt_display_text(input_text, pasted_spans)
            display_cursor = self._prompt_display_cursor(
                input_text,
                len(input_text),
                pasted_spans,
            )
            view_start = self._prompt_view_start(display_text, display_cursor, layout)
            clicked_line = view_start + (y - layout.prompt_line)
            display_click = (clicked_line * layout.input_width) + (x - layout.input_x)
            cursor = self._prompt_real_cursor(input_text, display_click, pasted_spans)
            return SessionMouseAction("cursor", cursor)
        return None

    def _activity_scroll_action_at(
        self,
        y: int,
        delta: int,
    ) -> SessionMouseAction | None:
        for action in self._click_targets.get(y, ()):
            if action.kind == "scroll_activity_item":
                return SessionMouseAction("scroll_activity_item", delta, action.text)
        return None

    def _click_target_action_at(
        self,
        x: int,
        y: int,
        button_state: int,
    ) -> SessionMouseAction | None:
        if not self._is_click_target_activation(button_state):
            return None
        for action in reversed(self._click_targets.get(y, ())):
            if action.kind == "scroll_activity_item":
                continue
            if not action.x_end or action.x_start <= x < action.x_end:
                return action
        return None

    def _session_selection_mouse_action(
        self,
        x: int,
        y: int,
        button_state: int,
    ) -> SessionMouseAction | None:
        if self._is_left_press(button_state):
            if y in self._click_targets:
                return None
            point = self._session_text_point_at(x, y)
            if point is None:
                self._clear_session_selection()
                return None
            self._session_selection = SessionTextSelection(point, point)
            self._session_selecting = True
            return SessionMouseAction("selection", 0)

        if self._session_selecting and self._is_mouse_drag(button_state):
            point = self._nearest_session_text_point(x, y)
            if point is not None and self._session_selection is not None:
                self._session_selection = SessionTextSelection(
                    self._session_selection.anchor,
                    point,
                )
            return SessionMouseAction("selection", 0)

        if self._session_selecting and self._is_left_release(button_state):
            point = self._nearest_session_text_point(x, y)
            if point is not None and self._session_selection is not None:
                self._session_selection = SessionTextSelection(
                    self._session_selection.anchor,
                    point,
                )
            self._session_selecting = False
            selected_text = self._selected_session_text()
            if not selected_text:
                self._clear_session_selection()
                return SessionMouseAction("selection", 0)
            self._copy_to_clipboard(selected_text)
            return SessionMouseAction("selection", len(selected_text))

        return None

    def _is_left_press(self, button_state: int) -> bool:
        pressed = getattr(curses, "BUTTON1_PRESSED", 0)
        return bool(pressed and button_state & pressed)

    def _is_left_release(self, button_state: int) -> bool:
        released = getattr(curses, "BUTTON1_RELEASED", 0)
        return bool(released and button_state & released)

    def _is_mouse_drag(self, button_state: int) -> bool:
        report = getattr(curses, "REPORT_MOUSE_POSITION", 0)
        return bool((report and button_state & report) or self._is_left_press(button_state))

    def _is_click_target_activation(self, button_state: int) -> bool:
        clicked = getattr(curses, "BUTTON1_CLICKED", 0)
        pressed = getattr(curses, "BUTTON1_PRESSED", 0)
        return bool((clicked and button_state & clicked) or (pressed and button_state & pressed))

    def _is_left_click(self, button_state: int) -> bool:
        return bool(
            button_state
            & (curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED | curses.BUTTON1_RELEASED)
        )
