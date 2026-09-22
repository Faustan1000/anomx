"""Bottom activity panels and prompt-adjacent chooser panels."""

from __future__ import annotations

import curses
import hashlib
import textwrap
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime

from anomx.agent.helpers.state import (
    AsyncProcessSnapshot,
)
from anomx.agent.ui.constants import (
    ACTIVITY_DETAIL_MAX_LINES,
)
from anomx.agent.ui.models import (
    ActivityDetailEntry,
    ActivityDetailRow,
    ActivityItem,
    BottomPanel,
    BottomPanelViewport,
    CommandSpec,
    CursesWindow,
    MenuChoice,
    PromptPasteSpan,
    SessionMouseAction,
)


class BottomBarComponentMixin:
    """Bottom activity panels and prompt-adjacent chooser panels."""

    def _process_activity_item(
        self,
        process: AsyncProcessSnapshot,
        frame: int,
    ) -> ActivityItem:
        active = process.status == "running"
        title = self._process_activity_title(process)
        if active:
            title = f"{title}{self._activity_dots(frame)}"
        return ActivityItem(
            key=f"process:{process.process_id}",
            title=title,
            right_text=self._process_right_text(process),
            details=self._process_activity_details(process),
            active=active,
            kill_process_id=process.process_id if active else "",
            marker=self._activity_marker(active, frame),
            kind="process",
        )

    def _activity_panel_height(
        self,
        items: tuple[ActivityItem, ...],
        width: int,
    ) -> int:
        if not items:
            return 0
        height = 1
        for item in items:
            height += 2
            if item.key in self._expanded_activity_items:
                height += self._activity_detail_view_height(item, width)
        return height

    def _draw_activity_panel(
        self,
        stdscr: CursesWindow,
        items: tuple[ActivityItem, ...],
        start_y: int,
    ) -> None:
        _, width = stdscr.getmaxyx()
        panel_width = max(1, width - 4)
        separator = "─" * panel_width
        y = start_y
        self._add(stdscr, y, 2, separator, panel_width, self._attr("light"))
        for item in items:
            y += 1
            expanded = item.key in self._expanded_activity_items
            self._draw_activity_title_row(stdscr, y, item, width, expanded)
            if item.open_agent_id:
                self._add_click_target(
                    y,
                    SessionMouseAction("open_subagent", 0, item.open_agent_id),
                )
            else:
                self._add_click_target(y, SessionMouseAction("toggle_activity_item", 0, item.key))
                self._add_click_target(y, SessionMouseAction("scroll_activity_item", 0, item.key))
            y += 1
            if expanded:
                for row in self._visible_activity_detail_rows(item, width):
                    self._draw_activity_detail_row(stdscr, y, row, width)
                    self._add_click_target(
                        y,
                        SessionMouseAction("scroll_activity_item", 0, item.key),
                    )
                    if row.entry_key:
                        self._add_click_target(
                            y,
                            SessionMouseAction(
                                "toggle_activity_entry",
                                0,
                                row.entry_key,
                            ),
                        )
                    y += 1
            self._add(stdscr, y, 2, separator, panel_width, self._attr("light"))

    def _draw_activity_title_row(
        self,
        stdscr: CursesWindow,
        y: int,
        item: ActivityItem,
        width: int,
        expanded: bool,
    ) -> None:
        bullet_attr = self._attr(item.accent) if item.accent != "light" else self._attr("light")
        title_attr = self._attr("bold") if expanded else self._attr("light")
        right_attr = self._attr("bold") if expanded else self._attr("light")
        self._add(stdscr, y, 4, item.marker, 1, bullet_attr)
        right_text = self._activity_title_right_text(item, expanded)
        right_x = max(8, width - len(right_text) - 4) if right_text else width
        title_width = max(1, right_x - 7)
        if item.badge:
            badge = f" {item.badge} "
            self._add(stdscr, y, 6, badge, title_width, self._attr("subagent_badge"))
            suffix_x = 6 + min(len(badge), title_width)
            suffix_width = max(0, title_width - min(len(badge), title_width))
            if suffix_width:
                self._add(
                    stdscr,
                    y,
                    suffix_x,
                    item.title_suffix,
                    suffix_width,
                    self._attr("subagent"),
                )
        else:
            self._add(stdscr, y, 6, item.title, title_width, title_attr)
        if right_text:
            self._add(stdscr, y, right_x, right_text, len(right_text), right_attr)

    def _activity_title_right_text(self, item: ActivityItem, expanded: bool) -> str:
        if item.kind == "subagent":
            return item.right_text
        action = "Collapse" if expanded else "Expand"
        if item.right_text:
            return f"{action} · {item.right_text}"
        return action

    def _activity_marker(self, active: bool, frame: int) -> str:
        del frame
        if not active:
            return "⏸"
        return "▶"

    def _draw_activity_detail_row(
        self,
        stdscr: CursesWindow,
        y: int,
        row: ActivityDetailRow,
        width: int,
    ) -> None:
        detail_width = max(1, width - 10)
        role = "work_box" if row.role == "work_box" else "activity_detail"
        attr = self._attr("work_box") if role == "work_box" else self._attr("light")
        self._add(stdscr, y, 6, row.text, detail_width, attr)

    def _activity_detail_view_height(self, item: ActivityItem, width: int) -> int:
        return min(
            ACTIVITY_DETAIL_MAX_LINES,
            len(self._activity_detail_rows(item, self._activity_detail_width(width))),
        )

    def _visible_activity_detail_rows(
        self,
        item: ActivityItem,
        width: int,
    ) -> tuple[ActivityDetailRow, ...]:
        detail_width = self._activity_detail_width(width)
        rows = self._activity_detail_rows(item, detail_width)
        if not rows:
            return ()
        max_scroll = max(0, len(rows) - ACTIVITY_DETAIL_MAX_LINES)
        scroll = max(0, min(self._activity_detail_scrolls.get(item.key, 0), max_scroll))
        if scroll != self._activity_detail_scrolls.get(item.key, 0):
            self._activity_detail_scrolls[item.key] = scroll
        return rows[scroll : scroll + ACTIVITY_DETAIL_MAX_LINES]

    def _activity_detail_rows(
        self,
        item: ActivityItem,
        width: int,
    ) -> tuple[ActivityDetailRow, ...]:
        entries = item.details or (
            ActivityDetailEntry(f"{item.key}:empty", "No logs yet.", "No logs yet."),
        )
        rows: list[ActivityDetailRow] = []
        for entry in entries:
            rows.append(ActivityDetailRow(entry.text, entry_key=entry.key))
            if entry.key in self._expanded_activity_entries:
                rows.extend(self._activity_entry_box_rows(entry, width))
        return tuple(rows)

    def _activity_entry_box_rows(
        self,
        entry: ActivityDetailEntry,
        width: int,
    ) -> tuple[ActivityDetailRow, ...]:
        safe_width = max(20, width)
        inner_width = max(1, safe_width - 4)
        border = "─" * max(1, safe_width - 2)
        rows = [ActivityDetailRow(f"╭{border}╮", "work_box", entry.key)]
        body = entry.detail_body.strip() or entry.text
        for line in self._work_box_content_lines(body, inner_width):
            content = line[:inner_width].ljust(inner_width)
            rows.append(ActivityDetailRow(f"│ {content} │", "work_box", entry.key))
        rows.append(ActivityDetailRow(f"╰{border}╯", "work_box", entry.key))
        return tuple(rows)

    def _activity_detail_width(self, width: int) -> int:
        return max(1, width - 10)

    def _activity_dots(self, frame: int) -> str:
        dots = "." * ((frame // 4) % 4)
        return dots

    def _activity_command_display_text(self, statement: str, command: str) -> str:
        del command
        return self._single_line_work_text(statement) or "Command"

    def _activity_command_detail_body(self, command: str, output: str) -> str:
        del output
        return command.strip()

    def _process_activity_title(self, process: AsyncProcessSnapshot) -> str:
        label = process.statement.strip() or "Running command"
        if process.owner_name:
            return f"{process.owner_name} › Command {label}"
        if process.source in {"worker_command", "subagent_command"}:
            owner = process.owner_name or process.owner_id or "Process"
            return f"{owner} › Command {label}"
        if process.source == "command":
            return f"Command {label}"
        return f"Process {label}"

    def _short_command_label(self, command: str) -> str:
        return " ".join(command.split()) or "command"

    def _process_activity_details(
        self,
        process: AsyncProcessSnapshot,
    ) -> tuple[ActivityDetailEntry, ...]:
        output = process.output.strip()
        lines = output.splitlines() if output else ["No logs yet."]
        entries: list[ActivityDetailEntry] = []
        for line in lines:
            text = line.rstrip()
            if not text:
                continue
            entries.append(
                ActivityDetailEntry(
                    self._activity_entry_key(process.process_id, "log", len(entries), text),
                    text,
                    text,
                )
            )
        return tuple(entries)

    def _activity_entry_key(
        self,
        owner_id: str,
        kind: str,
        index: int,
        text: str,
    ) -> str:
        digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"activity:{owner_id}:{kind}:{index}:{digest}"

    def _process_right_text(self, process: AsyncProcessSnapshot) -> str:
        if process.status == "running":
            return self._process_runtime_duration(process.started_at)
        return self._process_state_label(process)

    def _process_runtime_duration(self, started_at: str) -> str:
        if not started_at:
            return ""
        with suppress(ValueError):
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            seconds = max(0, int((datetime.now(tz=UTC) - started).total_seconds()))
            return self._format_duration(seconds)
        return ""

    def _process_state_label(self, process: AsyncProcessSnapshot) -> str:
        status = process.status.strip().lower()
        if status in {"ready", "complete", "completed", "done"}:
            return "Ready"
        if status == "ended":
            if process.exit_code is not None and process.exit_code < 0:
                return "Interrupted"
            if process.exit_code not in (None, 0):
                return "Failed"
            return "Ready"
        if status in {"interrupted", "killed", "stopped", "cancelled", "canceled"}:
            return "Interrupted"
        if status == "failed" or process.exit_code not in (None, 0):
            return "Failed"
        return status.title() if status else "Ready"

    def _add_click_target(self, y: int, action: SessionMouseAction) -> None:
        self._click_targets.setdefault(y, []).append(action)

    def _worker_left_text(self, worker: object, frame: int) -> str:
        return ""

    def _worker_right_text(self, worker: object) -> str:
        return ""

    def _worker_context_text(self, worker: object) -> str:
        return ""

    def _command_bottom_panel(
        self,
        suggestions: list[CommandSpec],
        selected: int,
    ) -> BottomPanel | None:
        if not suggestions:
            return None
        return BottomPanel(
            "Commands",
            "Choose a command to run",
            tuple(
                MenuChoice(
                    label=command.command,
                    value=command.command,
                    detail=command.description,
                )
                for command in suggestions
            ),
            selected,
        )

    def _file_reference_bottom_panel(
        self,
        suggestions: list[MenuChoice],
        selected: int,
        active: bool = False,
        searching: bool = False,
    ) -> BottomPanel | None:
        if not suggestions:
            if not active:
                return None
            return BottomPanel(
                "Files",
                "Searching…" if searching else "No matches found",
                tuple(),
                0,
                frame_attr="bold",
                title_attr="bold",
                subtitle_attr="bold",
                choice_attr="bold",
                selected_choice_attr="accent",
                highlight_attr="accent",
                selected_highlight_attr="accent",
            )
        return BottomPanel(
            "Files",
            "Choose a file to reference",
            tuple(suggestions),
            selected,
            frame_attr="bold",
            title_attr="bold",
            subtitle_attr="bold",
            choice_attr="bold",
            selected_choice_attr="accent",
            highlight_attr="accent",
            selected_highlight_attr="accent",
        )

    def _draw_bottom_panel(
        self,
        stdscr: CursesWindow,
        panel: BottomPanel,
        input_text: str = "",
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> None:
        layout = self._prompt_layout(stdscr, input_text, pasted_spans=pasted_spans)
        _, width = stdscr.getmaxyx()
        viewport = self._bottom_panel_viewport(stdscr, panel, input_text, pasted_spans)
        start_y = viewport.start_y
        panel_width = max(1, width - 4)
        for y in range(start_y, layout.top_line + 1):
            self._clear_row(stdscr, y)
        self._add(stdscr, start_y, 2, "─" * panel_width, panel_width, self._attr(panel.frame_attr))
        title_x = 4
        title_width = panel_width - 4
        if panel.title_prefix:
            prefix = f"{panel.title_prefix} "
            self._add(
                stdscr,
                start_y + 1,
                title_x,
                prefix,
                title_width,
                self._attr(panel.title_prefix_attr),
            )
            title_x += len(prefix)
            title_width = max(1, title_width - len(prefix))
        self._add(
            stdscr,
            start_y + 1,
            title_x,
            panel.title,
            title_width,
            self._attr(panel.title_attr),
        )
        if panel.title_suffix:
            suffix_width = len(panel.title_suffix)
            suffix_x = max(title_x, width - suffix_width - 4)
            self._add(
                stdscr,
                start_y + 1,
                suffix_x,
                panel.title_suffix,
                max(1, width - suffix_x - 2),
                self._attr(panel.title_suffix_attr),
            )
        for offset, line in enumerate(viewport.subtitle_lines):
            self._add(
                stdscr,
                start_y + 2 + offset,
                4,
                line,
                panel_width - 4,
                self._attr(panel.subtitle_attr),
            )
        choice_y = viewport.choice_y
        longest_label = max((len(choice.label) for choice in panel.choices), default=0)
        detail_x = min(width - 4, max(44, longest_label + 10))
        if viewport.show_overflow_counts:
            self._add(
                stdscr,
                choice_y,
                4,
                f"↑ {viewport.more_above} more above",
                width - 8,
                self._attr(panel.subtitle_attr),
            )
            choice_y += 1
        for row_offset, choice_index in enumerate(viewport.visible_indices):
            choice = panel.choices[choice_index]
            marker = "›" if choice_index == panel.selected else "•"
            attr_name = (
                panel.selected_choice_attr
                if choice_index == panel.selected
                else panel.choice_attr
            )
            attr = self._attr(attr_name) if attr_name else curses.A_NORMAL
            self._draw_bottom_panel_choice_label(
                stdscr,
                choice_y + row_offset,
                4,
                f"{marker} {choice.label}",
                max(1, detail_x - 6),
                attr,
                choice.highlight,
                choice_index == panel.selected,
                choice.highlight_spans,
                label_offset=2,
                highlight_attr=panel.highlight_attr,
                selected_highlight_attr=panel.selected_highlight_attr,
            )
            if choice.detail:
                detail_attr_name = (
                    panel.selected_choice_attr
                    if choice_index == panel.selected
                    else panel.detail_attr
                )
                self._add(
                    stdscr,
                    choice_y + row_offset,
                    detail_x,
                    choice.detail,
                    width - detail_x - 4,
                    self._attr(detail_attr_name) if detail_attr_name else curses.A_NORMAL,
                )
        if viewport.show_overflow_counts:
            self._add(
                stdscr,
                choice_y + len(viewport.visible_indices),
                4,
                f"↓ {viewport.more_below} more below",
                width - 8,
                self._attr(panel.subtitle_attr),
            )

    def _draw_bottom_panel_choice_label(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        attr: int,
        highlight: str = "",
        selected: bool = False,
        highlight_spans: Sequence[tuple[int, int]] = (),
        label_offset: int = 0,
        highlight_attr: str = "accent",
        selected_highlight_attr: str = "selected",
    ) -> None:
        if highlight_spans:
            self._draw_bottom_panel_choice_highlight_spans(
                stdscr,
                y,
                x,
                text,
                width,
                attr,
                highlight_spans,
                label_offset,
                self._attr(selected_highlight_attr if selected else highlight_attr),
            )
            return
        query = highlight.strip().lower()
        if not query:
            self._add(stdscr, y, x, text, width, attr)
            return
        visible = text[: max(0, width)]
        search = visible.lower()
        start = search.find(query)
        if start < 0:
            self._add(stdscr, y, x, visible, width, attr)
            return
        end = min(len(visible), start + len(query))
        if start > 0:
            self._add(stdscr, y, x, visible[:start], width, attr)
        active_highlight_attr = self._attr(selected_highlight_attr if selected else highlight_attr)
        self._add(stdscr, y, x + start, visible[start:end], width - start, active_highlight_attr)
        if end < len(visible):
            self._add(stdscr, y, x + end, visible[end:], width - end, attr)

    def _draw_bottom_panel_choice_highlight_spans(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        attr: int,
        highlight_spans: Sequence[tuple[int, int]],
        label_offset: int,
        highlight_attr: int,
    ) -> None:
        visible = text[: max(0, width)]
        if not visible:
            return
        adjusted = [
            (max(0, start + label_offset), max(0, end + label_offset))
            for start, end in highlight_spans
            if end > start
        ]
        spans = sorted(
            (
                (max(0, start), min(len(visible), end))
                for start, end in adjusted
                if start < len(visible) and end > 0
            ),
            key=lambda span: span[0],
        )
        cursor = 0
        for start, end in spans:
            if start > cursor:
                self._add(stdscr, y, x + cursor, visible[cursor:start], width - cursor, attr)
            if end > start:
                self._add(stdscr, y, x + start, visible[start:end], width - start, highlight_attr)
            cursor = max(cursor, end)
        if cursor < len(visible):
            self._add(stdscr, y, x + cursor, visible[cursor:], width - cursor, attr)

    def _bottom_panel_height(self, panel: BottomPanel, subtitle_line_count: int) -> int:
        return min(18, len(panel.choices) + subtitle_line_count + 5)

    def _bottom_panel_start(
        self,
        stdscr: CursesWindow,
        panel: BottomPanel,
        subtitle_line_count: int,
        input_text: str = "",
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> int:
        layout = self._prompt_layout(stdscr, input_text, pasted_spans=pasted_spans)
        return max(6, layout.top_line - self._bottom_panel_height(panel, subtitle_line_count))

    def _bottom_panel_viewport(
        self,
        stdscr: CursesWindow,
        panel: BottomPanel,
        input_text: str = "",
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> BottomPanelViewport:
        layout = self._prompt_layout(stdscr, input_text, pasted_spans=pasted_spans)
        _, width = stdscr.getmaxyx()
        full_subtitle_lines = self._panel_text_lines(panel.subtitle, max(1, width - 8))
        subtitle_limit = max(0, panel.subtitle_max_lines)
        if subtitle_limit:
            max_offset = max(0, len(full_subtitle_lines) - subtitle_limit)
            subtitle_offset = min(max(0, panel.subtitle_scroll), max_offset)
            subtitle_lines = tuple(
                full_subtitle_lines[subtitle_offset : subtitle_offset + subtitle_limit]
            )
        else:
            subtitle_lines = tuple(full_subtitle_lines)
        start_y = self._bottom_panel_start(
            stdscr,
            panel,
            len(subtitle_lines),
            input_text,
            pasted_spans,
        )
        first_choice_y = start_y + 4 + len(subtitle_lines)
        raw_visible_rows = max(1, layout.top_line - first_choice_y)
        show_overflow_counts = len(panel.choices) > raw_visible_rows and raw_visible_rows >= 3
        visible_rows = raw_visible_rows - 2 if show_overflow_counts else raw_visible_rows
        visible_rows = max(1, min(len(panel.choices), visible_rows))
        max_offset = max(0, len(panel.choices) - visible_rows)
        offset = min(max(panel.selected - visible_rows + 1, 0), max_offset)
        visible_indices = tuple(range(offset, min(len(panel.choices), offset + visible_rows)))
        return BottomPanelViewport(
            start_y=start_y,
            subtitle_lines=subtitle_lines,
            choice_y=first_choice_y,
            visible_indices=visible_indices,
            more_above=offset,
            more_below=max(0, len(panel.choices) - (offset + len(visible_indices))),
            show_overflow_counts=show_overflow_counts,
        )

    def _panel_text_lines(
        self,
        text: str,
        width: int,
        max_lines: int | None = None,
    ) -> list[str]:
        if not text:
            return []
        sanitized = " ".join(text.replace("\r", " ").replace("\n", " / ").split())
        lines = textwrap.wrap(
            sanitized,
            width=max(10, width),
        )
        if max_lines is None:
            return lines
        return lines[:max(0, max_lines)]

    def _bottom_panel_subtitle_hit(self, stdscr: CursesWindow, panel: BottomPanel, y: int) -> bool:
        viewport = self._bottom_panel_viewport(stdscr, panel)
        return viewport.start_y + 2 <= y < viewport.choice_y

    def _bottom_panel_mouse_choice(
        self,
        stdscr: CursesWindow,
        panel: BottomPanel,
        input_text: str = "",
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> int | None:
        with suppress(curses.error):
            _, _x, y, _, button_state = curses.getmouse()
            if not self._is_left_click(button_state):
                return None
            return self._bottom_panel_mouse_choice_at(stdscr, panel, y, input_text, pasted_spans)
        return None

    def _bottom_panel_mouse_choice_at(
        self,
        stdscr: CursesWindow,
        panel: BottomPanel,
        y: int,
        input_text: str = "",
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> int | None:
        viewport = self._bottom_panel_viewport(stdscr, panel, input_text, pasted_spans)
        choice_y = viewport.choice_y + (1 if viewport.show_overflow_counts else 0)
        index = y - choice_y
        if 0 <= index < len(viewport.visible_indices):
            return viewport.visible_indices[index]
        return None
