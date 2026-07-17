"""Transcript message rendering, wrapping, expansion, and selection helpers."""

from __future__ import annotations

import curses
import hashlib
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from anomx.agent.base.backends import strip_thinking_tags
from anomx.agent.helpers.terminal import CODE_END, CODE_START, markdown_to_terminal_rendered_lines
from anomx.agent.store import (
    SessionRecord,
)
from anomx.agent.ui.constants import (
    TABLE_BORDER_CHARS,
)
from anomx.agent.ui.models import (
    CursesWindow,
    MessageLine,
    SessionSelectionPoint,
    SessionTextRow,
    SessionTextSelection,
)


class MessagesComponentMixin:
    """Transcript message rendering, wrapping, expansion, and selection helpers."""

    INLINE_CODE_MARKER_RE = re.compile(f"{re.escape(CODE_START)}|{re.escape(CODE_END)}")

    def _line_attr(self, role: str) -> int:
        if role in {"user", "pinned_user"}:
            return self._attr("user")
        if role == "user_box":
            return self._attr("bold")
        if role == "meta_accent":
            return self._attr("accent")
        if role in {"meta", "thought", "tool", "work_summary", "approved", "notice"}:
            return self._attr("light")
        if role == "code":
            return self._attr("accent")
        if role == "bold":
            return self._attr("bold")
        if role == "warning":
            return self._attr("warning")
        if role == "work_box":
            return self._attr("work_box")
        if role == "work_box_danger":
            return self._attr("danger")
        if role == "table_header":
            return self._attr("table_header")
        if role == "table_border":
            return self._attr("table_border")
        if role == "system":
            return self._attr("danger")
        if role == "forbidden":
            return self._attr("light")
        return curses.A_NORMAL

    def _session_rendered_lines(
        self,
        session: SessionRecord,
        messages: list[MessageLine],
        width: int,
        working_text: str | None = None,
    ) -> list[MessageLine]:
        cache_key = self._session_cache_key(session.path)
        expanded_turns = self._expanded_work_turns_key()
        expanded_lines = self._expanded_work_lines_key()
        expanded_users = self._expanded_pinned_users_key()
        working_key = "" if working_text is None else working_text
        message_cache = self._message_line_cache.get(session.path)
        if (
            cache_key is not None
            and message_cache is not None
            and message_cache[0] == cache_key[0]
            and message_cache[1] == cache_key[1]
            and message_cache[2] == expanded_turns
            and message_cache[3] is messages
        ):
            rendered_cache = self._rendered_message_cache.get(session.path)
            if (
                rendered_cache is not None
                and rendered_cache[0] == cache_key[0]
                and rendered_cache[1] == cache_key[1]
                and rendered_cache[2] == expanded_turns
                and rendered_cache[3] == expanded_lines
                and rendered_cache[4] == expanded_users
                and rendered_cache[5] == width
                and rendered_cache[6] == working_key
            ):
                return rendered_cache[7]
            rendered_messages = self._messages_with_working_status(messages, working_text)
            rendered = self._render_messages(rendered_messages, width)
            self._rendered_message_cache[session.path] = (
                cache_key[0],
                cache_key[1],
                expanded_turns,
                expanded_lines,
                expanded_users,
                width,
                working_key,
                rendered,
            )
            return rendered

        rendered_messages = self._messages_with_working_status(messages, working_text)
        return self._render_messages(rendered_messages, width)

    def _messages_with_working_status(
        self,
        messages: list[MessageLine],
        working_text: str | None,
    ) -> list[MessageLine]:
        if working_text is None:
            return messages
        return [*messages, MessageLine("working", working_text), MessageLine("meta", "")]

    def _messages_with_transient_state(
        self,
        messages: list[MessageLine],
        active_turn_elapsed: float | None,
        streaming_text: str,
    ) -> list[MessageLine]:
        del active_turn_elapsed
        if not streaming_text:
            return messages
        rendered = list(messages)
        if streaming_text:
            rendered.append(MessageLine("agent", streaming_text))
        return rendered

    def _draw_working_line(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        frame: int,
    ) -> None:
        if self._is_waiting_status_text(text):
            dots = "." * (((frame // 4) % 3) + 1)
        else:
            dots = "." * ((frame // 4) % 4)
        self._add(stdscr, y, x, f"{text}{dots}", width, self._attr("light"))

    def _is_waiting_status_text(self, text: str) -> bool:
        return text == "Waiting" or text.startswith("Waiting ")

    def _draw_work_box_line(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        role: str = "work_box",
    ) -> None:
        self._add(stdscr, y, x, text.ljust(max(0, width)), width, self._line_attr(role))

    def _draw_user_box_line(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
    ) -> None:
        border_attr = self._attr("bold")
        content_attr = curses.A_NORMAL
        self._add(stdscr, y, x, text.ljust(max(0, width)), width, content_attr)
        for offset, character in enumerate(text[:width]):
            if character in {"╭", "╮", "╰", "╯", "│", "─"}:
                self._add(stdscr, y, x + offset, character, 1, border_attr)

    def _draw_table_line(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        role: str,
    ) -> None:
        content_attr = self._attr("table_header") if role == "table_header" else curses.A_NORMAL
        border_attr = self._attr("table_border")
        start = 0
        current_is_border = bool(text) and text[0] in TABLE_BORDER_CHARS

        for index, character in enumerate(text):
            is_border = character in TABLE_BORDER_CHARS
            if is_border == current_is_border:
                continue
            attr = border_attr if current_is_border else content_attr
            self._add(stdscr, y, x + start, text[start:index], width - start, attr)
            start = index
            current_is_border = is_border

        if start < len(text):
            attr = border_attr if current_is_border else content_attr
            self._add(stdscr, y, x + start, text[start:], width - start, attr)

    def _draw_line_with_inline_code(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        default_attr: int,
    ) -> None:
        if CODE_START not in text:
            self._add(stdscr, y, x, text, width, default_attr)
            return
        parts = self.INLINE_CODE_MARKER_RE.split(text)
        cursor = x
        for i, part in enumerate(parts):
            if not part:
                continue
            remaining = width - (cursor - x)
            if remaining <= 0:
                break
            attr = self._attr("accent") if i % 2 == 1 else default_attr
            self._add(stdscr, y, cursor, part, remaining, attr)
            cursor += len(part)

    def _draw_session_selection(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        line_index: int,
        text: str,
        width: int,
    ) -> None:
        span = self._selection_span_for_line(line_index, len(text))
        if span is None:
            return
        start, end = span
        if end <= start:
            return
        selected_width = min(end, width) - start
        if selected_width <= 0:
            return
        self._add(
            stdscr,
            y,
            x + start,
            text[start:end],
            selected_width,
            self._attr("selected"),
        )

    def _selection_span_for_line(
        self,
        line_index: int,
        text_length: int,
    ) -> tuple[int, int] | None:
        if self._session_selection is None:
            return None
        start, end = self._normalized_selection_points(self._session_selection)
        if line_index < start.line_index or line_index > end.line_index:
            return None
        if start.line_index == end.line_index:
            return (
                max(0, min(text_length, start.column)),
                max(0, min(text_length, end.column)),
            )
        if line_index == start.line_index:
            return max(0, min(text_length, start.column)), text_length
        if line_index == end.line_index:
            return 0, max(0, min(text_length, end.column))
        return 0, text_length

    def _normalized_selection_points(
        self,
        selection: SessionTextSelection,
    ) -> tuple[SessionSelectionPoint, SessionSelectionPoint]:
        if (
            selection.anchor.line_index,
            selection.anchor.column,
        ) <= (
            selection.focus.line_index,
            selection.focus.column,
        ):
            return selection.anchor, selection.focus
        return selection.focus, selection.anchor

    def _session_text_point_at(self, x: int, y: int) -> SessionSelectionPoint | None:
        row = self._session_text_rows.get(y)
        if row is None:
            return None
        return self._selection_point_for_row(row, x)

    def _nearest_session_text_point(self, x: int, y: int) -> SessionSelectionPoint | None:
        if not self._session_text_rows:
            return None
        row = self._session_text_rows.get(y)
        if row is None:
            row = min(
                self._session_text_rows.values(),
                key=lambda candidate: abs(candidate.y - y),
            )
        return self._selection_point_for_row(row, x)

    def _selection_point_for_row(
        self,
        row: SessionTextRow,
        x: int,
    ) -> SessionSelectionPoint:
        column = max(0, min(len(row.text), x - row.x))
        return SessionSelectionPoint(row.line_index, column)

    def _selected_session_text(self) -> str:
        if self._session_selection is None:
            return ""
        rows_by_index = {row.line_index: row.text for row in self._session_text_rows.values()}
        start, end = self._normalized_selection_points(self._session_selection)
        pieces: list[str] = []
        for line_index in range(start.line_index, end.line_index + 1):
            text = rows_by_index.get(line_index, "")
            if line_index == start.line_index == end.line_index:
                pieces.append(text[start.column : end.column])
            elif line_index == start.line_index:
                pieces.append(text[start.column :])
            elif line_index == end.line_index:
                pieces.append(text[: end.column])
            else:
                pieces.append(text)
        return "\n".join(pieces)

    def _clear_session_selection(self) -> None:
        self._session_selection = None
        self._session_selecting = False

    def _copy_to_clipboard(self, text: str) -> bool:
        if not text:
            return False
        clipboard_commands = (
            ("pbcopy",),
            ("wl-copy",),
            ("xclip", "-selection", "clipboard"),
            ("xsel", "--clipboard", "--input"),
        )
        for command in clipboard_commands:
            if shutil.which(command[0]) is None:
                continue
            try:
                subprocess.run(
                    command,
                    input=text,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    timeout=2,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            return True
        return False

    def _read_message_lines(self, session_path: Path) -> list[MessageLine]:
        cache_key = self._session_cache_key(session_path)
        expanded_turns = self._expanded_work_turns_key()
        if cache_key is not None:
            cached = self._message_line_cache.get(session_path)
            if (
                cached is not None
                and cached[0] == cache_key[0]
                and cached[1] == cache_key[1]
                and cached[2] == expanded_turns
            ):
                return cached[3]

        lines: list[MessageLine] = []
        turn_segments: dict[str, list[list[MessageLine]]] = {}
        turn_segment_by_key: dict[str, list[MessageLine]] = {}
        turn_segment_keys: dict[str, list[str]] = {}
        turn_summaries: dict[str, str] = {}
        current_turn_id = ""
        current_segment_key = ""

        def append_turn_line(turn_id: str, line: MessageLine) -> None:
            nonlocal current_segment_key, current_turn_id
            if not turn_id:
                lines.append(line)
                current_turn_id = ""
                current_segment_key = ""
                return
            if current_turn_id != turn_id or not current_segment_key:
                segments = turn_segments.setdefault(turn_id, [])
                current_segment_key = f"{turn_id}:{len(segments)}"
                current_turn_id = turn_id
                segments.append([])
                turn_segment_by_key[current_segment_key] = segments[-1]
                turn_segment_keys.setdefault(turn_id, []).append(current_segment_key)
                lines.append(MessageLine("__turn_placeholder__", turn_id, current_segment_key))
            turn_segment_by_key[current_segment_key].append(line)

        def append_turn_summary(turn_id: str, message: str) -> None:
            if not turn_id:
                lines.append(MessageLine("work_summary", f"{message} · expand"))
                return
            if turn_id not in turn_segments:
                append_turn_line(turn_id, MessageLine("meta", ""))
                turn_segment_by_key[current_segment_key].clear()
            turn_summaries[turn_id] = message

        for event_index, event in enumerate(self._session_events(session_path)):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = str(
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            message = str(payload.get("message", "")).strip()
            if event_type in {"user_message", "skill_invocation"} and message:
                turn_id = str(payload.get("turn_id", ""))
                if event_type == "user_message" and payload.get("intermediate") and turn_id:
                    append_turn_line(
                        turn_id,
                        MessageLine(
                            "user",
                            message,
                            turn_id,
                            expansion_key=self._session_user_message_key(event_index),
                        ),
                    )
                    continue
                current_turn_id = ""
                current_segment_key = ""
                lines.append(
                    MessageLine(
                        "user",
                        message,
                        expansion_key=self._session_user_message_key(event_index),
                    )
                )
            elif event_type == "agent_message" and message:
                turn_id = str(payload.get("turn_id", ""))
                role = "agent_intermediate" if payload.get("intermediate") else "agent"
                visible_message = strip_thinking_tags(message)
                if visible_message:
                    append_turn_line(turn_id, MessageLine(role, visible_message, turn_id))
            elif event_type == "system_message" and message:
                role = str(payload.get("role", "system"))
                if role == "question":
                    continue
                turn_id = str(payload.get("turn_id", ""))
                expansion_key = self._session_work_line_key(role, turn_id, event_index)
                message, detail_body, detail_title = self._message_display_parts(
                    role,
                    message,
                    str(payload.get("command", "")),
                )
                append_turn_line(
                    turn_id,
                    MessageLine(
                        role,
                        message,
                        turn_id,
                        expansion_key,
                        detail_title,
                        detail_body,
                    )
                    if turn_id
                    else MessageLine(
                        role,
                        message,
                        expansion_key=expansion_key,
                        detail_title=detail_title,
                        detail_body=detail_body,
                    ),
                )
            elif event_type == "work_message" and message:
                turn_id = str(payload.get("turn_id", ""))
                role = str(payload.get("role", "tool"))
                expansion_key = self._session_work_line_key(role, turn_id, event_index)
                message, detail_body, detail_title = self._message_display_parts(
                    role,
                    message,
                    str(payload.get("command", "")),
                )
                append_turn_line(
                    turn_id,
                    MessageLine(
                        role,
                        message,
                        turn_id,
                        expansion_key,
                        detail_title,
                        detail_body,
                    )
                    if turn_id
                    else MessageLine(
                        role,
                        message,
                        expansion_key=expansion_key,
                        detail_title=detail_title,
                        detail_body=detail_body,
                    ),
                )
            elif event_type == "work_summary" and message:
                turn_id = str(payload.get("turn_id", ""))
                append_turn_summary(turn_id, message)
        rendered_lines: list[MessageLine] = []
        collapsed_turns: set[str] = set()
        for line in lines:
            if line.role != "__turn_placeholder__":
                rendered_lines.append(line)
                continue
            turn_id = line.text
            segment_key = line.meta
            summary = turn_summaries.get(turn_id)
            if summary:
                if turn_id in self._expanded_work_turns:
                    rendered_lines.extend(turn_segment_by_key.get(segment_key, []))
                    if segment_key == (turn_segment_keys.get(turn_id) or [""])[-1]:
                        rendered_lines.append(
                            MessageLine("work_summary", f"{summary} · collapse", turn_id)
                        )
                elif turn_id not in collapsed_turns:
                    collapsed_turns.add(turn_id)
                    rendered_lines.append(
                        MessageLine("work_summary", f"{summary} · expand", turn_id)
                    )
                else:
                    continue
            else:
                rendered_lines.extend(turn_segment_by_key.get(segment_key, []))
        if cache_key is not None:
            self._message_line_cache[session_path] = (
                cache_key[0],
                cache_key[1],
                expanded_turns,
                rendered_lines,
            )
        return rendered_lines

    def _session_events(self, session_path: Path) -> list[dict[str, Any]]:
        cache_key = self._session_cache_key(session_path)
        if cache_key is None:
            return self.home.read_session_events(session_path)
        cached = self._session_event_cache.get(session_path)
        if cached is not None and cached[0] == cache_key[0] and cached[1] == cache_key[1]:
            return cached[2]
        events = self.home.read_session_events(session_path)
        self._session_event_cache[session_path] = (cache_key[0], cache_key[1], events)
        return events

    def _session_cache_key(self, session_path: Path) -> tuple[int, int] | None:
        try:
            stat = session_path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _expanded_work_turns_key(self) -> tuple[str, ...]:
        return tuple(sorted(self._expanded_work_turns))

    def _expanded_work_lines_key(self) -> tuple[str, ...]:
        return tuple(sorted(self._expanded_work_lines))

    def _expanded_pinned_users_key(self) -> tuple[str, ...]:
        return tuple(sorted(self._expanded_pinned_users))

    def _message_display_parts(
        self,
        role: str,
        message: str,
        command: str = "",
    ) -> tuple[str, str, str]:
        if role == "thought":
            return message, command, "Thought"
        if role != "forbidden":
            return message, command, ""

        lines = message.splitlines()
        statement = lines[0].strip() if lines else message.strip()
        detail_command = command.strip()
        reason = ""
        for line in lines[1:]:
            if line.startswith("Command: "):
                detail_command = line.removeprefix("Command: ").strip()
            elif line.startswith("Reason: "):
                reason = line.removeprefix("Reason: ").strip()
        if statement.startswith("Blocked: "):
            detail_title = f"Reason: {reason}" if reason else ""
            return statement, detail_command, detail_title

        if message.startswith("Blocked command: "):
            command, separator, reason = message.removeprefix("Blocked command: ").partition(" · ")
            detail_title = f"Reason: {reason.strip()}" if separator else ""
            detail_command = command.strip()
            return f"Blocked: {detail_command}", detail_command, detail_title

        return message, detail_command, ""

    def _session_work_line_key(self, role: str, turn_id: str, event_index: int) -> str:
        if not self._is_expandable_work_role(role):
            return ""
        namespace = turn_id or "session"
        return f"{namespace}:{event_index}"

    def _session_user_message_key(self, event_index: int) -> str:
        return f"user:{event_index}"

    def _render_messages(self, messages: list[MessageLine], width: int) -> list[MessageLine]:
        rendered: list[MessageLine] = []
        previous_kind: str | None = None
        for message in messages:
            kind = self._message_kind(message.role)
            if rendered and previous_kind is not None and kind != previous_kind:
                rendered.append(MessageLine("meta", ""))
            if message.role == "user":
                rendered.extend(self._render_user_message(message, width))
            elif self._is_expandable_work_role(message.role):
                rendered.extend(self._render_work_message(message, width))
            else:
                for line in markdown_to_terminal_rendered_lines(
                    message.text,
                    width=max(20, width),
                ):
                    rendered.append(
                        MessageLine(
                            self._terminal_line_role(message.role, line.style),
                            line.text,
                            message.meta,
                            message.expansion_key,
                            message.detail_title,
                            message.detail_body,
                        )
                    )
            previous_kind = kind
        return rendered

    def _render_user_message(self, message: MessageLine, width: int) -> list[MessageLine]:
        safe_width = max(20, width)
        expansion_key = message.expansion_key or self._fallback_user_line_key(message)
        if expansion_key in self._expanded_pinned_users:
            return self._expanded_user_box_lines(message, safe_width, expansion_key)

        display_text = self._single_line_work_text(message.text)
        toggle_text = " Expand"
        available = max(1, safe_width - len(toggle_text))
        collapsed_text = self._ellipsized_statement_text(display_text, available)
        return [
            MessageLine(
                "user",
                f"{collapsed_text.ljust(available)}{toggle_text}",
                message.meta,
                expansion_key,
                message.detail_title,
                message.detail_body,
            )
        ]

    def _expanded_user_box_lines(
        self,
        message: MessageLine,
        width: int,
        expansion_key: str,
    ) -> list[MessageLine]:
        safe_width = max(20, width)
        inner_width = max(1, safe_width - 4)
        collapse_text = " Collapse"
        top_width = max(1, safe_width - 2)
        top_border = "─" * max(1, top_width - len(collapse_text))
        lines = [
            MessageLine(
                "user_box",
                f"╭{top_border}{collapse_text}╮",
                message.meta,
                expansion_key,
            )
        ]
        for content_line in self._work_box_content_lines(message.text, inner_width):
            content = content_line[:inner_width].ljust(inner_width)
            lines.append(MessageLine("user_box", f"│ {content} │", message.meta, expansion_key))
        border = "─" * max(1, safe_width - 2)
        lines.append(MessageLine("user_box", f"╰{border}╯", message.meta, expansion_key))
        return lines

    def _terminal_line_role(self, fallback_role: str, style: str) -> str:
        if style in {"table_border", "table_header", "table_row"}:
            return style
        if style == "bold":
            return "bold"
        return fallback_role

    def _render_work_message(self, message: MessageLine, width: int) -> list[MessageLine]:
        safe_width = max(20, width)
        expansion_key = message.expansion_key or self._fallback_work_line_key(message)
        if expansion_key in self._expanded_work_lines:
            return self._expanded_work_box_lines(message, safe_width, expansion_key)

        display_text = self._ellipsized_statement_text(
            self._work_message_display_text(message),
            safe_width,
        )
        return [
            MessageLine(
                message.role,
                display_text,
                message.meta,
                expansion_key,
                message.detail_title,
                message.detail_body,
            )
        ]

    def _is_expandable_work_role(self, role: str) -> bool:
        return role in {"thought", "tool", "approved", "forbidden"}

    def _work_message_display_text(self, message: MessageLine) -> str:
        return self._single_line_work_text(message.text)

    def _single_line_work_text(self, text: str) -> str:
        return " ".join(text.replace("\r", " ").replace("\n", " ").split())

    def _ellipsized_statement_text(self, text: str, width: int) -> str:
        safe_width = max(1, width)
        if len(text) <= safe_width:
            return text
        if safe_width <= 3:
            return "." * safe_width
        return f"{text[: safe_width - 3].rstrip()}..."

    def _expanded_work_box_lines(
        self,
        message: MessageLine,
        width: int,
        expansion_key: str,
    ) -> list[MessageLine]:
        safe_width = max(20, width)
        inner_width = max(1, safe_width - 4)
        border = "─" * max(1, safe_width - 2)
        lines = [MessageLine("work_box", f"╭{border}╮", message.meta, expansion_key)]
        for title_line in self._work_box_content_lines(message.text, inner_width):
            title = title_line[:inner_width].ljust(inner_width)
            lines.append(MessageLine("work_box", f"│ {title} │", message.meta, expansion_key))
        if message.detail_title:
            for title_line in self._work_box_content_lines(message.detail_title, inner_width):
                title = title_line[:inner_width].ljust(inner_width)
                lines.append(MessageLine("work_box", f"│ {title} │", message.meta, expansion_key))
        if message.detail_body:
            lines.append(self._empty_work_box_line(inner_width, message.meta, expansion_key))
            for content_line in self._work_box_content_lines(message.detail_body, inner_width):
                content = content_line[:inner_width].ljust(inner_width)
                lines.append(MessageLine("work_box", f"│ {content} │", message.meta, expansion_key))
        lines.append(MessageLine("work_box", f"╰{border}╯", message.meta, expansion_key))
        return lines

    def _empty_work_box_line(
        self,
        inner_width: int,
        meta: str,
        expansion_key: str,
    ) -> MessageLine:
        return MessageLine(
            "work_box",
            f"│ {' ' * max(1, inner_width)} │",
            meta,
            expansion_key,
        )

    def _work_box_content_lines(self, text: str, width: int) -> list[str]:
        safe_width = max(1, width)
        lines: list[str] = []
        for raw_line in text.splitlines() or [""]:
            line = raw_line.replace("\t", "    ").rstrip()
            if not line:
                lines.append("")
                continue
            lines.extend(
                textwrap.wrap(
                    line,
                    width=safe_width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                or [""]
            )
        return lines

    def _fallback_work_line_key(self, message: MessageLine) -> str:
        identity = f"{message.text}\n{message.detail_body}\n{message.detail_title}"
        digest = hashlib.sha1(identity.encode("utf-8", errors="replace")).hexdigest()
        return f"{message.role}:{message.meta}:{digest}"

    def _fallback_user_line_key(self, message: MessageLine) -> str:
        digest = hashlib.sha1(message.text.encode("utf-8", errors="replace")).hexdigest()
        return f"user:{message.meta}:{digest}"

    def _message_kind(self, role: str) -> str:
        if role == "user":
            return "user"
        if role == "agent":
            return "agent"
        if role == "agent_intermediate":
            return "agent"
        return "working"
