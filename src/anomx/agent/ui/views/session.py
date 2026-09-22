"""Conversation session view."""

from __future__ import annotations

import curses
import math
import random
import textwrap
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from anomx.agent.helpers.state import (
    PlanStep,
    SubagentSnapshot,
    active_subagent_snapshots,
    latest_plan_steps,
    running_process_snapshots,
)
from anomx.agent.runtime import (
    format_token_count,
)
from anomx.agent.skills import (
    Skill,
)
from anomx.agent.store import (
    SessionRecord,
    normalize_thinking_intensity,
    thinking_intensity_options,
)
from anomx.agent.ui.constants import (
    ACTIVITY_FRAME_SECONDS,
    PLAN_STEP_REVEAL_SECONDS,
    START_HINT_REVEAL_SECONDS,
    STARTUP_MATRIX_ALPHABET,
)
from anomx.agent.ui.models import (
    BottomPanel,
    CommandSpec,
    CursesWindow,
    MenuChoice,
    MessageLine,
    PromptPasteSpan,
    SessionMouseAction,
    SessionTextRow,
    SessionViewportState,
)


class SessionViewMixin:
    """Conversation session view."""

    def _draw_continue_session_prompt(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        statement: str,
        selected: int,
    ) -> None:
        height, width = self._draw_shell(
            stdscr,
            "Continue session?",
            "Previous workspace session",
        )
        self._add(stdscr, 8, 4, str(self.workspace_root), width - 8, self._attr("bold"))
        y = 10
        self._add(stdscr, y, 4, session.title, width - 8, self._attr("bold"))
        y += 1
        detail = (
            f"{self._message_count_label(session.message_count)} · "
            f"{session.updated_at} · {session.provider}/{session.model}"
        )
        self._add(stdscr, y, 4, detail, width - 8, self._attr("light"))

        y += 3
        copy = statement.strip() or self._fallback_continue_session_statement(session)
        for line in textwrap.wrap(copy, width=max(24, width - 8)):
            self._add(stdscr, y, 4, line, width - 8)
            y += 1
        y += 1
        resume_copy = (
            "Anomx can resume the stored transcript and continue with the previous "
            "context, or start fresh in this workspace."
        )
        for line in textwrap.wrap(resume_copy, width=max(24, width - 8)):
            self._add(stdscr, y, 4, line, width - 8)
            y += 1

        y += 2
        choices = ("Yes, continue with session", "No, start new session")
        for index, choice in enumerate(choices):
            marker = "›" if index == selected else "•"
            attr = self._attr("accent") if index == selected else curses.A_NORMAL
            self._add(stdscr, y + index, 4, f"{marker} {index + 1}. {choice}", width - 8, attr)

        self._add(
            stdscr,
            min(height - 2, y + len(choices) + 2),
            4,
            "Enter to confirm · Esc to start new",
            width - 8,
            self._attr("light"),
        )
        stdscr.refresh()

    def _draw_info_panel(self, stdscr: CursesWindow, session: SessionRecord) -> None:
        config = self.home.load_config()
        model = str(config.get("model", session.model))
        header_lines = self._session_header_lines(session, model)
        body_lines: list[str] = []
        body_lines.append("Current location")
        for row in self._session_location_rows(session):
            body_lines.append(f"  {row.label}: {row.value}")
        body_lines.append("")
        body_lines.append("Approved commands")
        for row in self._approved_command_rows(session):
            body_lines.append(f"  {row.label}: {row.value}")
        self._draw_overlay(
            stdscr,
            title="Info",
            subtitle=" ".join(header_lines) if header_lines else "",
            body_lines=tuple(body_lines),
            footer="Esc Back · Enter Back",
        )

    def _draw_back_to_project_link(
        self,
        stdscr: CursesWindow,
        width: int,
        text: str = "Back to Project",
    ) -> None:
        if width < len(text) + 8:
            return
        y = 3
        x = max(4, width - len(text) - 5)
        self._add(stdscr, y, x, text, len(text), self._attr("bold"))
        esc_text = "esc"
        esc_x = max(4, width - len(esc_text) - 5)
        self._add(stdscr, y + 1, esc_x, esc_text, len(esc_text), self._attr("light"))
        self._add_click_target(
            y,
            SessionMouseAction(
                "back_project",
                0,
                x_start=x,
                x_end=x + len(text),
            ),
        )

    def _draw_session(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        messages: list[MessageLine],
        input_text: str,
        cursor: int,
        scroll: int,
        command_suggestions: list[CommandSpec] | None = None,
        command_selected: int = 0,
        file_suggestions: list[MenuChoice] | None = None,
        file_selected: int = 0,
        file_reference_searching: bool = False,
        file_references: Mapping[str, str] | None = None,
        image_attachments: Mapping[str, Mapping[str, str]] | None = None,
        bottom_panel: BottomPanel | None = None,
        working_text: str | None = None,
        working_deadline: float | None = None,
        working_frame: int = 0,
        anchor_line: int | None = None,
        sticky_anchor: bool = False,
        prompt_notice: str = "",
        prompt_notice_role: str = "light",
        prompt_hint_suffix: str = "",
        force_start_hints: bool = False,
        start_hint_reveal_progress: float | None = None,
        start_hint_removal_progress: float = 0.0,
        active_turn_elapsed: float | None = None,
        streaming_text: str = "",
        show_prompt_bar: bool = True,
        hide_plan: bool = False,
        title_override: str = "",
        back_link_text: str = "Back to Project",
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> SessionViewportState:
        config = self._load_config_cached()
        provider = str(config.get("provider", session.provider))
        model = str(config.get("model", session.model))
        session_events = self._session_events(session.path)
        plan_steps = (
            ()
            if hide_plan
            else self._visible_plan_steps(
                session_events,
                latest_plan_steps(session_events),
            )
        )
        plan_expanded = bool(plan_steps and session.path in self._expanded_plan_sessions)
        processes = running_process_snapshots(session_events) if bottom_panel is None else ()
        subagents = active_subagent_snapshots(session_events) if bottom_panel is None else ()
        working_text = self._effective_session_working_text(working_text, subagents)
        header_lines = self._session_header_lines(session, model)
        self._click_targets = {}
        height, width = self._draw_shell(
            stdscr,
            title_override or self._session_project_title(session),
            header_lines,
            plan_steps,
            header_meta=self._session_header_meta(session, provider, model),
            plan_expanded=plan_expanded,
            title_suffix=self._session_title_counter(active_turn_elapsed),
        )
        if back_link_text:
            self._draw_back_to_project_link(stdscr, width, back_link_text)
        layout = self._prompt_layout(stdscr, input_text, pasted_spans=pasted_spans)
        suggestions = command_suggestions or []
        activity_items = self._activity_items(subagents, processes, session_events, working_frame)
        working_status_text = self._working_status_text(working_text, working_deadline)
        base_activity_panel_height = self._activity_panel_height(activity_items, width)
        body_top = self._session_body_top(
            plan_steps,
            subtitle_line_count=len(header_lines),
            plan_expanded=plan_expanded,
        )
        prompt_top = layout.top_line if show_prompt_bar else max(0, height - 1)
        activity_panel_bottom = (
            layout.prompt_line if activity_items and show_prompt_bar else prompt_top
        )
        body_bottom = max(body_top + 1, activity_panel_bottom - base_activity_panel_height)
        body_height = max(1, body_bottom - body_top)
        command_panel = (
            self._command_bottom_panel(suggestions, command_selected)
            if bottom_panel is None and show_prompt_bar
            else None
        )
        file_panel = (
            self._file_reference_bottom_panel(
                file_suggestions or [],
                file_selected,
                active=bool(file_suggestions) or file_reference_searching,
                searching=file_reference_searching,
            )
            if bottom_panel is None and show_prompt_bar
            else None
        )
        active_bottom_panel = (
            (bottom_panel or file_panel or command_panel) if show_prompt_bar else None
        )
        display_messages = self._messages_with_transient_state(
            messages,
            active_turn_elapsed,
            streaming_text,
        )
        rendered = self._session_rendered_lines(
            session,
            display_messages,
            max(20, width - 8),
            None if streaming_text else working_status_text,
        )
        rendered_line_count = len(rendered)
        visible_rows: list[tuple[int, MessageLine]]
        if anchor_line is None:
            scroll = self._clamp_session_scroll(scroll, rendered_line_count, body_height)
            start = self._session_view_start(scroll, rendered_line_count, body_height)
            visible_rows = [
                (start + offset, line)
                for offset, line in enumerate(rendered[start : start + body_height])
            ]
        elif sticky_anchor and rendered_line_count:
            start = max(0, min(anchor_line, self._session_max_start(rendered_line_count)))
            pinned_rows, anchor_extent = self._sticky_anchor_rows(
                session,
                rendered,
                start,
                width - 8,
            )
            pinned_height = min(body_height, len(pinned_rows))
            visible_rows = [
                (start + min(offset, max(0, anchor_extent - 1)), line)
                for offset, line in enumerate(pinned_rows[:pinned_height])
            ]
            tail_start = min(rendered_line_count, start + anchor_extent)
            tail_height = max(0, body_height - pinned_height)
            if tail_height:
                tail_count = max(0, rendered_line_count - tail_start)
                scroll = self._clamp_session_scroll(scroll, tail_count, tail_height)
                relative_start = self._session_view_start(scroll, tail_count, tail_height)
                visible_rows.extend(
                    (tail_start + offset, line)
                    for offset, line in enumerate(
                        rendered[
                            tail_start + relative_start : tail_start + relative_start + tail_height
                        ]
                    )
                )
            else:
                scroll = 0
                visible_rows = [(start, rendered[start])]
        else:
            start = max(0, min(anchor_line, self._session_max_start(rendered_line_count)))
            scroll = self._session_scroll_for_start(start, rendered_line_count, body_height)
            visible_rows = [
                (start + offset, line)
                for offset, line in enumerate(rendered[start : start + body_height])
            ]
        self._session_text_rows = {}
        wave_frame = int(time.monotonic() / ACTIVITY_FRAME_SECONDS)
        for offset, (line_index, line) in enumerate(visible_rows):
            y = body_top + offset
            self._session_text_rows[y] = SessionTextRow(
                line_index=line_index,
                y=y,
                x=4,
                width=width - 8,
                text=line.text,
            )
            if line.role in {"user", "pinned_user", "user_box"} and line.expansion_key:
                self._add_click_target(
                    y,
                    SessionMouseAction("toggle_pinned_user", 0, line.expansion_key),
                )
            elif line.role in {"work_summary", "work_active"}:
                self._add_click_target(y, SessionMouseAction("toggle_work", 0, line.meta))
            elif line.expansion_key and (
                self._is_expandable_work_role(line.role)
                or line.role in {"work_box", "work_box_danger"}
            ):
                self._add_click_target(
                    y,
                    SessionMouseAction("toggle_work_line", 0, line.expansion_key),
                )
            if line.role == "working":
                self._draw_working_line(
                    stdscr,
                    y,
                    4,
                    line.text,
                    width - 8,
                    wave_frame,
                )
                self._draw_session_selection(stdscr, y, 4, line_index, line.text, width - 8)
                continue
            if line.role in {"work_box", "work_box_danger"}:
                self._draw_work_box_line(stdscr, y, 4, line.text, width - 8, line.role)
                if line.activity_wave:
                    self._draw_activity_wave(
                        stdscr, y, 6, line.text[2:-2].rstrip(), width - 12,
                        wave_frame, self._line_attr(line.role),
                    )
                self._draw_session_selection(stdscr, y, 4, line_index, line.text, width - 8)
                continue
            if line.role == "user_box":
                self._draw_user_box_line(stdscr, y, 4, line.text, width - 8)
                self._draw_session_selection(stdscr, y, 4, line_index, line.text, width - 8)
                continue
            if line.role in {"table_header", "table_row"}:
                self._draw_table_line(stdscr, y, 4, line.text, width - 8, line.role)
                self._draw_session_selection(stdscr, y, 4, line_index, line.text, width - 8)
                continue
            default_attr = self._line_attr(line.role)
            self._draw_line_with_inline_code(stdscr, y, 4, line.text, width - 8, default_attr)
            if line.activity_wave:
                self._draw_activity_wave(
                    stdscr, y, 4, line.text, width - 8, wave_frame, default_attr,
                )
            self._draw_session_selection(stdscr, y, 4, line_index, line.text, width - 8)

        should_draw_start_hints = self._should_draw_start_hints(
            messages,
            input_text,
            active_bottom_panel,
            working_text if show_prompt_bar else None,
            plan_steps,
        )
        if force_start_hints or should_draw_start_hints:
            reveal_progress = (
                self._start_hint_reveal_progress()
                if start_hint_reveal_progress is None
                else start_hint_reveal_progress
            )
            self._draw_start_hints(
                stdscr,
                body_top,
                body_bottom,
                width,
                working_frame,
                reveal_progress,
                start_hint_removal_progress,
            )
        else:
            self._start_hint_reveal_started_at = None

        if activity_items:
            self._draw_activity_panel(
                stdscr,
                activity_items,
                body_bottom,
            )
        if active_bottom_panel is not None:
            self._draw_bottom_panel(stdscr, active_bottom_panel, input_text, pasted_spans)
        if show_prompt_bar:
            self._draw_prompt_bar(
                stdscr,
                input_text,
                cursor,
                prompt_notice,
                prompt_notice_role,
                self._prompt_reference_labels(file_references, image_attachments),
                draw_top_rule=not activity_items,
                hint_suffix=prompt_hint_suffix,
                pasted_spans=pasted_spans,
            )
        stdscr.refresh()
        return SessionViewportState(start, scroll, body_height, rendered_line_count)

    def _sticky_anchor_rows(
        self,
        session: SessionRecord,
        rendered: list[MessageLine],
        anchor_line: int,
        width: int,
    ) -> tuple[list[MessageLine], int]:
        anchor = rendered[anchor_line]
        if anchor.role not in {"user", "user_box", "pinned_user"}:
            return [anchor], 1

        group_key = anchor.expansion_key
        group_end = anchor_line + 1
        if group_key:
            while group_end < len(rendered):
                candidate = rendered[group_end]
                if (
                    candidate.role not in {"user", "user_box", "pinned_user"}
                    or candidate.expansion_key != group_key
                ):
                    break
                group_end += 1

        group = rendered[anchor_line:group_end] or [anchor]
        anchor_extent = max(1, group_end - anchor_line)
        del session, width
        return group, anchor_extent

    def _session_title_counter(self, active_turn_elapsed: float | None) -> str:
        if active_turn_elapsed is None:
            return ""
        return self._format_duration(active_turn_elapsed)

    def _session_project_title(self, session: SessionRecord) -> str:
        project_name = self._current_project_name()
        title = session.title.strip() or "New session"
        return f"{project_name} › {title}" if project_name else title

    def _current_project_name(self) -> str:
        stored_project = self.home.project_for_path(self.project_path)
        if stored_project is not None and stored_project.name.strip():
            return stored_project.name.strip()
        return self._fallback_project_name()

    def _visible_plan_steps(
        self,
        events: Sequence[Mapping[str, object]],
        plan_steps: tuple[PlanStep, ...],
    ) -> tuple[PlanStep, ...]:
        if not plan_steps:
            return ()
        created_at = self._latest_plan_create_timestamp(events)
        if created_at is None:
            return plan_steps
        elapsed = max(0.0, (datetime.now(tz=UTC) - created_at).total_seconds())
        visible_count = min(
            len(plan_steps),
            max(1, int(elapsed // PLAN_STEP_REVEAL_SECONDS) + 1),
        )
        return plan_steps[:visible_count]

    def _latest_plan_create_timestamp(
        self,
        events: Sequence[Mapping[str, object]],
    ) -> datetime | None:
        created_at: datetime | None = None
        for event in events:
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = str(
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            if event_type != "plan_update":
                continue
            if str(payload.get("action", "")) != "create":
                created_at = None
                continue
            created_at = self._event_timestamp(event)
        return created_at

    def _event_timestamp(self, event: Mapping[str, object]) -> datetime | None:
        timestamp = str(event.get("timestamp", "")).strip()
        if not timestamp:
            return None
        with suppress(ValueError):
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return None

    def _start_hint_reveal_progress(self) -> float:
        now = time.monotonic()
        if self._start_hint_reveal_started_at is None:
            self._start_hint_reveal_started_at = now
        elapsed = now - self._start_hint_reveal_started_at
        return min(1.0, max(0.0, elapsed / START_HINT_REVEAL_SECONDS))

    def _start_hint_reveal_active(self) -> bool:
        if self._start_hint_reveal_started_at is None:
            return False
        return time.monotonic() - self._start_hint_reveal_started_at < START_HINT_REVEAL_SECONDS

    def _should_draw_start_hints(
        self,
        messages: list[MessageLine],
        input_text: str,
        active_bottom_panel: BottomPanel | None,
        working_text: str | None,
        plan_steps: tuple[PlanStep, ...],
    ) -> bool:
        del input_text
        return (
            not messages and active_bottom_panel is None and working_text is None and not plan_steps
        )

    def _draw_start_hints(
        self,
        stdscr: CursesWindow,
        body_top: int,
        body_bottom: int,
        width: int,
        frame: int = 0,
        reveal_progress: float = 1.0,
        removal_progress: float = 0.0,
    ) -> None:
        skills = self._starter_skills()
        if not skills:
            return

        available_height = max(1, body_bottom - body_top)
        card_height = 7
        gap = 3
        if width >= 96:
            card_width = min(34, max(24, (width - 8 - (gap * 2)) // 3))
            total_width = (card_width * len(skills)) + (gap * (len(skills) - 1))
            start_x = max(4, (width - total_width) // 2)
            start_y = body_top + max(0, (available_height - card_height) // 2)
            for index, skill in enumerate(skills):
                self._draw_start_hint_card(
                    stdscr,
                    start_y,
                    start_x + (index * (card_width + gap)),
                    card_width,
                    card_height,
                    skill,
                    frame,
                    reveal_progress,
                    removal_progress,
                )
            return

        card_width = min(max(24, width - 8), 42)
        total_height = (card_height * len(skills)) + (len(skills) - 1)
        start_x = max(4, (width - card_width) // 2)
        start_y = body_top + max(0, (available_height - total_height) // 2)
        for index, skill in enumerate(skills):
            self._draw_start_hint_card(
                stdscr,
                start_y + (index * (card_height + 1)),
                start_x,
                card_width,
                card_height,
                skill,
                frame,
                reveal_progress,
                removal_progress,
            )

    def _draw_start_hint_card(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        width: int,
        height: int,
        skill: Skill,
        frame: int = 0,
        reveal_progress: float = 1.0,
        removal_progress: float = 0.0,
    ) -> None:
        inner_width = max(1, width - 4)
        horizontal = "─" * max(1, width - 2)
        self._draw_start_hint_line(
            stdscr,
            y,
            x,
            f"╭{horizontal}╮",
            width,
            self._attr("accent"),
            frame,
            1.0,
            removal_progress,
        )
        for offset in range(1, height - 1):
            self._draw_start_hint_line(
                stdscr,
                y + offset,
                x,
                "│",
                1,
                self._attr("accent"),
                frame,
                1.0,
                removal_progress,
            )
            self._draw_start_hint_line(
                stdscr,
                y + offset,
                x + width - 1,
                "│",
                1,
                self._attr("accent"),
                frame,
                1.0,
                removal_progress,
            )
        self._draw_start_hint_line(
            stdscr,
            y + height - 1,
            x,
            f"╰{horizontal}╯",
            width,
            self._attr("accent"),
            frame,
            1.0,
            removal_progress,
        )
        for offset in range(height):
            self._add_click_target(
                y + offset,
                SessionMouseAction("skill", 0, skill.command, x, x + width),
            )

        self._draw_start_hint_line(
            stdscr,
            y + 1,
            x + 2,
            skill.title,
            inner_width,
            self._attr("bold"),
            frame,
            reveal_progress,
            removal_progress,
        )
        description_lines = textwrap.wrap(
            skill.description,
            width=inner_width,
            break_long_words=True,
            break_on_hyphens=False,
        )[:3]
        for index, line in enumerate(description_lines):
            self._draw_start_hint_line(
                stdscr,
                y + 3 + index,
                x + 2,
                line,
                inner_width,
                self._attr("light"),
                frame,
                reveal_progress,
                removal_progress,
            )

    def _draw_start_hint_line(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        attr: int,
        frame: int,
        reveal_progress: float,
        removal_progress: float,
    ) -> None:
        reveal_progress = min(1.0, max(0.0, reveal_progress))
        removal_progress = min(1.0, max(0.0, removal_progress))
        if reveal_progress >= 1.0 and removal_progress <= 0.0:
            self._add(stdscr, y, x, text, width, attr)
            return
        char_rng = random.Random((frame + 1) * 262_147 + y * 97 + x * 31)
        for offset, character in enumerate(text[:width]):
            cell_x = x + offset
            if self._start_hint_cell_hidden(
                cell_x,
                y,
                reveal_progress,
                removal_progress,
            ):
                continue
            visible_character = (
                char_rng.choice(STARTUP_MATRIX_ALPHABET)
                if reveal_progress < 1.0 and character.strip()
                else character
            )
            self._add(stdscr, y, cell_x, visible_character, 1, attr)

    def _start_hint_cell_hidden(
        self,
        x: int,
        y: int,
        reveal_progress: float,
        removal_progress: float,
    ) -> bool:
        if removal_progress > 0:
            return self._startup_cell_removed(x, y, removal_progress)
        if reveal_progress >= 1:
            return False
        if reveal_progress <= 0:
            return True
        threshold = int(reveal_progress * 10_000)
        return self._startup_cell_rank(x, y, 0xC4AD5) >= threshold

    def _session_max_start(self, rendered_line_count: int) -> int:
        return max(0, rendered_line_count - 1)

    def _session_bottom_start(self, rendered_line_count: int, body_height: int) -> int:
        return max(0, rendered_line_count - max(1, body_height))

    def _session_scroll_bounds(
        self,
        rendered_line_count: int,
        body_height: int,
    ) -> tuple[int, int]:
        max_start = self._session_max_start(rendered_line_count)
        bottom_start = self._session_bottom_start(rendered_line_count, body_height)
        return bottom_start - max_start, bottom_start

    def _clamp_session_scroll(
        self,
        scroll: int,
        rendered_line_count: int,
        body_height: int,
    ) -> int:
        min_scroll, max_scroll = self._session_scroll_bounds(rendered_line_count, body_height)
        return max(min_scroll, min(scroll, max_scroll))

    def _session_view_start(
        self,
        scroll: int,
        rendered_line_count: int,
        body_height: int,
    ) -> int:
        clamped_scroll = self._clamp_session_scroll(scroll, rendered_line_count, body_height)
        bottom_start = self._session_bottom_start(rendered_line_count, body_height)
        max_start = self._session_max_start(rendered_line_count)
        return max(0, min(max_start, bottom_start - clamped_scroll))

    def _session_scroll_for_start(
        self,
        start: int,
        rendered_line_count: int,
        body_height: int,
    ) -> int:
        max_start = self._session_max_start(rendered_line_count)
        clamped_start = max(0, min(start, max_start))
        bottom_start = self._session_bottom_start(rendered_line_count, body_height)
        return self._clamp_session_scroll(
            bottom_start - clamped_start,
            rendered_line_count,
            body_height,
        )

    def _working_status_text(
        self,
        working_text: str | None,
        working_deadline: float | None = None,
        now: float | None = None,
    ) -> str | None:
        if working_text is None:
            return None
        if working_deadline is None or self._is_waiting_status_text(working_text):
            return working_text
        current_time = time.monotonic() if now is None else now
        remaining = max(0, math.ceil(working_deadline - current_time))
        return f"{working_text} {remaining // 60:02d}:{remaining % 60:02d}"

    def _effective_session_working_text(
        self,
        working_text: str | None,
        subagents: tuple[SubagentSnapshot, ...],
    ) -> str | None:
        if not any(subagent.status in {"running", "working"} for subagent in subagents):
            return working_text
        normalized = "" if working_text is None else working_text.strip().lower()
        if normalized in {"", "thinking", "loading model"}:
            return "Waiting"
        return working_text

    def _session_header_lines(
        self,
        session: SessionRecord,
        model: str,
    ) -> tuple[str, ...]:
        del model
        return (self._session_location_line(session),)

    def _session_location_line(self, session: SessionRecord) -> str:
        return str(session.cwd or self.cwd)

    def _session_header_meta(self, session: SessionRecord, provider: str, model: str) -> str:
        model_label = self._model_header_label(provider, model)
        parts = [session.session_id[:8], f"{provider}/{model_label}"]
        context_status = self._context_status(session, model)
        if context_status:
            parts.append(context_status)
        return " · ".join(parts)

    def _model_header_label(self, provider: str, model: str) -> str:
        marker = self._thinking_intensity_marker(provider, model)
        return f"{model} {marker}" if marker else model

    def _thinking_intensity_marker(self, provider: str, model: str) -> str:
        config = self._load_config_cached()
        intensity = normalize_thinking_intensity(config.get("thinking_intensity"))
        supported = {option.value for option in thinking_intensity_options(provider, model)}
        if intensity not in supported:
            return ""
        return {
            "low": "(L)",
            "medium": "(M)",
            "high": "(H)",
            "xhigh": "(X)",
        }.get(intensity, "")

    def _context_status(self, session: SessionRecord, model: str) -> str:
        usage_snapshot = self._session_usage_snapshots.get(session.path)
        usage_tokens = usage_snapshot.context_tokens if usage_snapshot is not None else 0
        cache_key = self._session_cache_key(session.path)
        if cache_key is not None:
            cached = self._context_status_cache.get(session.path)
            if (
                cached is not None
                and cached[0] == cache_key[0]
                and cached[1] == cache_key[1]
                and cached[2] == model
                and cached[4] == usage_tokens
            ):
                return cached[3]

        if not self._has_user_messages(session.path):
            status = ""
        else:
            used_tokens = usage_tokens or self.runtime.estimate_session_context_tokens(session.path)
            status = format_token_count(used_tokens) if used_tokens > 0 else ""

        if cache_key is not None:
            self._context_status_cache[session.path] = (
                cache_key[0],
                cache_key[1],
                model,
                status,
                usage_tokens,
            )
        return status

    def _estimate_context_tokens(self, session_path: Path) -> int:
        return self.runtime.estimate_session_context_tokens(session_path)
