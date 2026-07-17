"""Reusable menus, overlays, text popovers, and approval/question popups."""

from __future__ import annotations

import curses
import textwrap
from collections.abc import Sequence
from contextlib import suppress

from anomx.agent.helpers.tool_manager import (
    ApprovalChoice,
    CommandApprovalRequest,
)
from anomx.agent.runtime import (
    QuestionOption,
    QuestionRequest,
    QuestionResponse,
)
from anomx.agent.store import (
    ProjectRecord,
    SessionRecord,
)
from anomx.agent.ui.models import (
    BottomPanel,
    CursesWindow,
    MenuChoice,
    PromptPasteEvent,
)


class PopupComponentMixin:
    """Reusable menus, overlays, text popovers, and approval/question popups."""

    @staticmethod
    def _insert_single_line_paste(value: str, cursor: int, text: str) -> tuple[str, int]:
        """Insert a terminal paste event into a single-line input.

        Clipboard tools commonly include a trailing newline.  A text popover is
        intentionally single-line, so keep the pasted secret or value intact
        while discarding line separators rather than treating them as submit
        keystrokes.
        """
        paste = text.replace("\r", "").replace("\n", "")
        return value[:cursor] + paste + value[cursor:], cursor + len(paste)

    def _menu(
        self,
        stdscr: CursesWindow,
        title: str,
        subtitle: str,
        choices: tuple[MenuChoice, ...],
    ) -> str | None:
        return self._run_overlay_menu(stdscr, title, subtitle, choices)

    def _bottom_menu(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        title: str,
        subtitle: str,
        choices: tuple[MenuChoice, ...],
        restore_nodelay: bool = False,
        autonomous_value: str | None = None,
        scroll: int = 0,
        anchor_line: int | None = None,
        prompt_notice: str = "",
        prompt_notice_role: str = "light",
    ) -> str | None:
        selected = 0
        current_scroll = scroll
        with suppress(curses.error):
            stdscr.nodelay(False)
        try:
            while True:
                messages = self._read_message_lines(session.path)
                panel = BottomPanel(title, subtitle, choices, selected)
                viewport = self._draw_session(
                    stdscr,
                    session,
                    messages,
                    "",
                    0,
                    current_scroll,
                    bottom_panel=panel,
                    anchor_line=anchor_line,
                    prompt_notice=prompt_notice,
                    prompt_notice_role=prompt_notice_role,
                )
                if viewport is not None:
                    current_scroll = viewport.scroll
                key = stdscr.get_wch()
                if self._is_escape(key) or self._is_ctrl_c(key):
                    return None
                if self._is_shift_tab(key):
                    self._cycle_agent_mode()
                    continue
                if key == curses.KEY_UP:
                    selected = max(0, selected - 1)
                elif key == curses.KEY_DOWN:
                    selected = min(len(choices) - 1, selected + 1)
                elif key == curses.KEY_PPAGE:
                    panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                    page_size = max(1, len(panel_viewport.visible_indices))
                    selected = max(0, selected - page_size)
                elif key == curses.KEY_NPAGE:
                    panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                    page_size = max(1, len(panel_viewport.visible_indices))
                    selected = min(len(choices) - 1, selected + page_size)
                elif key == curses.KEY_MOUSE:
                    choice = self._bottom_panel_mouse_choice(stdscr, panel)
                    if choice is not None:
                        return choices[choice].value
                elif self._is_enter(key):
                    return choices[selected].value
        finally:
            if restore_nodelay:
                with suppress(curses.error):
                    stdscr.nodelay(True)

    def _project_bottom_menu(
        self,
        stdscr: CursesWindow,
        project: ProjectRecord,
        title: str,
        subtitle: str,
        choices: tuple[MenuChoice, ...],
        *,
        sessions: Sequence[SessionRecord],
        session_selected: int,
        scroll: int = 0,
    ) -> str | None:
        if not choices:
            return None
        selected = 0
        current_scroll = scroll
        visible_sessions = list(sessions)
        with suppress(curses.error):
            stdscr.nodelay(False)
        while True:
            if not visible_sessions:
                visible_sessions = self._project_sessions(project.path)
            session_selected = (
                max(0, min(session_selected, len(visible_sessions) - 1)) if visible_sessions else 0
            )
            panel = BottomPanel(title, subtitle, choices, selected)
            current_scroll = self._draw_project(
                stdscr,
                project,
                visible_sessions,
                session_selected,
                current_scroll,
                "",
                0,
                "",
                "light",
                0,
                bottom_panel=panel,
            )
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                return None
            if self._is_shift_tab(key):
                self._cycle_agent_mode()
                continue
            if key == curses.KEY_UP:
                selected = max(0, selected - 1)
            elif key == curses.KEY_DOWN:
                selected = min(len(choices) - 1, selected + 1)
            elif key == curses.KEY_PPAGE:
                panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                page_size = max(1, len(panel_viewport.visible_indices))
                selected = max(0, selected - page_size)
            elif key == curses.KEY_NPAGE:
                panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                page_size = max(1, len(panel_viewport.visible_indices))
                selected = min(len(choices) - 1, selected + page_size)
            elif key == curses.KEY_MOUSE:
                choice = self._bottom_panel_mouse_choice(stdscr, panel)
                if choice is not None:
                    return choices[choice].value
            elif self._is_enter(key):
                return choices[selected].value

    def _run_overlay_menu(
        self,
        stdscr: CursesWindow,
        title: str,
        subtitle: str = "",
        choices: tuple[MenuChoice, ...] = (),
        footer: str = "Esc Back · ↑↓ Navigate · Enter Select",
    ) -> str | None:
        """Run an overlay menu loop."""
        if not choices:
            return None
        selected = 0
        while True:
            self._draw_overlay(
                stdscr,
                title=title,
                subtitle=subtitle,
                choices=choices,
                selected=selected,
                footer=footer,
            )
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                return None
            if key == curses.KEY_UP:
                selected = max(0, selected - 1)
            elif key == curses.KEY_DOWN:
                selected = min(len(choices) - 1, selected + 1)
            elif self._is_enter(key):
                return choices[selected].value

    def _run_overlay_text(
        self,
        stdscr: CursesWindow,
        title: str,
        subtitle: str = "",
        mask: bool = False,
        optional: bool = True,
        default: str = "",
        footer: str = "Esc Cancel · Enter Save",
    ) -> str | None:
        """Run an overlay single-line text input loop."""
        value = default
        cursor = len(value)
        while True:
            display_value = "*" * len(value) if mask else value
            self._draw_overlay(
                stdscr,
                title=title,
                subtitle=subtitle,
                input_value=display_value,
                input_cursor=cursor,
                footer=footer,
                show_input_cursor=True,
            )
            key = self._read_prompt_key(stdscr)
            if isinstance(key, PromptPasteEvent):
                value, cursor = self._insert_single_line_paste(value, cursor, key.text)
                continue
            if self._is_escape(key) or self._is_ctrl_c(key):
                return None if optional else ""
            if self._is_enter(key) and (value or optional):
                return value
            if self._is_option_left(key):
                cursor = self._previous_prompt_word(value, cursor)
            elif self._is_option_right(key):
                cursor = self._next_prompt_word(value, cursor)
            elif self._is_option_delete(key):
                value, cursor = self._delete_previous_prompt_word(value, cursor)
            elif key == curses.KEY_LEFT:
                cursor = max(0, cursor - 1)
            elif key == curses.KEY_RIGHT:
                cursor = min(len(value), cursor + 1)
            elif key == curses.KEY_HOME:
                cursor = 0
            elif key == curses.KEY_END:
                cursor = len(value)
            elif self._is_backspace(key):
                if cursor > 0:
                    value = value[: cursor - 1] + value[cursor:]
                    cursor -= 1
            elif isinstance(key, str) and key.isprintable():
                value = value[:cursor] + key + value[cursor:]
                cursor += 1
        height, width = stdscr.getmaxyx()
        attr = self._attr("background")
        for y in range(height):
            self._add(stdscr, y, 0, " " * max(1, width), width, attr)

    def _prompt_text(
        self,
        stdscr: CursesWindow,
        title: str,
        label: str,
        mask: bool = False,
        optional: bool = True,
        default: str = "",
    ) -> str | None:
        return self._run_overlay_text(
            stdscr,
            title,
            label,
            mask=mask,
            optional=optional,
            default=default,
        )

    def _prompt_popover_text(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        title: str,
        label: str,
        mask: bool = False,
        optional: bool = True,
        default: str = "",
    ) -> str | None:
        value = default
        cursor = len(value)
        while True:
            self._draw_session(
                stdscr,
                session,
                self._read_message_lines(session.path),
                "",
                0,
                0,
                prompt_hint_suffix=" · Esc Cancel · Enter Save",
            )
            display_value = "*" * len(value) if mask else value
            self._draw_text_popover(
                stdscr,
                title,
                label,
                display_value,
                cursor,
                "",
            )
            key = self._read_prompt_key(stdscr)
            if isinstance(key, PromptPasteEvent):
                value, cursor = self._insert_single_line_paste(value, cursor, key.text)
                continue
            if self._is_escape(key) or self._is_ctrl_c(key):
                return None if optional else ""
            if self._is_enter(key) and (value or optional):
                return value
            if self._is_option_left(key):
                cursor = self._previous_prompt_word(value, cursor)
            elif self._is_option_right(key):
                cursor = self._next_prompt_word(value, cursor)
            elif self._is_option_delete(key):
                value, cursor = self._delete_previous_prompt_word(value, cursor)
            elif key == curses.KEY_LEFT:
                cursor = max(0, cursor - 1)
            elif key == curses.KEY_RIGHT:
                cursor = min(len(value), cursor + 1)
            elif key == curses.KEY_HOME:
                cursor = 0
            elif key == curses.KEY_END:
                cursor = len(value)
            elif self._is_backspace(key):
                if cursor > 0:
                    value = value[: cursor - 1] + value[cursor:]
                    cursor -= 1
            elif isinstance(key, str) and key.isprintable():
                value = value[:cursor] + key + value[cursor:]
                cursor += 1

    def _prompt_project_popover_text(
        self,
        stdscr: CursesWindow,
        project: ProjectRecord,
        title: str,
        label: str,
        mask: bool = False,
        optional: bool = True,
        default: str = "",
        session_selected: int = 0,
        scroll: int = 0,
    ) -> str | None:
        value = default
        cursor = len(value)
        current_scroll = scroll
        while True:
            sessions = self._project_sessions(project.path)
            current_scroll = self._draw_project(
                stdscr,
                project,
                sessions,
                session_selected,
                current_scroll,
                "",
                0,
                "",
                "light",
                0,
                prompt_hint_suffix=" · Esc Cancel · Enter Save",
            )
            display_value = "*" * len(value) if mask else value
            self._draw_text_popover(
                stdscr,
                title,
                label,
                display_value,
                cursor,
                "",
            )
            key = self._read_prompt_key(stdscr)
            if isinstance(key, PromptPasteEvent):
                value, cursor = self._insert_single_line_paste(value, cursor, key.text)
                continue
            if self._is_escape(key) or self._is_ctrl_c(key):
                return None if optional else ""
            if self._is_enter(key) and (value or optional):
                return value
            if self._is_option_left(key):
                cursor = self._previous_prompt_word(value, cursor)
            elif self._is_option_right(key):
                cursor = self._next_prompt_word(value, cursor)
            elif self._is_option_delete(key):
                value, cursor = self._delete_previous_prompt_word(value, cursor)
            elif key == curses.KEY_LEFT:
                cursor = max(0, cursor - 1)
            elif key == curses.KEY_RIGHT:
                cursor = min(len(value), cursor + 1)
            elif key == curses.KEY_HOME:
                cursor = 0
            elif key == curses.KEY_END:
                cursor = len(value)
            elif self._is_backspace(key):
                if cursor > 0:
                    value = value[: cursor - 1] + value[cursor:]
                    cursor -= 1
            elif isinstance(key, str) and key.isprintable():
                value = value[:cursor] + key + value[cursor:]
                cursor += len(key)

    def _draw_text_popover(
        self,
        stdscr: CursesWindow,
        title: str,
        label: str,
        value: str,
        cursor: int,
        footer: str,
    ) -> None:
        layout = self._prompt_layout(stdscr, "")
        _, width = stdscr.getmaxyx()
        panel_width = max(1, width - 4)
        left = 2
        popover_rows = 6 if footer else 5
        start_y = max(4, layout.top_line - popover_rows)
        end_y = max(start_y, layout.top_line - 1)
        for y in range(start_y, end_y + 1):
            self._clear_row(stdscr, y)
        self._add(stdscr, start_y, left, "─" * panel_width, panel_width, self._attr("accent"))
        self._add(stdscr, start_y + 1, left + 2, title, panel_width - 4, self._attr("accent"))
        self._add(stdscr, start_y + 2, left + 2, label, panel_width - 4, self._attr("light"))
        input_y = start_y + 4
        self._add(stdscr, input_y, left + 2, value, panel_width - 4, curses.A_NORMAL)
        self._paint_visual_cursor(stdscr, input_y, left + 2, value, cursor)
        if footer:
            footer_y = start_y + 5
            self._add(
                stdscr,
                footer_y,
                left,
                " " * panel_width,
                panel_width,
                self._attr("selected"),
            )
            self._add(stdscr, footer_y, left + 2, footer, panel_width - 4, self._attr("selected"))
        stdscr.refresh()

    def _message(self, stdscr: CursesWindow, title: str, message: str) -> None:
        body_lines = tuple(textwrap.wrap(message, width=60) or [message])
        while True:
            self._draw_overlay(
                stdscr,
                title=title,
                body_lines=body_lines,
                footer="Esc Back · Enter Continue",
            )
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key) or self._is_enter(key):
                return

    def _request_command_approval(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        request: CommandApprovalRequest,
        scroll: int = 0,
        anchor_line: int | None = None,
    ) -> ApprovalChoice:
        self._approval_memory_reason = ""
        if request.evaluation is not None:
            return self._request_evaluated_command_approval(
                stdscr,
                session,
                request,
                scroll=scroll,
                anchor_line=anchor_line,
            )
        allowance_label = request.allowance_label or "matching commands"
        allowance_subject = request.allowance_subject or "this command"
        title = request.reason
        selected = self._bottom_menu(
            stdscr,
            session,
            title,
            request.command,
            (
                MenuChoice("Approve", ApprovalChoice.ALLOW.value, "Run this command once"),
                MenuChoice("Reject", ApprovalChoice.REJECT.value, "Do not run this command"),
                MenuChoice(
                    "Approve always",
                    ApprovalChoice.ALWAYS_ALLOW.value,
                    f"Trust {allowance_label} globally for {allowance_subject}",
                ),
                MenuChoice(
                    "Reject always, because ...",
                    ApprovalChoice.ALWAYS_REJECT.value,
                    f"Block {allowance_label} globally and save your reason as a memory",
                ),
            ),
            restore_nodelay=True,
            autonomous_value=ApprovalChoice.ALLOW.value,
            scroll=scroll,
            anchor_line=anchor_line,
        )
        if selected == ApprovalChoice.ALLOW.value:
            return ApprovalChoice.ALLOW
        if selected == ApprovalChoice.ALWAYS_ALLOW.value:
            return ApprovalChoice.ALWAYS_ALLOW
        if selected == ApprovalChoice.ALWAYS_REJECT.value:
            reason = self._request_approval_memory_reason(stdscr, request)
            if reason is None:
                return ApprovalChoice.REJECT
            self._approval_memory_reason = reason
            return ApprovalChoice.ALWAYS_REJECT
        return ApprovalChoice.REJECT

    def _request_evaluated_command_approval(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        request: CommandApprovalRequest,
        scroll: int = 0,
        anchor_line: int | None = None,
    ) -> ApprovalChoice:
        selected = 0
        current_scroll = scroll
        command_scroll = 0
        show_command = False
        choices = self._command_approval_choices(request)
        with suppress(curses.error):
            stdscr.nodelay(False)
        try:
            while True:
                panel = self._command_approval_panel(
                    request,
                    choices,
                    selected,
                    show_command=show_command,
                    command_scroll=command_scroll,
                )
                messages = self._read_message_lines(session.path)
                viewport = self._draw_session(
                    stdscr,
                    session,
                    messages,
                    "",
                    0,
                    current_scroll,
                    bottom_panel=panel,
                    anchor_line=anchor_line,
                )
                if viewport is not None:
                    current_scroll = viewport.scroll
                key = stdscr.get_wch()
                if self._is_escape(key) or self._is_ctrl_c(key):
                    return ApprovalChoice.REJECT
                if self._is_shift_tab(key):
                    self._cycle_agent_mode()
                    continue
                if key == curses.KEY_UP:
                    selected = max(0, selected - 1)
                elif key == curses.KEY_DOWN:
                    selected = min(len(choices) - 1, selected + 1)
                elif key == curses.KEY_PPAGE and show_command:
                    command_scroll = max(0, command_scroll - 5)
                elif key == curses.KEY_NPAGE and show_command:
                    command_scroll = min(
                        self._command_approval_command_max_scroll(stdscr, request),
                        command_scroll + 5,
                    )
                elif key == curses.KEY_PPAGE:
                    panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                    page_size = max(1, len(panel_viewport.visible_indices))
                    selected = max(0, selected - page_size)
                elif key == curses.KEY_NPAGE:
                    panel_viewport = self._bottom_panel_viewport(stdscr, panel)
                    page_size = max(1, len(panel_viewport.visible_indices))
                    selected = min(len(choices) - 1, selected + page_size)
                elif key == curses.KEY_MOUSE:
                    mouse = self._approval_panel_mouse_event()
                    if mouse is None:
                        continue
                    _x, y, button_state = mouse
                    if self._is_left_click(button_state) and self._bottom_panel_subtitle_hit(
                        stdscr,
                        panel,
                        y,
                    ):
                        show_command = not show_command
                        command_scroll = 0
                        continue
                    if self._is_left_click(button_state):
                        choice = self._bottom_panel_mouse_choice_at(stdscr, panel, y)
                        if choice is not None:
                            return self._approval_choice_for_value(
                                stdscr,
                                request,
                                choices[choice].value,
                            )
                elif self._is_enter(key):
                    return self._approval_choice_for_value(
                        stdscr,
                        request,
                        choices[selected].value,
                    )
        finally:
            with suppress(curses.error):
                stdscr.nodelay(True)

    def _command_approval_choices(
        self,
        request: CommandApprovalRequest,
    ) -> tuple[MenuChoice, ...]:
        allowance_label = request.allowance_label or "matching commands"
        allowance_subject = request.allowance_subject or "this command"
        return (
            MenuChoice("Approve", ApprovalChoice.ALLOW.value, "Run this command once"),
            MenuChoice("Reject", ApprovalChoice.REJECT.value, "Do not run this command"),
            MenuChoice(
                "Approve always",
                ApprovalChoice.ALWAYS_ALLOW.value,
                f"Trust {allowance_label} globally for {allowance_subject}",
            ),
            MenuChoice(
                "Reject always, because ...",
                ApprovalChoice.ALWAYS_REJECT.value,
                f"Block {allowance_label} globally and save your reason as a memory",
            ),
        )

    def _command_approval_panel(
        self,
        request: CommandApprovalRequest,
        choices: tuple[MenuChoice, ...],
        selected: int,
        *,
        show_command: bool,
        command_scroll: int,
    ) -> BottomPanel:
        evaluation = request.evaluation
        risk = evaluation.risk if evaluation is not None else ""
        subtitle = request.command if show_command else (
            evaluation.description if evaluation is not None else request.command
        )
        return BottomPanel(
            request.reason,
            subtitle,
            choices,
            selected,
            title_attr="bold",
            title_suffix="(Click for description)" if show_command else "(Click for command)",
            title_prefix=self._command_risk_label(risk),
            title_prefix_attr=self._command_risk_attr(risk),
            subtitle_max_lines=5 if show_command else 4,
            subtitle_scroll=command_scroll if show_command else 0,
        )

    def _approval_choice_for_value(
        self,
        stdscr: CursesWindow,
        request: CommandApprovalRequest,
        value: str,
    ) -> ApprovalChoice:
        if value == ApprovalChoice.ALLOW.value:
            return ApprovalChoice.ALLOW
        if value == ApprovalChoice.ALWAYS_ALLOW.value:
            return ApprovalChoice.ALWAYS_ALLOW
        if value == ApprovalChoice.ALWAYS_REJECT.value:
            reason = self._request_approval_memory_reason(stdscr, request)
            if reason is None:
                return ApprovalChoice.REJECT
            self._approval_memory_reason = reason
            return ApprovalChoice.ALWAYS_REJECT
        return ApprovalChoice.REJECT

    def _request_approval_memory_reason(
        self,
        stdscr: CursesWindow,
        request: CommandApprovalRequest,
    ) -> str | None:
        del request
        prompt = getattr(self, "_prompt_multiline_text", None)
        if callable(prompt):
            reason = prompt(
                stdscr,
                "Reject Always",
                (
                    "Why should matching commands be rejected? "
                    "Ctrl+S saves this reason as a memory."
                ),
                optional=False,
            )
        else:
            reason = self._run_overlay_text(
                stdscr,
                "Reject Always",
                "Why should matching commands be rejected?",
                optional=False,
            )
        if reason is None or not reason.strip():
            return None
        return reason.strip()

    def _command_risk_label(self, risk: str) -> str:
        normalized = risk.strip().lower()
        if normalized == "low":
            return "Low Risk"
        if normalized == "medium":
            return "Medium Risk"
        if normalized == "high":
            return "High Risk"
        return ""

    def _command_risk_attr(self, risk: str) -> str:
        normalized = risk.strip().lower()
        if normalized == "low":
            return "ok"
        if normalized == "medium":
            return "warning"
        if normalized == "high":
            return "danger"
        return "accent"

    def _command_approval_command_max_scroll(
        self,
        stdscr: CursesWindow,
        request: CommandApprovalRequest,
    ) -> int:
        _, width = stdscr.getmaxyx()
        lines = self._panel_text_lines(request.command, max(1, width - 8))
        return max(0, len(lines) - 5)

    def _approval_panel_mouse_event(self) -> tuple[int, int, int] | None:
        with suppress(curses.error):
            _mouse_id, x, y, _z, button_state = curses.getmouse()
            return x, y, button_state
        return None

    def _request_question(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        request: QuestionRequest,
        scroll: int = 0,
        anchor_line: int | None = None,
    ) -> QuestionResponse:
        if request.kind == "text":
            return self._bottom_text_question(
                stdscr,
                session,
                request,
                scroll=scroll,
                anchor_line=anchor_line,
            )
        if request.kind == "confirm":
            return self._bottom_select_question(
                stdscr,
                session,
                self._confirm_question_request(request),
                scroll=scroll,
                anchor_line=anchor_line,
            )
        return self._bottom_select_question(
            stdscr,
            session,
            request,
            scroll=scroll,
            anchor_line=anchor_line,
        )

    def _bottom_select_question(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        request: QuestionRequest,
        scroll: int = 0,
        anchor_line: int | None = None,
    ) -> QuestionResponse:
        if not request.options and request.allow_custom:
            return self._bottom_text_question(
                stdscr,
                session,
                request,
                scroll=scroll,
                anchor_line=anchor_line,
            )

        choices = [
            MenuChoice(option.label, option.value, option.description) for option in request.options
        ]
        custom_value = "__custom_question_answer__"
        if request.allow_custom:
            choices.append(
                MenuChoice(
                    "Custom response",
                    custom_value,
                    request.placeholder or "Type a custom answer",
                )
            )
        selected = self._bottom_menu(
            stdscr,
            session,
            "Question",
            request.question,
            tuple(choices),
            restore_nodelay=True,
            scroll=scroll,
            anchor_line=anchor_line,
        )
        if selected is None:
            return QuestionResponse(
                answered=False,
                kind=request.kind,
                cancelled=True,
            )
        if selected == custom_value:
            return self._bottom_text_question(
                stdscr,
                session,
                request,
                scroll=scroll,
                anchor_line=anchor_line,
            )

        option_by_value = {option.value: option for option in request.options}
        option = option_by_value.get(selected)
        return QuestionResponse(
            answered=True,
            answer=selected,
            selected_label=option.label if option is not None else selected,
            kind=request.kind,
        )

    def _bottom_text_question(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        request: QuestionRequest,
        scroll: int = 0,
        anchor_line: int | None = None,
    ) -> QuestionResponse:
        value = request.default
        cursor = len(value)
        current_scroll = scroll
        with suppress(curses.error):
            stdscr.nodelay(False)
        try:
            while True:
                viewport = self._draw_session(
                    stdscr,
                    session,
                    self._read_message_lines(session.path),
                    "",
                    0,
                    current_scroll,
                    anchor_line=anchor_line,
                )
                if viewport is not None:
                    current_scroll = viewport.scroll
                self._draw_text_popover(
                    stdscr,
                    "Question",
                    request.question,
                    value,
                    cursor,
                    "Esc Cancel · Enter Submit",
                )
                key = self._read_prompt_key(stdscr)
                if isinstance(key, PromptPasteEvent):
                    value, cursor = self._insert_single_line_paste(value, cursor, key.text)
                    continue
                if self._is_escape(key) or self._is_ctrl_c(key):
                    return QuestionResponse(
                        answered=False,
                        kind=request.kind,
                        cancelled=True,
                    )
                if self._is_enter(key):
                    return QuestionResponse(
                        answered=True,
                        answer=value.strip(),
                        selected_label="",
                        kind=request.kind,
                    )
                if self._is_option_left(key):
                    cursor = self._previous_prompt_word(value, cursor)
                elif self._is_option_right(key):
                    cursor = self._next_prompt_word(value, cursor)
                elif self._is_option_delete(key):
                    value, cursor = self._delete_previous_prompt_word(value, cursor)
                elif key == curses.KEY_LEFT:
                    cursor = max(0, cursor - 1)
                elif key == curses.KEY_RIGHT:
                    cursor = min(len(value), cursor + 1)
                elif key == curses.KEY_HOME:
                    cursor = 0
                elif key == curses.KEY_END:
                    cursor = len(value)
                elif self._is_backspace(key):
                    if cursor > 0:
                        value = value[: cursor - 1] + value[cursor:]
                        cursor -= 1
                elif isinstance(key, str) and key.isprintable():
                    value = value[:cursor] + key + value[cursor:]
                    cursor += len(key)
        finally:
            with suppress(curses.error):
                stdscr.nodelay(True)

    def _confirm_question_request(self, request: QuestionRequest) -> QuestionRequest:
        if request.options:
            return request
        return QuestionRequest(
            question=request.question,
            kind=request.kind,
            options=(
                QuestionOption("Yes", "yes", "Confirm and continue"),
                QuestionOption("No", "no", "Cancel or choose another path"),
            ),
            placeholder=request.placeholder,
            default=request.default,
            allow_custom=False,
        )

    def _append_question_context(
        self,
        session: SessionRecord,
        request: QuestionRequest,
        answer: QuestionResponse,
    ) -> None:
        if answer.cancelled:
            message = f"User cancelled question: {request.question}"
        else:
            message = f"Question: {request.question}\nAnswer: {answer.answer}"
        self.home.append_session_event(
            session.path,
            "system_message",
            {
                "message": message,
                "role": "question",
            },
        )
