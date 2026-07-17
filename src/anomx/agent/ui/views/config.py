"""Configuration, model, skill, and platform management views."""

from __future__ import annotations

import curses
import queue
import shutil
import textwrap
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.platform_client import (
    PlatformClientError,
    PlatformLoginResult,
    connect_platform,
    heartbeat_platform_connection,
    platform_domain,
)
from anomx.agent.helpers.tool_manager import command_allowance_display
from anomx.agent.memories import (
    MemoryKind,
    MemoryRecord,
    create_memory_record,
    load_memories,
    write_memory,
)
from anomx.agent.skills import (
    Skill,
    is_valid_skill_command,
    normalize_skill_command,
    sync_builtin_skills,
    write_user_skill,
)
from anomx.agent.store import (
    AI_PROVIDERS,
    ProjectRecord,
    ProviderOption,
    SessionRecord,
    model_detail,
    provider_by_key,
    thinking_intensity_options,
)
from anomx.agent.ui.constants import (
    STARTUP_FRAME_SECONDS,
)
from anomx.agent.ui.models import (
    AgentState,
    CursesWindow,
    InfoRow,
    MenuChoice,
    PlatformConnectionDraft,
    SkillFormDraft,
)


class ConfigViewMixin:
    """Configuration, model, skill, and platform management views."""

    def _run_skills_panel(self, stdscr: CursesWindow, current_session: SessionRecord) -> None:
        self.state = AgentState.SKILLS
        while True:
            selected = self._menu(
                stdscr,
                "Skills",
                "Create or open a user skill",
                self._skills_menu_choices(),
            )
            if selected is None:
                self.state = AgentState.NEW_SESSION
                return
            if selected == "__create_skill__":
                self._create_user_skill(stdscr)
                continue
            skill = self._user_skill_by_command(selected)
            if skill is not None:
                self._run_skill_detail_panel(stdscr, skill)

    def _run_memories_panel(self, stdscr: CursesWindow) -> None:
        self.state = AgentState.INFO
        while True:
            selected = self._menu(
                stdscr,
                "Memories",
                "Create or open a local agent memory",
                self._memories_menu_choices(),
            )
            if selected is None:
                self.state = AgentState.CONFIG
                return
            if selected == "__create_memory__":
                self._create_manual_memory(stdscr)
                continue
            memory = self._memory_by_path(selected)
            if memory is not None:
                self._run_memory_detail_panel(stdscr, memory)

    def _memories_menu_choices(self) -> tuple[MenuChoice, ...]:
        choices = [
            MenuChoice(
                "Create new Memory",
                "__create_memory__",
                "Write a durable local memory for future agent turns",
            )
        ]
        choices.extend(
            MenuChoice(
                memory.title,
                str(memory.path or ""),
                f"{memory.kind.value} · {memory.summary}",
            )
            for memory in load_memories(self.home.brain_dir)
        )
        return tuple(choices)

    def _create_manual_memory(self, stdscr: CursesWindow) -> MemoryRecord | None:
        content = self._prompt_multiline_text(
            stdscr,
            "Create Memory",
            "Memory content. Ctrl+S saves.",
            optional=False,
        )
        if content is None or not content.strip():
            return None
        context = {"source": "config"}
        metadata = self.runtime.suggest_memory_metadata(
            kind=MemoryKind.MANUAL,
            context=context,
            content=content,
        )
        record = create_memory_record(
            title=metadata.title,
            summary=metadata.summary,
            kind=MemoryKind.MANUAL,
            context=context,
            content=content,
        )
        saved = write_memory(self.home.brain_dir, record)
        self._message(stdscr, "Create Memory", f"Saved {saved.title}.")
        return saved

    def _memory_by_path(self, selected: str) -> MemoryRecord | None:
        for memory in load_memories(self.home.brain_dir):
            if str(memory.path or "") == selected:
                return memory
        return None

    def _run_memory_detail_panel(self, stdscr: CursesWindow, memory: MemoryRecord) -> None:
        current = memory
        while True:
            self._draw_memory_detail_panel(stdscr, current)
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key) or self._is_enter(key):
                return
            if self._is_ctrl_d(key) and self._delete_memory(stdscr, current):
                return

    def _draw_memory_detail_panel(self, stdscr: CursesWindow, memory: MemoryRecord) -> None:
        body_lines: list[str] = [
            f"Kind: {memory.kind.value}",
            f"Created: {memory.created_at}",
            f"Uses: {memory.uses}",
            f"Summary: {memory.summary}",
            "",
            "Context:",
        ]
        if memory.context:
            body_lines.extend(f"  {key}: {value}" for key, value in memory.context.items())
        else:
            body_lines.append("  none")
        body_lines.extend(["", "Content:"])
        body_lines.extend(memory.content.splitlines() or [""])
        self._draw_overlay(
            stdscr,
            title=f"Memory {memory.title}",
            subtitle=self._memory_detail_path_line(memory),
            body_lines=tuple(body_lines),
            footer="Esc Back · Enter Back · Ctrl+D Delete",
        )

    def _memory_detail_path_line(self, memory: MemoryRecord) -> str:
        if memory.path is None:
            return "Stored at: unknown"
        return f"Stored at: {memory.path}"

    def _delete_memory(self, stdscr: CursesWindow, memory: MemoryRecord) -> bool:
        selected = self._menu(
            stdscr,
            "Delete Memory",
            f"Delete {memory.title}?",
            (
                MenuChoice("Cancel", "cancel", "Keep this memory"),
                MenuChoice("Delete Memory", "delete", "Remove this local memory"),
            ),
        )
        if selected != "delete":
            return False
        if memory.path is not None:
            with suppress(FileNotFoundError):
                memory.path.unlink()
        self._message(stdscr, "Delete Memory", f"Deleted {memory.title}.")
        return True

    def _skills_menu_choices(self) -> tuple[MenuChoice, ...]:
        choices = [
            MenuChoice(
                "Create new Skill",
                "__create_skill__",
                "Define a global slash-command skill",
            )
        ]
        choices.extend(
            MenuChoice(f"/{skill.command}", skill.command, skill.description)
            for skill in self._user_skills()
            if not skill.hidden
        )
        return tuple(choices)

    def _create_user_skill(self, stdscr: CursesWindow) -> Skill | None:
        saved = self._run_skill_editor(stdscr, title="Create Skill")
        if saved is not None:
            self._message(stdscr, "Create Skill", f"Saved /{saved.command}.")
        return saved

    def _edit_user_skill(self, stdscr: CursesWindow, skill: Skill) -> Skill | None:
        return self._run_skill_editor(stdscr, title="Edit Skill", existing_skill=skill)

    def _run_skill_editor(
        self,
        stdscr: CursesWindow,
        *,
        title: str,
        existing_skill: Skill | None = None,
    ) -> Skill | None:
        draft = self._skill_form_draft(existing_skill)
        cursor = 0
        selected = 0
        while True:
            self._draw_skill_editor_panel(stdscr, title, draft, selected, cursor=cursor)
            stdscr.refresh()
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                return None
            if self._is_ctrl_s(key):
                saved = self._save_skill_draft(stdscr, draft, existing_skill)
                if saved is not None:
                    return saved
                continue
            if key == curses.KEY_UP:
                draft = self._skill_commit_field(draft, selected, cursor)
                selected = max(0, selected - 1)
                cursor = self._skill_field_cursor(draft, selected)
                continue
            if key == curses.KEY_DOWN:
                draft = self._skill_commit_field(draft, selected, cursor)
                selected = min(2, selected + 1)
                cursor = self._skill_field_cursor(draft, selected)
                continue
            if self._is_option_left(key):
                cursor = self._previous_prompt_word(
                    self._skill_editor_value(draft, selected),
                    cursor,
                )
                continue
            if self._is_option_right(key):
                cursor = self._next_prompt_word(
                    self._skill_editor_value(draft, selected),
                    cursor,
                )
                continue
            if self._is_option_delete(key):
                value = self._skill_editor_value(draft, selected)
                value, cursor = self._delete_previous_prompt_word(value, cursor)
                draft = self._update_skill_draft(
                    draft,
                    self._skill_editor_field(selected),
                    value,
                )
                continue
            if key == curses.KEY_LEFT:
                if cursor > 0:
                    cursor -= 1
                continue
            if key == curses.KEY_RIGHT:
                value = self._skill_editor_value(draft, selected)
                if cursor < len(value):
                    cursor += 1
                continue
            if key == curses.KEY_HOME:
                cursor = 0
                continue
            if key == curses.KEY_END:
                cursor = len(self._skill_editor_value(draft, selected))
                continue
            if self._is_enter(key):
                if selected == 2:
                    value = self._skill_editor_value(draft, selected)
                    draft = self._update_skill_draft(
                        draft,
                        self._skill_editor_field(selected),
                        value[:cursor] + "\n" + value[cursor:],
                    )
                    cursor += 1
                else:
                    draft = self._skill_commit_field(draft, selected, cursor)
                    selected = min(2, selected + 1)
                    cursor = self._skill_field_cursor(draft, selected)
            elif self._is_shift_enter(key):
                draft = self._skill_commit_field(draft, selected, cursor)
                cursor = self._find_previous_field_cursor(draft, selected)
                selected = max(0, selected - 1)
                cursor = self._skill_field_cursor(draft, selected)
            elif self._is_backspace(key):
                if cursor > 0:
                    value = self._skill_editor_value(draft, selected)
                    draft = self._update_skill_draft(
                        draft,
                        self._skill_editor_field(selected),
                        value[: cursor - 1] + value[cursor:],
                    )
                    cursor -= 1
            elif isinstance(key, str) and key.isprintable():
                value = self._skill_editor_value(draft, selected)
                draft = self._update_skill_draft(
                    draft,
                    self._skill_editor_field(selected),
                    value[:cursor] + key + value[cursor:],
                )
                cursor += 1

    def _skill_field_cursor(self, draft: SkillFormDraft, field_index: int) -> int:
        if field_index == 0:
            return draft.command_cursor
        if field_index == 1:
            return draft.description_cursor
        return draft.body_cursor

    def _skill_commit_field(
        self, draft: SkillFormDraft, field_index: int, cursor: int
    ) -> SkillFormDraft:
        if field_index == 0:
            return replace(draft, command_cursor=cursor)
        if field_index == 1:
            return replace(draft, description_cursor=cursor)
        return replace(draft, body_cursor=cursor)

    def _find_previous_field_cursor(self, draft: SkillFormDraft, field_index: int) -> int:
        if field_index == 0:
            return 0
        if field_index == 1:
            return draft.command_cursor
        return draft.description_cursor

    def _skill_form_draft(self, skill: Skill | None = None) -> SkillFormDraft:
        if skill is None:
            return SkillFormDraft()
        return SkillFormDraft(
            command=skill.command,
            description=skill.description,
            body=skill.body,
            path=skill.path,
        )

    def _save_skill_draft(
        self,
        stdscr: CursesWindow,
        draft: SkillFormDraft,
        existing_skill: Skill | None,
    ) -> Skill | None:
        command = normalize_skill_command(draft.command)
        if not command or not is_valid_skill_command(command):
            self._message(
                stdscr,
                "Skill",
                (
                    "Use letters, numbers, dashes, or underscores. "
                    "Commands must start with a letter or number."
                ),
            )
            return None
        original_command = existing_skill.command if existing_skill is not None else None
        if self._command_exists(command, exclude_command=original_command):
            self._message(stdscr, "Skill", f"/{command} already exists.")
            return None
        if not draft.description.strip():
            self._message(stdscr, "Skill", "Description is required.")
            return None
        if not draft.body.strip():
            self._message(stdscr, "Skill", "Skill instructions are required.")
            return None

        skill = Skill(
            command=command,
            title=command,
            description=draft.description.strip(),
            body=draft.body.strip(),
            source="user",
        )
        path = write_user_skill(self.home.skills_dir, skill)
        old_path = existing_skill.path if existing_skill is not None else draft.path
        if old_path is not None and old_path != path:
            with suppress(FileNotFoundError):
                if old_path.is_dir():
                    shutil.rmtree(old_path)
                else:
                    old_path.unlink()
        return Skill(
            command=skill.command,
            title=skill.title,
            description=skill.description,
            body=skill.body,
            source=skill.source,
            hidden=skill.hidden,
            path=path,
        )

    def _draw_skill_editor_panel(
        self,
        stdscr: CursesWindow,
        title: str,
        draft: SkillFormDraft,
        selected: int,
        cursor: int = 0,
    ) -> None:
        if selected == 2:
            self._draw_overlay(
                stdscr,
                title=title,
                subtitle=self._skill_editor_path_line(draft),
                editor_text=draft.body,
                editor_cursor=cursor,
                footer="Esc Cancel · Ctrl+S Save · ↑↓ Navigate · Enter Next",
            )
        else:
            rows = self._skill_editor_scalar_rows(draft)
            choices = tuple(MenuChoice(row.label, str(i), row.value) for i, row in enumerate(rows))
            self._draw_overlay(
                stdscr,
                title=title,
                subtitle=self._skill_editor_path_line(draft),
                choices=choices,
                selected=selected,
                input_value=draft.body if selected == 2 else rows[selected].value,
                input_cursor=cursor,
                footer="Esc Cancel · Ctrl+S Save · ↑↓ Navigate · Enter Next",
                show_input_cursor=True,
            )

    def _skill_editor_scalar_rows(self, draft: SkillFormDraft) -> tuple[InfoRow, ...]:
        return (
            InfoRow("Command", self._skill_form_display_value("Command", draft.command)),
            InfoRow("Description", draft.description),
        )

    def _skill_editor_label(self, selected: int) -> str:
        return ("Command", "Description", "Skill")[selected]

    def _skill_editor_field(self, selected: int) -> str:
        return ("command", "description", "body")[selected]

    def _skill_editor_value(self, draft: SkillFormDraft, selected: int) -> str:
        return str(getattr(draft, self._skill_editor_field(selected)))

    def _update_skill_draft(
        self,
        draft: SkillFormDraft,
        field_name: str,
        value: str,
    ) -> SkillFormDraft:
        if field_name == "command":
            return replace(draft, command=value)
        if field_name == "description":
            return replace(draft, description=value)
        return replace(draft, body=value)

    def _skill_editor_path_line(self, draft: SkillFormDraft) -> str:
        return f"Stored at: {self._skill_editor_path(draft)}"

    def _skill_editor_path(self, draft: SkillFormDraft) -> str:
        command = normalize_skill_command(draft.command)
        if command:
            return str(self.home.skills_dir / command / "README.md")
        if draft.path is not None:
            return str(draft.path)
        return str(self.home.skills_dir / "<command>" / "README.md")

    def _skill_form_display_value(self, active_label: str, active_value: str) -> str:
        if active_label == "Command":
            return f"/{active_value.removeprefix('/')}"
        return active_value

    def _run_skill_detail_panel(self, stdscr: CursesWindow, skill: Skill) -> None:
        current_skill = skill
        while True:
            self._draw_skill_detail_panel(stdscr, current_skill)
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key) or self._is_enter(key):
                return
            if self._is_ctrl_e(key) and self._skill_manageable(current_skill):
                edited = self._edit_user_skill(stdscr, current_skill)
                if edited is not None:
                    current_skill = edited
                continue
            should_delete = self._is_ctrl_d(key) and self._skill_manageable(current_skill)
            if should_delete and self._delete_user_skill(stdscr, current_skill):
                return

    def _draw_skill_detail_panel(self, stdscr: CursesWindow, skill: Skill) -> None:
        body_lines: list[str] = [
            f"Command: /{skill.command}",
            f"Description: {skill.description}",
            "",
            "Instructions:",
        ]
        body_lines.extend(skill.body.splitlines())
        footer = self._skill_detail_footer(skill)
        self._draw_overlay(
            stdscr,
            title=f"Skill /{skill.command}",
            subtitle=self._skill_detail_path_line(skill),
            body_lines=tuple(body_lines),
            footer=footer,
        )

    def _skill_manageable(self, skill: Skill) -> bool:
        return skill.source == "user" and skill.path is not None

    def _skill_detail_footer(self, skill: Skill) -> str:
        if self._skill_manageable(skill):
            return "Esc Back · Enter Back · Ctrl+E Edit · Ctrl+D Delete"
        return "Esc Back · Enter Back"

    def _skill_detail_path_line(self, skill: Skill) -> str:
        if skill.path is not None:
            return f"Stored at: {skill.path}"
        return "Stored at: bundled skill"

    def _delete_user_skill(self, stdscr: CursesWindow, skill: Skill) -> bool:
        selected = self._menu(
            stdscr,
            "Delete Skill",
            f"Delete /{skill.command}?",
            (
                MenuChoice("Cancel", "cancel", "Keep this skill"),
                MenuChoice("Delete Skill", "delete", "Remove this global skill"),
            ),
        )
        if selected != "delete":
            return False
        if skill.path is not None:
            with suppress(FileNotFoundError):
                if skill.path.is_dir():
                    shutil.rmtree(skill.path)
                else:
                    skill.path.unlink()
        self._message(stdscr, "Delete Skill", f"Deleted /{skill.command}.")
        return True

    def _prompt_multiline_text(
        self,
        stdscr: CursesWindow,
        title: str,
        label: str,
        optional: bool = True,
    ) -> str | None:
        value = ""
        cursor = 0
        while True:
            self._draw_overlay(
                stdscr,
                title=title,
                subtitle=label,
                editor_text=value,
                editor_cursor=cursor,
                footer="Esc Cancel · Enter New line · Ctrl+S Save",
            )
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                return None if optional else ""
            if self._is_ctrl_s(key):
                if value.strip() or optional:
                    return value
                continue
            content_width = max(20, (stdscr.getmaxyx()[1]) - 8)
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
            elif key == curses.KEY_UP:
                cursor = self._cursor_move_up(value, cursor, content_width)
            elif key == curses.KEY_DOWN:
                cursor = self._cursor_move_down(value, cursor, content_width)
            elif key == curses.KEY_HOME:
                cursor = self._cursor_line_start(value, cursor)
            elif key == curses.KEY_END:
                cursor = self._cursor_line_end(value, cursor)
            elif self._is_enter(key):
                value = value[:cursor] + "\n" + value[cursor:]
                cursor += 1
            elif self._is_backspace(key):
                if cursor > 0:
                    value = value[: cursor - 1] + value[cursor:]
                    cursor -= 1
            elif isinstance(key, str) and key.isprintable():
                value = value[:cursor] + key + value[cursor:]
                cursor += 1

    def _run_model_panel(
        self,
        stdscr: CursesWindow,
        current_session: SessionRecord,
        *,
        bottom_popover: bool = True,
    ) -> bool:
        self.state = AgentState.MODEL
        config = self.home.load_config()
        provider = provider_by_key(str(config.get("provider", "openai"))) or AI_PROVIDERS[0]
        provider = self._provider_with_discovered_models(provider)
        choices = [MenuChoice(model, model, model_detail(model)) for model in provider.models]
        if provider.allow_custom_model:
            choices.append(
                MenuChoice(
                    "Custom model",
                    "__custom__",
                    f"Use a custom {provider.label} model name",
                )
            )
        selected = (
            self._bottom_menu(
                stdscr,
                current_session,
                "Model",
                f"Provider: {provider.label}",
                tuple(choices),
            )
            if bottom_popover
            else self._menu(
                stdscr,
                "Model",
                f"Provider: {provider.label}",
                tuple(choices),
            )
        )
        if selected is None:
            self.state = AgentState.NEW_SESSION
            return False
        model = (
            (
                self._prompt_popover_text(
                    stdscr,
                    current_session,
                    "Model",
                    "Model name",
                    optional=False,
                )
                if bottom_popover
                else self._prompt_text(stdscr, "Model", "Model name", optional=False)
            )
            if selected == "__custom__"
            else selected
        )
        if model:
            thinking_intensity = self._select_thinking_intensity(
                stdscr,
                provider,
                model,
                current_session=current_session if bottom_popover else None,
            )
            if thinking_intensity is None:
                self.state = AgentState.NEW_SESSION
                return False
            config["provider"] = provider.key
            config["model"] = model
            config["thinking_intensity"] = thinking_intensity
            config["onboarding_complete"] = True
            self.home.save_config(config)
        self.state = AgentState.NEW_SESSION
        return bool(model)

    def _run_project_model_panel(
        self,
        stdscr: CursesWindow,
        project: ProjectRecord,
        sessions: Sequence[SessionRecord],
        session_selected: int,
        scroll: int = 0,
    ) -> bool:
        self.state = AgentState.MODEL
        config = self.home.load_config()
        provider = provider_by_key(str(config.get("provider", "openai"))) or AI_PROVIDERS[0]
        provider = self._provider_with_discovered_models(provider)
        choices = [MenuChoice(model, model, model_detail(model)) for model in provider.models]
        if provider.allow_custom_model:
            choices.append(
                MenuChoice(
                    "Custom model",
                    "__custom__",
                    f"Use a custom {provider.label} model name",
                )
            )
        selected = self._project_bottom_menu(
            stdscr,
            project,
            "Model",
            f"Provider: {provider.label}",
            tuple(choices),
            sessions=sessions,
            session_selected=session_selected,
            scroll=scroll,
        )
        if selected is None:
            self.state = AgentState.PROJECT
            return False
        model = (
            self._prompt_project_popover_text(
                stdscr,
                project,
                "Model",
                "Model name",
                optional=False,
                session_selected=session_selected,
                scroll=scroll,
            )
            if selected == "__custom__"
            else selected
        )
        if model:
            thinking_intensity = self._select_project_thinking_intensity(
                stdscr,
                project,
                provider,
                model,
                sessions=sessions,
                session_selected=session_selected,
                scroll=scroll,
            )
            if thinking_intensity is None:
                self.state = AgentState.PROJECT
                return False
            config["provider"] = provider.key
            config["model"] = model
            config["thinking_intensity"] = thinking_intensity
            config["onboarding_complete"] = True
            self.home.save_config(config)
        self.state = AgentState.PROJECT
        return bool(model)

    def _run_commands_panel(self, stdscr: CursesWindow, current_session: SessionRecord) -> None:
        self.state = AgentState.INFO
        selected = 0
        footer = "Esc Back · ↑↓ Navigate · Ctrl+D Delete"
        while True:
            commands = self._commands_panel_items()
            if commands and selected >= len(commands):
                selected = len(commands) - 1
            body_lines = ("No commands saved yet",) if not commands else ()
            self._draw_overlay(
                stdscr,
                title="Manage Commands",
                subtitle="Globally approved and rejected commands",
                body_lines=body_lines,
                choices=tuple(
                    MenuChoice(
                        self._command_panel_label(subject),
                        subject,
                        self._command_panel_detail(subject),
                    )
                    for subject in commands
                ),
                selected=selected if commands else 0,
                footer=footer,
            )
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                self.state = AgentState.CONFIG
                return
            if not commands:
                continue
            if key == curses.KEY_UP:
                selected = max(0, selected - 1)
            elif key == curses.KEY_DOWN:
                selected = min(len(commands) - 1, selected + 1)
            elif self._is_ctrl_d(key):
                subject = commands[selected]
                self._remove_global_command(subject)
                selected = max(0, min(selected, len(self._commands_panel_items()) - 1))
            elif self._is_enter(key):
                subject = commands[selected]
                self._toggle_global_command(subject)
                selected = max(0, min(selected, len(self._commands_panel_items()) - 1))

    def _commands_panel_items(self) -> list[str]:
        allowed = self._session_command_subjects(self.session_allowed_commands)
        rejected = self._session_command_subjects(self.session_rejected_commands)
        result: list[str] = []
        seen: set[str] = set()
        for subject in allowed:
            if subject not in seen:
                result.append(subject)
                seen.add(subject)
        for subject in rejected:
            if subject not in seen:
                result.append(subject)
                seen.add(subject)
        return result

    def _session_command_subject_is_allowed(self, subject: str) -> bool:
        for key in self.session_allowed_commands:
            if self._session_command_subject(key) == subject:
                return True
        return False

    def _command_panel_label(self, subject: str) -> str:
        return command_allowance_display(f"cmd:{subject}").command

    def _command_panel_detail(self, subject: str) -> str:
        display = command_allowance_display(f"cmd:{subject}")
        status = "Approved" if self._session_command_subject_is_allowed(subject) else "Rejected"
        return f"{status} · Parameters: {display.parameters}"

    def _remove_global_command(self, subject: str) -> None:
        key = self._allowance_key_for_subject(subject)
        if key:
            self.home.remove_global_allowed_command(key)
            self.home.remove_global_rejected_command(key)
            self.session_allowed_commands.discard(key)
            self.session_rejected_commands.discard(key)

    def _toggle_global_command(self, subject: str) -> None:
        is_allowed = self._session_command_subject_is_allowed(subject)
        key = self._allowance_key_for_subject(subject)
        if key is None:
            return
        if is_allowed:
            self.home.remove_global_allowed_command(key)
            self.session_allowed_commands.discard(key)
            self.home.add_global_rejected_command(key)
            self.session_rejected_commands.add(key)
        else:
            self.home.remove_global_rejected_command(key)
            self.session_rejected_commands.discard(key)
            self.home.add_global_allowed_command(key)
            self.session_allowed_commands.add(key)

    def _load_global_allowances(self) -> None:
        for key in self.home.load_global_allowed_commands():
            self.session_allowed_commands.add(key)
        for key in self.home.load_global_rejected_commands():
            self.session_rejected_commands.add(key)

    def _allowance_key_for_subject(self, subject: str) -> str | None:
        for key in self.session_allowed_commands:
            if self._session_command_subject(key) == subject:
                return key
        for key in self.session_rejected_commands:
            if self._session_command_subject(key) == subject:
                return key
        return None

    def _run_config_panel(self, stdscr: CursesWindow, current_session: SessionRecord) -> None:
        self.state = AgentState.CONFIG

        try:
            while True:
                choices = self._config_menu_choices()
                selected = self._menu(
                    stdscr,
                    "Config",
                    "Choose a setting to change",
                    choices,
                )
                if selected is None:
                    return
                if selected == "backend":
                    if self._configure_backend(stdscr):
                        return
                    continue
                if selected == "model":
                    if self._run_model_panel(
                        stdscr,
                        current_session,
                        bottom_popover=False,
                    ):
                        return
                    continue
                if selected == "platform":
                    self._configure_platform(stdscr, current_session)
                    continue
                if selected == "debug":
                    self._run_debug_panel(stdscr, current_session)
                    continue
                if selected == "skills":
                    self._run_skills_panel(stdscr, current_session)
                    continue
                if selected == "memories":
                    self._run_memories_panel(stdscr)
                    continue
                if selected == "manage_instructions":
                    self._run_manage_instructions_panel(stdscr)
                    continue
                if selected == "sandbox":
                    self._configure_sandbox(stdscr)
                    continue
                if selected == "commands":
                    self._run_commands_panel(stdscr, current_session)
                    continue
        finally:
            self.state = AgentState.NEW_SESSION

    def _config_menu_choices(self) -> tuple[MenuChoice, ...]:
        platform_connection = self.home.platform_connection()
        platform_choice = (
            MenuChoice(
                "Manage Platform",
                "platform",
                f"Connected to {platform_domain(platform_connection['url'])}",
            )
            if platform_connection is not None
            else MenuChoice(
                "Manage Platform",
                "platform",
                "Send agent activity, results, and findings to Anomx Platform",
            )
        )
        config = self.home.load_config()
        return (
            MenuChoice("Choose Backend", "backend", "Select provider and enter API key"),
            MenuChoice("Choose Model", "model", "Pick the model for the selected backend"),
            platform_choice,
            MenuChoice("Manage Debug Mode", "debug", self._debug_config_detail(config)),
            MenuChoice(
                "Manage Skills",
                "skills",
                "Create or open user slash-command skills",
            ),
            MenuChoice(
                "Manage Memories",
                "memories",
                "Create, view, or remove local agent memories",
            ),
            MenuChoice(
                "Manage Instructions",
                "manage_instructions",
                "Add, edit, view, or remove custom agent instructions",
            ),
            MenuChoice(
                "Manage Sandbox",
                "sandbox",
                self._sandbox_config_detail(config),
            ),
            MenuChoice(
                "Manage Commands",
                "commands",
                "Review globally approved and rejected commands",
            ),
        )

    def _debug_config_detail(self, config: Mapping[str, object]) -> str:
        debug_active = bool(config.get("debug_mode"))
        full_logs = bool(config.get("debug_full_session_logs"))
        if not debug_active:
            return "debug mode false"
        if full_logs:
            return "debug mode true · full session logs true"
        return "debug mode true · full session logs false"

    def _sandbox_config_detail(self, config: Mapping[str, object]) -> str:
        if not config.get("sandbox_enabled"):
            return "sandbox disabled"
        system = config.get("sandbox_system", "docker")
        from anomx.agent.helpers.local_sandbox import is_python_sandbox_system

        if is_python_sandbox_system(system):
            return "python · software workspace sandbox"
        method = config.get("sandbox_method", "mount")
        cpu = config.get("sandbox_cpu_limit", "2")
        ram = config.get("sandbox_ram_limit", "4g")
        return f"{system} · {method} · {cpu}c/{ram}"

    def _handle_sandbox_check(
        self,
        stdscr: CursesWindow,
        project: ProjectRecord,
    ) -> bool:
        """Check sandbox readiness and prompt for oversized projects.

        Returns False if the user wants to abort.
        """
        config = self.home.load_config()
        if not config.get("sandbox_enabled"):
            return True

        from anomx.agent.helpers.local_sandbox import is_python_sandbox_system

        if is_python_sandbox_system(config.get("sandbox_system")):
            return True

        from anomx.agent.helpers.sandbox import (
            detect_container_runtime,
            sandbox_config_from_dict,
        )

        runtime = detect_container_runtime()
        if runtime is None:
            self._message(
                stdscr,
                "Sandbox Not Available",
                f"Sandbox is enabled but '{config.get('sandbox_system', 'docker')}' "
                "was not found. Install the runtime or disable sandbox.",
            )
            return False

        if runtime != config.get("sandbox_system"):
            self._message(
                stdscr,
                "Runtime Changed",
                f"'{config['sandbox_system']}' not found. Using '{runtime}' instead.",
            )
            config["sandbox_system"] = runtime
            self.home.save_config(config)

        if config.get("sandbox_method") != "copy":
            return True

        sandbox_cfg = sandbox_config_from_dict(config)
        size = self._evaluate_project_size(stdscr, project.path)
        if size is None:
            return False
        if size <= sandbox_cfg.copy_threshold_bytes:
            return True

        size_gb = size / (1024**3)
        threshold_gb = sandbox_cfg.copy_threshold_bytes / (1024**3)
        choices = (
            MenuChoice(
                "Use mount instead",
                "mount",
                "Switch to bind-mount (no copy) for this project",
            ),
            MenuChoice(
                "Copy anyway",
                "copy",
                f"Copy {size_gb:.1f} GiB into the sandbox container",
            ),
        )
        picked = self._menu(
            stdscr,
            "Large Project Detected",
            f"Project is {size_gb:.1f} GiB (threshold: {threshold_gb:.1f} GiB). How to proceed?",
            choices,
        )
        if picked is None:
            return False
        if picked == "mount":
            config["sandbox_method"] = "mount"
            self.home.save_config(config)
        return True

    def _evaluate_project_size(
        self,
        stdscr: CursesWindow,
        project_path: Path,
    ) -> int | None:
        """Show 'Evaluating project size' loading screen with abort capability.

        Returns the project size in bytes, or None if aborted.
        """
        from anomx.agent.helpers.sandbox import project_size_bytes

        result: queue.SimpleQueue[int | None] = queue.SimpleQueue()
        abort_key = ""
        abort_deadline = 0.0
        sentinel: int | None = None

        def worker() -> None:
            try:
                result.put(project_size_bytes(project_path))
            except Exception:
                result.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        frame = 0
        with suppress(curses.error):
            stdscr.nodelay(True)
        try:
            while thread.is_alive():
                subtitle = f"  {spinner_chars[frame % len(spinner_chars)]}  Counting project files"
                self._draw_shell(stdscr, "Evaluating project size", subtitle)
                frame += 1
                with suppress(curses.error):
                    key = stdscr.get_wch()
                    if self._is_ctrl_c(key):
                        key_label = "Ctrl+C"
                        if abort_key == key_label and time.monotonic() <= abort_deadline:
                            sentinel = None
                            break
                        abort_key = key_label
                        abort_deadline = time.monotonic() + 3.0
                    elif self._is_escape(key):
                        sentinel = None
                        break
                time.sleep(STARTUP_FRAME_SECONDS)
        finally:
            with suppress(curses.error):
                stdscr.nodelay(False)
            stdscr.erase()
            stdscr.refresh()
        thread.join(timeout=5)
        with suppress(queue.Empty):
            value = result.get_nowait()
            if value is not None:
                return value
        return sentinel

    def _remove_all_sandbox_containers(self) -> None:
        try:
            from anomx.agent.helpers.sandbox import SandboxSession

            config = self.home.load_config()
            runtime_bin = str(config.get("sandbox_system", "docker"))
            from anomx.agent.helpers.local_sandbox import is_python_sandbox_system

            if is_python_sandbox_system(runtime_bin):
                return
            SandboxSession.remove_all(runtime=runtime_bin)
        except Exception:
            pass

    def _configure_sandbox(self, stdscr: CursesWindow) -> None:
        while True:
            config = self.home.load_config()
            selected = self._menu(
                stdscr,
                "Sandbox",
                "Configure container sandbox execution",
                self._sandbox_menu_choices(config),
            )
            if selected is None:
                self.state = AgentState.CONFIG
                return
            if selected == "toggle":
                currently_enabled = bool(config.get("sandbox_enabled"))
                if currently_enabled:
                    confirm = self._menu(
                        stdscr,
                        "Disable Sandbox",
                        "Disabling sandbox will remove all sandbox containers. Continue?",
                        (
                            MenuChoice("Cancel", "cancel"),
                            MenuChoice("Disable and remove containers", "confirm"),
                        ),
                    )
                    if confirm != "confirm":
                        continue
                    self._remove_all_sandbox_containers()
                config["sandbox_enabled"] = not currently_enabled
                self.home.save_config(config)
                if not currently_enabled:
                    self._activate_agent_mode(AgentMode.SANDBOX)
                else:
                    self._activate_agent(self.active_agent.kind)
                continue
            if selected == "system":
                system_choices = (
                    MenuChoice("Python", "python", "Use the software workspace sandbox"),
                    MenuChoice("Docker", "docker", "Use Docker as the container runtime"),
                    MenuChoice("Podman", "podman", "Use Podman as the container runtime"),
                )
                picked = self._menu(stdscr, "Sandbox Runtime", "Choose runtime", system_choices)
                if picked is not None:
                    config["sandbox_system"] = picked
                    self.home.save_config(config)
                continue
            if selected == "method":
                method_choices = (
                    MenuChoice(
                        "Mount",
                        "mount",
                        "Bind-mount the project directory (no copy)",
                    ),
                    MenuChoice(
                        "Copy",
                        "copy",
                        "Copy project files into the container (slower startup, isolated)",
                    ),
                )
                picked = self._menu(
                    stdscr,
                    "Project Method",
                    "How to provide files",
                    method_choices,
                )
                if picked is not None:
                    config["sandbox_method"] = picked
                    self.home.save_config(config)
                continue
            if selected == "limits":
                self._configure_sandbox_limits(stdscr)
                continue
            if selected == "strategy":
                strategy_choices = (
                    MenuChoice(
                        "Stop on exit",
                        "stop",
                        "Stop the container when exiting anomx, resume in next session",
                    ),
                    MenuChoice(
                        "Remove on exit",
                        "remove",
                        "Fully remove the container when exiting anomx",
                    ),
                )
                picked = self._menu(
                    stdscr,
                    "Container Handling",
                    "What to do with the container on exit",
                    strategy_choices,
                )
                if picked is not None:
                    config["sandbox_strategy"] = picked
                    self.home.save_config(config)
                continue

    def _sandbox_menu_choices(self, config: Mapping[str, object]) -> tuple[MenuChoice, ...]:
        toggle_label = "Disable Sandbox" if config.get("sandbox_enabled") else "Enable Sandbox"
        return (
            MenuChoice(
                toggle_label,
                "toggle",
                "Turn sandbox execution on or off",
            ),
            MenuChoice(
                "Sandbox Runtime",
                "system",
                f"Runtime: {config.get('sandbox_system', 'docker')}",
            ),
            MenuChoice(
                "Project Method",
                "method",
                f"How files are provided: {config.get('sandbox_method', 'mount')}",
            ),
            MenuChoice(
                "Configure Limits",
                "limits",
                "CPU, RAM, and disk limits for containers",
            ),
            MenuChoice(
                "Container Handling",
                "strategy",
                f"On exit: {config.get('sandbox_strategy', 'stop')}",
            ),
        )

    def _configure_sandbox_limits(self, stdscr: CursesWindow) -> None:
        while True:
            config = self.home.load_config()
            selected = self._menu(
                stdscr,
                "Sandbox Limits",
                "Set resource limits for sandbox containers",
                self._sandbox_limits_choices(config),
            )
            if selected is None:
                return
            if selected == "cpu":
                value = self._run_overlay_text(
                    stdscr,
                    "CPU Limit",
                    "Number of CPU cores (e.g. 2, 1.5)",
                    default=str(config.get("sandbox_cpu_limit", "2")),
                    optional=False,
                )
                if value is not None and value.strip():
                    config["sandbox_cpu_limit"] = value.strip()
                    self.home.save_config(config)
                continue
            if selected == "ram":
                value = self._run_overlay_text(
                    stdscr,
                    "RAM Limit",
                    "Memory limit (e.g. 4g, 8g, 2048m)",
                    default=str(config.get("sandbox_ram_limit", "4g")),
                    optional=False,
                )
                if value is not None and value.strip():
                    config["sandbox_ram_limit"] = value.strip()
                    self.home.save_config(config)
                continue
            if selected == "hd":
                value = self._run_overlay_text(
                    stdscr,
                    "Disk Limit",
                    "Storage limit (e.g. 10g, 20g, 51200m)",
                    default=str(config.get("sandbox_hd_limit", "10g")),
                    optional=False,
                )
                if value is not None and value.strip():
                    config["sandbox_hd_limit"] = value.strip()
                    self.home.save_config(config)
                continue

    def _sandbox_limits_choices(self, config: Mapping[str, object]) -> tuple[MenuChoice, ...]:
        return (
            MenuChoice("CPU Cores", "cpu", f"Current: {config.get('sandbox_cpu_limit', '2')}"),
            MenuChoice("RAM", "ram", f"Current: {config.get('sandbox_ram_limit', '4g')}"),
            MenuChoice("Disk", "hd", f"Current: {config.get('sandbox_hd_limit', '10g')}"),
        )

    def _run_debug_panel(
        self,
        stdscr: CursesWindow,
        current_session: SessionRecord,
    ) -> None:
        while True:
            config = self.home.load_config()
            selected = self._menu(
                stdscr,
                "Debug",
                "Configure debug mode and step logging",
                self._debug_menu_choices(config),
            )
            if selected is None:
                self.state = AgentState.CONFIG
                return
            if selected == "debug_mode":
                config["debug_mode"] = not bool(config.get("debug_mode"))
                self.home.save_config(config)
                continue
            if selected == "full_session_logs":
                config["debug_full_session_logs"] = not bool(config.get("debug_full_session_logs"))
                self.home.save_config(config)
                continue
            if selected == "full_session_logs_path":
                current_path = str(self.home.debug_location(config))
                value = self._prompt_text(
                    stdscr,
                    "Debug Location",
                    "Directory path (default: ~/.anomx)",
                    default=current_path,
                )
                if value is not None:
                    config["debug_full_session_logs_path"] = value.strip() or str(self.home.root)
                    self.home.save_config(config)

    def _debug_menu_choices(
        self,
        config: Mapping[str, object],
    ) -> tuple[MenuChoice, ...]:
        return (
            MenuChoice(
                "Debug mode active",
                "debug_mode",
                self._bool_config_detail(config.get("debug_mode")),
            ),
            MenuChoice(
                "Full session logs",
                "full_session_logs",
                self._bool_config_detail(config.get("debug_full_session_logs")),
            ),
            MenuChoice(
                "Debug location",
                "full_session_logs_path",
                str(self.home.debug_location(config)),
            ),
        )

    def _bool_config_detail(self, value: object) -> str:
        return "true" if bool(value) else "false"

    def _run_manage_instructions_panel(self, stdscr: CursesWindow) -> None:
        instruction_path = self.home.instructions_dir / "instruction.md"
        text = instruction_path.read_text(encoding="utf-8") if instruction_path.exists() else ""
        cursor_pos = 0
        footer = "Esc Cancel  \u00b7  Ctrl+S Save  \u00b7  \u2191\u2193 Home End"
        while True:
            self.state = AgentState.CONFIG
            self._draw_overlay(
                stdscr,
                title="Custom Instructions",
                editor_text=text,
                editor_cursor=cursor_pos,
                footer=footer,
            )
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                return
            if self._is_ctrl_s(key):
                instruction_path.parent.mkdir(parents=True, exist_ok=True)
                instruction_path.write_text(text, encoding="utf-8")
                self._message(stdscr, "Custom Instructions", "Custom instructions saved.")
                return
            content_width = max(20, (stdscr.getmaxyx()[1]) - 8)
            if key == curses.KEY_UP:
                cursor_pos = self._cursor_move_up(text, cursor_pos, content_width)
            elif key == curses.KEY_DOWN:
                cursor_pos = self._cursor_move_down(text, cursor_pos, content_width)
            elif self._is_option_left(key):
                cursor_pos = self._previous_prompt_word(text, cursor_pos)
            elif self._is_option_right(key):
                cursor_pos = self._next_prompt_word(text, cursor_pos)
            elif self._is_option_delete(key):
                text, cursor_pos = self._delete_previous_prompt_word(text, cursor_pos)
            elif key == curses.KEY_LEFT:
                cursor_pos = max(0, cursor_pos - 1)
            elif key == curses.KEY_RIGHT:
                cursor_pos = min(len(text), cursor_pos + 1)
            elif key == curses.KEY_HOME:
                cursor_pos = self._cursor_line_start(text, cursor_pos)
            elif key == curses.KEY_END:
                cursor_pos = self._cursor_line_end(text, cursor_pos)
            elif key == curses.KEY_PPAGE:
                cursor_pos = self._cursor_move_up(text, cursor_pos, content_width, 8)
            elif key == curses.KEY_NPAGE:
                cursor_pos = self._cursor_move_down(text, cursor_pos, content_width, 8)
            elif self._is_enter(key):
                text = text[:cursor_pos] + "\n" + text[cursor_pos:]
                cursor_pos += 1
            elif self._is_backspace(key):
                if cursor_pos > 0:
                    text = text[: cursor_pos - 1] + text[cursor_pos:]
                    cursor_pos -= 1
            elif isinstance(key, str) and key.isprintable():
                text = text[:cursor_pos] + key + text[cursor_pos:]
                cursor_pos += 1

    def _cursor_display_position(self, text: str, cursor_pos: int, line_width: int) -> int:
        """Return the display line index (0-based) where cursor_pos falls."""
        if not text:
            return 0
        display_line = 0
        pos = 0
        for raw_line in text.splitlines(keepends=True):
            line_content = raw_line.rstrip("\n")
            cleaned = line_content.replace("\t", "    ")
            wrapped = textwrap.wrap(
                cleaned,
                width=line_width,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            for segment in wrapped:
                if pos <= cursor_pos <= pos + len(segment):
                    return display_line
                display_line += 1
                pos += len(segment)
            if raw_line.endswith("\n"):
                if pos == cursor_pos:
                    return max(0, display_line - 1)
                pos += 1
        return max(0, display_line - 1)

    def _cursor_column_in_display_line(self, text: str, cursor_pos: int, line_width: int) -> int:
        """Return the column offset within the display line at cursor_pos."""
        if not text:
            return 0
        pos = 0
        for raw_line in text.splitlines(keepends=True):
            line_content = raw_line.rstrip("\n")
            cleaned = line_content.replace("\t", "    ")
            wrapped = textwrap.wrap(
                cleaned,
                width=line_width,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            for segment in wrapped:
                if pos <= cursor_pos <= pos + len(segment):
                    return cursor_pos - pos
                pos += len(segment)
            if raw_line.endswith("\n"):
                if pos == cursor_pos:
                    return 0
                pos += 1
        return 0

    def _cursor_move_up(self, text: str, cursor_pos: int, line_width: int, n: int = 1) -> int:
        """Move cursor n display lines up."""
        display_pos = self._cursor_display_position(text, cursor_pos, line_width)
        target = max(0, display_pos - n)
        if target == display_pos:
            return 0
        col = self._cursor_column_in_display_line(text, cursor_pos, line_width)
        return self._position_at_display_line(text, target, col, line_width)

    def _cursor_move_down(self, text: str, cursor_pos: int, line_width: int, n: int = 1) -> int:
        """Move cursor n display lines down."""
        total = self._total_display_lines(text, line_width)
        display_pos = self._cursor_display_position(text, cursor_pos, line_width)
        target = min(total - 1, display_pos + n)
        if target == display_pos:
            return len(text)
        col = self._cursor_column_in_display_line(text, cursor_pos, line_width)
        return self._position_at_display_line(text, target, col, line_width)

    def _total_display_lines(self, text: str, line_width: int) -> int:
        """Return total number of display lines for the text."""
        if not text:
            return 1
        count = 0
        for raw_line in text.splitlines(keepends=True):
            cleaned = raw_line.replace("\t", "    ").rstrip("\n")
            wrapped = textwrap.wrap(
                cleaned,
                width=line_width,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            count += len(wrapped)
        return max(1, count)

    def _position_at_display_line(
        self, text: str, target_line: int, col: int, line_width: int
    ) -> int:
        """Return the character position at the given display line."""
        display_line = 0
        pos = 0
        for raw_line in text.splitlines(keepends=True):
            line_content = raw_line.rstrip("\n")
            cleaned = line_content.replace("\t", "    ")
            wrapped = textwrap.wrap(
                cleaned,
                width=line_width,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            for segment in wrapped:
                if display_line == target_line:
                    return pos + min(col, len(segment))
                display_line += 1
                pos += len(segment)
            if raw_line.endswith("\n"):
                if display_line == target_line:
                    return pos
                display_line += 1
                pos += 1
        return pos

    def _cursor_line_start(self, text: str, cursor_pos: int) -> int:
        """Move cursor to the start of the current logical line."""
        line_start = text.rfind("\n", 0, cursor_pos)
        return line_start + 1 if line_start >= 0 else 0

    def _cursor_line_end(self, text: str, cursor_pos: int) -> int:
        """Move cursor to the end of the current logical line."""
        line_end = text.find("\n", cursor_pos)
        return line_end if line_end >= 0 else len(text)

    def _editor_mouse_position(
        self,
        stdscr: CursesWindow,
        editor_top: int,
        editor_width: int,
        editor_bottom: int,
        scroll_offset: int,
        text: str,
    ) -> int | None:
        """Convert a mouse click to a character position in the text."""
        try:
            _, mx, my, _bstate = curses.getmouse()
            line_width = max(1, editor_width)
            if my < editor_top or my > editor_bottom:
                return None
            click_line_in_view = my - editor_top
            doc_line = scroll_offset + click_line_in_view
            click_col = max(0, mx - 4)
            display_line = 0
            pos = 0
            for raw_line in text.splitlines(keepends=True):
                line_content = raw_line.rstrip("\n")
                cleaned = line_content.replace("\t", "    ")
                wrapped = textwrap.wrap(
                    cleaned,
                    width=line_width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                    break_on_hyphens=False,
                ) or [""]
                for segment in wrapped:
                    if display_line == doc_line:
                        return pos + min(click_col, len(segment))
                    display_line += 1
                    pos += len(segment)
                if raw_line.endswith("\n"):
                    if display_line == doc_line:
                        return pos
                    display_line += 1
                    pos += 1
            return len(text)
        except curses.error:
            return None

    def _configure_backend(self, stdscr: CursesWindow) -> bool:
        config = self.home.load_config()
        previous_provider = str(config.get("provider", ""))
        provider = self._select_provider(stdscr)
        if provider is None:
            return False
        if provider.key in {"openai", "anthropic", "blablador", "desy"}:
            should_prompt_api_key = True
            if self.home.has_api_key(provider.key):
                selected = self._menu(
                    stdscr,
                    provider.label,
                    "API key already configured",
                    (
                        MenuChoice("Keep API Key", "keep", "Use the saved API key"),
                        MenuChoice("New API Key", "new", "Replace the saved API key"),
                    ),
                )
                if selected is None:
                    return False
                should_prompt_api_key = selected == "new"
            if should_prompt_api_key:
                api_key = self._prompt_text(
                    stdscr,
                    title=provider.label,
                    label="API key",
                    mask=True,
                    optional=False,
                )
                if not api_key:
                    return False
                self.home.set_api_key(provider.key, api_key)
        provider = self._provider_with_discovered_models(provider, refresh=True)
        selected_model = str(config.get("model", ""))
        model_was_selected = False
        if provider.key != previous_provider:
            model = self._select_model(stdscr, provider)
            if model is None:
                return False
            selected_model = model
            model_was_selected = True
        elif not self._model_allowed(provider, selected_model):
            selected_model = provider.models[0]
            model_was_selected = True
        if model_was_selected:
            thinking_intensity = self._select_thinking_intensity(stdscr, provider, selected_model)
            if thinking_intensity is None:
                return False
            config["thinking_intensity"] = thinking_intensity
        config["provider"] = provider.key
        config["model"] = selected_model
        self.home.save_config(config)
        return True

    def _configure_platform(
        self,
        stdscr: CursesWindow,
        current_session: SessionRecord,
    ) -> bool:
        platform_connection = self.home.platform_connection()
        if platform_connection is not None:
            return self._run_platform_management_form(stdscr, platform_connection)

        result = self._run_platform_connection_form(stdscr)
        return bool(result)

    def _run_platform_connection_form(
        self,
        stdscr: CursesWindow,
    ) -> PlatformLoginResult | None:
        config = self.home.load_config()
        draft = PlatformConnectionDraft(
            url=str(config.get("platform_last_url") or ""),
            email=str(config.get("platform_last_email") or ""),
        )
        selected = 0 if not draft.url else 1 if not draft.email else 2
        error = ""
        while True:
            self._draw_platform_connection_form(stdscr, draft, selected, error=error)
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                self._save_platform_form_defaults(draft)
                return None
            if key == curses.KEY_UP or self._is_shift_enter(key):
                selected = max(0, selected - 1)
                continue
            if key == curses.KEY_DOWN or key == "\t":
                selected = min(2, selected + 1)
                continue
            if self._is_enter(key):
                if selected < 2:
                    selected += 1
                    continue
                missing_row = self._missing_platform_form_row(draft)
                if missing_row is not None:
                    selected = missing_row
                    error = "Domain, email, and password are required."
                    continue
                self._save_platform_form_defaults(draft)
                try:
                    result = self._connect_platform_with_loading(stdscr, draft)
                except PlatformClientError as exc:
                    self._save_platform_form_defaults(draft)
                    selected = 2
                    error = str(exc)
                    continue
                connection = {
                    "url": result.url,
                    "token": result.token,
                    "user_email": result.user_email,
                    "organization_url": result.organization_url,
                    "hostname": result.hostname,
                }
                self.home.set_platform_connection(
                    url=connection["url"],
                    token=connection["token"],
                    user_email=connection["user_email"],
                    organization_url=connection["organization_url"],
                    hostname=connection["hostname"],
                )
                sync_builtin_skills(self.home.skills_dir, include_system=True)
                self._run_platform_management_form(
                    stdscr,
                    connection,
                    initial_status="Connection alive.",
                    initial_status_role="ok",
                    check_connection=False,
                )
                return result
            if self._is_option_delete(key):
                value, _cursor = self._delete_previous_prompt_word(
                    self._platform_form_value(draft, selected),
                    len(self._platform_form_value(draft, selected)),
                )
                draft = self._update_platform_form_draft(draft, selected, value)
                error = ""
                self._save_platform_form_defaults(draft)
            elif self._is_backspace(key):
                draft = self._update_platform_form_draft(
                    draft,
                    selected,
                    self._platform_form_value(draft, selected)[:-1],
                )
                error = ""
                self._save_platform_form_defaults(draft)
            elif isinstance(key, str) and key.isprintable():
                draft = self._update_platform_form_draft(
                    draft,
                    selected,
                    self._platform_form_value(draft, selected) + key,
                )
                error = ""
                self._save_platform_form_defaults(draft)

    def _run_platform_management_form(
        self,
        stdscr: CursesWindow,
        connection: dict[str, str],
        *,
        initial_status: str = "Checking connection...",
        initial_status_role: str = "normal",
        check_connection: bool = True,
    ) -> bool:
        config = self.home.load_config()
        draft = PlatformConnectionDraft(
            url=str(connection.get("url") or config.get("platform_last_url") or ""),
            email=str(connection.get("user_email") or config.get("platform_last_email", "")),
            password="*****",
        )
        self._save_platform_form_defaults(draft)
        check_result: queue.SimpleQueue[bool] = queue.SimpleQueue()
        worker: threading.Thread | None = None

        def run_check() -> None:
            check_result.put(heartbeat_platform_connection(self.home))

        if check_connection:
            worker = threading.Thread(target=run_check, daemon=True)
            worker.start()
        frame = 0
        status = initial_status
        status_role = initial_status_role
        with suppress(curses.error):
            stdscr.nodelay(True)
        try:
            while True:
                if worker is not None and not worker.is_alive():
                    with suppress(queue.Empty):
                        ok = check_result.get_nowait()
                        if ok:
                            status = "Connection alive."
                            status_role = "ok"
                        else:
                            status = "Connection check failed."
                            status_role = "danger"
                    worker.join(timeout=0)
                    worker = None
                self._draw_platform_connection_form(
                    stdscr,
                    draft,
                    2,
                    title="Manage Platform",
                    status=status,
                    status_role=status_role,
                    frame=frame,
                    footer="Esc Back · Enter Back · Ctrl+D Logout",
                    editable=False,
                )
                frame += 1
                try:
                    key = stdscr.get_wch()
                except curses.error:
                    key = None
                if key is None:
                    time.sleep(0.08)
                    continue
                if self._is_ctrl_d(key):
                    with suppress(curses.error):
                        stdscr.nodelay(False)
                    if self._confirm_platform_logout(stdscr, draft):
                        self.home.clear_platform_connection()
                        self._wait_platform_form_status(
                            stdscr,
                            draft,
                            "Logged out.",
                            "ok",
                            title="Manage Platform",
                            footer="Esc Back · Enter Continue",
                            editable=False,
                        )
                        return True
                    with suppress(curses.error):
                        stdscr.nodelay(True)
                    continue
                if self._is_escape(key) or self._is_ctrl_c(key) or self._is_enter(key):
                    return False
                time.sleep(0.08)
        finally:
            with suppress(curses.error):
                stdscr.nodelay(False)

    def _confirm_platform_logout(
        self,
        stdscr: CursesWindow,
        draft: PlatformConnectionDraft,
    ) -> bool:
        while True:
            self._draw_platform_connection_form(
                stdscr,
                draft,
                2,
                title="Manage Platform",
                status="Logout this CLI agent?",
                status_role="normal",
                footer="Esc Cancel · Enter Logout",
                editable=False,
            )
            key = stdscr.get_wch()
            if self._is_enter(key):
                return True
            if self._is_escape(key) or self._is_ctrl_c(key):
                return False

    def _wait_platform_form_status(
        self,
        stdscr: CursesWindow,
        draft: PlatformConnectionDraft,
        status: str,
        status_role: str,
        *,
        title: str,
        footer: str = "Esc Back · Enter Continue",
        editable: bool = False,
    ) -> None:
        while True:
            self._draw_platform_connection_form(
                stdscr,
                draft,
                2,
                title=title,
                status=status,
                status_role=status_role,
                footer=footer,
                editable=editable,
            )
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key) or self._is_enter(key):
                return

    def _missing_platform_form_row(self, draft: PlatformConnectionDraft) -> int | None:
        for index, value in enumerate((draft.url, draft.email, draft.password)):
            if not value.strip():
                return index
        return None

    def _platform_form_field(self, selected: int) -> str:
        return ("url", "email", "password")[selected]

    def _platform_form_value(self, draft: PlatformConnectionDraft, selected: int) -> str:
        return str(getattr(draft, self._platform_form_field(selected)))

    def _update_platform_form_draft(
        self,
        draft: PlatformConnectionDraft,
        selected: int,
        value: str,
    ) -> PlatformConnectionDraft:
        field_name = self._platform_form_field(selected)
        if field_name == "url":
            return replace(draft, url=value)
        if field_name == "email":
            return replace(draft, email=value)
        return replace(draft, password=value)

    def _save_platform_form_defaults(self, draft: PlatformConnectionDraft) -> None:
        self.home.set_platform_form_defaults(url=draft.url, email=draft.email)

    def _draw_platform_connection_form(
        self,
        stdscr: CursesWindow,
        draft: PlatformConnectionDraft,
        selected: int,
        *,
        title: str = "Manage Platform",
        error: str = "",
        status: str = "",
        status_role: str = "normal",
        connecting: bool = False,
        frame: int = 0,
        footer: str | None = None,
        editable: bool = True,
    ) -> None:
        rows = (
            ("Domain", draft.url),
            ("Email", draft.email),
            ("Password", "*" * len(draft.password)),
        )
        choices = tuple(MenuChoice(label, str(i), value) for i, (label, value) in enumerate(rows))
        post_body_lines: list[str] = []
        if error:
            post_body_lines.extend(("", f"! {error}"))
        elif status:
            status_text = f"! {status}" if status_role == "danger" else status
            post_body_lines.extend(("", status_text))
        elif connecting:
            post_body_lines.extend(("", f"Connecting{'.' * ((frame // 4) % 4)}"))

        subtitle = "Set the URL of an Anomx Platform."
        if connecting:
            footer_text = "Please wait"
        else:
            footer_text = footer or (
                "Esc Cancel · ↑↓ Navigate · Enter for Login"
                if selected == 2
                else "Esc Cancel · ↑↓ Navigate · Enter Next"
            )
        input_val = rows[selected][1] if editable and not connecting else ""
        input_cur = len(input_val)
        self._draw_overlay(
            stdscr,
            title=title,
            subtitle=subtitle,
            choices=choices,
            selected=selected,
            input_value=input_val,
            input_cursor=input_cur,
            post_body_lines=tuple(post_body_lines),
            footer=footer_text,
            show_input_cursor=editable and not connecting,
        )

    def _connect_platform_with_loading(
        self,
        stdscr: CursesWindow,
        draft: PlatformConnectionDraft,
    ) -> PlatformLoginResult:
        results: queue.SimpleQueue[tuple[str, PlatformLoginResult | PlatformClientError]] = (
            queue.SimpleQueue()
        )

        def run_connect() -> None:
            try:
                results.put(("ok", connect_platform(draft.url, draft.email, draft.password)))
            except PlatformClientError as exc:
                results.put(("error", exc))
            except Exception as exc:
                error = PlatformClientError(f"Platform connection failed: {exc}")
                results.put(("error", error))

        worker = threading.Thread(target=run_connect, daemon=True)
        worker.start()
        frame = 0
        with suppress(curses.error):
            stdscr.nodelay(True)
        try:
            while worker.is_alive():
                self._draw_platform_connection_form(
                    stdscr,
                    draft,
                    2,
                    connecting=True,
                    frame=frame,
                )
                frame += 1
                with suppress(curses.error):
                    stdscr.get_wch()
                time.sleep(0.08)
        finally:
            with suppress(curses.error):
                stdscr.nodelay(False)

        worker.join(timeout=0)
        try:
            kind, payload = results.get_nowait()
        except queue.Empty as exc:
            raise PlatformClientError("Platform connection failed.") from exc
        if isinstance(payload, PlatformClientError):
            raise payload
        if kind == "error":
            raise PlatformClientError("Platform connection failed.")
        return payload

    def _draw_platform_connect_loading(
        self,
        stdscr: CursesWindow,
        frame: int,
    ) -> None:
        self._draw_platform_connection_form(
            stdscr,
            PlatformConnectionDraft(password=" "),
            2,
            connecting=True,
            frame=frame,
        )

    def _select_provider(self, stdscr: CursesWindow) -> ProviderOption | None:
        choices = tuple(
            MenuChoice(provider.label, provider.key, ", ".join(provider.models))
            for provider in AI_PROVIDERS
        )
        selected = self._menu(stdscr, "AI Backend", "Select provider", choices)
        return provider_by_key(selected) if selected is not None else None

    def _select_model(self, stdscr: CursesWindow, provider: ProviderOption) -> str | None:
        provider = self._provider_with_discovered_models(provider)
        choices = [MenuChoice(model, model, model_detail(model)) for model in provider.models]
        if provider.allow_custom_model:
            choices.append(
                MenuChoice(
                    "Custom model",
                    "__custom__",
                    f"Use a custom {provider.label} model name",
                )
            )
        selected = self._menu(stdscr, "Model", provider.label, tuple(choices))
        if selected is None:
            return None
        if selected == "__custom__":
            return self._prompt_text(stdscr, "Model", "Model name", optional=False)
        return selected

    def _select_thinking_intensity(
        self,
        stdscr: CursesWindow,
        provider: ProviderOption,
        model: str,
        *,
        current_session: SessionRecord | None = None,
    ) -> str | None:
        options = thinking_intensity_options(provider.key, model)
        if not options:
            return "auto"
        choices = tuple(MenuChoice(option.label, option.value, option.detail) for option in options)
        selected = (
            self._bottom_menu(
                stdscr,
                current_session,
                "Thinking Intensity",
                model,
                choices,
            )
            if current_session is not None
            else self._menu(stdscr, "Thinking Intensity", model, choices)
        )
        return selected

    def _select_project_thinking_intensity(
        self,
        stdscr: CursesWindow,
        project: ProjectRecord,
        provider: ProviderOption,
        model: str,
        *,
        sessions: Sequence[SessionRecord],
        session_selected: int,
        scroll: int = 0,
    ) -> str | None:
        options = thinking_intensity_options(provider.key, model)
        if not options:
            return "auto"
        choices = tuple(MenuChoice(option.label, option.value, option.detail) for option in options)
        return self._project_bottom_menu(
            stdscr,
            project,
            "Thinking Intensity",
            model,
            choices,
            sessions=sessions,
            session_selected=session_selected,
            scroll=scroll,
        )
