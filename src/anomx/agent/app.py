"""Full-screen terminal UI for the Anomx agent."""

from __future__ import annotations

import curses
import hashlib
import json
import os
import queue
import random
import re
import shlex
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from anomx.agent.base.agents import AgentKind, BaseAgent
from anomx.agent.file_reference_search import FileReferenceSearchClient
from anomx.agent.helpers.anomx_api import call_anomx_api, connection_from_home
from anomx.agent.helpers.mode import AgentMode, next_agent_mode
from anomx.agent.helpers.state import (
    latest_plan_steps,
    running_process_snapshots,
    running_subagent_snapshots,
)
from anomx.agent.helpers.tool_manager import (
    ApprovalChoice,
    CommandApprovalRequest,
    command_allowance_display,
    discover_workspace_root,
)
from anomx.agent.helpers.utils import agent_spec
from anomx.agent.memories import MemoryKind, create_memory_record, write_memory
from anomx.agent.model_catalog import discover_provider_models, merge_provider_models
from anomx.agent.runtime import (
    AgentRuntime,
    QuestionRequest,
    QuestionResponse,
    RuntimeCallbacks,
    StatusCallback,
    UsageSnapshot,
    backend_supports_image_input,
    image_mime_type,
)
from anomx.agent.runtime_process import RuntimeProcessClient
from anomx.agent.skills import (
    Skill,
    load_builtin_skills,
    load_user_skills,
    skill_invocation_prompt,
    sync_builtin_skills,
)
from anomx.agent.startup import CliStartupProcess
from anomx.agent.store import (
    AnomxHome,
    ProjectRecord,
    ProviderOption,
    SessionRecord,
    provider_by_key,
)
from anomx.agent.ui.components.bottom_bar import BottomBarComponentMixin
from anomx.agent.ui.components.info_box import InfoBoxComponentMixin
from anomx.agent.ui.components.messages import MessagesComponentMixin
from anomx.agent.ui.components.popup import PopupComponentMixin
from anomx.agent.ui.components.prompt_bar import PromptBarComponentMixin
from anomx.agent.ui.constants import (
    ABORT_AGENT_CONFIRM_NOTICE,
    ACTIVITY_FRAME_SECONDS,
    COMMANDS,
    EXIT_ANOMX_CONFIRM_NOTICE,
    FILE_REFERENCE_CACHE_SECONDS,
    FILE_REFERENCE_FIRST_LEVEL_LIMIT,
    FILE_REFERENCE_INDEX_REFRESH_SECONDS,
    FILE_REFERENCE_LIMIT,
    IGNORED_FILE_REFERENCE_DIRS,
    IMAGE_DROP_CANDIDATE_PATTERN,
    MANUAL_INTERRUPT_MESSAGE,
    PLAN_STEP_REVEAL_SECONDS,
    PROJECT_COMMANDS,
    PROMPT_PASTE_COLLAPSE_THRESHOLD,
    PROMPT_PLACEHOLDERS,
    RAW_MOUSE_RE,
    RAW_MOUSE_SUFFIX_RE,
    RUNNING_COMMAND_BLOCKED_NOTICE,
    RUNNING_NOTICE,
    RUNNING_SESSION_COMMANDS,
    START_HINT_WIPE_SECONDS,
    STARTUP_FRAME_SECONDS,
)
from anomx.agent.ui.models import (
    ActiveSessionTurn,
    AgentState,
    BackendTurnResult,
    CommandSpec,
    CursesWindow,
    InfoRow,
    MenuChoice,
    MessageLine,
    PromptPasteEvent,
    PromptPasteSpan,
    RunningKeyResult,
    RuntimeUiEvent,
    SessionMouseAction,
    SessionTextRow,
    SessionTextSelection,
    StartupPreparation,
)
from anomx.agent.ui.views.approve_update import ApproveUpdateViewMixin
from anomx.agent.ui.views.approve_workspace import ApproveWorkspaceViewMixin
from anomx.agent.ui.views.config import ConfigViewMixin
from anomx.agent.ui.views.matrix_loading import MatrixLoadingViewMixin
from anomx.agent.ui.views.onboarding import OnboardingViewMixin
from anomx.agent.ui.views.project import ProjectViewMixin
from anomx.agent.ui.views.session import SessionViewMixin
from anomx.agent.ui.views.subagent import SubagentViewMixin


class AnomxCliApp(
    MatrixLoadingViewMixin,
    OnboardingViewMixin,
    ApproveWorkspaceViewMixin,
    ApproveUpdateViewMixin,
    ProjectViewMixin,
    SessionViewMixin,
    SubagentViewMixin,
    ConfigViewMixin,
    InfoBoxComponentMixin,
    MessagesComponentMixin,
    PopupComponentMixin,
    PromptBarComponentMixin,
    BottomBarComponentMixin,
):
    """Full-screen state-machine based Anomx agent."""

    def __init__(
        self,
        home: AnomxHome | None = None,
        cwd: Path | None = None,
        startup_provider: str | None = None,
        startup_model: str | None = None,
        use_color: bool = True,
        isolate_runtime: bool = False,
    ) -> None:
        self.home = AnomxHome() if home is None else home
        self.home.ensure()
        self.isolate_runtime = isolate_runtime
        if not isolate_runtime:
            sync_builtin_skills(
                self.home.skills_dir,
                include_system=True,
            )
        self.cwd = (Path.cwd() if cwd is None else cwd).expanduser().resolve()
        self.project_path = self.cwd
        self.workspace_root = discover_workspace_root(self.cwd)
        self.startup_provider = startup_provider
        self.startup_model = startup_model
        self.use_color = use_color
        self.session_allowed_commands: set[str] = set()
        self.session_rejected_commands: set[str] = set()
        self._load_global_allowances()
        config = self.home.load_config()
        self.work_visualization = str(config["work_visualization"])
        self.active_agent = agent_spec(AgentKind.MAIN)
        configured_mode = AgentMode.parse(config.get("agent_mode"))
        self.agent_mode = (
            AgentMode.STANDARD
            if configured_mode == AgentMode.BACKGROUND
            else configured_mode
        )
        self.runtime = self._create_runtime(self.agent_mode)
        self._runtime_processes: list[RuntimeProcessClient] = []
        if isinstance(self.runtime, RuntimeProcessClient):
            self._runtime_processes.append(self.runtime)
        self.state = AgentState.ONBOARDING
        self._colors: dict[str, int] = {}
        self._activity_wave_palettes: dict[bool, tuple[int, ...]] = {}
        self._accent_attr_name = "accent"
        self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
        self._modal_prompt_text = ""
        self._modal_prompt_cursor = 0
        self._modal_prompt_pasted_spans: tuple[PromptPasteSpan, ...] = ()
        self._modal_prompt_file_references: dict[str, str] = {}
        self._modal_prompt_image_attachments: dict[str, dict[str, str]] = {}
        self._expanded_work_turns: set[str] = set()
        self._expanded_work_lines: set[str] = set()
        self._expanded_pinned_users: set[str] = set()
        self._expanded_plan_sessions: set[Path] = set()
        self._expanded_activity_items: set[str] = set()
        self._expanded_activity_entries: set[str] = set()
        self._activity_detail_scrolls: dict[str, int] = {}
        self._session_event_cache: dict[Path, tuple[int, int, list[dict[str, Any]]]] = {}
        self._message_line_cache: dict[
            Path,
            tuple[int, int, tuple[str, ...], list[MessageLine]],
        ] = {}
        self._rendered_message_cache: dict[
            Path,
            tuple[
                int,
                int,
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
                int,
                str,
                list[MessageLine],
            ],
        ] = {}
        self._context_status_cache: dict[Path, tuple[int, int, str, str, int]] = {}
        self._session_usage_snapshots: dict[Path, UsageSnapshot] = {}
        self._config_cache: tuple[int, int, dict[str, Any]] | None = None
        self._start_hint_reveal_started_at: float | None = None
        self._click_targets: dict[int, list[SessionMouseAction]] = {}
        self._session_text_rows: dict[int, SessionTextRow] = {}
        self._session_selection: SessionTextSelection | None = None
        self._session_selecting = False
        self._title_events: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()
        self._title_jobs: set[str] = set()
        self._file_reference_cache_at = 0.0
        self._file_reference_cache: tuple[str, ...] = ()
        self._file_reference_full_cache: tuple[str, ...] = ()
        self._file_reference_full_cache_at = 0.0
        self._file_reference_index_lock = threading.Lock()
        self._file_reference_index_loaded = False
        self._file_reference_index_thread: threading.Thread | None = None
        self._file_reference_index_started_at = 0.0
        self._file_reference_search = FileReferenceSearchClient(
            self.workspace_root,
            self._file_reference_index_path(),
        )
        self._prepare_startup_during_loading = False
        self._startup_preparation: StartupPreparation | None = None
        self._active_session_turns: dict[Path, ActiveSessionTurn] = {}
        self._discovered_provider_models: dict[str, tuple[str, ...]] = {}
        self._startup_tasks: CliStartupProcess | None = None

    def prepare_background_startup(self, *, wait_for_runtime: bool = False) -> None:
        """Warm external startup work without blocking terminal rendering."""

        if self._startup_tasks is None:
            self._startup_tasks = CliStartupProcess(self.home.root)
            self._startup_tasks.start()
        self._file_reference_search.start()
        if isinstance(self.runtime, RuntimeProcessClient):
            self.runtime.start()
            if wait_for_runtime:
                self.runtime.wait_until_ready()

    def prepare_startup_config(self) -> None:
        """Apply command-line startup provider/model overrides."""

        self.home.ensure()
        provider_key = self.startup_provider
        model = self.startup_model
        if provider_key is None and model is None:
            return

        config = self.home.load_config()
        if provider_key is None:
            provider_key = str(config.get("provider") or "ollama")
        provider = provider_by_key(provider_key)
        if provider is None:
            raise ValueError(f"unknown provider: {provider_key}")

        selected_model = model or str(config.get("model") or provider.models[0])
        if not self._model_allowed(provider, selected_model):
            selected_model = provider.models[0]

        config["onboarding_complete"] = True
        config["provider"] = provider.key
        config["model"] = selected_model
        self.home.save_config(config)

    def _provider_with_discovered_models(
        self,
        provider: ProviderOption,
        *,
        refresh: bool = False,
    ) -> ProviderOption:
        """Return a provider enriched with models advertised by its API."""

        if refresh or provider.key not in self._discovered_provider_models:
            api_keys = self.home.load_auth().get("api_keys")
            api_key = api_keys.get(provider.key) if isinstance(api_keys, dict) else None
            discovered = discover_provider_models(provider.key, api_key)
            self._discovered_provider_models[provider.key] = merge_provider_models(
                provider.models,
                discovered,
            )
        return replace(
            provider,
            models=self._discovered_provider_models.get(provider.key, provider.models),
        )

    def run(self) -> int:
        """Run the full-screen terminal UI."""

        self.prepare_startup_config()
        self.prepare_background_startup()
        try:
            return int(curses.wrapper(self._run))
        finally:
            self._disable_bracketed_paste()
            if self._startup_tasks is not None:
                self._startup_tasks.close()
            self._file_reference_search.close()
            if isinstance(self.runtime, RuntimeProcessClient):
                self.runtime.close()

    def _run(self, stdscr: CursesWindow) -> int:
        self._configure_terminal(stdscr)
        self._poll_startup_tasks()

        while True:
            config = self.home.load_config()
            if not self._onboarding_complete(config):
                if not self._run_onboarding(stdscr):
                    return 1
                config = self.home.load_config()

            should_check_access = not self.home.is_repo_trusted(self.workspace_root)
            if should_check_access and not self._run_access_check(stdscr):
                return 1

            self.state = AgentState.PROJECT
            project = self._startup_project(stdscr, self._startup_preparation)
            self._cleanup_stale_project_sessions(project)
            if not self._handle_sandbox_check(stdscr, project):
                return 1
            try:
                return self._run_project(stdscr, project)
            finally:
                self._shutdown_active_session_turns()
                self._shutdown_sandbox_containers()
                self.runtime.shutdown()

    def _poll_startup_tasks(self) -> None:
        if self._startup_tasks is not None:
            self._startup_tasks.poll()

    def _configure_terminal(self, stdscr: CursesWindow) -> None:
        self._activity_wave_palettes.clear()
        stdscr.keypad(True)
        curses.noecho()
        with suppress(curses.error):
            curses.raw()
        with suppress(curses.error):
            curses.set_escdelay(120)
        with suppress(curses.error):
            curses.curs_set(0)
        with suppress(curses.error):
            curses.mousemask(curses.ALL_MOUSE_EVENTS | getattr(curses, "REPORT_MOUSE_POSITION", 0))
            curses.mouseinterval(0)
        if self.use_color and curses.has_colors():
            curses.start_color()
            default_fg, default_bg = self._terminal_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, default_bg)
            curses.init_pair(2, default_fg, curses.COLOR_CYAN)
            curses.init_pair(3, curses.COLOR_YELLOW, default_bg)
            curses.init_pair(4, curses.COLOR_GREEN, default_bg)
            curses.init_pair(5, curses.COLOR_RED, default_bg)
            curses.init_pair(6, default_fg, default_bg)
            curses.init_pair(7, default_fg, default_bg)
            brand_dot_pair = 3
            warning_badge_pair = 3
            subagent_pair = 3
            subagent_badge_pair = 3
            if getattr(curses, "COLORS", 0) > 208 and getattr(curses, "COLOR_PAIRS", 0) > 8:
                with suppress(curses.error):
                    curses.init_pair(8, 208, default_bg)
                    brand_dot_pair = 8
                    subagent_pair = 8
            if getattr(curses, "COLOR_PAIRS", 0) > 9:
                with suppress(curses.error):
                    curses.init_pair(9, curses.COLOR_BLACK, curses.COLOR_YELLOW)
                    warning_badge_pair = 9
            if getattr(curses, "COLOR_PAIRS", 0) > 10:
                with suppress(curses.error):
                    curses.init_pair(10, default_fg, default_bg)
            if getattr(curses, "COLORS", 0) > 208 and getattr(curses, "COLOR_PAIRS", 0) > 11:
                with suppress(curses.error):
                    curses.init_pair(11, curses.COLOR_BLACK, 208)
                    subagent_badge_pair = 11
            self._colors = {
                "accent": curses.color_pair(1) | curses.A_BOLD,
                "match": curses.color_pair(subagent_pair) | curses.A_BOLD,
                "subagent": curses.color_pair(subagent_pair) | curses.A_BOLD,
                "subagent_badge": curses.color_pair(subagent_badge_pair) | curses.A_BOLD,
                "selected": curses.color_pair(7) | curses.A_REVERSE | curses.A_BOLD,
                "cursor": curses.color_pair(7) | curses.A_REVERSE,
                "background": curses.color_pair(7),
                "muted": curses.color_pair(7) | curses.A_DIM,
                "light": curses.color_pair(6) | curses.A_DIM,
                "user": (
                    curses.color_pair(10)
                    if getattr(curses, "COLOR_PAIRS", 0) > 10
                    else curses.color_pair(7)
                )
                | curses.A_REVERSE,
                "warning": curses.color_pair(3) | curses.A_BOLD,
                "warning_badge": curses.color_pair(warning_badge_pair) | curses.A_BOLD,
                "ok": curses.color_pair(4) | curses.A_BOLD,
                "danger": curses.color_pair(5) | curses.A_BOLD,
                "bold": curses.color_pair(6) | curses.A_BOLD,
                "work_box": curses.color_pair(6) | curses.A_REVERSE,
                "table_header": curses.color_pair(6) | curses.A_REVERSE | curses.A_BOLD,
                "table_border": curses.color_pair(6) | curses.A_DIM,
                "matrix_dim": curses.color_pair(6) | curses.A_DIM,
                "matrix_function": curses.color_pair(1) | curses.A_BOLD,
                "matrix_blue": curses.color_pair(1) | curses.A_BOLD,
                "matrix_wave": curses.color_pair(6) | curses.A_BOLD,
                "matrix_hot": curses.color_pair(6) | curses.A_BOLD,
                "matrix_brand": curses.color_pair(1) | curses.A_BOLD,
                "brand_dot": curses.color_pair(brand_dot_pair) | curses.A_BOLD,
            }
            self._configure_activity_wave_colors(default_fg, default_bg)
        else:
            self._colors = {
                "accent": curses.A_BOLD,
                "match": curses.A_BOLD,
                "subagent": curses.A_BOLD,
                "subagent_badge": curses.A_REVERSE | curses.A_BOLD,
                "selected": curses.A_REVERSE,
                "cursor": curses.A_REVERSE,
                "background": curses.A_NORMAL,
                "light": curses.A_DIM,
                "muted": curses.A_DIM,
                "user": curses.A_REVERSE,
                "warning": curses.A_BOLD,
                "warning_badge": curses.A_REVERSE | curses.A_BOLD,
                "ok": curses.A_BOLD,
                "danger": curses.A_BOLD,
                "bold": curses.A_BOLD,
                "work_box": curses.A_REVERSE,
                "table_header": curses.A_REVERSE | curses.A_BOLD,
                "table_border": curses.A_DIM,
                "matrix_dim": curses.A_DIM,
                "matrix_function": curses.A_BOLD,
                "matrix_blue": curses.A_NORMAL,
                "matrix_wave": curses.A_BOLD,
                "matrix_hot": curses.A_BOLD,
                "matrix_brand": curses.A_BOLD,
                "brand_dot": curses.A_BOLD,
            }
        with suppress(curses.error):
            stdscr.bkgd(" ", self._attr("background"))
        self._enable_bracketed_paste()

    def _terminal_default_colors(self) -> tuple[int, int]:
        """Return curses foreground/background values that preserve the terminal theme."""
        use_default_colors = getattr(curses, "use_default_colors", None)
        if callable(use_default_colors):
            with suppress(curses.error):
                use_default_colors()
                return -1, -1
        return curses.COLOR_WHITE, curses.COLOR_BLACK

    def _enable_bracketed_paste(self) -> None:
        with suppress(OSError):
            os.write(1, b"\x1b[?2004h")

    def _disable_bracketed_paste(self) -> None:
        with suppress(OSError):
            os.write(1, b"\x1b[?2004l")

    def _prepare_startup_state(self) -> StartupPreparation:
        """Precompute project metadata before the loading animation ends."""

        config = self.home.load_config()
        if not self._onboarding_complete(config):
            return StartupPreparation()
        provider = provider_by_key(str(config.get("provider") or ""))
        if provider is not None:
            self._provider_with_discovered_models(provider, refresh=True)
        project = self._ensure_project()
        sandbox_runtime: str | None = None
        if config.get("sandbox_enabled"):
            from anomx.agent.helpers.sandbox import detect_container_runtime

            sandbox_runtime = detect_container_runtime()
        return StartupPreparation(project=project, sandbox_runtime=sandbox_runtime)

    def _startup_project(
        self,
        stdscr: CursesWindow,
        preparation: StartupPreparation | None = None,
    ) -> ProjectRecord:
        prepared_project = preparation.project if preparation is not None else None
        if prepared_project is not None and prepared_project.path == self.project_path:
            return prepared_project
        stored_project = self.home.project_for_path(self.project_path)
        if stored_project is not None:
            return stored_project
        return self._ensure_project_with_loading(stdscr)

    def _ensure_project_with_loading(self, stdscr: CursesWindow) -> ProjectRecord:
        if self.isolate_runtime:
            return self._ensure_project()
        result: queue.SimpleQueue[ProjectRecord] = queue.SimpleQueue()

        def run_project_setup() -> None:
            result.put(self._ensure_project())

        worker = threading.Thread(target=run_project_setup, daemon=True)
        worker.start()
        frame = 0
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        with suppress(curses.error):
            stdscr.nodelay(True)
        try:
            while worker.is_alive():
                self._draw_shell(
                    stdscr,
                    "Preparing project",
                    f"  {spinner_chars[frame % len(spinner_chars)]}  Analyzing workspace",
                )
                frame += 1
                with suppress(curses.error):
                    stdscr.get_wch()
                time.sleep(STARTUP_FRAME_SECONDS)
        finally:
            with suppress(curses.error):
                stdscr.nodelay(False)
            stdscr.erase()
            stdscr.refresh()
        worker.join(timeout=0)
        with suppress(queue.Empty):
            return result.get_nowait()
        return self.home.save_project(self.project_path, self._fallback_project_name())

    def _ensure_project(self) -> ProjectRecord:
        stored_project = self.home.project_for_path(self.project_path)
        if stored_project is not None:
            return stored_project
        if self.isolate_runtime:
            return self.home.save_project(
                self.project_path,
                self._fallback_project_name(),
            )
        name = ""
        outline = self._project_directory_outline(self.project_path)
        with suppress(Exception):
            name = self.runtime.suggest_project_name(self.project_path, outline) or ""
        return self.home.save_project(
            self.project_path,
            name or self._fallback_project_name(),
        )

    def _fallback_project_name(self) -> str:
        name = self.project_path.name.strip() or "Anomx Project"
        words = [word for word in re.split(r"[\s._-]+", name) if word.strip()]
        if not words:
            return "Anomx Project"
        if len(words) == 1:
            words.append("Project")
        return " ".join(word[:1].upper() + word[1:] for word in words[:3])

    def _project_directory_outline(self, root: Path, max_depth: int = 3) -> str:
        lines: list[str] = []

        def visit(path: Path, depth: int, prefix: str) -> None:
            if depth > max_depth:
                return
            try:
                entries = sorted(
                    path.iterdir(),
                    key=lambda item: (not item.is_dir(), item.name.lower()),
                )
            except OSError:
                return
            visible = [entry for entry in entries if entry.name not in IGNORED_FILE_REFERENCE_DIRS][
                :24
            ]
            for entry in visible:
                marker = "/" if entry.is_dir() else ""
                lines.append(f"{prefix}{entry.name}{marker}")
                if entry.is_dir():
                    visit(entry, depth + 1, f"{prefix}  ")

        lines.append(f"{root.name or root}/")
        visit(root, 1, "  ")
        return "\n".join(lines[:160])

    def _run_project(self, stdscr: CursesWindow, project: ProjectRecord) -> int:
        input_text = ""
        cursor = 0
        selected = 0
        scroll = 0
        command_selected = 0
        file_references: dict[str, str] = {}
        pasted_spans: list[PromptPasteSpan] = []
        file_selected = 0
        delete_pending_index: int | None = None
        frame = 0
        exit_confirm_deadline = 0.0
        prompt_notice = ""
        prompt_notice_role = "light"

        while True:
            if exit_confirm_deadline and time.monotonic() > exit_confirm_deadline:
                exit_confirm_deadline = 0.0
                prompt_notice = ""
            self._drain_active_session_turns(stdscr)
            sessions = self._project_sessions(project.path)
            selected = max(0, min(selected, len(sessions) - 1)) if sessions else 0
            if delete_pending_index is not None and delete_pending_index >= len(sessions):
                delete_pending_index = None
            command_token = self._active_command_token(input_text, cursor)
            command_suggestions = (
                self._filtered_project_commands(command_token[2])
                if command_token is not None
                else []
            )
            if command_suggestions:
                command_selected = min(command_selected, len(command_suggestions) - 1)
            else:
                command_selected = 0
            file_reference_token = self._active_file_reference_token(input_text, cursor)
            file_suggestions = (
                self._background_file_reference_suggestions(file_reference_token[2])
                if file_reference_token is not None
                else []
            )
            file_selected = min(file_selected, len(file_suggestions) - 1) if file_suggestions else 0
            file_search_active = (
                file_reference_token is not None
                and self._file_reference_search.is_searching(file_reference_token[2])
            )
            scroll = self._draw_project(
                stdscr,
                project,
                sessions,
                selected,
                scroll,
                input_text,
                cursor,
                prompt_notice,
                prompt_notice_role,
                frame,
                command_suggestions=command_suggestions,
                command_selected=command_selected,
                delete_pending_index=delete_pending_index,
                file_suggestions=file_suggestions if file_reference_token else [],
                file_selected=file_selected,
                file_references=file_references,
                file_reference_active=file_reference_token is not None,
                file_reference_searching=file_search_active,
                pasted_spans=pasted_spans,
            )
            animated = self._project_animation_active(sessions) or file_search_active
            if animated:
                with suppress(curses.error, AttributeError):
                    stdscr.nodelay(True)
                key = self._read_nonblocking_key(stdscr)
                if key is None:
                    time.sleep(0.08)
                    frame += 1
                    continue
                with suppress(curses.error, AttributeError):
                    stdscr.nodelay(False)
            else:
                with suppress(curses.error, AttributeError):
                    stdscr.nodelay(False)
                key = self._read_prompt_key(stdscr)
            frame += 1

            if self._is_shift_tab(key):
                self._cycle_agent_mode()
                continue
            if self._is_ctrl_c(key):
                if input_text:
                    input_text = ""
                    cursor = 0
                    pasted_spans = []
                    command_selected = 0
                    file_references = {}
                    file_selected = 0
                    prompt_notice = ""
                    continue
                now = time.monotonic()
                if exit_confirm_deadline and now <= exit_confirm_deadline:
                    return 0
                exit_confirm_deadline = now + 3.0
                prompt_notice = EXIT_ANOMX_CONFIRM_NOTICE
                prompt_notice_role = "light"
                continue
            if self._is_escape(key):
                delete_pending_index = None
                input_text = ""
                cursor = 0
                pasted_spans = []
                command_selected = 0
                file_references = {}
                file_selected = 0
                prompt_notice = ""
                continue
            if key == curses.KEY_UP:
                if file_suggestions:
                    file_selected = max(0, file_selected - 1)
                elif command_suggestions:
                    command_selected = max(0, command_selected - 1)
                elif input_text:
                    cursor = self._move_prompt_cursor_row(
                        stdscr,
                        input_text,
                        cursor,
                        -1,
                        pasted_spans,
                    )
                elif sessions:
                    selected = max(0, selected - 1)
                    delete_pending_index = None
                continue
            if key == curses.KEY_DOWN:
                if file_suggestions:
                    file_selected = min(len(file_suggestions) - 1, file_selected + 1)
                elif command_suggestions:
                    command_selected = min(len(command_suggestions) - 1, command_selected + 1)
                elif input_text:
                    cursor = self._move_prompt_cursor_row(
                        stdscr,
                        input_text,
                        cursor,
                        1,
                        pasted_spans,
                    )
                elif sessions:
                    selected = min(len(sessions) - 1, selected + 1)
                    delete_pending_index = None
                continue
            if key == curses.KEY_PPAGE:
                scroll = max(0, scroll - 5)
                selected = max(0, selected - 5)
                delete_pending_index = None
                continue
            if key == curses.KEY_NPAGE:
                selected = min(max(0, len(sessions) - 1), selected + 5)
                scroll += 5
                delete_pending_index = None
                continue
            if self._is_option_left(key):
                cursor = self._previous_prompt_word(input_text, cursor)
                continue
            if self._is_option_right(key):
                cursor = self._next_prompt_word(input_text, cursor)
                continue
            if key == curses.KEY_LEFT:
                cursor = max(0, cursor - 1)
                continue
            if key == curses.KEY_RIGHT:
                cursor = min(len(input_text), cursor + 1)
                continue
            if key == curses.KEY_HOME:
                cursor = 0
                continue
            if key == curses.KEY_END:
                cursor = len(input_text)
                continue
            if key == curses.KEY_MOUSE:
                action = self._project_mouse_action(
                    stdscr,
                    input_text,
                    command_suggestions,
                    command_selected,
                    file_suggestions=file_suggestions if file_reference_token else [],
                    file_selected=file_selected,
                    pasted_spans=pasted_spans,
                )
                if action is None:
                    continue
                if action.kind == "cursor":
                    cursor = action.value
                elif action.kind == "scroll":
                    scroll = max(0, scroll + action.value)
                elif action.kind == "open_project_session":
                    delete_pending_index = None
                    opened = self._open_project_session(stdscr, sessions[action.value])
                    if isinstance(opened, int):
                        return opened
                    input_text = ""
                    cursor = 0
                    pasted_spans = []
                    command_selected = 0
                elif action.kind == "file_reference":
                    if file_reference_token is not None:
                        input_text, cursor = self._insert_file_reference(
                            input_text,
                            cursor,
                            file_reference_token,
                            file_suggestions[action.value],
                            file_references,
                            pasted_spans,
                        )
                        file_selected = 0
                elif action.kind == "command":
                    suggestion = command_suggestions[action.value]
                    if command_token is not None and self._command_token_is_inline(
                        input_text,
                        command_token,
                    ):
                        input_text, cursor = self._remove_command_token(
                            input_text,
                            command_token,
                            pasted_spans,
                        )
                        result = self._handle_project_command_with_prompt(
                            stdscr,
                            suggestion.command,
                            suggestion.command,
                            project,
                            sessions,
                            selected,
                            scroll,
                            input_text,
                            cursor,
                            pasted_spans,
                            file_references,
                        )
                        if result == "exit":
                            return 0
                        if isinstance(result, SessionRecord):
                            opened = self._open_project_session(stdscr, result)
                            if isinstance(opened, int):
                                return opened
                        sessions = self._project_sessions(project.path)
                        selected = max(0, min(selected, len(sessions) - 1)) if sessions else 0
                        command_selected = 0
                    else:
                        command = suggestion.command
                        result = self._handle_project_command(
                            stdscr,
                            command,
                            command,
                            project,
                            sessions,
                            selected,
                            scroll,
                        )
                        if result == "exit":
                            return 0
                        if isinstance(result, SessionRecord):
                            opened = self._open_project_session(stdscr, result)
                            if isinstance(opened, int):
                                return opened
                        sessions = self._project_sessions(project.path)
                        selected = max(0, min(selected, len(sessions) - 1)) if sessions else 0
                        delete_pending_index = None
                        input_text = ""
                        cursor = 0
                        pasted_spans = []
                        command_selected = 0
                continue
            if self._is_raw_mouse_fragment_key(key):
                continue
            if isinstance(key, PromptPasteEvent):
                if key.text:
                    input_text, cursor = self._insert_prompt_text(
                        input_text,
                        cursor,
                        key.text,
                        pasted_spans,
                        pasted=True,
                    )
                    command_selected = 0
                    file_selected = 0
                    delete_pending_index = None
                    prompt_notice = ""
                continue
            if self._is_ctrl_d(key):
                if sessions:
                    delete_pending_index = selected
                    prompt_notice = ""
                continue
            if self._is_shift_enter(key):
                input_text = input_text[:cursor] + "\n" + input_text[cursor:]
                pasted_spans[:] = self._prompt_spans_after_replacement(
                    pasted_spans,
                    cursor,
                    cursor,
                    1,
                )
                cursor += 1
                command_selected = 0
                delete_pending_index = None
                prompt_notice = ""
                continue
            if self._is_enter(key):
                if file_suggestions and file_reference_token is not None:
                    input_text, cursor = self._insert_file_reference(
                        input_text,
                        cursor,
                        file_reference_token,
                        file_suggestions[file_selected],
                        file_references,
                        pasted_spans,
                    )
                    file_selected = 0
                    continue
                if (
                    command_token is not None
                    and command_suggestions
                    and self._command_token_is_inline(input_text, command_token)
                ):
                    suggestion = command_suggestions[command_selected]
                    input_text, cursor = self._remove_command_token(
                        input_text,
                        command_token,
                        pasted_spans,
                    )
                    result = self._handle_project_command_with_prompt(
                        stdscr,
                        suggestion.command,
                        suggestion.command,
                        project,
                        sessions,
                        selected,
                        scroll,
                        input_text,
                        cursor,
                        pasted_spans,
                        file_references,
                    )
                    if result == "exit":
                        return 0
                    if isinstance(result, SessionRecord):
                        opened = self._open_project_session(stdscr, result)
                        if isinstance(opened, int):
                            return opened
                    sessions = self._project_sessions(project.path)
                    selected = max(0, min(selected, len(sessions) - 1)) if sessions else 0
                    command_selected = 0
                    continue
                submitted = input_text.strip()
                if submitted:
                    if submitted.startswith("/"):
                        command = self._submitted_command(
                            submitted,
                            command_suggestions,
                            command_selected,
                        )
                        result = self._handle_project_command(
                            stdscr,
                            command,
                            submitted,
                            project,
                            sessions,
                            selected,
                            scroll,
                        )
                        if result == "exit":
                            return 0
                        if isinstance(result, SessionRecord):
                            opened = self._open_project_session(stdscr, result)
                            if isinstance(opened, int):
                                return opened
                    else:
                        backend_message = self._backend_message_for_prompt(
                            submitted,
                            file_references,
                        )
                        prompt_session = self._start_project_prompt_session(
                            submitted,
                            backend_message=backend_message,
                            file_references=dict(file_references),
                        )
                        opened = self._open_project_session(stdscr, prompt_session)
                        if isinstance(opened, int):
                            return opened
                        sessions = self._project_sessions(project.path)
                        selected = max(0, min(selected, len(sessions) - 1)) if sessions else 0
                        prompt_notice = ""
                    input_text = ""
                    cursor = 0
                    pasted_spans = []
                    command_selected = 0
                    file_references = {}
                    file_selected = 0
                    delete_pending_index = None
                    continue
                if delete_pending_index == selected and sessions:
                    self._delete_project_session(sessions[selected])
                    sessions = self._project_sessions(project.path)
                    selected = max(0, min(selected, len(sessions) - 1)) if sessions else 0
                    delete_pending_index = None
                    prompt_notice = ""
                    continue
                if sessions:
                    delete_pending_index = None
                    opened = self._open_project_session(stdscr, sessions[selected])
                    if isinstance(opened, int):
                        return opened
                continue
            if self._is_option_delete(key):
                word_start = self._previous_prompt_word(input_text, cursor)
                input_text, cursor = self._replace_prompt_range(
                    input_text,
                    word_start,
                    cursor,
                    "",
                    pasted_spans,
                )
                command_selected = 0
                file_selected = 0
                delete_pending_index = None
                continue
            if self._is_backspace(key):
                if cursor > 0:
                    input_text, cursor = self._delete_prompt_backward(
                        input_text,
                        cursor,
                        pasted_spans,
                        file_references,
                    )
                    command_selected = 0
                    file_selected = 0
                    delete_pending_index = None
                continue
            if isinstance(key, str) and key.isprintable():
                input_text, cursor = self._insert_prompt_text(
                    input_text,
                    cursor,
                    key,
                    pasted_spans,
                )
                command_selected = 0
                file_selected = 0
                delete_pending_index = None
                prompt_notice = ""

    def _open_project_session(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
    ) -> int | None:
        opened = self._prepare_session_for_opening(session)
        result = self._run_session(stdscr, opened)
        return None if result == "project" else int(result)

    def _start_project_prompt_session(
        self,
        message: str,
        backend_message: str = "",
        file_references: Mapping[str, str] | None = None,
    ) -> SessionRecord:
        session = self._create_session()
        payload: dict[str, object] = {"message": message}
        if backend_message:
            payload["backend_message"] = backend_message
        if file_references:
            payload["file_references"] = file_references
        self.home.append_session_event(session.path, "user_message", payload)
        self._maybe_start_session_rename(session)
        turn = self._start_session_turn(session)
        turn.anchor_expansion_key = self._latest_root_user_expansion_key(session.path)
        return session

    def _start_project_skill_session(
        self,
        skill: Skill,
        submitted: str,
    ) -> SessionRecord:
        session = self._create_session()
        self._append_skill_invocation_event(session, skill, submitted)
        self._maybe_start_session_rename(session)
        turn = self._start_session_turn(session)
        turn.anchor_expansion_key = self._latest_root_user_expansion_key(session.path)
        return session

    def _handle_project_command(
        self,
        stdscr: CursesWindow,
        command: str,
        submitted: str,
        project: ProjectRecord,
        sessions: Sequence[SessionRecord],
        selected: int,
        scroll: int = 0,
    ) -> str | SessionRecord | None:
        if command == "/exit":
            return "exit"
        if command == "/new":
            return self._create_session()
        if not self._project_command_allowed(command):
            self._message(
                stdscr,
                "Unavailable Command",
                f"{command} is available from inside a session.",
            )
            return None
        if command == "/model":
            self._run_project_model_panel(
                stdscr,
                project,
                sessions,
                selected,
                scroll,
            )
            return None
        if command == "/effort":
            if not self._apply_effort_argument(stdscr, submitted):
                self._run_project_effort_panel(
                    stdscr,
                    project,
                    sessions,
                    selected,
                    scroll,
                )
            return None
        if command == "/feedback":
            current_session = (
                self._project_command_session(sessions, selected) or self._ephemeral_session()
            )
            self._send_cli_feedback(stdscr, current_session, submitted)
            return None
        skill = self._skill_for_command(command)
        if skill is not None:
            return self._start_project_skill_session(skill, submitted or command)
        current_session = self._project_command_session(sessions, selected)
        if current_session is None and command == "/config":
            current_session = self._ephemeral_session()
        elif current_session is None:
            current_session = self._create_session()
        return self._handle_command(stdscr, command, current_session, submitted)

    def _handle_project_command_with_prompt(
        self,
        stdscr: CursesWindow,
        command: str,
        submitted: str,
        project: ProjectRecord,
        sessions: Sequence[SessionRecord],
        selected: int,
        scroll: int,
        input_text: str,
        cursor: int,
        pasted_spans: Sequence[PromptPasteSpan],
        file_references: Mapping[str, str],
    ) -> str | SessionRecord | None:
        with self._preserve_modal_prompt(
            input_text,
            cursor,
            pasted_spans,
            file_references,
        ):
            return self._handle_project_command(
                stdscr,
                command,
                submitted,
                project,
                sessions,
                selected,
                scroll,
            )

    def _project_command_allowed(self, command: str) -> bool:
        return command in {spec.command for spec in self._project_command_specs()}

    def _project_command_session(
        self,
        sessions: Sequence[SessionRecord],
        selected: int,
    ) -> SessionRecord | None:
        if not sessions:
            return None
        index = max(0, min(selected, len(sessions) - 1))
        return sessions[index]

    def _delete_project_session(self, session: SessionRecord) -> None:
        turn = self._active_turn_for_session(session)
        if turn is not None:
            turn.runtime.shutdown(session.path)
            turn.completed = True
            self._active_session_turns.pop(self._session_turn_key(session), None)
        self.home.delete_session(session.path)
        with suppress(KeyError):
            self._session_event_cache.pop(session.path.expanduser().resolve())
        with suppress(KeyError):
            self._message_line_cache.pop(session.path.expanduser().resolve())
        with suppress(KeyError):
            self._rendered_message_cache.pop(session.path.expanduser().resolve())

    def _project_sessions(self, project_path: Path) -> list[SessionRecord]:
        project_key = str(project_path.expanduser().resolve())
        sessions = [
            session
            for session in self.home.list_sessions(limit=None)
            if session.cwd.strip() == project_key
        ]
        sessions.sort(
            key=lambda session: (session.last_user_at or session.created_at, session.created_at),
            reverse=True,
        )
        return sessions

    def _cleanup_stale_project_sessions(self, project: ProjectRecord) -> None:
        if self.isolate_runtime:
            threading.Thread(
                target=self._cleanup_stale_project_sessions_in_runtime,
                args=(project,),
                name="anomx-runtime-cleanup",
                daemon=True,
            ).start()
            return
        self._cleanup_stale_project_sessions_in_runtime(project)

    def _cleanup_stale_project_sessions_in_runtime(self, project: ProjectRecord) -> None:
        for session in self._project_sessions(project.path):
            if self._active_turn_for_session(session) is None:
                self.runtime.cleanup_session_runtime_state(session.path)

    def _project_animation_active(self, sessions: Sequence[SessionRecord]) -> bool:
        return any(self._active_turn_for_session(session) is not None for session in sessions)

    def _startup_session(
        self,
        stdscr: CursesWindow,
        preparation: StartupPreparation | None = None,
    ) -> SessionRecord:
        previous_session = (
            preparation.previous_session
            if preparation is not None and preparation.previous_session is not None
            else self._latest_continuable_session()
        )
        if previous_session is None:
            return self._create_session()

        statement = (
            preparation.continuation_statement
            if preparation is not None
            and preparation.previous_session is not None
            and preparation.previous_session.session_id == previous_session.session_id
            and preparation.continuation_statement
            else self._continue_session_statement(previous_session)
        )
        if self._run_continue_session_prompt(stdscr, previous_session, statement):
            return self._prepare_session_for_opening(previous_session)
        return self._create_session()

    def _prepare_session_for_opening(self, session: SessionRecord) -> SessionRecord:
        if session.unread:
            self.home.set_session_unread(session.path, False)
            session = replace(session, unread=False)
        if self._active_turn_for_session(session) is None:
            if self.isolate_runtime:
                threading.Thread(
                    target=self.runtime.cleanup_session_runtime_state,
                    args=(session.path,),
                    name="anomx-runtime-session-cleanup",
                    daemon=True,
                ).start()
            else:
                self.runtime.cleanup_session_runtime_state(session.path)
        return session

    def _latest_continuable_session(self) -> SessionRecord | None:
        for session in self.home.list_sessions(limit=None):
            if self._session_belongs_to_workspace(session):
                return session
        return None

    def _session_belongs_to_workspace(self, session: SessionRecord) -> bool:
        cwd = session.cwd.strip()
        if not cwd:
            return False
        with suppress(OSError, RuntimeError):
            return discover_workspace_root(Path(cwd)) == self.workspace_root
        return False

    def _continue_session_statement(self, session: SessionRecord) -> str:
        workspace_name = self.workspace_root.name or str(self.workspace_root)
        if self.isolate_runtime:
            return self._fallback_continue_session_statement(session)
        with suppress(Exception):
            return self.runtime.suggest_session_continuation(
                session.path,
                workspace_name,
            )
        return self._fallback_continue_session_statement(session)

    def _fallback_continue_session_statement(self, session: SessionRecord) -> str:
        title = session.title.strip()
        if title and title != "New session":
            return f"Do you want to continue the previous session titled {title}?"
        workspace_name = self.workspace_root.name or str(self.workspace_root)
        return f"Do you want to continue the previous Anomx session in {workspace_name}?"

    def _run_continue_session_prompt(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        statement: str,
    ) -> bool:
        self.state = AgentState.CONTINUE_SESSION
        selected = 0
        try:
            while True:
                self._draw_continue_session_prompt(stdscr, session, statement, selected)
                key = stdscr.get_wch()
                if self._is_escape(key) or self._is_ctrl_c(key):
                    return False
                if key == curses.KEY_UP:
                    selected = max(0, selected - 1)
                elif key == curses.KEY_DOWN:
                    selected = min(1, selected + 1)
                elif self._is_enter(key):
                    return selected == 0
        finally:
            self.state = AgentState.NEW_SESSION

    def _create_session(self) -> SessionRecord:
        config = self.home.load_config()
        return self.home.create_session(
            self.cwd,
            provider=str(config.get("provider", "openai")),
            model=str(config.get("model", "gpt-5.5")),
            mode=self.agent_mode,
            agent_kind=self.active_agent.kind,
        )

    def _restore_session_model(self, session: SessionRecord) -> None:
        """Make the chat's stored provider/model the active selection on open."""
        provider = str(getattr(session, "provider", "") or "").strip()
        model = str(getattr(session, "model", "") or "").strip()
        if not provider or not model:
            return
        config = self.home.load_config()
        if str(config.get("provider", "")) == provider and str(config.get("model", "")) == model:
            return
        config["provider"] = provider
        config["model"] = model
        self.home.save_config(config)

    def _ephemeral_session(self) -> SessionRecord:
        """Return a non-persisted placeholder session for backdrop rendering."""
        return SessionRecord(
            session_id="",
            path=Path(os.devnull),
            created_at="",
            updated_at="",
            cwd=str(self.cwd),
            provider="",
            model="",
            title="",
        )

    def _run_session(self, stdscr: CursesWindow, session: SessionRecord) -> int | str:
        current_session = session
        self._activate_agent(current_session.agent_kind)
        self._restore_session_model(current_session)
        input_text = ""
        cursor = 0
        file_references: dict[str, str] = {}
        image_attachments: dict[str, dict[str, str]] = {}
        pasted_spans: list[PromptPasteSpan] = []
        scroll = 0
        command_selected = 0
        file_selected = 0
        pinned_anchor: int | None = None
        exit_confirm_deadline = 0.0
        exit_notice = ""
        running_notice = RUNNING_NOTICE
        running_notice_role = "light"
        running_abort_key = ""
        running_abort_deadline = 0.0
        frame = 0

        while True:
            if exit_confirm_deadline and time.monotonic() > exit_confirm_deadline:
                exit_confirm_deadline = 0.0
                exit_notice = ""
            current_session = self._process_title_events(stdscr, current_session)
            active_turn = self._active_turn_for_session(current_session)
            active_turn_running = (
                active_turn is not None
                and active_turn.worker is not None
                and active_turn.worker.is_alive()
            )
            if active_turn_running and active_turn is not None:
                pinned_anchor = self._active_turn_anchor_line(
                    stdscr,
                    current_session,
                    active_turn,
                    pinned_anchor,
                )
                self._drain_session_turn_events(
                    stdscr,
                    active_turn,
                    anchor_line=pinned_anchor,
                    input_text=input_text,
                    cursor=cursor,
                    pasted_spans=pasted_spans,
                    prompt_notice=running_notice,
                    prompt_notice_role=running_notice_role,
                    scroll=scroll,
                    active_turn_elapsed=time.monotonic() - active_turn.started_at,
                    sticky_anchor=pinned_anchor is not None,
                )
            elif active_turn is not None:
                self._complete_session_turn(
                    stdscr,
                    active_turn,
                    anchor_line=pinned_anchor,
                    scroll=scroll,
                    input_text=input_text,
                    cursor=cursor,
                    pasted_spans=pasted_spans,
                    render_final=True,
                )
                active_turn = None
                active_turn_running = False
                running_notice = RUNNING_NOTICE
                running_notice_role = "light"
                running_abort_key = ""
                running_abort_deadline = 0.0
            messages = self._read_message_lines(current_session.path)
            command_token = self._active_command_token(input_text, cursor)
            command_suggestions = (
                (
                    self._filtered_running_commands(command_token[2])
                    if active_turn_running
                    else self._filtered_commands(command_token[2])
                )
                if command_token is not None
                else []
            )
            file_reference_token = self._active_file_reference_token(input_text, cursor)
            file_suggestions = (
                self._background_file_reference_suggestions(file_reference_token[2])
                if file_reference_token is not None and not active_turn_running
                else []
            )
            if command_suggestions:
                command_selected = min(command_selected, len(command_suggestions) - 1)
            else:
                command_selected = 0
            file_selected = min(file_selected, len(file_suggestions) - 1) if file_suggestions else 0
            file_search_active = (
                file_reference_token is not None
                and not active_turn_running
                and self._file_reference_search.is_searching(file_reference_token[2])
            )
            viewport = self._draw_session(
                stdscr,
                current_session,
                messages,
                input_text,
                cursor,
                scroll,
                command_suggestions,
                command_selected,
                file_suggestions=file_suggestions,
                file_selected=file_selected,
                file_reference_searching=file_search_active,
                file_references=file_references,
                image_attachments=image_attachments,
                anchor_line=pinned_anchor,
                prompt_notice=running_notice if active_turn_running else exit_notice,
                prompt_notice_role=running_notice_role if active_turn_running else "light",
                working_text=(
                    active_turn.working_text if active_turn_running and active_turn else None
                ),
                working_deadline=(
                    active_turn.working_deadline if active_turn_running and active_turn else None
                ),
                working_frame=frame,
                sticky_anchor=active_turn_running and pinned_anchor is not None,
                active_turn_elapsed=(
                    time.monotonic() - active_turn.started_at
                    if active_turn_running and active_turn
                    else None
                ),
                pasted_spans=pasted_spans,
            )
            if viewport is not None:
                scroll = viewport.scroll
            if (
                active_turn_running
                or self._session_animation_active(current_session.path)
                or file_search_active
            ):
                with suppress(curses.error, AttributeError):
                    stdscr.nodelay(True)
                key = self._read_nonblocking_key(stdscr)
                if key is None:
                    time.sleep(ACTIVITY_FRAME_SECONDS if active_turn_running else 0.08)
                    frame += 1
                    continue
                with suppress(curses.error, AttributeError):
                    stdscr.nodelay(False)
            else:
                with suppress(curses.error, AttributeError):
                    stdscr.nodelay(False)
                key = self._read_prompt_key(stdscr)
            frame += 1

            if active_turn_running and active_turn is not None:
                key_result = self._handle_running_key(
                    stdscr,
                    current_session,
                    key,
                    input_text,
                    cursor,
                    running_abort_key,
                    running_abort_deadline,
                    command_suggestions,
                    command_selected,
                    pasted_spans,
                )
                input_text = key_result.input_text
                cursor = key_result.cursor
                running_notice = key_result.notice
                running_notice_role = key_result.notice_role
                running_abort_key = key_result.abort_key
                running_abort_deadline = key_result.abort_deadline
                command_selected = key_result.command_selected
                if key_result.clear_anchor and not key_result.scroll_delta:
                    pinned_anchor = None
                if key_result.scroll_delta:
                    scroll += key_result.scroll_delta
                if key_result.command:
                    with suppress(curses.error):
                        stdscr.nodelay(False)
                    try:
                        command_result = self._handle_command_with_prompt(
                            stdscr,
                            key_result.command,
                            current_session,
                            key_result.submitted,
                            input_text,
                            cursor,
                            pasted_spans,
                            file_references,
                            image_attachments,
                        )
                    finally:
                        with suppress(curses.error):
                            stdscr.nodelay(True)
                    running_notice = RUNNING_NOTICE
                    running_notice_role = "light"
                    if command_result == "exit":
                        self._interrupt_session_turn(
                            stdscr,
                            active_turn,
                            anchor_line=pinned_anchor,
                            scroll=scroll,
                            input_text=input_text,
                            cursor=cursor,
                            pasted_spans=pasted_spans,
                        )
                        return 0
                    if isinstance(command_result, SessionRecord):
                        current_session = command_result
                        self._activate_agent(current_session.agent_kind)
                        self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
                        scroll = 0
                        pinned_anchor = None
                    continue
                if key_result.exit_requested:
                    self._interrupt_session_turn(
                        stdscr,
                        active_turn,
                        anchor_line=pinned_anchor,
                        scroll=scroll,
                        input_text=input_text,
                        cursor=cursor,
                        pasted_spans=pasted_spans,
                    )
                    running_notice = RUNNING_NOTICE
                    running_notice_role = "light"
                    running_abort_key = ""
                    running_abort_deadline = 0.0
                    continue
                if key_result.back_requested:
                    return "project"
                if key_result.submitted_message:
                    self._interrupt_and_requeue_session_turn(
                        stdscr,
                        active_turn,
                        current_session,
                        key_result.submitted_message,
                        anchor_line=pinned_anchor,
                        scroll=scroll,
                    )
                    input_text = ""
                    cursor = 0
                    pasted_spans = []
                    command_selected = 0
                    file_selected = 0
                    file_references = {}
                    image_attachments = {}
                    running_notice = RUNNING_NOTICE
                    running_notice_role = "light"
                    running_abort_key = ""
                    running_abort_deadline = 0.0
                    continue
                continue

            if self._is_shift_tab(key):
                current_session = replace(
                    current_session,
                    mode=self._cycle_agent_mode(current_session),
                    agent_kind=self.active_agent.kind,
                )
                continue
            if self._is_ctrl_c(key):
                if input_text:
                    input_text = ""
                    cursor = 0
                    pasted_spans = []
                    command_selected = 0
                    file_selected = 0
                    file_references = {}
                    image_attachments = {}
                    exit_confirm_deadline = 0.0
                    exit_notice = ""
                    continue
                now = time.monotonic()
                if exit_confirm_deadline and now <= exit_confirm_deadline:
                    return 0
                exit_confirm_deadline = now + 3.0
                exit_notice = EXIT_ANOMX_CONFIRM_NOTICE
                continue
            if self._is_option_left(key):
                cursor = self._previous_prompt_word(input_text, cursor)
                continue
            if self._is_option_right(key):
                cursor = self._next_prompt_word(input_text, cursor)
                continue

            if self._is_escape(key):
                exit_confirm_deadline = 0.0
                exit_notice = ""
                if input_text:
                    input_text = ""
                    cursor = 0
                    pasted_spans = []
                    command_selected = 0
                    file_selected = 0
                    file_references = {}
                    image_attachments = {}
                elif pinned_anchor is not None:
                    pinned_anchor = None
                elif self._session_selection is not None:
                    self._clear_session_selection()
                else:
                    return "project"
                continue

            if key == curses.KEY_UP:
                if file_suggestions:
                    file_selected = max(0, file_selected - 1)
                    continue
                if command_suggestions:
                    command_selected = max(0, command_selected - 1)
                    continue
                moved_cursor = self._move_prompt_cursor_row(
                    stdscr,
                    input_text,
                    cursor,
                    direction=-1,
                    pasted_spans=pasted_spans,
                )
                if moved_cursor != cursor:
                    cursor = moved_cursor
                    continue
                pinned_anchor = None
                scroll += 1
                continue
            if key == curses.KEY_DOWN:
                if file_suggestions:
                    file_selected = min(len(file_suggestions) - 1, file_selected + 1)
                    continue
                if command_suggestions:
                    command_selected = min(len(command_suggestions) - 1, command_selected + 1)
                    continue
                moved_cursor = self._move_prompt_cursor_row(
                    stdscr,
                    input_text,
                    cursor,
                    direction=1,
                    pasted_spans=pasted_spans,
                )
                if moved_cursor != cursor:
                    cursor = moved_cursor
                    continue
                pinned_anchor = None
                scroll -= 1
                continue
            if key == curses.KEY_PPAGE:
                pinned_anchor = None
                scroll += 5
                continue
            if key == curses.KEY_NPAGE:
                pinned_anchor = None
                scroll -= 5
                continue
            if key == curses.KEY_LEFT:
                cursor = max(0, cursor - 1)
                continue
            if key == curses.KEY_RIGHT:
                cursor = min(len(input_text), cursor + 1)
                continue
            if key == curses.KEY_HOME:
                cursor = 0
                continue
            if key == curses.KEY_END:
                cursor = len(input_text)
                continue
            if key == curses.KEY_MOUSE:
                mouse_action = self._session_mouse_action(
                    stdscr,
                    input_text,
                    command_suggestions,
                    command_selected,
                    file_suggestions,
                    file_selected,
                    pasted_spans,
                )
                if mouse_action is None:
                    continue
                if mouse_action.kind == "cursor":
                    cursor = mouse_action.value
                elif mouse_action.kind == "back_project":
                    return "project"
                elif mouse_action.kind == "scroll":
                    pinned_anchor = None
                    scroll += mouse_action.value
                elif mouse_action.kind == "toggle_work":
                    self._toggle_work_turn(mouse_action.text)
                elif mouse_action.kind == "toggle_pinned_user":
                    self._toggle_pinned_user(mouse_action.text)
                elif mouse_action.kind == "toggle_plan":
                    self._toggle_plan(current_session.path)
                elif mouse_action.kind == "toggle_work_line":
                    self._toggle_work_line(mouse_action.text)
                elif mouse_action.kind == "toggle_activity_item":
                    self._toggle_activity_item(mouse_action.text)
                elif mouse_action.kind == "open_subagent":
                    self._open_subagent_session(stdscr, current_session, mouse_action.text)
                elif mouse_action.kind == "toggle_activity_entry":
                    self._toggle_activity_entry(mouse_action.text)
                elif mouse_action.kind == "scroll_activity_item":
                    self._scroll_activity_item(mouse_action.text, mouse_action.value)
                elif mouse_action.kind == "kill_process":
                    self._runtime_for_session(current_session).end_process(
                        mouse_action.text,
                        current_session.path,
                    )
                elif mouse_action.kind == "command":
                    suggestion = command_suggestions[mouse_action.value]
                    if command_token is not None and self._command_token_is_inline(
                        input_text,
                        command_token,
                    ):
                        input_text, cursor = self._remove_command_token(
                            input_text,
                            command_token,
                            pasted_spans,
                        )
                        command_result = self._handle_command_with_prompt(
                            stdscr,
                            suggestion.command,
                            current_session,
                            suggestion.command,
                            input_text,
                            cursor,
                            pasted_spans,
                            file_references,
                            image_attachments,
                        )
                        if command_result == "exit":
                            return 0
                        if isinstance(command_result, SessionRecord):
                            current_session = command_result
                            self._activate_agent(current_session.agent_kind)
                            self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
                            scroll = 0
                            pinned_anchor = None
                        command_selected = 0
                    else:
                        command = suggestion.command
                        command_result = self._handle_command(stdscr, command, current_session)
                        if command_result == "exit":
                            return 0
                        if isinstance(command_result, SessionRecord):
                            current_session = command_result
                            self._activate_agent(current_session.agent_kind)
                            self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
                            scroll = 0
                            pinned_anchor = None
                        input_text = ""
                        cursor = 0
                        pasted_spans = []
                        command_selected = 0
                        file_selected = 0
                        file_references = {}
                        image_attachments = {}
                elif mouse_action.kind == "file_reference":
                    if file_reference_token is None:
                        continue
                    input_text, cursor = self._insert_file_reference(
                        input_text,
                        cursor,
                        file_reference_token,
                        file_suggestions[mouse_action.value],
                        file_references,
                        pasted_spans,
                    )
                    file_selected = 0
                elif mouse_action.kind == "skill":
                    skill = self._skill_for_command(f"/{mouse_action.text}")
                    if skill is None:
                        continue
                    command_result = self._invoke_skill(
                        stdscr,
                        current_session,
                        skill,
                        skill.slash_command,
                    )
                    if command_result == "exit":
                        return 0
                    input_text = ""
                    cursor = 0
                    pasted_spans = []
                    command_selected = 0
                    file_selected = 0
                    file_references = {}
                    image_attachments = {}
                continue
            raw_mouse_action = self._raw_mouse_action(
                key,
                stdscr,
                input_text,
                command_suggestions,
                command_selected,
                file_suggestions,
                file_selected,
                pasted_spans,
            )
            if raw_mouse_action is not None:
                if raw_mouse_action.kind == "cursor":
                    cursor = raw_mouse_action.value
                elif raw_mouse_action.kind == "back_project":
                    return "project"
                elif raw_mouse_action.kind == "scroll":
                    pinned_anchor = None
                    scroll += raw_mouse_action.value
                elif raw_mouse_action.kind == "toggle_work":
                    self._toggle_work_turn(raw_mouse_action.text)
                elif raw_mouse_action.kind == "toggle_pinned_user":
                    self._toggle_pinned_user(raw_mouse_action.text)
                elif raw_mouse_action.kind == "toggle_plan":
                    self._toggle_plan(current_session.path)
                elif raw_mouse_action.kind == "toggle_work_line":
                    self._toggle_work_line(raw_mouse_action.text)
                elif raw_mouse_action.kind == "toggle_activity_item":
                    self._toggle_activity_item(raw_mouse_action.text)
                elif raw_mouse_action.kind == "open_subagent":
                    self._open_subagent_session(stdscr, current_session, raw_mouse_action.text)
                elif raw_mouse_action.kind == "toggle_activity_entry":
                    self._toggle_activity_entry(raw_mouse_action.text)
                elif raw_mouse_action.kind == "scroll_activity_item":
                    self._scroll_activity_item(raw_mouse_action.text, raw_mouse_action.value)
                elif raw_mouse_action.kind == "kill_process":
                    self._runtime_for_session(current_session).end_process(
                        raw_mouse_action.text,
                        current_session.path,
                    )
                elif raw_mouse_action.kind == "command":
                    suggestion = command_suggestions[raw_mouse_action.value]
                    if command_token is not None and self._command_token_is_inline(
                        input_text,
                        command_token,
                    ):
                        input_text, cursor = self._remove_command_token(
                            input_text,
                            command_token,
                            pasted_spans,
                        )
                        command_result = self._handle_command_with_prompt(
                            stdscr,
                            suggestion.command,
                            current_session,
                            suggestion.command,
                            input_text,
                            cursor,
                            pasted_spans,
                            file_references,
                            image_attachments,
                        )
                        if command_result == "exit":
                            return 0
                        if isinstance(command_result, SessionRecord):
                            current_session = command_result
                            self._activate_agent(current_session.agent_kind)
                            self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
                            scroll = 0
                            pinned_anchor = None
                        command_selected = 0
                    else:
                        command = suggestion.command
                        command_result = self._handle_command(stdscr, command, current_session)
                        if command_result == "exit":
                            return 0
                        if isinstance(command_result, SessionRecord):
                            current_session = command_result
                            self._activate_agent(current_session.agent_kind)
                            self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
                            scroll = 0
                            pinned_anchor = None
                        input_text = ""
                        cursor = 0
                        pasted_spans = []
                        command_selected = 0
                        file_selected = 0
                        file_references = {}
                        image_attachments = {}
                elif raw_mouse_action.kind == "file_reference":
                    if file_reference_token is None:
                        continue
                    input_text, cursor = self._insert_file_reference(
                        input_text,
                        cursor,
                        file_reference_token,
                        file_suggestions[raw_mouse_action.value],
                        file_references,
                        pasted_spans,
                    )
                    file_selected = 0
                elif raw_mouse_action.kind == "skill":
                    skill = self._skill_for_command(f"/{raw_mouse_action.text}")
                    if skill is None:
                        continue
                    command_result = self._invoke_skill(
                        stdscr,
                        current_session,
                        skill,
                        skill.slash_command,
                    )
                    if command_result == "exit":
                        return 0
                    input_text = ""
                    cursor = 0
                    pasted_spans = []
                    command_selected = 0
                    file_selected = 0
                    file_references = {}
                    image_attachments = {}
                continue
            if self._is_raw_mouse_fragment_key(key):
                continue
            if isinstance(key, PromptPasteEvent):
                if key.text:
                    input_text, cursor = self._insert_prompt_text(
                        input_text,
                        cursor,
                        key.text,
                        pasted_spans,
                        pasted=True,
                    )
                    input_text, cursor, added_images = self._consume_dropped_images(
                        input_text,
                        cursor,
                        image_attachments,
                        pasted_spans,
                    )
                    if added_images:
                        self._append_unsupported_image_notice(
                            current_session,
                            added_images,
                        )
                    command_selected = 0
                    file_selected = 0
                continue
            if self._is_enter(key):
                if file_suggestions and file_reference_token is not None:
                    input_text, cursor = self._insert_file_reference(
                        input_text,
                        cursor,
                        file_reference_token,
                        file_suggestions[file_selected],
                        file_references,
                        pasted_spans,
                    )
                    file_selected = 0
                    continue
                if (
                    command_token is not None
                    and command_suggestions
                    and self._command_token_is_inline(input_text, command_token)
                ):
                    suggestion = command_suggestions[command_selected]
                    input_text, cursor = self._remove_command_token(
                        input_text,
                        command_token,
                        pasted_spans,
                    )
                    command_result = self._handle_command_with_prompt(
                        stdscr,
                        suggestion.command,
                        current_session,
                        suggestion.command,
                        input_text,
                        cursor,
                        pasted_spans,
                        file_references,
                        image_attachments,
                    )
                    if command_result == "exit":
                        return 0
                    if isinstance(command_result, SessionRecord):
                        current_session = command_result
                        self._activate_agent(current_session.agent_kind)
                        self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
                        scroll = 0
                        pinned_anchor = None
                    command_selected = 0
                    continue
                submitted = input_text.strip()
                submitted_image_attachments = self._active_image_attachments(
                    submitted,
                    image_attachments,
                )
                backend_message = self._backend_message_for_prompt(
                    submitted,
                    file_references,
                )
                submitted_file_references = dict(file_references)
                selected_command = command_selected
                if not submitted:
                    input_text = ""
                    cursor = 0
                    pasted_spans = []
                    file_references = {}
                    image_attachments = {}
                    command_selected = 0
                    file_selected = 0
                    continue
                if submitted.startswith("/"):
                    input_text = ""
                    cursor = 0
                    pasted_spans = []
                    file_references = {}
                    image_attachments = {}
                    command_selected = 0
                    file_selected = 0
                    command = self._submitted_command(
                        submitted,
                        command_suggestions,
                        selected_command,
                    )
                    command_result = self._handle_command(
                        stdscr,
                        command,
                        current_session,
                        submitted,
                    )
                    if command_result == "exit":
                        return 0
                    if isinstance(command_result, SessionRecord):
                        current_session = command_result
                        self._activate_agent(current_session.agent_kind)
                        self._prompt_placeholder = random.choice(PROMPT_PLACEHOLDERS)
                        scroll = 0
                        pinned_anchor = None
                    continue
                if self._start_hints_visible_for_prompt(
                    current_session.path,
                    messages,
                    input_text,
                    command_suggestions,
                    file_suggestions,
                ):
                    self._animate_start_hints_exit(
                        stdscr,
                        current_session,
                        messages,
                        scroll,
                        pinned_anchor,
                    )
                input_text = ""
                cursor = 0
                pasted_spans = []
                file_references = {}
                image_attachments = {}
                command_selected = 0
                file_selected = 0
                self.home.append_session_event(
                    current_session.path,
                    "user_message",
                    {
                        "message": submitted,
                        "backend_message": backend_message,
                        "file_references": submitted_file_references,
                        "image_attachments": list(submitted_image_attachments.values()),
                    },
                )
                self._maybe_start_session_rename(current_session)
                anchor_line = self._latest_user_anchor_line(stdscr, current_session)
                self._animate_message_anchor(stdscr, current_session, anchor_line)
                turn = self._start_session_turn(current_session)
                turn.anchor_expansion_key = self._latest_root_user_expansion_key(
                    current_session.path
                )
                pinned_anchor = anchor_line
                # A prior turn may have left the view scrolled away from the bottom
                # (e.g. the user scrolled up to read earlier output). Without this,
                # the new response streams in below the visible area and the view
                # never follows it until the user manually scrolls down.
                scroll = 0
                running_notice = RUNNING_NOTICE
                running_notice_role = "light"
                running_abort_key = ""
                running_abort_deadline = 0.0
                continue

            if self._is_option_delete(key):
                word_start = self._previous_prompt_word(input_text, cursor)
                input_text, cursor = self._replace_prompt_range(
                    input_text,
                    word_start,
                    cursor,
                    "",
                    pasted_spans,
                )
                command_selected = 0
                file_selected = 0
                continue
            if self._is_backspace(key):
                if cursor > 0:
                    input_text, cursor = self._delete_prompt_backward(
                        input_text,
                        cursor,
                        pasted_spans,
                        file_references,
                        image_attachments,
                    )
                    command_selected = 0
                    file_selected = 0
                continue
            if isinstance(key, str) and key.isprintable():
                input_text, cursor = self._insert_prompt_text(
                    input_text,
                    cursor,
                    key,
                    pasted_spans,
                )
                input_text, cursor, added_images = self._consume_dropped_images(
                    input_text,
                    cursor,
                    image_attachments,
                    pasted_spans,
                )
                if added_images:
                    self._append_unsupported_image_notice(
                        current_session,
                        added_images,
                    )
                command_selected = 0
                file_selected = 0

    def _session_animation_active(self, session_path: Path) -> bool:
        return (
            self._start_hint_reveal_active()
            or self._has_running_session_activity(session_path)
            or self._plan_reveal_active(session_path)
        )

    def _has_running_session_activity(self, session_path: Path) -> bool:
        events = self._session_events(session_path)
        return bool(running_process_snapshots(events) or running_subagent_snapshots(events))

    def _plan_reveal_active(self, session_path: Path) -> bool:
        events = self._session_events(session_path)
        plan_steps = latest_plan_steps(events)
        if not plan_steps:
            return False
        created_at = self._latest_plan_create_timestamp(events)
        if created_at is None:
            return False
        elapsed = max(0.0, (datetime.now(tz=UTC) - created_at).total_seconds())
        visible_count = min(
            len(plan_steps),
            max(1, int(elapsed // PLAN_STEP_REVEAL_SECONDS) + 1),
        )
        return visible_count < len(plan_steps)

    def _start_hints_visible_for_prompt(
        self,
        session_path: Path,
        messages: list[MessageLine],
        input_text: str,
        command_suggestions: Sequence[CommandSpec],
        file_suggestions: Sequence[MenuChoice],
    ) -> bool:
        active_bottom_panel = (
            self._command_bottom_panel(list(command_suggestions), 0)
            if command_suggestions
            else None
        )
        if active_bottom_panel is None and file_suggestions:
            active_bottom_panel = self._file_reference_bottom_panel(list(file_suggestions), 0)
        return self._should_draw_start_hints(
            messages,
            input_text,
            active_bottom_panel,
            None,
            latest_plan_steps(self._session_events(session_path)),
        )

    def _animate_start_hints_exit(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        messages: list[MessageLine],
        scroll: int,
        anchor_line: int | None,
    ) -> None:
        started_at = time.monotonic()
        frame = 0
        while True:
            progress = min(1.0, (time.monotonic() - started_at) / START_HINT_WIPE_SECONDS)
            self._draw_session(
                stdscr,
                session,
                messages,
                "",
                0,
                scroll,
                anchor_line=anchor_line,
                working_frame=frame,
                force_start_hints=True,
                start_hint_reveal_progress=1.0,
                start_hint_removal_progress=progress,
            )
            if progress >= 1.0:
                self._start_hint_reveal_started_at = None
                return
            frame += 1
            time.sleep(STARTUP_FRAME_SECONDS)

    def _maybe_start_session_rename(self, session: SessionRecord) -> None:
        if session.title != "New session" or not self._has_user_messages(session.path):
            return
        if session.session_id in self._title_jobs:
            return

        self._title_jobs.add(session.session_id)

        def suggest_title() -> None:
            title = self.runtime.suggest_session_title(session.path)
            if title is None or title == session.title:
                return
            self.home.update_session_title(session.path, title)
            self._title_events.put((session.session_id, title))

        threading.Thread(target=suggest_title, daemon=True).start()

    def _process_title_events(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
    ) -> SessionRecord:
        current_session = session
        while True:
            try:
                session_id, title = self._title_events.get_nowait()
            except queue.Empty:
                return current_session
            if session_id != current_session.session_id:
                continue
            self._animate_session_title(stdscr, current_session, title)
            current_session = replace(current_session, title=title)

    def _animate_session_title(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        new_title: str,
    ) -> None:
        messages = self._read_message_lines(session.path)
        old_title = session.title
        for index in range(len(old_title), -1, -1):
            self._draw_session(
                stdscr,
                replace(session, title=old_title[:index]),
                messages,
                "",
                0,
                0,
            )
            time.sleep(0.012)
        for index in range(1, len(new_title) + 1):
            self._draw_session(
                stdscr,
                replace(session, title=new_title[:index]),
                messages,
                "",
                0,
                0,
            )
            time.sleep(0.012)

    def _has_user_messages(self, session_path: Path) -> bool:
        for event in self._session_events(session_path):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = (
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            if (
                event_type in {"user_message", "skill_invocation"}
                and str(
                    payload.get("message", ""),
                ).strip()
            ):
                return True
        return False

    def _handle_command(
        self,
        stdscr: CursesWindow,
        command: str,
        current_session: SessionRecord,
        submitted: str = "",
    ) -> str | SessionRecord | None:
        if command == "/exit":
            return "exit"
        if command == "/rename":
            return self._rename_session(stdscr, current_session, submitted)
        if command == "/config":
            self._run_config_panel(stdscr, current_session)
            return None
        if command == "/model":
            self._run_model_panel(stdscr, current_session)
            return None
        if command == "/effort":
            if not self._apply_effort_argument(stdscr, submitted):
                self._run_effort_panel(stdscr, current_session)
            return None
        if command == "/feedback":
            self._send_cli_feedback(stdscr, current_session, submitted)
            return None
        if command == "/new":
            return self._create_session()
        skill = self._skill_for_command(command)
        if skill is not None:
            return self._invoke_skill(stdscr, current_session, skill, submitted or command)

        self._message(stdscr, "Unknown Command", command)
        return current_session

    def _handle_command_with_prompt(
        self,
        stdscr: CursesWindow,
        command: str,
        current_session: SessionRecord,
        submitted: str,
        input_text: str,
        cursor: int,
        pasted_spans: Sequence[PromptPasteSpan],
        file_references: Mapping[str, str] | None = None,
        image_attachments: Mapping[str, dict[str, str]] | None = None,
    ) -> str | SessionRecord | None:
        with self._preserve_modal_prompt(
            input_text,
            cursor,
            pasted_spans,
            file_references,
            image_attachments,
        ):
            return self._handle_command(
                stdscr,
                command,
                current_session,
                submitted,
            )

    def _open_session_panel(
        self,
        stdscr: CursesWindow,
        current_session: SessionRecord,
    ) -> SessionRecord | None:
        self.state = AgentState.OPEN_SESSION
        sessions = self.home.list_sessions(limit=None)
        if not sessions:
            self._message(stdscr, "Open Session", "No stored sessions")
            self.state = AgentState.NEW_SESSION
            return None

        selected = 0
        delete_pending_index: int | None = None
        with suppress(curses.error):
            stdscr.nodelay(False)
        try:
            while True:
                selected = min(selected, len(sessions) - 1)
                choices = self._open_session_choices(
                    sessions,
                    selected,
                    delete_pending_index,
                )
                _, _, _, content_height = self._draw_overlay(
                    stdscr,
                    title="Open Session",
                    subtitle=self._open_session_subtitle(delete_pending_index),
                    choices=choices,
                    selected=selected,
                    footer="Esc Back · ↑↓ Navigate · Enter Open · Ctrl+D Delete",
                )
                key = stdscr.get_wch()
                if self._is_escape(key):
                    if delete_pending_index is not None:
                        delete_pending_index = None
                        continue
                    return None
                if self._is_ctrl_c(key):
                    return None
                if self._is_shift_tab(key):
                    current_session = replace(
                        current_session,
                        mode=self._cycle_agent_mode(current_session),
                        agent_kind=self.active_agent.kind,
                    )
                    continue
                if key == curses.KEY_UP:
                    selected = max(0, selected - 1)
                    delete_pending_index = None
                    continue
                if key == curses.KEY_DOWN:
                    selected = min(len(sessions) - 1, selected + 1)
                    delete_pending_index = None
                    continue
                if key == curses.KEY_PPAGE:
                    page_size = max(1, content_height - 2)
                    selected = max(0, selected - page_size)
                    delete_pending_index = None
                    continue
                if key == curses.KEY_NPAGE:
                    page_size = max(1, content_height - 2)
                    selected = min(len(sessions) - 1, selected + page_size)
                    delete_pending_index = None
                    continue
                if self._is_ctrl_d(key):
                    delete_pending_index = selected
                    continue
                if self._is_enter(key):
                    if delete_pending_index == selected:
                        deleted_session = sessions[selected]
                        deleted_current = (
                            deleted_session.path.expanduser().resolve()
                            == current_session.path.expanduser().resolve()
                        )
                        self.home.delete_session(deleted_session.path)
                        sessions = self.home.list_sessions(limit=None)
                        delete_pending_index = None
                        if not sessions:
                            return self._create_session()
                        selected = min(selected, len(sessions) - 1)
                        if deleted_current:
                            return self._prepare_session_for_opening(sessions[selected])
                        continue
                    return self._prepare_session_for_opening(sessions[selected])
        finally:
            self.state = AgentState.NEW_SESSION

    def _open_session_choices(
        self,
        sessions: list[SessionRecord],
        selected: int,
        delete_pending_index: int | None = None,
    ) -> tuple[MenuChoice, ...]:
        del selected, delete_pending_index
        choices = tuple(
            MenuChoice(
                label=session.title,
                detail=self._open_session_detail(session),
                value=str(index),
            )
            for index, session in enumerate(sessions)
        )
        return choices

    def _open_session_subtitle(self, delete_pending_index: int | None = None) -> str:
        action = "Enter to confirm" if delete_pending_index is not None else "ctrl+d Delete"
        return f"Choose a stored session · {action}"

    def _open_session_detail(
        self,
        session: SessionRecord,
    ) -> str:
        return (
            f"{self._message_count_label(session.message_count)} · "
            f"{session.created_at} · {self._session_location_label(session)}"
        )

    def _session_location_label(self, session: SessionRecord) -> str:
        return session.cwd.strip() or "Unknown location"

    def _rename_session(
        self,
        stdscr: CursesWindow,
        current_session: SessionRecord,
        submitted: str = "",
    ) -> SessionRecord | None:
        parts = submitted.strip().split(maxsplit=1)
        title = parts[1].strip() if len(parts) > 1 else ""
        if not title:
            if hasattr(stdscr, "getmaxyx"):
                prompted_title = self._prompt_popover_text(
                    stdscr,
                    current_session,
                    title="Rename Session",
                    label="Title",
                    default=current_session.title,
                )
            else:
                prompted_title = self._prompt_text(
                    stdscr,
                    title="Rename Session",
                    label="Title",
                    default=current_session.title,
                )
            if prompted_title is None:
                return None
            title = prompted_title.strip()
        if not title or title == current_session.title:
            return None
        self.home.update_session_title(current_session.path, title)
        return replace(current_session, title=title)

    def _message_count_label(self, count: int) -> str:
        noun = "message" if count == 1 else "messages"
        return f"{count} {noun}"

    def _send_cli_feedback(
        self,
        stdscr: CursesWindow,
        current_session: SessionRecord,
        submitted: str = "",
    ) -> None:
        connection = connection_from_home(self.home)
        if connection is None:
            self._message(
                stdscr,
                "Feedback unavailable",
                "Connect this CLI agent to an Anomx Platform in /config first.",
            )
            return

        parts = submitted.strip().split(maxsplit=1)
        comment = parts[1].strip() if len(parts) > 1 else ""
        if not comment:
            if self.state == AgentState.PROJECT:
                prompted_comment = self._prompt_text(
                    stdscr,
                    title="Send Feedback",
                    label="Short comment",
                    optional=False,
                )
            else:
                prompted_comment = self._prompt_popover_text(
                    stdscr,
                    current_session,
                    title="Send Feedback",
                    label="Short comment",
                    optional=False,
                )
            if prompted_comment is None:
                return
            comment = prompted_comment.strip()
        if not comment:
            return

        result = call_anomx_api(
            connection,
            method="POST",
            path="/feedback/entries",
            body={
                "feature_key": "anomx-cli",
                "source": "global",
                "urgency": "low",
                "message": comment,
            },
            output_name="cli-feedback",
        )
        if result.get("ok"):
            self._message(stdscr, "Feedback sent", "Thank you for helping improve Anomx.")
            return
        self._message(
            stdscr,
            "Feedback failed",
            f"The platform returned status {result.get('status_code', 0)}.",
        )

    def _start_session_turn(
        self,
        session: SessionRecord,
        runtime: AgentRuntime | RuntimeProcessClient | None = None,
    ) -> ActiveSessionTurn:
        existing = self._active_turn_for_session(session)
        if existing is not None:
            return existing

        events: queue.SimpleQueue[RuntimeUiEvent] = queue.SimpleQueue()
        result: dict[str, str] = {}
        self._activate_agent(session.agent_kind)
        turn_mode = self.agent_mode
        turn_agent_kind = self.active_agent.kind
        turn_runtime = runtime or self._new_session_runtime(turn_mode)
        if not isinstance(turn_runtime, RuntimeProcessClient):
            turn_runtime.set_agent(turn_agent_kind)
            turn_runtime.set_mode(turn_mode)
        turn_id = uuid4().hex
        started_at = time.monotonic()

        def status_callback(message: str) -> None:
            events.put(RuntimeUiEvent("status", message))

        def message_callback(message: str) -> None:
            events.put(RuntimeUiEvent("message", message))

        def tool_message_callback(message: str) -> None:
            events.put(RuntimeUiEvent("tool_message", message))

        def thought_callback(thought: str) -> None:
            events.put(RuntimeUiEvent("thought", thought))

        def command_callback(statement: str, command: str, _output: str) -> None:
            events.put(RuntimeUiEvent("command", statement, command=command))

        def delta_callback(delta: str) -> None:
            events.put(RuntimeUiEvent("delta", delta))

        def system_message_callback(role: str, message: str) -> None:
            events.put(RuntimeUiEvent("system_message", message, role))

        def approval_callback(request: CommandApprovalRequest) -> ApprovalChoice:
            response: queue.SimpleQueue[ApprovalChoice] = queue.SimpleQueue()
            evaluation = turn_runtime.evaluate_command_request(session.path, request)
            approval_request = (
                replace(request, evaluation=evaluation) if evaluation is not None else request
            )
            if evaluation is not None and turn_mode.policy.auto_approves_risk(evaluation.risk):
                return ApprovalChoice.ALLOW
            events.put(
                RuntimeUiEvent(
                    "approval",
                    approval_request=approval_request,
                    approval_response=response,
                )
            )
            choice = response.get()
            if choice in {ApprovalChoice.ALWAYS_ALLOW, ApprovalChoice.ALWAYS_REJECT}:
                allowance_key = request.allowance_key or f"cmd:{request.canonical_command}"
                if choice == ApprovalChoice.ALWAYS_ALLOW:
                    self.home.add_global_allowed_command(allowance_key)
                else:
                    self.home.add_global_rejected_command(allowance_key)
                    reason = str(getattr(self, "_approval_memory_reason", "") or "").strip()
                    if reason:
                        self._save_approval_memory(turn_runtime, request, allowance_key, reason)
            return choice

        def question_callback(request: QuestionRequest) -> QuestionResponse:
            response: queue.SimpleQueue[QuestionResponse] = queue.SimpleQueue()
            events.put(
                RuntimeUiEvent(
                    "question",
                    question_request=request,
                    question_response=response,
                )
            )
            return response.get()

        def finish_callback(final_text: str) -> None:
            events.put(RuntimeUiEvent("finish", final_text))

        def usage_callback(snapshot: UsageSnapshot) -> None:
            self._session_usage_snapshots[session.path] = snapshot

        def run_backend() -> None:
            try:
                turn_runtime.set_agent(turn_agent_kind)
                turn_runtime.set_mode(turn_mode)
                turn_runtime.init_sandbox(status_callback=status_callback)
                result["response"] = turn_runtime.backend_response(
                    session.path,
                    callbacks=RuntimeCallbacks(
                        status=status_callback,
                        message=message_callback,
                        tool_message=tool_message_callback,
                        thought=thought_callback,
                        command=command_callback,
                        delta=delta_callback,
                        approval=approval_callback,
                        system_message=system_message_callback,
                        question=question_callback,
                        finish=finish_callback,
                        usage=usage_callback,
                    ),
                )
            except Exception as error:  # pragma: no cover - defensive thread boundary
                crash_path: Path | None = None
                with suppress(OSError, TypeError, ValueError):
                    crash_path = self.home.write_crash_log(
                        error,
                        context={
                            "session_path": str(session.path),
                            "workspace_root": str(self.workspace_root),
                        },
                    )
                message = f"Agent crashed: {type(error).__name__}: {error}"
                if crash_path is not None:
                    message = f"{message}\nCrash log: {crash_path}"
                result["response"] = message

        worker = threading.Thread(target=run_backend, daemon=True)
        turn = ActiveSessionTurn(
            session=session,
            runtime=turn_runtime,
            events=events,
            result=result,
            turn_id=turn_id,
            started_at=started_at,
            worker=worker,
            mode=turn_mode,
            agent_symbol=self.active_agent.symbol,
            anchor_expansion_key=self._latest_root_user_expansion_key(session.path),
        )
        self._active_session_turns[self._session_turn_key(session)] = turn
        worker.start()
        return turn

    def _create_runtime(
        self,
        mode: AgentMode | None = None,
    ) -> AgentRuntime | RuntimeProcessClient:
        runtime_mode = self.agent_mode if mode is None else mode
        if self.isolate_runtime:
            return RuntimeProcessClient(
                self.home,
                self.cwd,
                self.session_allowed_commands,
                self.session_rejected_commands,
                runtime_mode,
                agent_kind=self.active_agent.kind,
                workspace_root=self.workspace_root,
            )
        return AgentRuntime(
            self.home,
            self.cwd,
            self.session_allowed_commands,
            self.session_rejected_commands,
            runtime_mode,
            agent_kind=self.active_agent.kind,
            workspace_root=self.workspace_root,
        )

    def _new_session_runtime(
        self,
        mode: AgentMode | None = None,
    ) -> AgentRuntime | RuntimeProcessClient:
        runtime = self._create_runtime(mode)
        if isinstance(runtime, RuntimeProcessClient):
            self._runtime_processes.append(runtime)
        return runtime

    def _foreground_session_runtime(
        self,
        mode: AgentMode | None = None,
    ) -> AgentRuntime | RuntimeProcessClient:
        for turn in self._active_session_turns.values():
            if turn.runtime is self.runtime and turn.worker is not None and turn.worker.is_alive():
                return self._new_session_runtime(mode)
        if mode is not None and not isinstance(self.runtime, RuntimeProcessClient):
            self.runtime.set_agent(self.active_agent.kind)
            self.runtime.set_mode(mode)
        return self.runtime

    def _active_turn_for_session(self, session: SessionRecord) -> ActiveSessionTurn | None:
        turn = self._active_session_turns.get(self._session_turn_key(session))
        if turn is None or turn.completed:
            return None
        return turn

    def _runtime_for_session(
        self,
        session: SessionRecord,
    ) -> AgentRuntime | RuntimeProcessClient:
        turn = self._active_turn_for_session(session)
        return turn.runtime if turn is not None else self.runtime

    def _session_turn_key(self, session: SessionRecord) -> Path:
        return session.path.expanduser().resolve()

    def _drain_active_session_turns(self, stdscr: CursesWindow) -> None:
        for turn in list(self._active_session_turns.values()):
            self._drain_session_turn_events(
                stdscr,
                turn,
                render_events=False,
            )
            if turn.worker is not None and not turn.worker.is_alive():
                response = self._complete_session_turn(
                    stdscr,
                    turn,
                    render_final=False,
                )
                if response:
                    self.home.set_session_unread(turn.session.path, True)

    def _drain_session_turn_events(
        self,
        stdscr: CursesWindow,
        turn: ActiveSessionTurn,
        *,
        anchor_line: int | None = None,
        input_text: str = "",
        cursor: int = 0,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
        prompt_notice: str = "",
        prompt_notice_role: str = "light",
        scroll: int = 0,
        active_turn_elapsed: float | None = None,
        render_events: bool = True,
        sticky_anchor: bool = False,
    ) -> None:
        (
            turn.working_text,
            turn.working_deadline,
            turn.final_text,
            turn.work_count,
        ) = self._process_runtime_events(
            stdscr,
            turn.session,
            turn.events,
            turn.working_text,
            turn.working_deadline,
            turn.final_text,
            turn.turn_id,
            turn.work_count,
            anchor_line,
            input_text,
            cursor,
            pasted_spans,
            prompt_notice,
            prompt_notice_role,
            scroll,
            active_turn_elapsed=active_turn_elapsed,
            render_events=render_events,
            sticky_anchor=sticky_anchor,
            deferred_events=turn.pending_events,
            defer_interactive=not render_events,
        )

    def _complete_session_turn(
        self,
        stdscr: CursesWindow,
        turn: ActiveSessionTurn,
        *,
        anchor_line: int | None = None,
        scroll: int = 0,
        input_text: str = "",
        cursor: int = 0,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
        render_final: bool = True,
    ) -> str:
        if turn.completed:
            return ""
        turn.worker.join(timeout=0)
        self._drain_session_turn_events(
            stdscr,
            turn,
            anchor_line=anchor_line,
            input_text=input_text,
            cursor=cursor,
            pasted_spans=pasted_spans,
            scroll=scroll,
            active_turn_elapsed=time.monotonic() - turn.started_at,
            render_events=render_final,
            sticky_anchor=anchor_line is not None,
        )
        response = turn.final_text or turn.result.get("response", "")
        if not turn.work_summary_appended:
            turn.work_summary_appended = self._append_work_summary(
                turn.session,
                turn.turn_id,
                turn.started_at,
                turn.work_count,
            )
        if response:
            if render_final and (not turn.final_text or not turn.streamed_final_displayed):
                self._fake_type_message(
                    stdscr,
                    turn.session,
                    response,
                    anchor_line=anchor_line,
                    scroll=scroll,
                    input_text=input_text,
                    cursor=cursor,
                    pasted_spans=pasted_spans,
                    active_turn_elapsed=time.monotonic() - turn.started_at,
                )
            self.home.append_session_event(
                turn.session.path,
                "agent_message",
                {"message": response},
            )
        self._append_turn_outcome_notice(turn.session, response, turn.work_count)
        turn.completed = True
        self._active_session_turns.pop(self._session_turn_key(turn.session), None)
        return response

    def _interrupt_session_turn(
        self,
        stdscr: CursesWindow,
        turn: ActiveSessionTurn,
        *,
        anchor_line: int | None = None,
        scroll: int = 0,
        input_text: str = "",
        cursor: int = 0,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> None:
        with suppress(Exception):
            turn.runtime.abort_current_turn(turn.session.path)
        with suppress(Exception):
            turn.runtime.shutdown(turn.session.path)
        self._drain_session_turn_events(
            stdscr,
            turn,
            anchor_line=anchor_line,
            input_text=input_text,
            cursor=cursor,
            pasted_spans=pasted_spans,
            scroll=scroll,
            active_turn_elapsed=time.monotonic() - turn.started_at,
            render_events=False,
            sticky_anchor=anchor_line is not None,
        )
        if not turn.work_summary_appended:
            turn.work_summary_appended = self._append_interrupted_summary(
                turn.session,
                turn.turn_id,
                turn.started_at,
            )
        self.home.append_session_event(
            turn.session.path,
            "agent_message",
            {"message": MANUAL_INTERRUPT_MESSAGE},
        )
        turn.completed = True
        self._active_session_turns.pop(self._session_turn_key(turn.session), None)

    def _interrupt_and_requeue_session_turn(
        self,
        stdscr: CursesWindow,
        turn: ActiveSessionTurn,
        session: SessionRecord,
        new_message: str,
        *,
        anchor_line: int | None = None,
        scroll: int = 0,
    ) -> None:
        """Abort the current turn and immediately start a new one with an
        appended user message. Leaves running processes/subagents alive."""

        turn.runtime.abort_current_turn(session.path)
        turn.worker.join(timeout=5)
        self._drain_session_turn_events(
            stdscr,
            turn,
            anchor_line=anchor_line,
            input_text="",
            cursor=0,
            scroll=scroll,
            active_turn_elapsed=time.monotonic() - turn.started_at,
            render_events=False,
            sticky_anchor=anchor_line is not None,
        )
        turn.completed = True
        self._active_session_turns.pop(self._session_turn_key(turn.session), None)
        self.home.append_session_event(
            session.path,
            "user_message",
            {
                "message": new_message,
                "turn_id": turn.turn_id,
                "intermediate": True,
            },
        )
        replacement_turn = self._start_session_turn(session)
        if replacement_turn is not None:
            replacement_turn.turn_id = turn.turn_id

    def _shutdown_active_session_turns(self) -> None:
        for turn in list(self._active_session_turns.values()):
            turn.runtime.shutdown(turn.session.path)
            turn.completed = True
        self._active_session_turns.clear()
        for runtime in self._runtime_processes:
            if runtime is not self.runtime:
                runtime.close()
        self._runtime_processes = [
            runtime for runtime in self._runtime_processes if runtime is self.runtime
        ]

    def _shutdown_sandbox_containers(self) -> None:
        try:
            from anomx.agent.helpers.sandbox import SandboxSession

            config = self.home.load_config()
            runtime_bin = str(config.get("sandbox_system", "docker"))
            strategy = str(config.get("sandbox_strategy", "stop"))
            if strategy == "remove":
                SandboxSession.remove_all(runtime=runtime_bin)
            else:
                SandboxSession.stop_all(runtime=runtime_bin)
        except Exception:
            pass

    def _run_backend_turn(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        anchor_line: int | None = None,
    ) -> BackendTurnResult:
        self._activate_agent(session.agent_kind)
        turn_mode = self.agent_mode
        turn = self._start_session_turn(
            replace(session, mode=turn_mode, agent_kind=self.active_agent.kind),
            runtime=self._foreground_session_runtime(turn_mode),
        )
        input_text = ""
        cursor = 0
        pasted_spans: list[PromptPasteSpan] = []
        prompt_notice = RUNNING_NOTICE
        prompt_notice_role = "light"
        abort_key = ""
        abort_deadline = 0.0
        frame = 0
        running_anchor = anchor_line
        running_scroll = 0
        command_selected = 0
        with suppress(curses.error):
            stdscr.nodelay(True)
        try:
            while turn.worker is not None and turn.worker.is_alive():
                running_anchor = self._active_turn_anchor_line(
                    stdscr,
                    session,
                    turn,
                    running_anchor,
                )
                command_token = self._active_command_token(input_text, cursor)
                command_suggestions = (
                    self._filtered_running_commands(command_token[2])
                    if command_token is not None
                    else []
                )
                if command_suggestions:
                    command_selected = min(command_selected, len(command_suggestions) - 1)
                else:
                    command_selected = 0
                key = self._read_nonblocking_key(stdscr)
                if key is not None:
                    key_result = self._handle_running_key(
                        stdscr,
                        session,
                        key,
                        input_text,
                        cursor,
                        abort_key,
                        abort_deadline,
                        command_suggestions,
                        command_selected,
                        pasted_spans,
                    )
                    input_text = key_result.input_text
                    cursor = key_result.cursor
                    prompt_notice = key_result.notice
                    prompt_notice_role = key_result.notice_role
                    abort_key = key_result.abort_key
                    abort_deadline = key_result.abort_deadline
                    command_selected = key_result.command_selected
                    if key_result.clear_anchor and not key_result.scroll_delta:
                        running_anchor = None
                    if key_result.scroll_delta:
                        running_scroll += key_result.scroll_delta
                    if key_result.command:
                        self._drain_session_turn_events(
                            stdscr,
                            turn,
                            anchor_line=running_anchor,
                            input_text=input_text,
                            cursor=cursor,
                            pasted_spans=pasted_spans,
                            prompt_notice=prompt_notice,
                            prompt_notice_role=prompt_notice_role,
                            scroll=running_scroll,
                            active_turn_elapsed=time.monotonic() - turn.started_at,
                            sticky_anchor=running_anchor is not None,
                        )
                        with suppress(curses.error):
                            stdscr.nodelay(False)
                        try:
                            command_result = self._handle_command_with_prompt(
                                stdscr,
                                key_result.command,
                                session,
                                key_result.submitted,
                                input_text,
                                cursor,
                                pasted_spans,
                            )
                        finally:
                            with suppress(curses.error):
                                stdscr.nodelay(True)
                        prompt_notice = RUNNING_NOTICE
                        prompt_notice_role = "light"
                        if command_result == "exit":
                            self._interrupt_session_turn(
                                stdscr,
                                turn,
                                anchor_line=running_anchor,
                                input_text=input_text,
                                cursor=cursor,
                                pasted_spans=pasted_spans,
                                scroll=running_scroll,
                            )
                            return BackendTurnResult(
                                input_text,
                                cursor,
                                exit_requested=True,
                                anchor_line=running_anchor,
                                scroll=running_scroll,
                            )
                    if key_result.exit_requested:
                        self._interrupt_session_turn(
                            stdscr,
                            turn,
                            anchor_line=running_anchor,
                            input_text=input_text,
                            cursor=cursor,
                            pasted_spans=pasted_spans,
                            scroll=running_scroll,
                        )
                        return BackendTurnResult(
                            input_text,
                            cursor,
                            anchor_line=running_anchor,
                            scroll=running_scroll,
                        )
                    if key_result.back_requested:
                        self._drain_session_turn_events(
                            stdscr,
                            turn,
                            anchor_line=running_anchor,
                            input_text=input_text,
                            cursor=cursor,
                            pasted_spans=pasted_spans,
                            prompt_notice=prompt_notice,
                            prompt_notice_role=prompt_notice_role,
                            scroll=running_scroll,
                            active_turn_elapsed=time.monotonic() - turn.started_at,
                            sticky_anchor=running_anchor is not None,
                        )
                        return BackendTurnResult(
                            input_text,
                            cursor,
                            anchor_line=running_anchor,
                            scroll=running_scroll,
                            back_requested=True,
                        )
                    if key_result.submitted_message:
                        self._interrupt_and_requeue_session_turn(
                            stdscr,
                            turn,
                            session,
                            key_result.submitted_message,
                            anchor_line=running_anchor,
                            scroll=running_scroll,
                        )
                        replacement_turn = self._active_turn_for_session(session)
                        if replacement_turn is None:
                            return BackendTurnResult(
                                "",
                                0,
                                anchor_line=running_anchor,
                                scroll=running_scroll,
                            )
                        turn = replacement_turn
                        input_text = ""
                        cursor = 0
                        pasted_spans.clear()
                        prompt_notice = RUNNING_NOTICE
                        prompt_notice_role = "light"
                        abort_key = ""
                        abort_deadline = 0.0
                        command_selected = 0
                        continue
                elif abort_key and time.monotonic() > abort_deadline:
                    abort_key = ""
                    prompt_notice = RUNNING_NOTICE
                    prompt_notice_role = "light"

                self._drain_session_turn_events(
                    stdscr,
                    turn,
                    anchor_line=running_anchor,
                    input_text=input_text,
                    cursor=cursor,
                    pasted_spans=pasted_spans,
                    prompt_notice=prompt_notice,
                    prompt_notice_role=prompt_notice_role,
                    scroll=running_scroll,
                    active_turn_elapsed=time.monotonic() - turn.started_at,
                    sticky_anchor=running_anchor is not None,
                )
                viewport = self._draw_session(
                    stdscr,
                    session,
                    self._read_message_lines(session.path),
                    input_text,
                    cursor,
                    running_scroll,
                    working_text=turn.working_text,
                    working_deadline=turn.working_deadline,
                    working_frame=frame,
                    anchor_line=running_anchor,
                    sticky_anchor=running_anchor is not None,
                    command_suggestions=(
                        self._filtered_running_commands(input_text)
                        if input_text.startswith("/")
                        else []
                    ),
                    command_selected=command_selected,
                    prompt_notice=prompt_notice,
                    prompt_notice_role=prompt_notice_role,
                    active_turn_elapsed=time.monotonic() - turn.started_at,
                    pasted_spans=pasted_spans,
                )
                if viewport is not None:
                    running_scroll = viewport.scroll
                time.sleep(ACTIVITY_FRAME_SECONDS)
                frame += 1
        finally:
            with suppress(curses.error):
                stdscr.nodelay(False)
        self._complete_session_turn(
            stdscr,
            turn,
            anchor_line=running_anchor,
            scroll=running_scroll,
            input_text=input_text,
            cursor=cursor,
            pasted_spans=pasted_spans,
            render_final=True,
        )
        viewport = self._draw_session(
            stdscr,
            session,
            self._read_message_lines(session.path),
            input_text,
            cursor,
            running_scroll,
            anchor_line=running_anchor,
            pasted_spans=pasted_spans,
        )
        if viewport is not None:
            running_scroll = viewport.scroll
        return BackendTurnResult(
            input_text,
            cursor,
            anchor_line=running_anchor,
            scroll=running_scroll,
        )

    def _append_work_summary(
        self,
        session: SessionRecord,
        turn_id: str,
        started_at: float,
        work_count: int,
    ) -> bool:
        return self._append_turn_summary(session, turn_id, started_at, "Worked for")

    def _append_interrupted_summary(
        self,
        session: SessionRecord,
        turn_id: str,
        started_at: float,
    ) -> bool:
        return self._append_turn_summary(session, turn_id, started_at, "Interrupted after")

    def _append_turn_summary(
        self,
        session: SessionRecord,
        turn_id: str,
        started_at: float,
        prefix: str,
    ) -> bool:
        duration = self._format_duration(time.monotonic() - started_at)
        self.home.append_session_event(
            session.path,
            "work_summary",
            {
                "turn_id": turn_id,
                "message": f"{prefix} {duration}",
            },
        )
        return True

    def _read_nonblocking_key(self, stdscr: CursesWindow) -> str | int | PromptPasteEvent | None:
        try:
            key = stdscr.get_wch()
        except curses.error:
            return None
        completed = self._complete_escape_key(stdscr, key, restore_blocking=False)
        if completed == "\x1b[200~":
            return self._read_bracketed_paste(stdscr, restore_blocking=False)
        return completed

    def _read_prompt_key(self, stdscr: CursesWindow) -> str | int | PromptPasteEvent:
        key = stdscr.get_wch()
        completed = self._complete_escape_key(stdscr, key, restore_blocking=True)
        if completed == "\x1b[200~":
            return self._read_bracketed_paste(stdscr, restore_blocking=True)
        return completed

    def _read_bracketed_paste(
        self,
        stdscr: CursesWindow,
        restore_blocking: bool,
    ) -> PromptPasteEvent:
        if not hasattr(stdscr, "nodelay"):
            return PromptPasteEvent("")
        parts: list[str] = []
        with suppress(curses.error):
            stdscr.nodelay(True)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                next_key = stdscr.get_wch()
            except curses.error:
                time.sleep(0.001)
                continue
            completed = self._complete_escape_key(stdscr, next_key, restore_blocking=False)
            if completed == "\x1b[201~":
                break
            if isinstance(completed, str):
                parts.append(completed)
                deadline = time.monotonic() + 0.1
        if restore_blocking:
            with suppress(curses.error):
                stdscr.nodelay(False)
        return PromptPasteEvent("".join(parts))

    def _insert_prompt_text(
        self,
        input_text: str,
        cursor: int,
        text: str,
        pasted_spans: list[PromptPasteSpan],
        *,
        pasted: bool = False,
    ) -> tuple[str, int]:
        collapse_paste = pasted and len(text) > PROMPT_PASTE_COLLAPSE_THRESHOLD
        replacement = f"\n\n{text}\n\n" if collapse_paste else text
        return self._replace_prompt_range(
            input_text,
            cursor,
            cursor,
            replacement,
            pasted_spans,
            pasted=collapse_paste,
            pasted_character_count=len(text) if collapse_paste else None,
        )

    def _replace_prompt_range(
        self,
        input_text: str,
        start: int,
        end: int,
        replacement: str,
        pasted_spans: list[PromptPasteSpan],
        *,
        pasted: bool = False,
        pasted_character_count: int | None = None,
    ) -> tuple[str, int]:
        bounded_start = max(0, min(start, len(input_text)))
        bounded_end = max(bounded_start, min(end, len(input_text)))
        updated = input_text[:bounded_start] + replacement + input_text[bounded_end:]
        pasted_spans[:] = self._prompt_spans_after_replacement(
            pasted_spans,
            bounded_start,
            bounded_end,
            len(replacement),
            pasted=pasted,
            pasted_character_count=pasted_character_count,
        )
        return updated, bounded_start + len(replacement)

    def _delete_prompt_backward(
        self,
        input_text: str,
        cursor: int,
        pasted_spans: list[PromptPasteSpan],
        file_references: dict[str, str] | None = None,
        image_attachments: dict[str, dict[str, str]] | None = None,
    ) -> tuple[str, int]:
        """Delete one character, or one complete highlighted prompt token."""

        if cursor <= 0:
            return input_text, cursor
        labels = self._prompt_reference_labels(file_references, image_attachments)
        start, end = self._prompt_atomic_backspace_range(
            input_text,
            cursor,
            pasted_spans,
            labels,
        )
        updated, updated_cursor = self._replace_prompt_range(
            input_text,
            start,
            end,
            "",
            pasted_spans,
        )
        for references in (file_references, image_attachments):
            if references is None:
                continue
            for label in tuple(references):
                if not self._file_reference_label_pattern(label).search(updated):
                    references.pop(label, None)
        return updated, updated_cursor

    def _prompt_atomic_backspace_range(
        self,
        input_text: str,
        cursor: int,
        pasted_spans: Sequence[PromptPasteSpan],
        reference_labels: Mapping[str, str] | None = None,
    ) -> tuple[int, int]:
        bounded_cursor = max(0, min(cursor, len(input_text)))
        for span in self._normalized_prompt_paste_spans(input_text, pasted_spans):
            if span.end == bounded_cursor:
                return span.start, span.end

        for label in sorted(reference_labels or {}, key=len, reverse=True):
            for match in self._file_reference_label_pattern(label).finditer(input_text):
                if match.end() == bounded_cursor:
                    end = match.end()
                    if end < len(input_text) and input_text[end].isspace():
                        end += 1
                    return match.start(), end
        return max(0, bounded_cursor - 1), bounded_cursor

    def _prompt_spans_after_replacement(
        self,
        spans: Sequence[PromptPasteSpan],
        start: int,
        end: int,
        replacement_length: int,
        *,
        pasted: bool = False,
        pasted_character_count: int | None = None,
    ) -> list[PromptPasteSpan]:
        delta = replacement_length - (end - start)
        updated: list[PromptPasteSpan] = []
        for span in spans:
            if span.end <= start:
                updated.append(span)
            elif span.start >= end:
                updated.append(
                    PromptPasteSpan(
                        span.start + delta,
                        span.end + delta,
                        span.character_count,
                    ),
                )
            else:
                if span.start < start:
                    updated.append(PromptPasteSpan(span.start, start))
                if span.end > end:
                    updated.append(
                        PromptPasteSpan(
                            start + replacement_length,
                            span.end + delta,
                        ),
                    )
        if pasted and replacement_length:
            updated.append(
                PromptPasteSpan(
                    start,
                    start + replacement_length,
                    pasted_character_count or replacement_length,
                ),
            )
        return self._merge_prompt_paste_spans(updated)

    def _merge_prompt_paste_spans(
        self,
        spans: Sequence[PromptPasteSpan],
    ) -> list[PromptPasteSpan]:
        merged: list[PromptPasteSpan] = []
        for span in sorted(spans, key=lambda item: (item.start, item.end)):
            if span.start >= span.end:
                continue
            if merged and span.start <= merged[-1].end:
                previous = merged[-1]
                character_count = None
                if previous.character_count is not None or span.character_count is not None:
                    character_count = (
                        previous.character_count or previous.end - previous.start
                    ) + (span.character_count or span.end - span.start)
                merged[-1] = PromptPasteSpan(
                    previous.start,
                    max(previous.end, span.end),
                    character_count,
                )
            else:
                merged.append(span)
        return merged

    def _complete_escape_key(
        self,
        stdscr: CursesWindow,
        key: str | int,
        restore_blocking: bool,
    ) -> str | int:
        if not self._is_escape(key):
            return key
        suffix = self._read_escape_suffix(stdscr, restore_blocking)
        return f"\x1b{suffix}" if suffix else key

    def _read_escape_suffix(self, stdscr: CursesWindow, restore_blocking: bool) -> str:
        if not hasattr(stdscr, "nodelay"):
            return ""
        suffix = ""
        with suppress(curses.error):
            stdscr.nodelay(True)
        deadline = time.monotonic() + 0.04
        while time.monotonic() < deadline and len(suffix) < 24:
            try:
                next_key = stdscr.get_wch()
            except curses.error:
                time.sleep(0.001)
                continue
            if isinstance(next_key, str):
                suffix += next_key
                if self._is_complete_escape_suffix(suffix):
                    break
            else:
                break
        if restore_blocking:
            with suppress(curses.error):
                stdscr.nodelay(False)
        return suffix

    def _is_complete_escape_suffix(self, suffix: str) -> bool:
        if RAW_MOUSE_SUFFIX_RE.match(suffix):
            return True
        if suffix in {"b", "B", "f", "F"}:
            return True
        return suffix.endswith(("C", "D", "~", "u", "Z"))

    def _is_raw_mouse_fragment_key(self, key: str | int) -> bool:
        if not isinstance(key, str):
            return False
        return bool(
            RAW_MOUSE_RE.match(key)
            or RAW_MOUSE_SUFFIX_RE.match(key)
            or re.match(r"^<?\d+;\d+;\d+[mM]$", key)
        )

    def _handle_running_key(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        key: str | int | PromptPasteEvent,
        input_text: str,
        cursor: int,
        abort_key: str,
        abort_deadline: float,
        command_suggestions: list[CommandSpec] | None = None,
        command_selected: int = 0,
        pasted_spans: list[PromptPasteSpan] | None = None,
    ) -> RunningKeyResult:
        now = time.monotonic()
        suggestions = command_suggestions or []
        active_pasted_spans = pasted_spans if pasted_spans is not None else []
        if self._is_shift_tab(key):
            new_mode = self._cycle_agent_mode(session)
            turn = self._active_turn_for_session(session)
            if turn is not None:
                turn.mode = new_mode
                turn.agent_symbol = self.active_agent.symbol
                turn.runtime.set_agent(self.active_agent.kind)
                turn.runtime.set_mode(new_mode)
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )

        if self._is_option_left(key):
            return RunningKeyResult(
                input_text,
                self._previous_prompt_word(input_text, cursor),
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )
        if self._is_option_right(key):
            return RunningKeyResult(
                input_text,
                self._next_prompt_word(input_text, cursor),
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )

        if self._is_escape(key):
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
                back_requested=True,
            )

        if self._is_ctrl_c(key) and input_text:
            active_pasted_spans.clear()
            return RunningKeyResult(
                "",
                0,
                RUNNING_NOTICE,
                "light",
                "",
                0.0,
                0,
            )

        if self._is_ctrl_c(key) or self._is_ctrl_x(key):
            key_label = self._running_interrupt_key_label(key)
            if abort_key == key_label and now <= abort_deadline:
                return RunningKeyResult(
                    input_text,
                    cursor,
                    "Aborting agent",
                    "light",
                    key_label,
                    now,
                    command_selected,
                    exit_requested=True,
                )

            notice = ABORT_AGENT_CONFIRM_NOTICE.format(key_label=key_label)
            return RunningKeyResult(
                input_text,
                cursor,
                notice,
                "light",
                key_label,
                now + 3.0,
                command_selected,
            )

        if self._is_enter(key):
            command_token = self._active_command_token(input_text, cursor)
            if (
                command_token is not None
                and suggestions
                and self._command_token_is_inline(input_text, command_token)
            ):
                suggestion = suggestions[command_selected]
                updated_input, updated_cursor = self._remove_command_token(
                    input_text,
                    command_token,
                    active_pasted_spans,
                )
                return RunningKeyResult(
                    updated_input,
                    updated_cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    0,
                    suggestion.command,
                    suggestion.command,
                )
            submitted = input_text.strip()
            if not submitted:
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if submitted.startswith("/"):
                command = self._submitted_running_command(
                    submitted,
                    suggestions,
                    command_selected,
                )
                if self._is_running_session_command(command):
                    active_pasted_spans.clear()
                    return RunningKeyResult(
                        "",
                        0,
                        RUNNING_NOTICE,
                        "light",
                        abort_key,
                        abort_deadline,
                        0,
                        command,
                        submitted,
                    )
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_COMMAND_BLOCKED_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
                submitted_message=submitted,
            )

        if key == curses.KEY_UP:
            if suggestions:
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    max(0, command_selected - 1),
                )
            moved_cursor = self._move_prompt_cursor_row(
                stdscr,
                input_text,
                cursor,
                direction=-1,
                pasted_spans=active_pasted_spans,
            )
            if moved_cursor != cursor:
                return RunningKeyResult(
                    input_text,
                    moved_cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
                scroll_delta=1,
                clear_anchor=True,
            )
        if key == curses.KEY_DOWN:
            if suggestions:
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    min(len(suggestions) - 1, command_selected + 1),
                )
            moved_cursor = self._move_prompt_cursor_row(
                stdscr,
                input_text,
                cursor,
                direction=1,
                pasted_spans=active_pasted_spans,
            )
            if moved_cursor != cursor:
                return RunningKeyResult(
                    input_text,
                    moved_cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
                scroll_delta=-1,
                clear_anchor=True,
            )
        if key == curses.KEY_PPAGE:
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
                scroll_delta=5,
                clear_anchor=True,
            )
        if key == curses.KEY_NPAGE:
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
                scroll_delta=-5,
                clear_anchor=True,
            )

        if key == curses.KEY_LEFT:
            return RunningKeyResult(
                input_text,
                max(0, cursor - 1),
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )
        if key == curses.KEY_RIGHT:
            return RunningKeyResult(
                input_text,
                min(len(input_text), cursor + 1),
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )
        if key == curses.KEY_HOME:
            return RunningKeyResult(
                input_text,
                0,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )
        if key == curses.KEY_END:
            return RunningKeyResult(
                input_text,
                len(input_text),
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )
        if key == curses.KEY_MOUSE:
            action = self._session_mouse_action(
                stdscr,
                input_text,
                suggestions,
                pasted_spans=active_pasted_spans,
            )
            if action is not None and action.kind == "back_project":
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                    back_requested=True,
                )
            if action is not None and action.kind == "cursor":
                return RunningKeyResult(
                    input_text,
                    action.value,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "scroll":
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                    scroll_delta=action.value,
                    clear_anchor=True,
                )
            if action is not None and action.kind == "toggle_work":
                self._toggle_work_turn(action.text)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "toggle_pinned_user":
                self._toggle_pinned_user(action.text)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "toggle_plan":
                self._toggle_plan(session.path)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "toggle_work_line":
                self._toggle_work_line(action.text)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "toggle_activity_item":
                self._toggle_activity_item(action.text)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "open_subagent":
                self._open_subagent_session(stdscr, session, action.text)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "toggle_activity_entry":
                self._toggle_activity_entry(action.text)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "scroll_activity_item":
                self._scroll_activity_item(action.text, action.value)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "kill_process":
                self._runtime_for_session(session).end_process(action.text, session.path)
                return RunningKeyResult(
                    input_text,
                    cursor,
                    "Process ended",
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if action is not None and action.kind == "command":
                suggestion = suggestions[action.value]
                command_token = self._active_command_token(input_text, cursor)
                if command_token is not None and self._command_token_is_inline(
                    input_text,
                    command_token,
                ):
                    updated_input, updated_cursor = self._remove_command_token(
                        input_text,
                        command_token,
                        active_pasted_spans,
                    )
                    return RunningKeyResult(
                        updated_input,
                        updated_cursor,
                        RUNNING_NOTICE,
                        "light",
                        abort_key,
                        abort_deadline,
                        0,
                        suggestion.command,
                        suggestion.command,
                    )
                command = suggestion.command
                active_pasted_spans.clear()
                return RunningKeyResult(
                    "",
                    0,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    0,
                    command,
                    command,
                )
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )
        raw_mouse_action = self._raw_mouse_action(
            key,
            stdscr,
            input_text,
            suggestions,
            command_selected,
            pasted_spans=active_pasted_spans,
        )
        if raw_mouse_action is not None:
            if raw_mouse_action.kind == "back_project":
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                    back_requested=True,
                )
            if raw_mouse_action.kind == "cursor":
                return RunningKeyResult(
                    input_text,
                    raw_mouse_action.value,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            if raw_mouse_action.kind == "scroll":
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                    scroll_delta=raw_mouse_action.value,
                    clear_anchor=True,
                )
            if raw_mouse_action.kind == "toggle_work":
                self._toggle_work_turn(raw_mouse_action.text)
            elif raw_mouse_action.kind == "toggle_pinned_user":
                self._toggle_pinned_user(raw_mouse_action.text)
            elif raw_mouse_action.kind == "toggle_plan":
                self._toggle_plan(session.path)
            elif raw_mouse_action.kind == "toggle_work_line":
                self._toggle_work_line(raw_mouse_action.text)
            elif raw_mouse_action.kind == "toggle_activity_item":
                self._toggle_activity_item(raw_mouse_action.text)
            elif raw_mouse_action.kind == "open_subagent":
                self._open_subagent_session(stdscr, session, raw_mouse_action.text)
            elif raw_mouse_action.kind == "toggle_activity_entry":
                self._toggle_activity_entry(raw_mouse_action.text)
            elif raw_mouse_action.kind == "scroll_activity_item":
                self._scroll_activity_item(raw_mouse_action.text, raw_mouse_action.value)
            elif raw_mouse_action.kind == "kill_process":
                self._runtime_for_session(session).end_process(
                    raw_mouse_action.text,
                    session.path,
                )
                return RunningKeyResult(
                    input_text,
                    cursor,
                    "Process ended",
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            elif raw_mouse_action.kind == "command":
                suggestion = suggestions[raw_mouse_action.value]
                command_token = self._active_command_token(input_text, cursor)
                if command_token is not None and self._command_token_is_inline(
                    input_text,
                    command_token,
                ):
                    updated_input, updated_cursor = self._remove_command_token(
                        input_text,
                        command_token,
                        active_pasted_spans,
                    )
                    return RunningKeyResult(
                        updated_input,
                        updated_cursor,
                        RUNNING_NOTICE,
                        "light",
                        abort_key,
                        abort_deadline,
                        0,
                        suggestion.command,
                        suggestion.command,
                    )
                command = suggestion.command
                active_pasted_spans.clear()
                return RunningKeyResult(
                    "",
                    0,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    0,
                    command,
                    command,
                )
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )
        if self._is_raw_mouse_fragment_key(key):
            return RunningKeyResult(
                input_text,
                cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                command_selected,
            )
        if isinstance(key, PromptPasteEvent):
            if not key.text:
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            updated, updated_cursor = self._insert_prompt_text(
                input_text,
                cursor,
                key.text,
                active_pasted_spans,
                pasted=True,
            )
            return RunningKeyResult(
                updated,
                updated_cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                0,
            )
        if self._is_option_delete(key):
            word_start = self._previous_prompt_word(input_text, cursor)
            updated_input, updated_cursor = self._replace_prompt_range(
                input_text,
                word_start,
                cursor,
                "",
                active_pasted_spans,
            )
            return RunningKeyResult(
                updated_input,
                updated_cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                0,
            )
        if self._is_backspace(key):
            if cursor <= 0:
                return RunningKeyResult(
                    input_text,
                    cursor,
                    RUNNING_NOTICE,
                    "light",
                    abort_key,
                    abort_deadline,
                    command_selected,
                )
            updated, updated_cursor = self._delete_prompt_backward(
                input_text,
                cursor,
                active_pasted_spans,
            )
            return RunningKeyResult(
                updated,
                updated_cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                0,
            )
        if isinstance(key, str) and key.isprintable():
            updated, updated_cursor = self._insert_prompt_text(
                input_text,
                cursor,
                key,
                active_pasted_spans,
            )
            return RunningKeyResult(
                updated,
                updated_cursor,
                RUNNING_NOTICE,
                "light",
                abort_key,
                abort_deadline,
                0,
            )
        return RunningKeyResult(
            input_text,
            cursor,
            RUNNING_NOTICE,
            "light",
            abort_key,
            abort_deadline,
            command_selected,
        )

    def _append_system_message(
        self,
        session: SessionRecord,
        role: str,
        message: str,
    ) -> None:
        self.home.append_session_event(
            session.path,
            "system_message",
            {
                "message": message,
                "role": role,
            },
        )

    def _append_turn_outcome_notice(
        self,
        session: SessionRecord,
        response: str,
        work_count: int,
    ) -> None:
        """Show a visible marker when a turn ended abnormally.

        Normal completions already render their answer plus the work summary.
        Crashes, step-limit stops, and empty runs would otherwise be
        indistinguishable from a normal answer (or show nothing at all).
        """

        text = response.strip()
        if text.startswith("Agent crashed:"):
            self._append_system_message(
                session,
                "system",
                "The agent crashed and the run was stopped.",
            )
        elif "step limit for a single turn" in text or "tool loop stopped after" in text:
            self._append_system_message(
                session,
                "system",
                "Run stopped at the step limit — send a follow-up message to continue.",
            )
        elif not text and not work_count:
            self._append_system_message(
                session,
                "system",
                "anomx ended the run without a response.",
            )

    def _backend_response(
        self,
        session_path: Path,
        status_callback: StatusCallback | None = None,
    ) -> str:
        callbacks = RuntimeCallbacks(status=status_callback)
        return self.runtime.backend_response(session_path, callbacks=callbacks)

    def _openai_response(self, session_path: Path, model: str) -> str:
        return self.runtime.openai_response(session_path, model)

    def _conversation_messages(self, session_path: Path) -> list[dict[str, str]]:
        return self.runtime.conversation_messages(session_path)

    def _extract_openai_text(self, data: dict[str, object]) -> str:
        return self.runtime.extract_openai_text(data)

    def _process_runtime_events(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        events: queue.SimpleQueue[RuntimeUiEvent],
        working_text: str | None,
        working_deadline: float | None,
        final_text: str,
        turn_id: str,
        work_count: int,
        anchor_line: int | None,
        input_text: str = "",
        cursor: int = 0,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
        prompt_notice: str = "",
        prompt_notice_role: str = "light",
        scroll: int = 0,
        *,
        active_turn_elapsed: float | None = None,
        render_events: bool = True,
        sticky_anchor: bool = False,
        deferred_events: list[RuntimeUiEvent] | None = None,
        defer_interactive: bool = False,
    ) -> tuple[str | None, float | None, str, int]:
        current_working = working_text
        current_deadline = working_deadline
        current_final = final_text
        current_work_count = work_count
        while True:
            if deferred_events:
                event = deferred_events.pop(0)
            else:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    return current_working, current_deadline, current_final, current_work_count

            if defer_interactive and event.kind in {"approval", "question"}:
                if deferred_events is not None:
                    deferred_events.insert(0, event)
                return current_working, current_deadline, current_final, current_work_count

            if event.kind == "status":
                status_text, status_seconds = self._parse_runtime_status(event.text)
                if status_text == "Waiting":
                    current_working = "Waiting"
                    current_deadline = (
                        time.monotonic() + status_seconds if status_seconds is not None else None
                    )
                elif status_text in {
                    "Automatic Context Compression",
                    "Starting Sandbox",
                    "Pulling sandbox image",
                    "Starting sandbox container",
                    "Sandbox startup completed",
                    "Evaluating project size",
                    "Reconnecting",
                }:
                    current_working = status_text
                    current_deadline = None
                else:
                    current_working = "Thinking"
                    current_deadline = None
                if self._should_persist_status_statement(status_text):
                    self.home.append_session_event(
                        session.path,
                        "work_message",
                        {
                            "message": status_text,
                            "role": "tool",
                            "turn_id": turn_id,
                        },
                    )
                    current_work_count += 1
            elif event.kind == "delta":
                current_final += event.text
            elif event.kind == "thought" and event.text:
                self.home.append_session_event(
                    session.path,
                    "work_message",
                    {
                        "message": "Created a thought",
                        "role": "thought",
                        "command": event.text.strip(),
                        "turn_id": turn_id,
                    },
                )
                current_work_count += 1
            elif event.kind in {"message", "tool_message", "command"} and event.text:
                clean_text = event.text.strip()
                if current_final and event.kind in {"tool_message", "command"}:
                    clean_current = current_final.strip()
                    if clean_current:
                        self.home.append_session_event(
                            session.path,
                            "agent_message",
                            {"message": clean_current, "turn_id": turn_id, "intermediate": True},
                        )
                        current_work_count += 1
                role = "tool" if event.kind == "tool_message" else "agent"
                if event.kind == "command":
                    role = "tool"
                current_final = ""
                if clean_text and render_events and self.work_visualization == "extended":
                    self._fake_type_message(
                        stdscr,
                        session,
                        clean_text,
                        role=role,
                        anchor_line=anchor_line,
                        input_text=input_text,
                        cursor=cursor,
                        pasted_spans=pasted_spans,
                        prompt_notice=prompt_notice,
                        prompt_notice_role=prompt_notice_role,
                        scroll=scroll,
                        animate=event.kind == "message",
                        active_turn_elapsed=active_turn_elapsed,
                        sticky_anchor=sticky_anchor,
                    )
                if event.kind == "message":
                    self.home.append_session_event(
                        session.path,
                        "agent_message",
                        {
                            "message": clean_text,
                            "turn_id": turn_id,
                            "intermediate": True,
                        },
                    )
                    current_work_count += 1
                else:
                    payload = {
                        "message": clean_text,
                        "role": role,
                        "turn_id": turn_id,
                    }
                    if event.kind == "command" and event.command:
                        payload["command"] = event.command
                    self.home.append_session_event(session.path, "work_message", payload)
                    current_work_count += 1
            elif event.kind == "approval":
                approval_request = event.approval_request
                approval_response = event.approval_response
                if approval_request is not None and approval_response is not None:
                    choice = self._request_command_approval(
                        stdscr,
                        session,
                        approval_request,
                        scroll=scroll,
                        anchor_line=anchor_line,
                    )
                    approval_response.put(choice)
                    approval_message = self._approval_work_message(
                        approval_request,
                        choice,
                    )
                    if approval_message:
                        approval_role = (
                            "forbidden"
                            if choice in {ApprovalChoice.REJECT, ApprovalChoice.ALWAYS_REJECT}
                            else "tool"
                        )
                        self.home.append_session_event(
                            session.path,
                            "work_message",
                            {
                                "message": approval_message,
                                "role": approval_role,
                                "turn_id": turn_id,
                            },
                        )
                        current_work_count += 1
            elif event.kind == "question":
                question_request = event.question_request
                question_response = event.question_response
                if question_request is not None and question_response is not None:
                    answer = self._request_question(
                        stdscr,
                        session,
                        question_request,
                        scroll=scroll,
                        anchor_line=anchor_line,
                    )
                    question_response.put(answer)
                    self._append_question_context(session, question_request, answer)
            elif event.kind == "system_message" and event.text:
                self.home.append_session_event(
                    session.path,
                    "system_message",
                    {
                        "message": event.text,
                        "role": event.role or "system",
                        "turn_id": turn_id,
                    },
                )
                current_work_count += 1
            elif event.kind == "finish" and event.text:
                current_final = event.text.strip()

    def _should_persist_status_statement(self, status_text: str) -> bool:
        normalized = status_text.strip()
        if not normalized:
            return False
        return normalized not in {
            "Thinking",
            "Waiting",
            "Loading model",
            "Starting Sandbox",
            "Pulling sandbox image",
            "Starting sandbox container",
            "Sandbox startup completed",
            "Evaluating project size",
            "Reconnecting",
        }

    def _parse_runtime_status(self, message: str) -> tuple[str, float | None]:
        text = message or "Thinking"
        if text.startswith("Waiting:"):
            raw_seconds = text.partition(":")[2]
            with suppress(ValueError):
                return "Waiting", max(0.0, float(raw_seconds))
            return "Waiting", None
        return text, None

    def _approval_work_message(
        self,
        request: CommandApprovalRequest,
        choice: ApprovalChoice,
    ) -> str:
        if choice in {ApprovalChoice.ALLOW, ApprovalChoice.ALWAYS_ALLOW}:
            return ""

        statement = request.statement.strip() or "command"
        if choice == ApprovalChoice.ALWAYS_REJECT:
            reason = (
                f"{request.allowance_label or 'Matching commands'} blocked globally by user policy."
            )
        else:
            reason = "Rejected by user."
        return self._blocked_statement_message(statement, request.command, reason)

    def _save_approval_memory(
        self,
        runtime: AgentRuntime | RuntimeProcessClient,
        request: CommandApprovalRequest,
        allowance_key: str,
        reason: str,
    ) -> None:
        context = {
            "command": request.command,
            "canonical_command": request.canonical_command,
            "statement": request.statement,
            "approval_reason": request.reason,
            "allowance_key": allowance_key,
            "allowance_label": request.allowance_label,
            "allowance_subject": request.allowance_subject,
            "agent_id": request.agent_id,
            "agent_name": request.agent_name,
        }
        metadata = runtime.suggest_memory_metadata(
            kind=MemoryKind.APPROVAL,
            context=context,
            content=reason,
        )
        record = create_memory_record(
            title=metadata.title,
            summary=metadata.summary,
            kind=MemoryKind.APPROVAL,
            context=context,
            content=reason,
        )
        write_memory(self.home.brain_dir, record)

    def _blocked_statement_message(self, statement: str, command: str, reason: str) -> str:
        return f"Blocked: {statement}\nCommand: {command}\nReason: {reason}"

    def _fake_type_message(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        message: str,
        role: str = "agent",
        anchor_line: int | None = None,
        input_text: str = "",
        cursor: int = 0,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
        prompt_notice: str = "",
        prompt_notice_role: str = "light",
        scroll: int = 0,
        animate: bool = True,
        active_turn_elapsed: float | None = None,
        sticky_anchor: bool = False,
    ) -> None:
        local_scroll = scroll
        local_anchor = anchor_line
        local_sticky_anchor = sticky_anchor
        if self.isolate_runtime:
            animate = False
        if not animate:
            self._draw_session(
                stdscr,
                session,
                [
                    *self._read_message_lines(session.path),
                    MessageLine(role, message),
                ],
                input_text,
                cursor,
                local_scroll,
                anchor_line=local_anchor,
                sticky_anchor=local_sticky_anchor and local_anchor is not None,
                prompt_notice=prompt_notice,
                prompt_notice_role=prompt_notice_role,
                active_turn_elapsed=active_turn_elapsed,
                pasted_spans=pasted_spans,
            )
            return
        with suppress(curses.error, AttributeError):
            stdscr.nodelay(True)
        rendered = ""
        for character in message:
            rendered += character
            key = self._read_nonblocking_key(stdscr) if hasattr(stdscr, "get_wch") else None
            if key is not None:
                local_anchor, local_scroll, local_sticky_anchor = self._handle_fake_type_key(
                    stdscr,
                    session,
                    key,
                    input_text,
                    cursor,
                    local_anchor,
                    local_scroll,
                    local_sticky_anchor,
                    pasted_spans,
                )
            viewport = self._draw_session(
                stdscr,
                session,
                [
                    *self._read_message_lines(session.path),
                    MessageLine(role, rendered),
                ],
                input_text,
                cursor,
                local_scroll,
                anchor_line=local_anchor,
                sticky_anchor=local_sticky_anchor and local_anchor is not None,
                prompt_notice=prompt_notice,
                prompt_notice_role=prompt_notice_role,
                active_turn_elapsed=active_turn_elapsed,
                pasted_spans=pasted_spans,
            )
            local_scroll = viewport.scroll
            time.sleep(0.003)

    def _handle_fake_type_key(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        key: str | int | PromptPasteEvent,
        input_text: str,
        cursor: int,
        anchor_line: int | None,
        scroll: int,
        sticky_anchor: bool,
        pasted_spans: Sequence[PromptPasteSpan] | None = None,
    ) -> tuple[int | None, int, bool]:
        del cursor
        if key == curses.KEY_UP:
            if sticky_anchor and anchor_line is not None:
                return anchor_line, scroll + 1, True
            return None, scroll + 1, False
        if key == curses.KEY_DOWN:
            if sticky_anchor and anchor_line is not None:
                return anchor_line, scroll - 1, True
            return None, scroll - 1, False
        if key == curses.KEY_PPAGE:
            if sticky_anchor and anchor_line is not None:
                return anchor_line, scroll + 5, True
            return None, scroll + 5, False
        if key == curses.KEY_NPAGE:
            if sticky_anchor and anchor_line is not None:
                return anchor_line, scroll - 5, True
            return None, scroll - 5, False
        if key == curses.KEY_MOUSE:
            action = self._session_mouse_action(
                stdscr,
                input_text,
                [],
                pasted_spans=pasted_spans,
            )
            return self._handle_fake_type_mouse_action(
                session,
                action,
                anchor_line,
                scroll,
                sticky_anchor,
            )
        action = self._raw_mouse_action(
            key,
            stdscr,
            input_text,
            [],
            pasted_spans=pasted_spans,
        )
        if action is not None:
            return self._handle_fake_type_mouse_action(
                session,
                action,
                anchor_line,
                scroll,
                sticky_anchor,
            )
        return anchor_line, scroll, sticky_anchor

    def _handle_fake_type_mouse_action(
        self,
        session: SessionRecord,
        action: SessionMouseAction | None,
        anchor_line: int | None,
        scroll: int,
        sticky_anchor: bool,
    ) -> tuple[int | None, int, bool]:
        if action is None:
            return anchor_line, scroll, sticky_anchor
        if action.kind == "scroll":
            if sticky_anchor and anchor_line is not None:
                return anchor_line, scroll + action.value, True
            return None, scroll + action.value, False
        if action.kind == "toggle_work":
            self._toggle_work_turn(action.text)
        elif action.kind == "toggle_pinned_user":
            self._toggle_pinned_user(action.text)
        elif action.kind == "toggle_plan":
            self._toggle_plan(session.path)
        elif action.kind == "toggle_work_line":
            self._toggle_work_line(action.text)
        elif action.kind == "toggle_activity_item":
            self._toggle_activity_item(action.text)
        elif action.kind == "toggle_activity_entry":
            self._toggle_activity_entry(action.text)
        elif action.kind == "scroll_activity_item":
            self._scroll_activity_item(action.text, action.value)
        elif action.kind == "kill_process":
            self._runtime_for_session(session).end_process(action.text, session.path)
        return anchor_line, scroll, sticky_anchor

    def _load_config_cached(self) -> dict[str, Any]:
        cache_key = self._config_cache_key()
        if cache_key is None:
            return self.home.load_config()
        if (
            self._config_cache is not None
            and self._config_cache[0] == cache_key[0]
            and self._config_cache[1] == cache_key[1]
        ):
            return self._config_cache[2]
        config = self.home.load_config()
        self._config_cache = (cache_key[0], cache_key[1], config)
        return config

    def _config_cache_key(self) -> tuple[int, int] | None:
        try:
            stat = self.home.config_path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _toggle_work_turn(self, turn_id: str) -> None:
        if turn_id in self._expanded_work_turns:
            self._expanded_work_turns.remove(turn_id)
        elif turn_id:
            self._expanded_work_turns.add(turn_id)

    def _toggle_plan(self, session_path: Path) -> None:
        if session_path in self._expanded_plan_sessions:
            self._expanded_plan_sessions.remove(session_path)
        else:
            self._expanded_plan_sessions.add(session_path)

    def _toggle_work_line(self, expansion_key: str) -> None:
        if expansion_key in self._expanded_work_lines:
            self._expanded_work_lines.remove(expansion_key)
        elif expansion_key:
            self._expanded_work_lines.add(expansion_key)

    def _toggle_pinned_user(self, expansion_key: str) -> None:
        if expansion_key in self._expanded_pinned_users:
            self._expanded_pinned_users.remove(expansion_key)
        elif expansion_key:
            self._expanded_pinned_users.add(expansion_key)

    def _toggle_activity_item(self, activity_key: str) -> None:
        if activity_key in self._expanded_activity_items:
            self._expanded_activity_items.remove(activity_key)
            self._remove_activity_entry_expansions(activity_key)
        elif activity_key:
            self._expanded_activity_items = {activity_key}
            self._expanded_activity_entries.clear()
            self._activity_detail_scrolls.setdefault(activity_key, 0)

    def _toggle_activity_entry(self, entry_key: str) -> None:
        if entry_key in self._expanded_activity_entries:
            self._expanded_activity_entries.remove(entry_key)
        elif entry_key:
            self._expanded_activity_entries.add(entry_key)

    def _scroll_activity_item(self, activity_key: str, delta: int) -> None:
        if not activity_key:
            return
        current = self._activity_detail_scrolls.get(activity_key, 0)
        self._activity_detail_scrolls[activity_key] = max(0, current - delta)

    def _remove_activity_entry_expansions(self, activity_key: str) -> None:
        owner_id = activity_key.partition(":")[2]
        if not owner_id:
            return
        prefix = f"activity:{owner_id}:"
        self._expanded_activity_entries = {
            entry_key
            for entry_key in self._expanded_activity_entries
            if not entry_key.startswith(prefix)
        }

    def _activate_agent(self, agent_kind: AgentKind | str) -> BaseAgent:
        active_agent = agent_spec(agent_kind)
        self.active_agent = active_agent
        if not isinstance(self.runtime, RuntimeProcessClient):
            self.runtime.set_agent(active_agent.kind)
            self.runtime.set_mode(self.agent_mode)
        return active_agent

    def _activate_agent_mode(self, mode: AgentMode | str) -> AgentMode:
        """Compatibility hook for config flows that still update approval mode."""

        agent_mode = AgentMode.parse(mode, self.agent_mode)
        if agent_mode == AgentMode.BACKGROUND:
            agent_mode = AgentMode.STANDARD
        self.agent_mode = agent_mode
        if not isinstance(self.runtime, RuntimeProcessClient):
            self.runtime.set_mode(agent_mode)
        return agent_mode

    def _cycle_agent_mode(self, session: SessionRecord | None = None) -> AgentMode:
        """Cycle interactive execution modes."""

        next_mode = next_agent_mode(
            self.agent_mode,
            platform_connected=self.home.has_platform_connection(),
        )
        self._activate_agent_mode(next_mode)
        if session is not None:
            self.home.update_session_agent(session.path, AgentKind.MAIN, next_mode)
        else:
            config = self.home.load_config()
            config["agent_kind"] = AgentKind.MAIN.value
            config["agent_mode"] = next_mode.value
            self.home.save_config(config)
        return next_mode

    def _mode_hint_attr_name(self) -> str:
        return self.agent_mode.policy.ui_attr

    def _sandbox_configured(self) -> bool:
        return bool(self._load_config_cached().get("sandbox_enabled"))

    def _sandbox_is_active(self) -> bool:
        if isinstance(self.runtime, RuntimeProcessClient):
            return self._sandbox_configured()
        if self.runtime.is_sandbox_active():
            return True
        return self._sandbox_configured()

    def _latest_user_anchor_line(self, stdscr: CursesWindow, session: SessionRecord) -> int | None:
        _, width = stdscr.getmaxyx()
        rendered = self._session_rendered_lines(
            session,
            self._read_message_lines(session.path),
            max(20, width - 8),
        )
        user_lines = [
            index for index, line in enumerate(rendered) if line.role == "user" and line.text
        ]
        return user_lines[-1] if user_lines else None

    def _active_turn_anchor_line(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        turn: ActiveSessionTurn,
        fallback_anchor: int | None,
    ) -> int | None:
        anchor_key = turn.anchor_expansion_key or self._latest_root_user_expansion_key(session.path)
        if not anchor_key:
            return fallback_anchor
        turn.anchor_expansion_key = anchor_key
        resolved = self._user_anchor_line_for_key(stdscr, session, anchor_key)
        return resolved if resolved is not None else fallback_anchor

    def _user_anchor_line_for_key(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        expansion_key: str,
    ) -> int | None:
        _, width = stdscr.getmaxyx()
        rendered = self._session_rendered_lines(
            session,
            self._read_message_lines(session.path),
            max(20, width - 8),
        )
        for index, line in enumerate(rendered):
            if (
                line.role in {"user", "user_box", "pinned_user"}
                and line.expansion_key == expansion_key
            ):
                return index
        return None

    def _latest_root_user_expansion_key(self, session_path: Path) -> str:
        latest_key = ""
        for event_index, event in enumerate(self._session_events(session_path)):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = str(
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            message = str(payload.get("message", "")).strip()
            if event_type not in {"user_message", "skill_invocation"} or not message:
                continue
            if event_type == "user_message" and payload.get("intermediate"):
                continue
            latest_key = self._session_user_message_key(event_index)
        return latest_key

    def _animate_message_anchor(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        anchor_line: int | None,
    ) -> None:
        if anchor_line is None:
            return
        steps = min(10, max(1, anchor_line))
        for step in range(1, steps + 1):
            interpolated = round(anchor_line * (step / steps))
            self._draw_session(
                stdscr,
                session,
                self._read_message_lines(session.path),
                "",
                0,
                0,
                anchor_line=interpolated,
            )
            time.sleep(0.015)

    def _format_duration(self, seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        minutes, remaining_seconds = divmod(total_seconds, 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"

    def _session_location_rows(self, session: SessionRecord) -> tuple[InfoRow, ...]:
        return (
            InfoRow("Path", session.cwd or str(self.cwd)),
            InfoRow("Workspace root", str(self.workspace_root)),
        )

    def _approved_command_rows(self, session: SessionRecord) -> tuple[InfoRow, ...]:
        subjects = self._session_command_subjects(self.session_allowed_commands)
        rows = [InfoRow("Always approved", subject) for subject in subjects]
        rows.extend(
            InfoRow("Approved", command)
            for command in self._approved_commands_from_session(session)
        )
        if not rows:
            return (InfoRow("Session", "No commands approved yet"),)
        return tuple(rows)

    def _approved_commands_from_session(self, session: SessionRecord) -> tuple[str, ...]:
        commands: list[str] = []
        seen: set[str] = set()
        for event in self._session_events(session.path):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = str(
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            if event_type != "work_message":
                continue
            command = self._approved_command_from_message(str(payload.get("message", "")))
            if not command or command in seen:
                continue
            seen.add(command)
            commands.append(command)
        return tuple(commands)

    def _approved_command_from_message(self, message: str) -> str:
        text = message.strip()
        if text.startswith("Approved command:"):
            return text.partition(":")[2].strip()
        if text.startswith("Always approved "):
            return text.partition(":")[2].strip()
        return ""

    def _session_command_subjects(self, keys: set[str]) -> list[str]:
        return sorted({self._session_command_subject(key) for key in keys})

    def _session_command_subject(self, key: str) -> str:
        return command_allowance_display(key).subject

    def _all_skills(self) -> tuple[Skill, ...]:
        skills: list[Skill] = []
        seen: set[str] = set()
        for skill in (*load_builtin_skills(), *self._user_skills()):
            if skill.system:
                continue
            if skill.command in seen:
                continue
            seen.add(skill.command)
            skills.append(skill)
        return tuple(skills)

    def _user_skills(self) -> tuple[Skill, ...]:
        return load_user_skills(self.home.skills_dir)

    def _starter_skills(self) -> tuple[Skill, ...]:
        return tuple(skill for skill in self._user_skills() if not skill.system)[:3]

    def _skill_for_command(self, command: str) -> Skill | None:
        command_name = command.removeprefix("/")
        return next(
            (skill for skill in self._all_skills() if skill.command == command_name),
            None,
        )

    def _user_skill_by_command(self, command: str) -> Skill | None:
        return next((skill for skill in self._user_skills() if skill.command == command), None)

    def _command_exists(self, command: str, exclude_command: str | None = None) -> bool:
        slash_command = f"/{command}"
        excluded = f"/{exclude_command}" if exclude_command else ""
        return any(
            spec.command == slash_command and spec.command != excluded
            for spec in self._command_specs()
        )

    def _command_specs(self) -> tuple[CommandSpec, ...]:
        skill_specs = tuple(
            CommandSpec(skill.slash_command, f"{skill.title} · {skill.description}")
            for skill in self._all_skills()
        )
        commands = tuple(
            spec
            for spec in COMMANDS
            if spec.command != "/feedback" or self.home.has_platform_connection()
        )
        return (*commands, *skill_specs)

    def _project_command_specs(self) -> tuple[CommandSpec, ...]:
        skill_specs = tuple(
            CommandSpec(skill.slash_command, f"{skill.title} · {skill.description}")
            for skill in self._all_skills()
        )
        project_specs = tuple(
            spec
            for spec in COMMANDS
            if spec.command in PROJECT_COMMANDS
            and (spec.command != "/feedback" or self.home.has_platform_connection())
        )
        return (*project_specs, *skill_specs)

    def _invoke_skill(
        self,
        stdscr: CursesWindow,
        session: SessionRecord,
        skill: Skill,
        submitted: str,
    ) -> str | None:
        self._append_skill_invocation_event(session, skill, submitted)
        self._maybe_start_session_rename(session)
        anchor_line = self._latest_user_anchor_line(stdscr, session)
        self._animate_message_anchor(stdscr, session, anchor_line)
        turn = self._start_session_turn(session)
        turn.anchor_expansion_key = self._latest_root_user_expansion_key(session.path)
        return None

    def _append_skill_invocation_event(
        self,
        session: SessionRecord,
        skill: Skill,
        submitted: str,
    ) -> None:
        arguments = self._skill_arguments(skill, submitted)
        display_message = f"/{skill.command} {arguments}".strip()
        self.home.append_session_event(
            session.path,
            "skill_invocation",
            {
                "message": display_message,
                "command": skill.command,
                "title": skill.title,
                "description": skill.description,
                "prompt": skill_invocation_prompt(skill, arguments),
            },
        )

    def _skill_arguments(self, skill: Skill, submitted: str) -> str:
        parts = submitted.strip().split(maxsplit=1)
        if not parts:
            return ""
        if parts[0] != skill.slash_command:
            return ""
        return parts[1] if len(parts) > 1 else ""

    def _filtered_commands(self, input_text: str) -> list[CommandSpec]:
        return self._filtered_command_specs(input_text, self._command_specs())

    def _filtered_project_commands(self, input_text: str) -> list[CommandSpec]:
        return self._filtered_command_specs(input_text, self._project_command_specs())

    def _filtered_running_commands(self, input_text: str) -> list[CommandSpec]:
        return self._filtered_command_specs(input_text, self._running_command_specs())

    def _active_command_token(
        self,
        input_text: str,
        cursor: int,
    ) -> tuple[int, int, str] | None:
        """Return the slash-command token surrounding the cursor, anywhere in the prompt."""

        bounded_cursor = max(0, min(cursor, len(input_text)))
        token_start = input_text.rfind("/", 0, bounded_cursor)
        if token_start < 0 or any(
            character.isspace() for character in input_text[token_start:bounded_cursor]
        ):
            return None
        fragment_start = token_start
        while fragment_start > 0 and not input_text[fragment_start - 1].isspace():
            fragment_start -= 1
        if input_text[fragment_start:token_start].startswith("@"):
            return None
        token_end = bounded_cursor
        while token_end < len(input_text) and not input_text[token_end].isspace():
            token_end += 1
        token = input_text[token_start:bounded_cursor]
        return token_start, token_end, token

    def _command_token_is_inline(
        self,
        input_text: str,
        token: tuple[int, int, str],
    ) -> bool:
        return bool(input_text[: token[0]].strip())

    def _remove_command_token(
        self,
        input_text: str,
        token: tuple[int, int, str],
        pasted_spans: list[PromptPasteSpan],
    ) -> tuple[str, int]:
        start, end, _query = token
        if start > 0 and input_text[start - 1].isspace():
            start -= 1
        elif end < len(input_text) and input_text[end].isspace():
            end += 1
        return self._replace_prompt_range(
            input_text,
            start,
            end,
            "",
            pasted_spans,
        )

    @contextmanager
    def _preserve_modal_prompt(
        self,
        input_text: str,
        cursor: int,
        pasted_spans: Sequence[PromptPasteSpan],
        file_references: Mapping[str, str] | None = None,
        image_attachments: Mapping[str, dict[str, str]] | None = None,
    ) -> Iterator[None]:
        """Keep the current draft visible while a command popover is open."""

        previous = (
            self._modal_prompt_text,
            self._modal_prompt_cursor,
            self._modal_prompt_pasted_spans,
            self._modal_prompt_file_references,
            self._modal_prompt_image_attachments,
        )
        self._modal_prompt_text = input_text
        self._modal_prompt_cursor = max(0, min(cursor, len(input_text)))
        self._modal_prompt_pasted_spans = tuple(pasted_spans)
        self._modal_prompt_file_references = dict(file_references or {})
        self._modal_prompt_image_attachments = dict(image_attachments or {})
        try:
            yield
        finally:
            (
                self._modal_prompt_text,
                self._modal_prompt_cursor,
                self._modal_prompt_pasted_spans,
                self._modal_prompt_file_references,
                self._modal_prompt_image_attachments,
            ) = previous

    def _filtered_command_specs(
        self,
        input_text: str,
        command_specs: tuple[CommandSpec, ...],
    ) -> list[CommandSpec]:
        query = input_text.removeprefix("/").strip().lower()
        if not query:
            return list(command_specs)

        prefix_matches: list[CommandSpec] = []
        contains_matches: list[CommandSpec] = []
        for command in command_specs:
            command_name = command.command.removeprefix("/").lower()
            searchable = f"{command_name} {command.description}".lower()
            if command_name.startswith(query):
                prefix_matches.append(command)
            elif query in searchable:
                contains_matches.append(command)
        if prefix_matches:
            return prefix_matches[:5]
        return contains_matches[:5]

    def _active_file_reference_token(
        self,
        input_text: str,
        cursor: int,
    ) -> tuple[int, int, str] | None:
        bounded_cursor = max(0, min(cursor, len(input_text)))
        token_start = bounded_cursor
        while token_start > 0 and not input_text[token_start - 1].isspace():
            token_start -= 1
        token_end = bounded_cursor
        while token_end < len(input_text) and not input_text[token_end].isspace():
            token_end += 1
        token = input_text[token_start:bounded_cursor]
        if not token.startswith("@"):
            return None
        return (token_start, token_end, token.removeprefix("@"))

    def _background_file_reference_suggestions(self, query: str) -> list[MenuChoice]:
        return [
            MenuChoice(
                result.label,
                result.value,
                "",
                result.query,
                result.highlight_spans,
            )
            for result in self._file_reference_search.suggestions(query)
        ]

    def _filtered_file_references(self, query: str) -> list[MenuChoice]:
        normalized_query = query.strip()
        if normalized_query.endswith("/") and normalized_query.strip("/"):
            return self._filtered_file_reference_children(normalized_query)

        matches: list[tuple[int, int, int, str, MenuChoice]] = []
        references = self._prioritized_workspace_reference_paths(normalized_query)
        for relative in references:
            label = self._file_reference_display_path(relative)

            if not normalized_query:
                matches.append(
                    (0, 0, len(label), label.lower(), MenuChoice(label, relative, "", ""))
                )
                continue

            match = self._file_reference_query_match(normalized_query, relative)
            if match is None:
                continue
            rank, score, spans = match

            matches.append(
                (
                    rank,
                    score,
                    len(label),
                    label.lower(),
                    MenuChoice(label, relative, "", normalized_query, spans),
                )
            )

        matches.sort(key=lambda match: (match[0], match[1], match[2]))
        return [match[4] for match in matches[:FILE_REFERENCE_LIMIT]]

    def _prioritized_workspace_reference_paths(self, query: str) -> tuple[str, ...]:
        paths: list[str] = []
        seen: set[str] = set()
        for relative in (
            *self._workspace_reference_prefix_paths(query),
            *self._workspace_reference_paths(query=query),
        ):
            normalized = relative.strip("/")
            if relative.endswith("/"):
                normalized = f"{normalized}/"
            if normalized in seen:
                continue
            seen.add(normalized)
            paths.append(normalized)
        return tuple(paths)

    def _filtered_file_reference_children(self, query: str) -> list[MenuChoice]:
        directory_match = self._best_file_reference_directory(query)
        if directory_match is None:
            return []
        directory, _rank, _score, parent_spans = directory_match
        choices: list[MenuChoice] = []
        for relative in self._workspace_reference_children(directory):
            label = self._file_reference_display_path(relative)
            spans = tuple(
                (start, end)
                for start, end in parent_spans
                if start < len(label) and end <= len(label)
            )
            choices.append(
                MenuChoice(
                    label,
                    relative,
                    "",
                    query,
                    spans,
                )
            )
        return choices[:FILE_REFERENCE_LIMIT]

    def _best_file_reference_directory(
        self,
        query: str,
    ) -> tuple[Path, int, int, tuple[tuple[int, int], ...]] | None:
        literal = (self.workspace_root / query).resolve()
        with suppress(ValueError):
            literal.relative_to(self.workspace_root)
            if literal.is_dir():
                relative = self._relative_workspace_path(literal)
                if not relative.endswith("/"):
                    relative = f"{relative}/"
                match = self._file_reference_query_match(query, relative)
                if match is not None:
                    rank, score, spans = match
                    return literal, rank, score, spans

        matches: list[tuple[int, int, int, str, Path, tuple[tuple[int, int], ...]]] = []
        for relative in self._workspace_reference_paths(query=query):
            if not relative.endswith("/"):
                continue
            match = self._file_reference_query_match(query, relative)
            if match is None:
                continue
            rank, score, spans = match
            path = (self.workspace_root / relative.rstrip("/")).resolve()
            matches.append((rank, score, len(relative), relative.lower(), path, spans))
        if not matches:
            return None
        rank, score, _length, _relative, path, spans = min(
            matches,
            key=lambda match: (match[0], match[1], match[2], match[3]),
        )
        return path, rank, score, spans

    def _workspace_reference_prefix_paths(self, query: str) -> tuple[str, ...]:
        normalized_query = query.strip().lstrip("/").rstrip("/")
        if "/" not in normalized_query:
            return ()
        parent_text, leaf_prefix = normalized_query.rsplit("/", 1)
        if not parent_text or not leaf_prefix:
            return ()
        parent = (self.workspace_root / parent_text).resolve()
        try:
            parent.relative_to(self.workspace_root)
        except ValueError:
            return ()
        if not parent.is_dir():
            return ()
        matches: list[str] = []
        leaf_prefix_lower = leaf_prefix.lower()
        try:
            for entry in sorted(
                parent.iterdir(),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            ):
                if not entry.name.lower().startswith(leaf_prefix_lower):
                    continue
                if entry.is_dir():
                    if self._ignore_file_reference_dir(entry.name):
                        continue
                    matches.append(f"{self._relative_workspace_path(entry)}/")
                elif entry.is_file():
                    if self._ignore_file_reference_file(entry.name):
                        continue
                    matches.append(self._relative_workspace_path(entry))
        except OSError:
            return ()
        return tuple(matches)

    def _file_reference_query_match(
        self,
        query: str,
        relative_path: str,
    ) -> tuple[int, int, tuple[tuple[int, int], ...]] | None:
        normalized_query = query.strip().lstrip("/")
        query_segments = tuple(
            segment for segment in normalized_query.strip("/").split("/") if segment
        )
        if not query_segments:
            return 0, 0, ()
        display_path = self._file_reference_display_path(relative_path)
        candidates = (
            (relative_path, len(display_path) - len(relative_path)),
            (display_path, 0),
        )
        matches: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
        for path_text, display_offset in candidates:
            match = self._file_reference_segment_path_match(
                normalized_query,
                query_segments,
                path_text,
                display_offset,
            )
            if match is None:
                continue
            matches.append(match)
        if not matches:
            return None
        return min(matches, key=lambda match: (match[0], match[1]))

    def _file_reference_segment_path_match(
        self,
        query: str,
        query_segments: tuple[str, ...],
        path_text: str,
        display_offset: int,
    ) -> tuple[int, int, tuple[tuple[int, int], ...]] | None:
        path_segments = self._file_reference_path_segments(path_text)
        if len(query_segments) > len(path_segments):
            return None
        query_lower = tuple(segment.lower() for segment in query_segments)
        requires_trailing_separator = query.endswith("/")
        matches: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
        last_start = len(path_segments) - len(query_segments)
        for start_index in range(last_start + 1):
            if (
                len(query_segments) > 1
                and not requires_trailing_separator
                and start_index + len(query_segments) != len(path_segments)
            ):
                continue
            window = path_segments[start_index : start_index + len(query_segments)]
            rank = 0
            for query_segment, (path_segment, _segment_start, _segment_end) in zip(
                query_lower[:-1],
                window[:-1],
                strict=True,
            ):
                if path_segment.lower() != query_segment:
                    break
            else:
                final_query = query_lower[-1]
                final_segment, final_start, final_end = window[-1]
                final_lower = final_segment.lower()
                if final_lower == final_query:
                    match_end = final_end
                elif final_lower.startswith(final_query):
                    rank += 2
                    match_end = final_start + len(final_query)
                elif len(query_segments) == 1 and final_query in final_lower:
                    contains_at = final_lower.find(final_query)
                    rank += 4
                    final_start += contains_at
                    match_end = final_start + len(final_query)
                else:
                    continue

                if requires_trailing_separator:
                    if final_end >= len(path_text) or path_text[final_end] != "/":
                        continue
                    match_end = final_end + 1

                match_start = final_start if len(query_segments) == 1 else window[0][1]
                display_start = display_offset + match_start
                display_end = display_offset + match_end
                matches.append((rank, display_start, ((display_start, display_end),)))
        if not matches:
            return None
        return min(matches, key=lambda match: (match[0], match[1]))

    def _file_reference_display_path(self, relative_path: str) -> str:
        return relative_path

    def _single_file_reference_segment_match(
        self,
        query: str,
        path_segments: tuple[tuple[str, int, int], ...],
    ) -> tuple[int, int, tuple[tuple[int, int], ...]] | None:
        terminal = path_segments[-1]
        terminal_match = self._file_reference_segment_match(query, terminal[0])
        if terminal_match is not None:
            rank, score, spans = terminal_match
            return rank, score, self._offset_file_reference_spans(spans, terminal[1])

        matches: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
        for index, (segment, start, _end) in enumerate(path_segments[:-1]):
            segment_match = self._file_reference_segment_match(query, segment)
            if segment_match is None:
                continue
            rank, score, spans = segment_match
            matches.append(
                (
                    3 + rank,
                    score + index,
                    start,
                    self._offset_file_reference_spans(spans, start),
                )
            )
        if not matches:
            return None
        rank, score, _start, spans = min(matches, key=lambda match: (match[0], match[1], match[2]))
        return rank, score, spans

    def _file_reference_segment_window_match(
        self,
        query_segments: tuple[str, ...],
        path_segments: tuple[tuple[str, int, int], ...],
    ) -> tuple[int, int, tuple[tuple[int, int], ...]] | None:
        if len(query_segments) > len(path_segments):
            return None
        matches: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
        last_start = len(path_segments) - len(query_segments)
        for start_index in range(last_start + 1):
            score = start_index * 4
            rank = 0 if start_index == 0 else 1
            spans: list[tuple[int, int]] = []
            for offset, query_segment in enumerate(query_segments):
                path_segment, segment_start, _segment_end = path_segments[start_index + offset]
                segment_match = self._file_reference_segment_match(query_segment, path_segment)
                if segment_match is None:
                    break
                segment_rank, segment_score, segment_spans = segment_match
                rank += segment_rank
                score += segment_score
                spans.extend(self._offset_file_reference_spans(segment_spans, segment_start))
            else:
                matches.append((rank, score, start_index, self._merge_file_reference_spans(spans)))
        if not matches:
            return None
        rank, score, _start_index, merged_spans = min(
            matches,
            key=lambda match: (match[0], match[1], match[2]),
        )
        return rank, score, merged_spans

    def _file_reference_segment_match(
        self,
        query: str,
        segment: str,
    ) -> tuple[int, int, tuple[tuple[int, int], ...]] | None:
        normalized_query = query.lower()
        normalized_segment = segment.lower()
        if not normalized_query:
            return 0, 0, ()
        if normalized_segment.startswith(normalized_query):
            return 0, 0, ((0, len(normalized_query)),)
        contains_at = normalized_segment.find(normalized_query)
        if contains_at >= 0:
            end = contains_at + len(normalized_query)
            return 1, contains_at, ((contains_at, end),)
        subsequence_spans = self._file_reference_subsequence_spans(
            normalized_query,
            normalized_segment,
        )
        if subsequence_spans:
            return 2, 4 + len(subsequence_spans), subsequence_spans
        edit_span = self._file_reference_one_edit_prefix_span(
            normalized_query,
            normalized_segment,
        )
        if edit_span is not None:
            return 3, 6, (edit_span,)
        return None

    def _file_reference_one_edit_prefix_span(
        self,
        query: str,
        segment: str,
    ) -> tuple[int, int] | None:
        if len(query) < 3:
            return None
        prefix = segment[: len(query)]
        if len(prefix) >= len(query) - 1 and self._file_reference_edit_distance_at_most_one(
            query,
            prefix,
        ):
            return 0, len(prefix)
        if len(segment) >= len(query) and self._file_reference_edit_distance_at_most_one(
            query,
            segment,
        ):
            return 0, len(segment)
        return None

    def _file_reference_edit_distance_at_most_one(self, left: str, right: str) -> bool:
        if abs(len(left) - len(right)) > 1:
            return False
        if left == right:
            return True
        if len(left) == len(right):
            return sum(1 for a, b in zip(left, right, strict=True) if a != b) <= 1
        if len(left) > len(right):
            left, right = right, left
        index_left = 0
        index_right = 0
        edits = 0
        while index_left < len(left) and index_right < len(right):
            if left[index_left] == right[index_right]:
                index_left += 1
                index_right += 1
                continue
            edits += 1
            if edits > 1:
                return False
            index_right += 1
        return True

    def _file_reference_subsequence_spans(
        self,
        query: str,
        segment: str,
    ) -> tuple[tuple[int, int], ...]:
        positions: list[int] = []
        search_from = 0
        for char in query:
            position = segment.find(char, search_from)
            if position < 0:
                return ()
            positions.append(position)
            search_from = position + 1
        spans: list[tuple[int, int]] = []
        span_start = positions[0]
        previous = positions[0]
        for position in positions[1:]:
            if position == previous + 1:
                previous = position
                continue
            spans.append((span_start, previous + 1))
            span_start = position
            previous = position
        spans.append((span_start, previous + 1))
        return tuple(spans)

    def _file_reference_path_segments(
        self,
        relative_path: str,
    ) -> tuple[tuple[str, int, int], ...]:
        segments: list[tuple[str, int, int]] = []
        start = 0
        for part in relative_path.split("/"):
            end = start + len(part)
            if part:
                segments.append((part, start, end))
            start = end + 1
        return tuple(segments)

    def _offset_file_reference_spans(
        self,
        spans: Sequence[tuple[int, int]],
        offset: int,
    ) -> tuple[tuple[int, int], ...]:
        return tuple((start + offset, end + offset) for start, end in spans)

    def _merge_file_reference_spans(
        self,
        spans: Sequence[tuple[int, int]],
    ) -> tuple[tuple[int, int], ...]:
        merged: list[tuple[int, int]] = []
        for start, end in sorted(spans):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
                continue
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        return tuple(merged)

    def _workspace_reference_paths(self, query: str = "") -> tuple[str, ...]:
        now = time.monotonic()
        self._ensure_file_reference_index()

        # Empty query (just "@"): show first-level entries only (instant, no full walk)
        if not query:
            if (
                self._file_reference_cache
                and now - self._file_reference_cache_at < FILE_REFERENCE_CACHE_SECONDS
            ):
                return self._file_reference_cache
            paths: list[str] = []
            try:
                for entry in sorted(
                    Path(self.workspace_root).iterdir(),
                    key=lambda e: (not e.is_dir(), e.name.lower()),
                ):
                    if entry.is_dir():
                        if self._ignore_file_reference_dir(entry.name):
                            continue
                        paths.append(f"{self._relative_workspace_path(entry)}/")
                    elif entry.is_file():
                        if self._ignore_file_reference_file(entry.name):
                            continue
                        paths.append(self._relative_workspace_path(entry))
                    if len(paths) >= FILE_REFERENCE_FIRST_LEVEL_LIMIT:
                        break
            except OSError:
                pass
            self._file_reference_cache = tuple(sorted(paths, key=str.lower))
            self._file_reference_cache_at = now
            return self._file_reference_cache

        self._load_file_reference_index()
        return self._file_reference_full_cache

    def _ensure_file_reference_index(self) -> None:
        self._load_file_reference_index()
        if self._file_reference_index_fresh():
            return
        if self._file_reference_index_running():
            return
        worker = threading.Thread(
            target=self._refresh_file_reference_index,
            name="anomx-file-reference-index",
            daemon=True,
        )
        with self._file_reference_index_lock:
            if self._file_reference_index_thread is not None and (
                self._file_reference_index_thread.is_alive()
            ):
                return
            self._file_reference_index_thread = worker
            self._file_reference_index_started_at = time.monotonic()
        worker.start()

    def _file_reference_index_running(self) -> bool:
        thread = self._file_reference_index_thread
        return thread is not None and thread.is_alive()

    def _file_reference_index_fresh(self) -> bool:
        if not self._file_reference_full_cache_at:
            return False
        return (
            time.monotonic() - self._file_reference_full_cache_at
            < FILE_REFERENCE_INDEX_REFRESH_SECONDS
        )

    def _load_file_reference_index(self) -> None:
        with self._file_reference_index_lock:
            if self._file_reference_index_loaded:
                return
            self._file_reference_index_loaded = True

        cache_path = self._file_reference_index_path()
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        if int(payload.get("version", 0) or 0) != 1:
            return
        if str(payload.get("project_path", "")) != str(self.workspace_root):
            return
        raw_paths = payload.get("paths")
        if not isinstance(raw_paths, list):
            return
        paths = tuple(
            path
            for path in (self._normalized_file_reference_entry(item) for item in raw_paths)
            if path
        )
        generated_at = self._file_reference_generated_monotonic_time(payload)
        with self._file_reference_index_lock:
            self._file_reference_full_cache = paths
            self._file_reference_full_cache_at = generated_at

    def _file_reference_generated_monotonic_time(self, payload: Mapping[str, object]) -> float:
        generated_at = payload.get("generated_at")
        if not isinstance(generated_at, int | float):
            return time.monotonic()
        age = max(0.0, time.time() - float(generated_at))
        return time.monotonic() - age

    def _refresh_file_reference_index(self) -> None:
        paths = self._scan_file_reference_index()
        payload = {
            "version": 1,
            "project_path": str(self.workspace_root),
            "generated_at": time.time(),
            "paths": list(paths),
        }
        cache_path = self._file_reference_index_path()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            tmp_path.replace(cache_path)
        except OSError:
            pass
        with self._file_reference_index_lock:
            self._file_reference_full_cache = paths
            self._file_reference_full_cache_at = time.monotonic()
            self._file_reference_index_loaded = True

    def _scan_file_reference_index(self) -> tuple[str, ...]:
        root_path = self.workspace_root
        paths: list[str] = []
        for root, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [
                dirname for dirname in dirnames if not self._ignore_file_reference_dir(dirname)
            ]
            dirnames.sort(key=str.lower)
            filenames.sort(key=str.lower)
            root_relative = self._walk_root_relative_path(Path(root))
            for dirname in dirnames:
                paths.append(f"{root_relative}{dirname}/")
            for filename in filenames:
                if self._ignore_file_reference_file(filename):
                    continue
                paths.append(f"{root_relative}{filename}")
        return tuple(sorted(set(paths), key=str.lower))

    def _walk_root_relative_path(self, root: Path) -> str:
        with suppress(ValueError):
            relative = root.resolve().relative_to(self.workspace_root)
            if not relative.parts:
                return ""
            return f"{relative.as_posix()}/"
        return ""

    def _file_reference_index_path(self) -> Path:
        project_path = str(self.workspace_root)
        digest = hashlib.sha256(project_path.encode("utf-8")).hexdigest()[:24]
        raw_name = self.workspace_root.name.strip() or "workspace"
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw_name).strip("-._") or "workspace"
        return self.home.search_dir / f"{name[:48]}-{digest}.json"

    def _normalized_file_reference_entry(self, value: object) -> str:
        if not isinstance(value, str):
            return ""
        normalized = value.strip().lstrip("/")
        if not normalized:
            return ""
        if normalized.endswith("/"):
            return f"{normalized.strip('/')}/"
        return normalized.rstrip("/")

    def _file_reference_walk_dir_sort_key(self, dirname: str, query_head: str) -> tuple[int, str]:
        normalized = dirname.lower()
        if query_head and (normalized.startswith(query_head) or query_head in normalized):
            return (0, normalized)
        return (1, normalized)

    def _workspace_reference_children(self, directory: Path) -> tuple[str, ...]:
        paths: list[str] = []
        try:
            for entry in sorted(
                directory.iterdir(),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            ):
                if entry.is_dir():
                    if self._ignore_file_reference_dir(entry.name):
                        continue
                    paths.append(f"{self._relative_workspace_path(entry)}/")
                elif entry.is_file():
                    if self._ignore_file_reference_file(entry.name):
                        continue
                    paths.append(self._relative_workspace_path(entry))
                if len(paths) >= FILE_REFERENCE_FIRST_LEVEL_LIMIT:
                    break
        except OSError:
            pass
        return tuple(sorted(paths, key=str.lower))

    def _reference_label(self, path: Path) -> str:
        suffix = "/" if path.is_dir() else ""
        return f"{path.name}{suffix}"

    def _ignore_file_reference_dir(self, dirname: str) -> bool:
        return dirname in IGNORED_FILE_REFERENCE_DIRS or dirname.startswith(".")

    def _ignore_file_reference_file(self, filename: str) -> bool:
        return filename.endswith((".pyc", ".pyo", ".DS_Store"))

    def _relative_workspace_path(self, path: Path) -> str:
        with suppress(ValueError):
            return path.resolve().relative_to(self.workspace_root).as_posix()
        return path.as_posix()

    def _file_reference_detail(self, relative_path: str) -> str:
        parent = Path(relative_path).parent.as_posix()
        return parent if parent != "." else relative_path

    def _prompt_reference_labels(
        self,
        file_references: Mapping[str, str] | None,
        image_attachments: Mapping[str, Mapping[str, str]] | None,
    ) -> dict[str, str]:
        labels = dict(file_references or {})
        for token in image_attachments or {}:
            labels[token] = ""
        return labels

    def _consume_dropped_images(
        self,
        input_text: str,
        cursor: int,
        image_attachments: dict[str, dict[str, str]],
        pasted_spans: list[PromptPasteSpan] | None = None,
    ) -> tuple[str, int, tuple[dict[str, str], ...]]:
        matches = list(IMAGE_DROP_CANDIDATE_PATTERN.finditer(input_text))
        if not matches:
            return input_text, cursor, ()

        updated = input_text
        updated_cursor = cursor
        added: list[dict[str, str]] = []
        for match in reversed(matches):
            raw_path = match.group("path")
            path = self._dropped_image_path(raw_path)
            if path is None:
                continue
            label = self._image_attachment_label(path, image_attachments)
            token = f"[image: {label}]"
            if token in image_attachments:
                continue
            payload = self._image_attachment_payload(path, label, token)
            if payload is None:
                continue

            start, end = match.span("path")
            suffix = "" if end < len(updated) and updated[end].isspace() else " "
            replacement = f"{token}{suffix}"
            updated = updated[:start] + replacement + updated[end:]
            if pasted_spans is not None:
                pasted_spans[:] = self._prompt_spans_after_replacement(
                    pasted_spans,
                    start,
                    end,
                    len(replacement),
                )
            delta = len(replacement) - (end - start)
            if updated_cursor >= end:
                updated_cursor += delta
            elif updated_cursor > start:
                updated_cursor = start + len(replacement)
            image_attachments[token] = payload
            added.append(payload)

        return updated, max(0, min(updated_cursor, len(updated))), tuple(reversed(added))

    def _dropped_image_path(self, raw_path: str) -> Path | None:
        for candidate in self._dropped_path_candidates(raw_path):
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = self.cwd / path
            with suppress(OSError):
                resolved = path.resolve()
                if resolved.is_file() and image_mime_type(resolved) is not None:
                    return resolved
        return None

    def _dropped_path_candidates(self, raw_path: str) -> tuple[str, ...]:
        stripped = raw_path.strip().strip("\"'")
        if not stripped:
            return ()
        if stripped.startswith("file://"):
            parsed = urlparse(stripped)
            path = unquote(parsed.path)
            if parsed.netloc and not path.startswith(f"//{parsed.netloc}"):
                path = f"//{parsed.netloc}{path}"
            return (path,)

        candidates = [stripped, stripped.replace("\\ ", " ")]
        with suppress(ValueError):
            parts = shlex.split(stripped)
            if len(parts) == 1:
                candidates.append(parts[0])
        unique: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in unique:
                unique.append(candidate)
        return tuple(unique)

    def _image_attachment_label(
        self,
        path: Path,
        image_attachments: Mapping[str, Mapping[str, str]],
    ) -> str:
        base = path.name or "image"
        existing = {
            str(payload.get("label", ""))
            for payload in image_attachments.values()
            if isinstance(payload, Mapping)
        }
        if base not in existing:
            return base
        stem = path.stem or "image"
        suffix = path.suffix
        index = 2
        while f"{stem}-{index}{suffix}" in existing:
            index += 1
        return f"{stem}-{index}{suffix}"

    def _image_attachment_payload(
        self,
        path: Path,
        label: str,
        token: str,
    ) -> dict[str, str] | None:
        mime_type = image_mime_type(path)
        if mime_type is None:
            return None
        return {
            "label": label,
            "token": token,
            "path": path.as_posix(),
            "mime_type": mime_type,
        }

    def _active_image_attachments(
        self,
        prompt: str,
        image_attachments: Mapping[str, dict[str, str]],
    ) -> dict[str, dict[str, str]]:
        return {
            token: payload
            for token, payload in image_attachments.items()
            if self._file_reference_label_pattern(token).search(prompt)
        }

    def _append_unsupported_image_notice(
        self,
        session: SessionRecord,
        images: tuple[dict[str, str], ...],
    ) -> None:
        if not images:
            return
        config = self.home.load_config()
        provider_key = str(config.get("provider", session.provider))
        model = str(config.get("model", session.model))
        if backend_supports_image_input(provider_key, model):
            return
        provider = provider_by_key(provider_key)
        provider_label = provider.label if provider is not None else provider_key
        labels = ", ".join(image["label"] for image in images if image.get("label"))
        self.home.append_session_event(
            session.path,
            "system_message",
            {
                "message": (
                    f"{provider_label}/{model} does not support image input. "
                    f"Dropped image attachments will be kept in the prompt but sent "
                    f"to this backend as text only: {labels}."
                ),
                "role": "warning",
            },
        )

    def _insert_file_reference(
        self,
        input_text: str,
        cursor: int,
        token: tuple[int, int, str],
        choice: MenuChoice,
        file_references: dict[str, str],
        pasted_spans: list[PromptPasteSpan] | None = None,
    ) -> tuple[str, int]:
        del cursor
        start, end, _query = token
        display_label = self._file_reference_prompt_label(choice)
        suffix = "" if end < len(input_text) and input_text[end].isspace() else " "
        replacement = f"{display_label}{suffix}"
        if pasted_spans is None:
            updated = input_text[:start] + replacement + input_text[end:]
            cursor = start + len(replacement)
        else:
            updated, cursor = self._replace_prompt_range(
                input_text,
                start,
                end,
                replacement,
                pasted_spans,
            )
        file_references[display_label] = choice.value
        return updated, cursor

    def _file_reference_prompt_label(self, choice: MenuChoice) -> str:
        value = choice.value.rstrip("/")
        label = Path(value).name or Path(choice.label.rstrip("/")).name or choice.label
        return f"{label}/" if choice.value.endswith("/") else label

    def _backend_message_for_prompt(
        self,
        prompt: str,
        file_references: Mapping[str, str],
    ) -> str:
        expanded = prompt
        for label, path in sorted(
            file_references.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if not label or not path:
                continue
            replacement = path if label == path else f"{label} [{path}]"
            expanded = self._file_reference_label_pattern(label).sub(replacement, expanded)
        return expanded

    def _file_reference_label_pattern(self, label: str) -> re.Pattern[str]:
        escaped = re.escape(label)
        return re.compile(rf"(?<![\w./-]){escaped}(?=$|\s|[,.!?;:)\]}}])")

    def _submitted_command(
        self,
        submitted: str,
        suggestions: list[CommandSpec],
        selected: int,
    ) -> str:
        exact_command = submitted.split(maxsplit=1)[0]
        if any(command.command == exact_command for command in self._command_specs()):
            return exact_command
        if suggestions:
            return suggestions[min(selected, len(suggestions) - 1)].command
        return submitted

    def _submitted_running_command(
        self,
        submitted: str,
        suggestions: list[CommandSpec],
        selected: int,
    ) -> str:
        exact_command = submitted.split(maxsplit=1)[0]
        if self._is_running_session_command(exact_command):
            return exact_command
        if suggestions:
            return suggestions[min(selected, len(suggestions) - 1)].command
        return exact_command

    def _running_command_specs(self) -> tuple[CommandSpec, ...]:
        return tuple(
            command for command in COMMANDS if self._is_running_session_command(command.command)
        )

    def _is_running_session_command(self, command: str) -> bool:
        return command in RUNNING_SESSION_COMMANDS

    def _config_value(self, field: str, config: dict[str, object]) -> str:
        if field == "home":
            return str(self.home.root)
        if field.endswith("_api_key"):
            provider_key = field.removesuffix("_api_key")
            return "configured" if self.home.has_api_key(provider_key) else "not configured"
        return str(config.get(field, ""))

    def _onboarding_complete(self, config: dict[str, object]) -> bool:
        provider = provider_by_key(str(config.get("provider", "")))
        if provider is None:
            return False
        model = str(config.get("model", ""))
        return bool(config.get("onboarding_complete")) and self._model_allowed(provider, model)

    def _model_allowed(self, provider: ProviderOption, model: str) -> bool:
        return model in provider.models or (provider.allow_custom_model and bool(model))

    def _footer(self, stdscr: CursesWindow, text: str) -> None:
        height, width = stdscr.getmaxyx()
        self._add(stdscr, height - 1, 0, " " * max(1, width), width, self._attr("selected"))
        self._add(stdscr, height - 1, 2, text, width - 4, self._attr("selected"))

    def _add(
        self,
        stdscr: CursesWindow,
        y: int,
        x: int,
        text: str,
        width: int,
        attr: int = 0,
    ) -> None:
        height, max_width = stdscr.getmaxyx()
        if y < 0 or y >= height or x >= max_width:
            return
        safe_width = max(0, min(width, max_width - x))
        if safe_width <= 0:
            return
        with suppress(curses.error):
            stdscr.addnstr(y, x, text, safe_width, attr)

    def _attr(self, name: str) -> int:
        if name == "accent" and self._accent_attr_name != "accent":
            name = self._accent_attr_name
        return self._colors.get(name, curses.A_NORMAL)

    def _is_escape(self, key: str | int) -> bool:
        return key == "\x1b" or key == 27

    def _is_ctrl_c(self, key: str | int) -> bool:
        return key == "\x03" or key == 3

    def _is_ctrl_x(self, key: str | int) -> bool:
        return key == "\x18" or key == 24

    def _is_ctrl_d(self, key: str | int) -> bool:
        return key == "\x04" or key == 4

    def _is_ctrl_e(self, key: str | int) -> bool:
        return key == "\x05" or key == 5

    def _is_ctrl_s(self, key: str | int) -> bool:
        return key == "\x13" or key == 19

    def _running_interrupt_key_label(self, key: str | int) -> str:
        if self._is_ctrl_c(key):
            return "Ctrl+C"
        if self._is_ctrl_x(key):
            return "Ctrl+X"
        return "Esc"

    def _is_shift_tab(self, key: str | int) -> bool:
        return key == "\x1b[Z" or key == getattr(curses, "KEY_BTAB", 353)

    def _is_option_left(self, key: str | int) -> bool:
        return key in {
            "\x1bb",
            "\x1bB",
            "\x1b[1;3D",
            "\x1b[1;9D",
            "\x1b[3D",
        }

    def _is_option_right(self, key: str | int) -> bool:
        return key in {
            "\x1bf",
            "\x1bF",
            "\x1b[1;3C",
            "\x1b[1;9C",
            "\x1b[3C",
        }

    def _is_shift_enter(self, key: str | int) -> bool:
        return key in {"\x1b[13;2u", "\x1b[13;2~"}

    def _is_enter(self, key: str | int) -> bool:
        return key in {"\n", "\r", curses.KEY_ENTER}

    def _is_backspace(self, key: str | int) -> bool:
        return key in {"\b", "\x7f", curses.KEY_BACKSPACE, 127, 8}

    def _is_option_delete(self, key: str | int) -> bool:
        return key in {
            "\x1b\x7f",  # Option+Backspace in Terminal.app
            "\x1b\b",  # Option+Backspace in some terminals
            "\x17",  # Ctrl+W / word erase, often emitted by terminal profiles
            "\x1b[3;3~",  # Option+Delete/Backspace in iTerm2, many terminals
            "\x1b[3;5~",  # Ctrl+Delete (also common)
        }
