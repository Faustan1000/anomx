"""Shared data structures for the Anomx terminal UI."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.tool_manager import ApprovalChoice, CommandApprovalRequest
from anomx.agent.runtime import AgentRuntime, QuestionRequest, QuestionResponse
from anomx.agent.runtime_process import RuntimeProcessClient
from anomx.agent.store import ProjectRecord, SessionRecord


class AgentState(StrEnum):
    """Top-level CLI states."""

    ONBOARDING = "Onboarding"
    VERSION_CHECK = "Version Check"
    ACCESS_CHECK = "Access Check"
    PROJECT = "Project"
    CONTINUE_SESSION = "Continue Session"
    NEW_SESSION = "New Session"
    OPEN_SESSION = "Open Session"
    CONFIG = "Config"
    MODEL = "Model"
    INFO = "Info"
    SKILLS = "Skills"
    EXIT = "Exit"


class CursesWindow(Protocol):
    """Subset of curses window methods used by the UI."""

    def erase(self) -> None: ...
    def refresh(self) -> None: ...
    def getmaxyx(self) -> tuple[int, int]: ...
    def addnstr(self, y: int, x: int, text: str, n: int, attr: int = 0) -> None: ...
    def get_wch(self) -> str | int: ...
    def keypad(self, flag: bool) -> None: ...
    def move(self, y: int, x: int) -> None: ...
    def clrtoeol(self) -> None: ...
    def bkgd(self, ch: str, attr: int = 0) -> None: ...
    def nodelay(self, flag: bool) -> None: ...


@dataclass(frozen=True)
class MenuChoice:
    """Full-screen menu row, optionally used as a non-selectable section label."""

    label: str
    value: str
    detail: str = ""
    highlight: str = ""
    highlight_spans: tuple[tuple[int, int], ...] = ()
    selectable: bool = True


@dataclass(frozen=True)
class MessageLine:
    """Rendered session transcript line."""

    role: str
    text: str
    meta: str = ""
    expansion_key: str = dataclass_field(default="", compare=False)
    detail_title: str = dataclass_field(default="", compare=False)
    detail_body: str = dataclass_field(default="", compare=False)
    activity_wave: bool = dataclass_field(default=False, compare=False)


@dataclass(frozen=True)
class CommandSpec:
    """Prompt command entry."""

    command: str
    description: str


@dataclass(frozen=True)
class InfoRow:
    """A label/value row shown on the session info screen."""

    label: str
    value: str


@dataclass(frozen=True)
class SkillFormDraft:
    """Editable skill form state."""

    command: str = ""
    description: str = ""
    body: str = ""
    path: Path | None = None
    command_cursor: int = 0
    description_cursor: int = 0
    body_cursor: int = 0


@dataclass(frozen=True)
class PlatformConnectionDraft:
    """Editable platform connection form state."""

    url: str = ""
    email: str = ""
    password: str = ""


@dataclass(frozen=True)
class BottomPanel:
    """Inline command panel rendered above the prompt bar."""

    title: str
    subtitle: str
    choices: tuple[MenuChoice, ...]
    selected: int
    frame_attr: str = "accent"
    title_attr: str = "accent"
    title_prefix: str = ""
    title_prefix_attr: str = "accent"
    title_suffix: str = ""
    title_suffix_attr: str = "light"
    subtitle_attr: str = "light"
    subtitle_max_lines: int = 4
    subtitle_scroll: int = 0
    choice_attr: str = ""
    selected_choice_attr: str = "accent"
    highlight_attr: str = "accent"
    selected_highlight_attr: str = "selected"
    detail_attr: str = "light"


@dataclass(frozen=True)
class BottomPanelViewport:
    """Visible bottom-panel rows for the current selection."""

    start_y: int
    subtitle_lines: tuple[str, ...]
    choice_y: int
    visible_indices: tuple[int, ...]
    more_above: int
    more_below: int
    show_overflow_counts: bool


@dataclass(frozen=True)
class RuntimeUiEvent:
    """Event emitted by the runtime worker for the TUI loop."""

    kind: str
    text: str = ""
    role: str = ""
    command: str = ""
    approval_request: CommandApprovalRequest | None = None
    approval_response: queue.SimpleQueue[ApprovalChoice] | None = None
    question_request: QuestionRequest | None = None
    question_response: queue.SimpleQueue[QuestionResponse] | None = None


@dataclass
class ActiveSessionTurn:
    """A model turn that may keep running outside the focused session view."""

    session: SessionRecord
    runtime: AgentRuntime | RuntimeProcessClient
    events: queue.SimpleQueue[RuntimeUiEvent]
    result: dict[str, str]
    turn_id: str
    started_at: float
    worker: threading.Thread
    mode: AgentMode
    working_text: str | None = "Thinking"
    working_deadline: float | None = None
    final_text: str = ""
    work_count: int = 0
    work_summary_appended: bool = False
    streamed_final_displayed: bool = False
    completed: bool = False
    agent_symbol: str = ""
    anchor_expansion_key: str = ""
    pending_events: list[RuntimeUiEvent] = dataclass_field(default_factory=list)


@dataclass(frozen=True)
class StartupPreparation:
    """Initial session state prepared while the startup animation is visible."""

    previous_session: SessionRecord | None = None
    continuation_statement: str = ""
    project: ProjectRecord | None = None
    sandbox_runtime: str | None = None


@dataclass(frozen=True)
class PromptLayout:
    """Prompt bar geometry."""

    top_line: int
    prompt_line: int
    bottom_line: int
    hint_line: int
    input_x: int
    input_width: int
    prompt_height: int


@dataclass(frozen=True)
class PromptPasteSpan:
    """Real prompt text range that should render as a pasted-content marker."""

    start: int
    end: int
    character_count: int | None = None


@dataclass(frozen=True)
class PromptPasteEvent:
    """Bracketed paste payload read from the terminal."""

    text: str


@dataclass(frozen=True)
class SessionViewportState:
    """Resolved transcript viewport state for the current frame."""

    start: int
    scroll: int
    body_height: int
    rendered_line_count: int


@dataclass(frozen=True)
class SessionMouseAction:
    """Mouse action resolved from a curses mouse event."""

    kind: str
    value: int
    text: str = ""
    x_start: int = 0
    x_end: int = 0


@dataclass(frozen=True)
class SessionTextRow:
    """A rendered transcript row that can be selected with the mouse."""

    line_index: int
    y: int
    x: int
    width: int
    text: str


@dataclass(frozen=True)
class SessionSelectionPoint:
    """A transcript text coordinate."""

    line_index: int
    column: int


@dataclass(frozen=True)
class SessionTextSelection:
    """Active or retained transcript text selection."""

    anchor: SessionSelectionPoint
    focus: SessionSelectionPoint


@dataclass(frozen=True)
class BackendTurnResult:
    """State returned from a non-blocking backend turn."""

    input_text: str
    cursor: int
    exit_requested: bool = False
    anchor_line: int | None = None
    scroll: int = 0
    back_requested: bool = False


@dataclass(frozen=True)
class RunningKeyResult:
    """Input state update while a backend turn is active."""

    input_text: str
    cursor: int
    notice: str
    notice_role: str
    abort_key: str
    abort_deadline: float
    command_selected: int = 0
    command: str = ""
    submitted: str = ""
    submitted_message: str = ""
    exit_requested: bool = False
    scroll_delta: int = 0
    clear_anchor: bool = False
    back_requested: bool = False


@dataclass(frozen=True)
class ActivityDetailEntry:
    """A clickable line inside an expanded activity item."""

    key: str
    text: str
    detail_body: str = ""


@dataclass(frozen=True)
class ActivityDetailRow:
    """A rendered row inside an expanded activity item."""

    text: str
    role: str = "activity_detail"
    entry_key: str = ""


@dataclass(frozen=True)
class ActivityItem:
    """A bottom activity-panel row for workers and async processes."""

    key: str
    title: str
    right_text: str
    details: tuple[ActivityDetailEntry, ...]
    active: bool = False
    kill_process_id: str = ""
    marker: str = "•"
    kind: str = "process"
    badge: str = ""
    title_suffix: str = ""
    accent: str = "light"
    open_agent_id: str = ""
