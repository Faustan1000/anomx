"""Constants for the Anomx terminal UI."""

from __future__ import annotations

import re

from anomx.agent.runtime import IMAGE_FILE_EXTENSIONS
from anomx.agent.ui.models import CommandSpec

PROMPT_PLACEHOLDERS = (
    "Let's find the anomalies in the data of this repo",
    "Inspect this repository and map the data pipeline",
    "Type /config to setup things like sandbox, skills or generic instructions",
    "Look for time-series signals that need anomaly detection",
    "Summarize the datasets and suggest detection jobs",
)

COMMANDS = (
    CommandSpec("/new", "Start a new session"),
    CommandSpec("/rename", "Rename the current session"),
    CommandSpec("/config", "Edit configuration"),
    CommandSpec("/model", "Change model"),
    CommandSpec("/effort", "Change reasoning effort"),
    CommandSpec("/feedback", "Send feedback to the connected Anomx Platform"),
    CommandSpec("/exit", "Exit Anomx"),
)
PROJECT_COMMANDS = frozenset({"/new", "/config", "/model", "/effort", "/feedback", "/exit"})
RUNNING_SESSION_COMMANDS = frozenset({"/config", "/model", "/effort"})

RUNNING_NOTICE = (
    "Agent is working · messages blocked · Esc back · Ctrl+C/Ctrl+X interrupt"
)
RUNNING_MESSAGE_BLOCKED_NOTICE = (
    "Agent is working · sending messages is blocked until the response finishes"
)
RUNNING_COMMAND_BLOCKED_NOTICE = (
    "Agent is working · this slash command is unavailable until the response finishes"
)
ABORT_AGENT_CONFIRM_NOTICE = (
    "Do you really want to abort the agent? Press {key_label} again to confirm."
)
INTERRUPTED_AGENT_NOTICE = "You have interrupted anomx."
MANUAL_INTERRUPT_MESSAGE = "You interrupted anomx manually"
EXIT_ANOMX_CONFIRM_NOTICE = "Do you really want to exit anomx? Press Ctrl+C again to confirm."
TABLE_BORDER_CHARS = frozenset("│┌┬┐├┼┤└┴┘─")
ACTIVITY_DETAIL_MAX_LINES = 10
RAW_MOUSE_RE = re.compile(r"^\x1b\[<(?P<button>\d+);(?P<x>\d+);(?P<y>\d+)(?P<suffix>[mM])$")
RAW_MOUSE_SUFFIX_RE = re.compile(r"^\[<\d+;\d+;\d+[mM]$")
FILE_REFERENCE_LIMIT = 10
FILE_REFERENCE_INDEX_REFRESH_SECONDS = 30.0
FILE_REFERENCE_SCAN_LIMIT = 5000
FILE_REFERENCE_CACHE_SECONDS = 2.0
FILE_REFERENCE_FIRST_LEVEL_LIMIT = 200
IMAGE_DROP_EXTENSION_PATTERN = "|".join(re.escape(ext) for ext in IMAGE_FILE_EXTENSIONS)
IMAGE_DROP_CANDIDATE_PATTERN = re.compile(
    rf"(?P<path>(?:file://)?(?:~|/)[^\r\n]*?(?:{IMAGE_DROP_EXTENSION_PATTERN}))"
    r"(?=$|[\s\"'`),;!?])",
    re.IGNORECASE,
)
STARTUP_PHASE_SECONDS = 2.0
STARTUP_LOADING_SECONDS = STARTUP_PHASE_SECONDS * 3
STARTUP_REVEAL_SECONDS = 1.2
STARTUP_OVERLAY_DELAY_SECONDS = 0.35
STARTUP_LINE_REVEAL_SECONDS = 2.15
STARTUP_BRAND_REVEAL_SECONDS = 1.9
STARTUP_WIPE_SECONDS = 0.7
STARTUP_FRAME_SECONDS = 0.065
STARTUP_COLUMN_WIDTH = 1
STARTUP_MATRIX_ALPHABET = "0123456789"  # "αβγδεζηθικλμνξοπρσςτυφχψω"
START_HINT_REVEAL_SECONDS = 1.0
START_HINT_WIPE_SECONDS = 0.45
PLAN_STEP_REVEAL_SECONDS = 1.0
BRAND_NAME = "Anomx"
BRAND_DOT = "."
AGENT_DESCRIPTOR = "Data Analysis and Anomaly Detection Agent"
STARTUP_ANOMX_GLYPH = (
    "  #    #   #   ###   #   #  #   #",
    " # #   ##  #  #   #  ## ##  #   #",
    "#   #  # # #  #   #  # # #   # # ",
    "#####  #  ##  #   #  #   #    #  ",
    "#   #  #   #  #   #  #   #   # # ",
    "#   #  #   #  #   #  #   #  #   #",
    "#   #  #   #   ###   #   #  #   #",
)
IGNORED_FILE_REFERENCE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "out",
        "venv",
    }
)
