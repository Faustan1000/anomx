"""CLI tool safety manager for the Anomx agent."""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, MutableSet, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from anomx.agent.helpers.mode import AgentMode


class CommandSafety(StrEnum):
    """Command safety class."""

    ALLOW = "allow"
    APPROVE = "approve"
    FORBIDDEN = "forbidden"


class ApprovalChoice(StrEnum):
    """User decision for a command approval request."""

    REJECT = "reject"
    ALLOW = "allow"
    ALWAYS_ALLOW = "always_allow"
    ALWAYS_REJECT = "always_reject"


@dataclass(frozen=True)
class CommandRiskEvaluation:
    """AI-generated risk summary for a pending command approval request."""

    risk: str
    description: str


@dataclass(frozen=True)
class CommandPolicy:
    """Safety decision for a CLI command."""

    safety: CommandSafety
    reason: str
    canonical_command: str
    allowance_key: str = ""
    allowance_label: str = ""
    allowance_subject: str = ""


@dataclass(frozen=True)
class CommandApprovalRequest:
    """Approval request shown in the chat UI."""

    command: str
    statement: str
    reason: str
    canonical_command: str
    allowance_key: str = ""
    allowance_label: str = ""
    allowance_subject: str = ""
    evaluation: CommandRiskEvaluation | None = None
    agent_id: str = ""
    agent_name: str = ""


@dataclass(frozen=True)
class CommandResult:
    """Result returned to the model after a command tool call."""

    output: str
    approved: bool
    safety: CommandSafety
    command: str
    reason: str
    blocked_by_mode: bool = False


@dataclass(frozen=True)
class CommandProcessResult:
    """Result returned after preparing or starting an async command process."""

    process: subprocess.Popen[str] | None
    output: str
    approved: bool
    safety: CommandSafety
    command: str
    reason: str
    blocked_by_mode: bool = False


ApprovalCallback = Callable[[CommandApprovalRequest], ApprovalChoice]
LongRunningCommandCallback = Callable[[subprocess.Popen[str]], str | None]
LONG_RUNNING_COMMAND_SECONDS = 2.0

ALLOW_COMMANDS = (
    "pwd",
    "ls",
    "cd <path inside workspace>",
    "cat",
    "grep",
    "head",
    "tail",
    "wc",
    "which",
    "whoami",
    "echo",
    "find without -exec/-delete/-fprint",
    "rg without --pre/--search-zip",
    "sed -n <range>p",
    "git status/log/diff/show/branch",
)

APPROVE_COMMANDS = (
    "rm",
    "rmdir",
    "unlink",
    "truncate",
    "sh",
    "bash",
    "zsh",
    "python",
    "python3",
    "pytest",
    "pip",
    "npm",
    "pnpm",
    "yarn",
    "make",
    "docker",
    "touch",
    "mkdir",
    "cp",
    "mv",
    "curl",
    "unknown commands",
)

SERIOUS_COMMANDS = (
    "shred",
    "sudo",
    "su",
    "chmod",
    "chown",
    "reboot",
    "shutdown",
    "halt",
    "poweroff",
    "kill",
    "killall",
    "pkill",
    "launchctl",
    "diskutil",
    "mkfs",
    "dd",
    "osascript",
    "systemctl",
    "service",
    "mount",
    "umount",
    "crontab",
)

SANDBOX_SERIOUS_COMMANDS = (
    "git",
    "svn",
    "hg",
    "cvs",
)

SANDBOX_SERIOUS_COMMAND_NAMES = frozenset(SANDBOX_SERIOUS_COMMANDS)

SHELL_METACHARS = frozenset({"&", ";", ">", "<", "`", "$", "\n"})
PIPE_OPERATOR = "|"
APPROVAL_COMMAND_NAMES = frozenset(APPROVE_COMMANDS)
SERIOUS_COMMAND_NAMES = frozenset(SERIOUS_COMMANDS)
READ_ONLY_COMMAND_NAMES = frozenset(
    {
        "cat",
        "cut",
        "echo",
        "expr",
        "false",
        "grep",
        "head",
        "id",
        "ls",
        "nl",
        "paste",
        "pwd",
        "rev",
        "seq",
        "sort",
        "stat",
        "tail",
        "tr",
        "true",
        "uname",
        "uniq",
        "wc",
        "which",
        "whoami",
    }
)
ALLOW_GIT_SUBCOMMANDS = frozenset({"status", "log", "diff", "show", "branch"})
READ_ONLY_GIT_BRANCH_FLAGS = frozenset(
    {
        "--list",
        "-l",
        "--show-current",
        "-a",
        "--all",
        "-r",
        "--remotes",
        "-v",
        "-vv",
        "--verbose",
    }
)
UNSAFE_FIND_OPTIONS = frozenset(
    {
        "-exec",
        "-execdir",
        "-ok",
        "-okdir",
        "-delete",
        "-fls",
        "-fprint",
        "-fprint0",
        "-fprintf",
    }
)
UNSAFE_RG_OPTIONS = frozenset({"--pre", "--hostname-bin", "--search-zip", "-z"})
SED_PRINT_ONLY_RE = re.compile(r"^(?:\d+|/.+/)(?:,(?:\d+|/.+/))?p$")
PATTERN_VALUE_OPTIONS = frozenset(
    {
        "-e",
        "--expression",
        "--regexp",
    }
)
PATH_VALUE_OPTIONS = frozenset(
    {
        "-f",
        "--file",
        "--files-from",
    }
)
NON_PATH_VALUE_OPTIONS = frozenset(
    {
        "-A",
        "-B",
        "-C",
        "-D",
        "-M",
        "-d",
        "-g",
        "-m",
        "-t",
        "--after-context",
        "--before-context",
        "--binary-files",
        "--color",
        "--colors",
        "--context",
        "--context-separator",
        "--encoding",
        "--engine",
        "--field-context-separator",
        "--field-match-separator",
        "--glob",
        "--glob-case-insensitive",
        "--heading",
        "--iglob",
        "--ignore-file-case-insensitive",
        "--json-seq",
        "--max-columns",
        "--max-count",
        "--max-depth",
        "--max-filesize",
        "--mmap",
        "--multiline-dotall",
        "--passthru",
        "--path-separator",
        "--sort",
        "--sort-files",
        "--threads",
        "--type",
        "--type-add",
        "--type-clear",
        "--type-list",
    }
)
COMMAND_TIMEOUT_SECONDS = 300
MAX_COMMAND_OUTPUT_ROWS = 400
VCS_ROOT_MARKERS = (".git", ".hg")
PROJECT_ROOT_MARKERS = (
    "pyproject.toml",
    "package.json",
    "pnpm-workspace.yaml",
    "Cargo.toml",
    "go.mod",
)
SUBCOMMAND_ALLOWANCE_COMMANDS = frozenset(
    {
        "docker",
        "git",
        "make",
        "npm",
        "pip",
        "pip3",
        "pnpm",
        "yarn",
    }
)
SCRIPT_RUNNERS = frozenset({"npm", "pnpm", "yarn"})
OPTION_VALUE_ALLOWANCE_FLAGS = frozenset({"-m", "--module"})


@dataclass(frozen=True)
class CommandAllowanceDisplay:
    """Human-readable parts of a persisted command allowance key."""

    command: str
    parameters: str
    subject: str


def discover_workspace_root(start: Path) -> Path:
    """Return the trusted project root for a launch directory."""

    resolved = start.expanduser().resolve()
    if resolved.is_file():
        resolved = resolved.parent

    search_paths = (resolved, *resolved.parents)
    for path in search_paths:
        if any((path / marker).exists() for marker in VCS_ROOT_MARKERS):
            return path
    for path in search_paths:
        if any((path / marker).exists() for marker in PROJECT_ROOT_MARKERS):
            return path
    return resolved


def command_allowance_display(key: str) -> CommandAllowanceDisplay:
    """Return display parts for a persisted command allowance key."""

    subject = key.removeprefix("cmd:").strip() if key.startswith("cmd:") else key.strip()
    if not subject:
        return CommandAllowanceDisplay("this command", "none", "this command")

    with suppress(ValueError):
        parts = shlex.split(subject)
        if parts:
            command = parts[0]
            parameters = " ".join(parts[1:]) if len(parts) > 1 else "none"
            return CommandAllowanceDisplay(command, parameters, subject)

    parts = subject.split(maxsplit=1)
    command = parts[0] if parts else subject
    parameters = parts[1] if len(parts) > 1 else "none"
    return CommandAllowanceDisplay(command, parameters, subject)


class CliToolManager:
    """Classify, approve, and execute CLI commands inside a trusted workspace."""

    def __init__(
        self,
        root: Path,
        session_allowed_commands: MutableSet[str] | None = None,
        session_rejected_commands: MutableSet[str] | None = None,
        mode: AgentMode = AgentMode.CONFIRM,
        *,
        current_dir: Path | None = None,
        cancel_event: threading.Event | None = None,
        subprocess_env: Mapping[str, str] | None = None,
        strict_workspace: bool = False,
        trusted_roots: Sequence[Path] | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.trusted_roots = self._normalize_trusted_roots(trusted_roots)
        self.current_dir = (
            self.root if current_dir is None else current_dir.expanduser().resolve()
        )
        if not self._inside_workspace(self.current_dir):
            self.current_dir = self.root
        self.session_allowed_commands = session_allowed_commands
        self.session_rejected_commands = session_rejected_commands
        self.mode = mode
        self.cancel_event = cancel_event
        self.subprocess_env = dict(subprocess_env) if subprocess_env is not None else None
        self.strict_workspace = strict_workspace

    def set_mode(self, mode: AgentMode) -> None:
        """Set the active command execution mode."""

        self.mode = mode

    def set_trusted_roots(self, trusted_roots: Sequence[Path] | None) -> None:
        """Replace the trusted roots used by strict workspace checks."""

        self.trusted_roots = self._normalize_trusted_roots(trusted_roots)
        if not self._inside_workspace(self.current_dir):
            self.current_dir = self.root

    def resolve_trusted_path(self, raw_path: str) -> Path:
        """Resolve a path using the same HOME and workspace rules as commands."""

        return self._resolve_path(raw_path)

    def path_inside_workspace(self, path: Path) -> bool:
        """Return whether a path is inside one of the trusted workspace roots."""

        return self._inside_workspace(path)

    def run_command(
        self,
        command: str,
        statement: str,
        approval_callback: ApprovalCallback | None,
        long_running_callback: LongRunningCommandCallback | None = None,
    ) -> CommandResult:
        """Run a command after policy checks and optional user approval."""

        authorization = self._authorize_command(command, statement, approval_callback)
        if isinstance(authorization, CommandResult):
            return authorization

        policy = authorization
        return CommandResult(
            self._execute(command, long_running_callback=long_running_callback),
            approved=True,
            safety=policy.safety,
            command=policy.canonical_command,
            reason=policy.reason,
        )

    def start_process(
        self,
        command: str,
        statement: str,
        approval_callback: ApprovalCallback | None,
    ) -> CommandProcessResult:
        """Start a long-running command after policy checks and optional approval."""

        authorization = self._authorize_command(command, statement, approval_callback)
        if isinstance(authorization, CommandResult):
            return CommandProcessResult(
                process=None,
                output=authorization.output,
                approved=authorization.approved,
                safety=authorization.safety,
                command=authorization.command,
                reason=authorization.reason,
                blocked_by_mode=authorization.blocked_by_mode,
            )

        policy = authorization
        try:
            process = self._start_subprocess(command)
        except (OSError, ValueError) as error:
            return CommandProcessResult(
                process=None,
                output=f"Process could not be started: {error}",
                approved=False,
                safety=policy.safety,
                command=policy.canonical_command,
                reason=policy.reason,
            )
        return CommandProcessResult(
            process=process,
            output="Process started.",
            approved=True,
            safety=policy.safety,
            command=policy.canonical_command,
            reason=policy.reason,
        )

    def _authorize_command(
        self,
        command: str,
        statement: str,
        approval_callback: ApprovalCallback | None,
    ) -> CommandPolicy | CommandResult:
        """Return an executable policy or an immediate denial result."""

        policy = self.classify(
            command,
            include_session_allowances=True,
        )
        return self._authorize_policy(
            policy,
            policy.canonical_command,
            statement,
            approval_callback,
        )

    def _authorize_policy(
        self,
        policy: CommandPolicy,
        command: str,
        statement: str,
        approval_callback: ApprovalCallback | None,
    ) -> CommandPolicy | CommandResult:
        """Apply the active mode and optional user approval to a classified policy."""

        if policy.safety == CommandSafety.FORBIDDEN:
            return CommandResult(
                self._user_blocked_output(policy.reason),
                approved=False,
                safety=policy.safety,
                command=policy.canonical_command,
                reason=policy.reason,
            )

        serious_token = self._serious_token_in_command(policy.canonical_command)
        if (
            self.mode == AgentMode.AUTONOMOUS
            and policy.safety == CommandSafety.APPROVE
            and serious_token is not None
        ):
            reason = f"{serious_token} can modify or control the host system."
            return CommandResult(
                self._user_blocked_output(reason),
                approved=False,
                safety=CommandSafety.FORBIDDEN,
                command=policy.canonical_command,
                reason=reason,
                blocked_by_mode=True,
            )

        if self._mode_allows_policy(policy):
            policy = CommandPolicy(
                CommandSafety.ALLOW,
                (
                    f"{self.mode.label} allowed command that would normally require approval: "
                    f"{policy.reason}"
                ),
                policy.canonical_command,
                policy.allowance_key,
                policy.allowance_label,
                policy.allowance_subject,
            )

        if policy.safety == CommandSafety.APPROVE:
            if approval_callback is None:
                return CommandResult(
                    "Command requires approval.",
                    approved=False,
                    safety=policy.safety,
                    command=policy.canonical_command,
                    reason=policy.reason,
                )
            decision = approval_callback(
                CommandApprovalRequest(
                    command=command,
                    statement=statement,
                    reason=policy.reason,
                    canonical_command=policy.canonical_command,
                    allowance_key=policy.allowance_key,
                    allowance_label=policy.allowance_label,
                    allowance_subject=policy.allowance_subject,
                )
            )
            if decision == ApprovalChoice.REJECT:
                return CommandResult(
                    self._user_blocked_output(
                        "The user rejected this command. Do not retry it unchanged.",
                    ),
                    approved=False,
                    safety=policy.safety,
                    command=policy.canonical_command,
                    reason=policy.reason,
                )
            if decision == ApprovalChoice.ALWAYS_REJECT:
                if self.session_rejected_commands is not None:
                    self.session_rejected_commands.add(
                        policy.allowance_key or policy.canonical_command
                    )
                reason = self._session_rejection_reason(
                    policy.allowance_key or policy.canonical_command
                )
                return CommandResult(
                    self._user_blocked_output(reason),
                    approved=False,
                    safety=CommandSafety.FORBIDDEN,
                    command=policy.canonical_command,
                    reason=reason,
                )
            if (
                decision == ApprovalChoice.ALWAYS_ALLOW
                and self.session_allowed_commands is not None
            ):
                self.session_allowed_commands.add(
                    policy.allowance_key or policy.canonical_command
                )

        return policy

    def _user_blocked_output(self, reason: str) -> str:
        return (
            "The user does not allow you to do this. "
            "Do not retry this command or command family unchanged. "
            f"Reason: {reason}"
        )

    def _mode_allows_policy(self, policy: CommandPolicy) -> bool:
        """Return whether the active mode auto-allows an approval policy."""

        if policy.safety != CommandSafety.APPROVE:
            return False
        serious_token = self._serious_token_in_command(policy.canonical_command)
        if serious_token is not None:
            return False
        if self.mode == AgentMode.SANDBOX:
            return self._sandbox_serious_token_in_command(policy.canonical_command) is None
        if self.mode == AgentMode.AUTONOMOUS:
            return True
        if self._contains_approval_only_shell_syntax(policy.canonical_command):
            return False
        if self.mode == AgentMode.AUTO:
            return False
        return False

    def run_cli_command(
        self,
        command: str,
        statement: str,
        approval_callback: ApprovalCallback | None,
    ) -> CommandResult:
        """Compatibility alias for older agent tool calls."""

        return self.run_command(command, statement, approval_callback)

    def classify(
        self,
        command: str,
        *,
        include_session_allowances: bool = True,
    ) -> CommandPolicy:
        """Classify a command as auto-allow, approval-required, or forbidden."""

        normalized = self._normalize_command(command)
        if not normalized:
            return CommandPolicy(CommandSafety.FORBIDDEN, "Empty command.", normalized)

        if self.mode == AgentMode.SANDBOX:
            return self._classify_sandbox(normalized)

        policy_source = self._strip_heredoc_bodies(normalized)
        if self._session_rejects_command(normalized, include_session_allowances):
            return CommandPolicy(
                CommandSafety.FORBIDDEN,
                self._session_rejection_reason(self._allowance_key(normalized) or normalized),
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )
        serious_token = self._serious_token_in_command(normalized)
        if serious_token is not None:
            return CommandPolicy(
                CommandSafety.APPROVE,
                f"{serious_token} can modify or control the host system.",
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )
        if self._has_pipe_operator(policy_source):
            policy = self._classify_pipeline(policy_source, include_session_allowances)
            return CommandPolicy(
                policy.safety,
                policy.reason,
                normalized,
                policy.allowance_key,
                policy.allowance_label,
                policy.allowance_subject,
            )
        if self._session_allows_command(normalized, include_session_allowances):
            path_error = self._allowanced_shell_path_error(normalized)
            if path_error is not None:
                return self._path_approval_policy(path_error, normalized)
            return CommandPolicy(
                CommandSafety.ALLOW,
                "Allowed command family for this session.",
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )
        if self._has_shell_syntax(policy_source):
            return self._classify_shell_compound(normalized, include_session_allowances)

        try:
            parts = shlex.split(normalized)
        except ValueError as error:
            return CommandPolicy(CommandSafety.FORBIDDEN, str(error), normalized)
        if not parts:
            return CommandPolicy(CommandSafety.FORBIDDEN, "Empty command.", normalized)

        executable = Path(parts[0]).name
        path_error = self._path_error(parts)
        if path_error is not None:
            return self._path_approval_policy(path_error, normalized)

        if executable in SERIOUS_COMMAND_NAMES:
            return CommandPolicy(
                CommandSafety.APPROVE,
                f"{executable} can modify or control the host system.",
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )

        if self._session_allows_command(normalized, include_session_allowances):
            return CommandPolicy(
                CommandSafety.ALLOW,
                "Allowed command family for this session.",
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )

        if executable == "cd":
            return self._classify_cd(parts, normalized)
        if self._is_known_read_only_command(executable, parts):
            return CommandPolicy(
                CommandSafety.ALLOW,
                "Known read-only command.",
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )
        if executable in APPROVAL_COMMAND_NAMES:
            return CommandPolicy(
                CommandSafety.APPROVE,
                f"{executable} may read, compute, install, or modify files.",
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )
        return CommandPolicy(
            CommandSafety.APPROVE,
            f"{executable} is not in the automatic allow list.",
            normalized,
            self._allowance_key(normalized),
            self._allowance_label(normalized),
            self._allowance_subject(normalized),
        )

    def _execute(
        self,
        command: str,
        *,
        long_running_callback: LongRunningCommandCallback | None = None,
    ) -> str:
        normalized = self._normalize_command(command)
        policy_source = self._strip_heredoc_bodies(normalized)
        if self._has_pipe_operator(policy_source):
            if (
                policy_source == normalized
                and self._classify_pipeline(policy_source).safety == CommandSafety.ALLOW
            ):
                return self._execute_pipeline(
                    normalized,
                    long_running_callback=long_running_callback,
                )
            return self._execute_shell_command(
                normalized,
                long_running_callback=long_running_callback,
            )
        if self._has_shell_syntax(normalized):
            return self._execute_shell_command(
                normalized,
                long_running_callback=long_running_callback,
            )
        parts = shlex.split(normalized)
        if parts[0] == "cd":
            target = self._resolve_path(parts[1] if len(parts) > 1 else ".")
            self.current_dir = target
            return str(self.current_dir)
        output = self._execute_subprocess(
            parts,
            long_running_callback=long_running_callback,
        )
        assert isinstance(output, str)
        return output

    def _classify_cd(self, parts: list[str], normalized: str) -> CommandPolicy:
        if len(parts) > 2:
            return CommandPolicy(
                CommandSafety.FORBIDDEN,
                "cd accepts at most one path.",
                normalized,
            )
        target = self._resolve_path(parts[1] if len(parts) == 2 else ".")
        if not target.exists() or not target.is_dir():
            return CommandPolicy(
                CommandSafety.FORBIDDEN,
                "cd target is not a directory.",
                normalized,
            )
        return CommandPolicy(CommandSafety.ALLOW, "Directory stays inside workspace.", normalized)

    def _classify_pipeline(
        self,
        normalized: str,
        include_session_allowances: bool = True,
    ) -> CommandPolicy:
        segments = self._pipeline_segments(normalized)
        if len(segments) < 2:
            return CommandPolicy(
                CommandSafety.APPROVE,
                "Shell pipe requires explicit approval.",
                normalized,
            )

        policies = [
            self.classify(
                segment,
                include_session_allowances=include_session_allowances,
            )
            for segment in segments
        ]
        forbidden = next(
            (policy for policy in policies if policy.safety == CommandSafety.FORBIDDEN),
            None,
        )
        if forbidden is not None:
            return CommandPolicy(CommandSafety.FORBIDDEN, forbidden.reason, normalized)
        if all(policy.safety == CommandSafety.ALLOW for policy in policies):
            return CommandPolicy(CommandSafety.ALLOW, "Known read-only pipeline.", normalized)
        return CommandPolicy(
            CommandSafety.APPROVE,
            "Pipeline includes commands that require approval.",
            normalized,
        )

    def _classify_shell_compound(
        self,
        normalized: str,
        include_session_allowances: bool = True,
    ) -> CommandPolicy:
        policy_source = self._strip_heredoc_bodies(normalized)
        path_error = self._allowanced_shell_path_error(policy_source)
        if path_error is not None:
            return self._path_approval_policy(path_error, normalized)

        unsafe_redirection = self._has_unsafe_redirection(policy_source)
        stripped = self._strip_null_redirections(policy_source)
        segments = self._shell_compound_segments(stripped)
        if len(segments) == 1 and segments[0] != policy_source:
            policy = self._classify_compound_segment(
                segments[0],
                include_session_allowances=include_session_allowances,
            )
            if policy.safety == CommandSafety.ALLOW:
                return CommandPolicy(
                    CommandSafety.ALLOW,
                    "Known read-only command with safe redirection.",
                    normalized,
                    self._allowance_key(normalized),
                    self._allowance_label(normalized),
                    self._allowance_subject(normalized),
                )
            return CommandPolicy(
                policy.safety,
                policy.reason,
                normalized,
                policy.allowance_key,
                policy.allowance_label,
                policy.allowance_subject,
            )

        if len(segments) >= 2:
            policies = [
                self._classify_compound_segment(
                    segment,
                    include_session_allowances=include_session_allowances,
                )
                for segment in segments
            ]
            forbidden = next(
                (policy for policy in policies if policy.safety == CommandSafety.FORBIDDEN),
                None,
            )
            if forbidden is not None:
                return CommandPolicy(CommandSafety.FORBIDDEN, forbidden.reason, normalized)
            allowance_key, allowance_label, allowance_subject = (
                self._compound_allowance_metadata(normalized, policies)
            )
            if all(policy.safety == CommandSafety.ALLOW for policy in policies):
                return CommandPolicy(
                    CommandSafety.ALLOW,
                    "Known read-only shell compound.",
                    normalized,
                    allowance_key,
                    allowance_label,
                    allowance_subject,
                )
            reason = (
                "Shell redirection requires explicit approval."
                if unsafe_redirection
                else "Shell compound includes commands that require approval."
            )
            return CommandPolicy(
                CommandSafety.APPROVE,
                reason,
                normalized,
                allowance_key,
                allowance_label,
                allowance_subject,
            )

        if unsafe_redirection:
            return CommandPolicy(
                CommandSafety.APPROVE,
                "Shell redirection requires explicit approval.",
                normalized,
                self._allowance_key(policy_source),
                self._allowance_label(policy_source),
                self._allowance_subject(policy_source),
            )
        if len(segments) < 2:
            return CommandPolicy(
                CommandSafety.APPROVE,
                "Shell operators require explicit approval.",
                normalized,
                self._allowance_key(policy_source),
                self._allowance_label(policy_source),
                self._allowance_subject(policy_source),
            )

        return CommandPolicy(
            CommandSafety.APPROVE,
            "Shell operators require explicit approval.",
            normalized,
        )

    def _classify_compound_segment(
        self,
        command: str,
        *,
        include_session_allowances: bool = True,
    ) -> CommandPolicy:
        """Classify one executable segment without recursively splitting compounds."""

        normalized = self._normalize_command(command)
        if not normalized:
            return CommandPolicy(CommandSafety.FORBIDDEN, "Empty command.", normalized)
        if self._session_rejects_command(normalized, include_session_allowances):
            return CommandPolicy(
                CommandSafety.FORBIDDEN,
                self._session_rejection_reason(self._allowance_key(normalized) or normalized),
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )

        serious_token = self._serious_token_in_command(normalized)
        if serious_token is not None:
            return CommandPolicy(
                CommandSafety.APPROVE,
                f"{serious_token} can modify or control the host system.",
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )

        if self._has_pipe_operator(normalized):
            return self._classify_pipeline(normalized, include_session_allowances)

        if self._session_allows_command(normalized, include_session_allowances):
            path_error = self._allowanced_shell_path_error(normalized)
            if path_error is not None:
                return self._path_approval_policy(path_error, normalized)
            return CommandPolicy(
                CommandSafety.ALLOW,
                "Allowed command family for this session.",
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )

        if self._has_shell_syntax(normalized):
            path_error = self._allowanced_shell_path_error(normalized)
            if path_error is not None:
                return self._path_approval_policy(path_error, normalized)
            if self._has_unsafe_redirection(normalized):
                return CommandPolicy(
                    CommandSafety.APPROVE,
                    "Shell redirection requires explicit approval.",
                    normalized,
                    self._allowance_key(normalized),
                    self._allowance_label(normalized),
                    self._allowance_subject(normalized),
                )
            stripped = self._strip_null_redirections(normalized)
            if stripped != normalized:
                policy = self._classify_compound_segment(
                    stripped,
                    include_session_allowances=include_session_allowances,
                )
                return CommandPolicy(
                    policy.safety,
                    policy.reason,
                    normalized,
                    policy.allowance_key,
                    policy.allowance_label,
                    policy.allowance_subject,
                )
            return CommandPolicy(
                CommandSafety.APPROVE,
                "Shell operators require explicit approval.",
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )

        try:
            parts = shlex.split(normalized)
        except ValueError as error:
            return CommandPolicy(CommandSafety.FORBIDDEN, str(error), normalized)
        if not parts:
            return CommandPolicy(CommandSafety.FORBIDDEN, "Empty command.", normalized)

        executable = Path(parts[0]).name
        path_error = self._path_error(parts)
        if path_error is not None:
            return self._path_approval_policy(path_error, normalized)
        if executable in SERIOUS_COMMAND_NAMES:
            return CommandPolicy(
                CommandSafety.APPROVE,
                f"{executable} can modify or control the host system.",
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )
        if executable == "cd":
            return self._classify_cd(parts, normalized)
        if self._is_known_read_only_command(executable, parts):
            return CommandPolicy(
                CommandSafety.ALLOW,
                "Known read-only command.",
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )
        if executable in APPROVAL_COMMAND_NAMES:
            return CommandPolicy(
                CommandSafety.APPROVE,
                f"{executable} may read, compute, install, or modify files.",
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )
        return CommandPolicy(
            CommandSafety.APPROVE,
            f"{executable} is not in the automatic allow list.",
            normalized,
            self._allowance_key(normalized),
            self._allowance_label(normalized),
            self._allowance_subject(normalized),
        )

    def _compound_allowance_metadata(
        self,
        normalized: str,
        policies: list[CommandPolicy],
    ) -> tuple[str, str, str]:
        approval_policies = [
            policy for policy in policies if policy.safety == CommandSafety.APPROVE
        ]
        approval_keys = {
            policy.allowance_key
            for policy in approval_policies
            if policy.allowance_key.startswith("cmd:")
        }
        if len(approval_keys) == 1:
            key = next(iter(approval_keys))
            subject = self._session_policy_subject(key)
            return key, f"{subject} commands", subject
        return normalized, "this exact command", "this command"

    def _classify_sandbox(self, normalized: str) -> CommandPolicy:
        """Classify a command in sandbox mode.

        In sandbox mode, most commands are allowed. Only sandbox-serious
        commands (git, svn, etc.) and standard serious host-control commands
        require approval.
        """
        if self.strict_workspace:
            if self._has_shell_syntax(normalized):
                path_error = self._allowanced_shell_path_error(normalized)
                if path_error is not None:
                    return CommandPolicy(
                        CommandSafety.FORBIDDEN,
                        path_error,
                        normalized,
                        self._allowance_key(normalized),
                        self._allowance_label(normalized),
                        self._allowance_subject(normalized),
                    )
            with suppress(ValueError):
                path_error = self._path_error(shlex.split(normalized))
                if path_error is not None:
                    return CommandPolicy(
                        CommandSafety.FORBIDDEN,
                        path_error,
                        normalized,
                        self._allowance_key(normalized),
                        self._allowance_label(normalized),
                        self._allowance_subject(normalized),
                    )
        if self._session_rejects_command(normalized, include_session_allowances=True):
            return CommandPolicy(
                CommandSafety.FORBIDDEN,
                self._session_rejection_reason(self._allowance_key(normalized) or normalized),
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )
        sandbox_serious = self._sandbox_serious_token_in_command(normalized)
        if sandbox_serious is not None:
            return CommandPolicy(
                CommandSafety.APPROVE,
                f"{sandbox_serious} can modify version control history.",
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )
        serious_token = self._serious_token_in_command(normalized)
        if serious_token is not None:
            return CommandPolicy(
                CommandSafety.APPROVE,
                f"{serious_token} can modify or control the host system.",
                normalized,
                self._allowance_key(normalized),
                self._allowance_label(normalized),
                self._allowance_subject(normalized),
            )
        return CommandPolicy(
            CommandSafety.ALLOW,
            "Sandbox mode auto-allows this command.",
            normalized,
        )

    def _path_error(self, parts: list[str]) -> str | None:
        for part in self._path_candidate_arguments(parts):
            if part.startswith("-") or "://" in part:
                continue
            if self._is_null_redirection_target(part):
                continue
            path = self._path_from_command_argument(part)
            if path.is_absolute() or ".." in path.parts:
                resolved = self._resolve_path(part)
                if not self._inside_workspace(resolved):
                    return f"Path is outside the trusted workspace: {part}"
        return None

    def _path_approval_policy(self, reason: str, normalized: str) -> CommandPolicy:
        safety = CommandSafety.FORBIDDEN if self.strict_workspace else CommandSafety.APPROVE
        resolved_reason = (
            f"{reason} Strict sandbox mode does not permit path approvals."
            if self.strict_workspace
            else reason
        )
        return CommandPolicy(
            safety,
            resolved_reason,
            normalized,
            self._allowance_key(normalized),
            self._allowance_label(normalized),
            self._allowance_subject(normalized),
        )

    def _path_candidate_arguments(self, parts: list[str]) -> list[str]:
        """Return command arguments that are intended to name files or directories."""

        if not parts:
            return []

        executable = Path(parts[0]).name
        if executable == "sed":
            return self._sed_path_candidate_arguments(parts[1:])
        if executable in {"grep", "egrep", "fgrep", "rg"}:
            return self._search_path_candidate_arguments(parts[1:])
        return parts[1:]

    def _sed_path_candidate_arguments(self, arguments: list[str]) -> list[str]:
        candidates: list[str] = []
        saw_script = False
        index = 0
        while index < len(arguments):
            argument = arguments[index]
            if argument == "--":
                if not saw_script and index + 1 < len(arguments):
                    index += 2
                    saw_script = True
                else:
                    index += 1
                candidates.extend(arguments[index:])
                break
            if argument == "-n":
                index += 1
                continue
            if argument in PATTERN_VALUE_OPTIONS:
                index += 2
                saw_script = True
                continue
            if argument in PATH_VALUE_OPTIONS:
                if index + 1 < len(arguments):
                    candidates.append(arguments[index + 1])
                index += 2
                saw_script = True
                continue
            if argument.startswith("-e") and argument != "-e":
                saw_script = True
                index += 1
                continue
            if argument.startswith("-f") and argument != "-f":
                candidates.append(argument[2:])
                saw_script = True
                index += 1
                continue
            if argument.startswith("-"):
                index += 1
                continue
            if not saw_script:
                saw_script = True
                index += 1
                continue
            candidates.append(argument)
            index += 1
        return candidates

    def _search_path_candidate_arguments(self, arguments: list[str]) -> list[str]:
        candidates: list[str] = []
        positional: list[str] = []
        has_explicit_pattern = False
        index = 0
        while index < len(arguments):
            argument = arguments[index]
            if argument == "--":
                positional.extend(arguments[index + 1 :])
                break
            if argument in PATTERN_VALUE_OPTIONS:
                has_explicit_pattern = True
                index += 2
                continue
            if argument in PATH_VALUE_OPTIONS:
                if index + 1 < len(arguments):
                    candidates.append(arguments[index + 1])
                index += 2
                continue
            if argument in NON_PATH_VALUE_OPTIONS:
                index += 2
                continue
            if any(
                argument.startswith(f"{option}=")
                for option in PATTERN_VALUE_OPTIONS | NON_PATH_VALUE_OPTIONS
            ):
                if any(argument.startswith(f"{option}=") for option in PATTERN_VALUE_OPTIONS):
                    has_explicit_pattern = True
                index += 1
                continue
            if any(argument.startswith(f"{option}=") for option in PATH_VALUE_OPTIONS):
                candidates.append(argument.split("=", 1)[1])
                index += 1
                continue
            if argument.startswith("-"):
                index += 1
                continue
            positional.append(argument)
            index += 1

        if has_explicit_pattern:
            candidates.extend(positional)
        elif len(positional) > 1:
            candidates.extend(positional[1:])
        return candidates

    def _allowanced_shell_path_error(self, normalized: str) -> str | None:
        policy_source = self._strip_heredoc_bodies(normalized)
        if not self._has_shell_syntax(policy_source):
            return None

        command_segments = self._shell_segments(
            policy_source,
            split_operators=frozenset({";", "&&", "||", "|", "\n"}),
        )
        for segment in command_segments or [policy_source]:
            command_prefixes = self._shell_segments(
                segment,
                split_operators=frozenset(SHELL_METACHARS),
            )
            command_prefix = command_prefixes[0] if command_prefixes else ""
            if command_prefix:
                with suppress(ValueError):
                    path_error = self._path_error(shlex.split(command_prefix))
                    if path_error is not None:
                        return path_error

        for target in self._redirection_targets(policy_source):
            if self._is_null_redirection_target(target):
                continue
            path = self._path_from_command_argument(target)
            if path.is_absolute() or ".." in path.parts:
                resolved = self._resolve_path(target)
                if not self._inside_workspace(resolved):
                    return f"Path is outside the trusted workspace: {target}"
        return None

    def _redirection_targets(self, normalized: str) -> list[str]:
        targets: list[str] = []
        for start, end, operator in self._shell_operator_spans(normalized):
            if operator not in {"<", ">"}:
                continue
            if normalized[start : start + 2] == "<<":
                continue
            suffix = normalized[end:].lstrip()
            lexer = shlex.shlex(suffix, posix=True)
            lexer.whitespace_split = True
            with suppress(ValueError, StopIteration):
                target = next(lexer)
                if target.startswith("&") or target.startswith("-"):
                    continue
                targets.append(target)
        return targets

    def _has_unsafe_redirection(self, normalized: str) -> bool:
        targets = self._redirection_targets(normalized)
        return any(not self._is_null_redirection_target(target) for target in targets)

    def _is_null_redirection_target(self, target: str) -> bool:
        stripped = target.strip()
        stripped = stripped.rstrip(";&|").strip()
        stripped = stripped.strip("'\"")
        if stripped == "/dev/null":
            return True
        return bool(
            re.fullmatch(
                r"(?:\d*(?:<>|>>|>|<)|&>)\s*/dev/null",
                stripped,
            )
        )

    def _strip_heredoc_bodies(self, normalized: str) -> str:
        if "<<" not in normalized or "\n" not in normalized:
            return normalized

        retained_lines: list[str] = []
        pending_delimiters: list[tuple[str, bool]] = []
        for line in normalized.splitlines():
            if pending_delimiters:
                delimiter, strip_tabs = pending_delimiters[0]
                candidate = line.lstrip("\t") if strip_tabs else line
                if candidate == delimiter:
                    pending_delimiters.pop(0)
                continue

            retained_lines.append(line)
            pending_delimiters.extend(self._heredoc_delimiters(line))

        return "\n".join(retained_lines).strip()

    def _heredoc_delimiters(self, line: str) -> list[tuple[str, bool]]:
        if "<<" not in line:
            return []

        delimiters: list[tuple[str, bool]] = []
        for start, _end, operator in self._shell_operator_spans(line):
            if operator != "<" or line[start : start + 2] != "<<":
                continue
            if line[start : start + 3] == "<<<":
                continue

            cursor = start + 2
            strip_tabs = False
            if cursor < len(line) and line[cursor] == "-":
                strip_tabs = True
                cursor += 1
            while cursor < len(line) and line[cursor].isspace():
                cursor += 1
            if cursor >= len(line):
                continue

            lexer = shlex.shlex(line[cursor:], posix=True)
            lexer.whitespace_split = True
            lexer.commenters = ""
            with suppress(ValueError, StopIteration):
                delimiter = next(lexer)
                if delimiter:
                    delimiters.append((delimiter, strip_tabs))
        return delimiters

    def _session_allows_command(
        self,
        normalized: str,
        include_session_allowances: bool,
    ) -> bool:
        if not include_session_allowances or self.session_allowed_commands is None:
            return False
        return (
            normalized in self.session_allowed_commands
            or self._allowance_key(normalized) in self.session_allowed_commands
        )

    def _session_rejects_command(
        self,
        normalized: str,
        include_session_allowances: bool,
    ) -> bool:
        if not include_session_allowances or self.session_rejected_commands is None:
            return False
        return (
            normalized in self.session_rejected_commands
            or self._allowance_key(normalized) in self.session_rejected_commands
        )

    def _allowance_key(self, normalized: str) -> str:
        parts = self._allowance_parts(normalized)
        if not parts:
            return normalized
        return f"cmd:{' '.join(parts)}"

    def _allowance_label(self, normalized: str) -> str:
        display = command_allowance_display(self._allowance_key(normalized))
        if display.subject == "this command":
            return "matching commands"
        return f"{display.subject} commands"

    def _allowance_subject(self, normalized: str) -> str:
        return command_allowance_display(self._allowance_key(normalized)).subject

    def _allowance_parts(self, normalized: str) -> list[str]:
        segment = self._command_segment_for_allowance(normalized)
        if not segment:
            return []
        with suppress(ValueError):
            parts = shlex.split(segment)
            if parts:
                executable = Path(parts[0]).name
                important = self._important_allowance_parameters(executable, parts[1:])
                important.extend(self._redirection_allowance_parameters(normalized))
                return [executable, *important]
        executable = self._command_executable(normalized)
        if not executable:
            return []
        return [executable, *self._redirection_allowance_parameters(normalized)]

    def _command_segment_for_allowance(self, normalized: str) -> str:
        policy_source = self._strip_heredoc_bodies(normalized)
        segments = self._shell_segments(
            policy_source,
            split_operators=frozenset(SHELL_METACHARS | {PIPE_OPERATOR}),
        )
        return segments[0] if segments else policy_source

    def _important_allowance_parameters(
        self,
        executable: str,
        arguments: list[str],
    ) -> list[str]:
        important: list[str] = []
        index = 0
        while index < len(arguments):
            argument = arguments[index]
            if not argument:
                index += 1
                continue
            if argument in OPTION_VALUE_ALLOWANCE_FLAGS:
                important.append(argument)
                if index + 1 < len(arguments):
                    important.append(arguments[index + 1])
                    index += 2
                    continue
            elif argument.startswith("-") and argument != "-":
                important.append(argument)
            elif executable in SCRIPT_RUNNERS and argument == "run":
                important.append(argument)
                if index + 1 < len(arguments) and not arguments[index + 1].startswith("-"):
                    important.append(arguments[index + 1])
                    index += 2
                    continue
            elif executable in SUBCOMMAND_ALLOWANCE_COMMANDS and not important:
                important.append(argument)
            index += 1
        return important

    def _redirection_allowance_parameters(self, normalized: str) -> list[str]:
        if not self._has_unsafe_redirection(normalized):
            return []
        operators: list[str] = []
        for start, end, operator in self._shell_operator_spans(normalized):
            if operator == "<" and normalized[start : start + 2] == "<<":
                continue
            suffix = normalized[end:].lstrip()
            lexer = shlex.shlex(suffix, posix=True)
            lexer.whitespace_split = True
            target = ""
            with suppress(ValueError, StopIteration):
                target = next(lexer)
            if (
                not target
                or target.startswith("&")
                or target.startswith("-")
                or self._is_null_redirection_target(target)
            ):
                continue
            if operator == ">":
                token = ">>" if normalized[start : start + 2] == ">>" else ">"
            elif operator == "<":
                token = "<"
            else:
                continue
            if token not in operators:
                operators.append(token)
        return operators

    def _session_rejection_reason(self, allowance_key: str) -> str:
        subject = self._session_policy_subject(allowance_key)
        if subject == "this command":
            return "This command is blocked for this session by user policy."
        return f"{subject} is blocked for this session by user policy."

    def session_policy_prompt_lines(self) -> list[str]:
        """Return session-scoped command policy lines for agent instructions."""

        approved = self._session_policy_subjects(self.session_allowed_commands)
        rejected = self._session_policy_subjects(self.session_rejected_commands)
        if not approved and not rejected:
            return []

        lines = ["Session command policy:"]
        if approved:
            lines.append(
                "- Already approved for this session: "
                f"{', '.join(approved)}. These command families do not need approval again."
            )
        if rejected:
            lines.append(
                "- Never call run_command with these command families in this session: "
                f"{', '.join(rejected)}. The user explicitly rejected them."
            )
        return lines

    def workspace_prompt_lines(self) -> list[str]:
        """Return workspace path policy lines for agent instructions."""

        lines = [
            "Workspace access:",
            f"- Trusted workspace root: {self.root}",
            f"- Shell starts in: {self.current_dir}",
            (
                "- Relative and absolute paths are allowed only when they resolve "
                "inside a trusted workspace root."
            ),
        ]
        extra_roots = [root for root in self.trusted_roots if root != self.root]
        if extra_roots:
            lines.append("- Additional trusted agent data roots:")
            lines.extend(f"  - {root}" for root in extra_roots)
        if self.strict_workspace:
            lines.append(
                "- Strict sandbox mode is active. You cannot request path approvals "
                "or use shell syntax to leave the trusted workspace roots."
            )
        return lines

    def _session_policy_subjects(
        self,
        keys: MutableSet[str] | None,
    ) -> list[str]:
        if not keys:
            return []
        return sorted({self._session_policy_subject(key) for key in keys})

    def _session_policy_subject(self, key: str) -> str:
        return command_allowance_display(key).subject

    def _command_executable(self, normalized: str) -> str:
        segment = self._shell_segments(normalized, split_operators=frozenset(SHELL_METACHARS))
        if not segment:
            return ""
        with suppress(ValueError):
            parts = shlex.split(segment[0])
            if parts:
                return Path(parts[0]).name
        match = re.match(r"\s*([^\s;&|><`$\n]+)", normalized)
        if match is None:
            return ""
        return Path(match.group(1)).name

    def _serious_token_in_command(self, normalized: str) -> str | None:
        policy_source = self._strip_heredoc_bodies(normalized)
        for segment in self._shell_segments(
            policy_source,
            split_operators=frozenset({";", "&&", "||", "|", "\n"}),
        ):
            with suppress(ValueError):
                parts = shlex.split(segment)
                if parts and Path(parts[0]).name in SERIOUS_COMMAND_NAMES:
                    return Path(parts[0]).name
        return None

    def _sandbox_serious_token_in_command(self, normalized: str) -> str | None:
        policy_source = self._strip_heredoc_bodies(normalized)
        for segment in self._shell_segments(
            policy_source,
            split_operators=frozenset({";", "&&", "||", "|", "\n"}),
        ):
            with suppress(ValueError):
                parts = shlex.split(segment)
                if parts and Path(parts[0]).name in SANDBOX_SERIOUS_COMMAND_NAMES:
                    return Path(parts[0]).name
        return None

    def _is_known_read_only_command(self, executable: str, parts: list[str]) -> bool:
        if executable in READ_ONLY_COMMAND_NAMES:
            return True
        if executable == "find":
            return not any(arg in UNSAFE_FIND_OPTIONS for arg in parts[1:])
        if executable == "rg":
            return not any(
                arg in UNSAFE_RG_OPTIONS
                or any(arg.startswith(f"{option}=") for option in ("--pre", "--hostname-bin"))
                for arg in parts[1:]
            )
        if executable == "sed":
            return (
                len(parts) <= 4
                and len(parts) >= 3
                and parts[1] == "-n"
                and SED_PRINT_ONLY_RE.match(parts[2]) is not None
            )
        if executable == "git":
            return self._is_read_only_git(parts)
        return False

    def _is_read_only_git(self, parts: list[str]) -> bool:
        if len(parts) < 2 or parts[1] not in ALLOW_GIT_SUBCOMMANDS:
            return False
        subcommand = parts[1]
        args = parts[2:]
        if subcommand in {"status", "log", "diff", "show"}:
            return not any(arg in {"--output", "-o"} for arg in args)
        if subcommand == "branch":
            if not args:
                return True
            return all(
                arg in READ_ONLY_GIT_BRANCH_FLAGS or arg.startswith("--format=") for arg in args
            )
        return False

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = self._path_from_command_argument(raw_path)
        if not candidate.is_absolute():
            candidate = self.current_dir / candidate
        return candidate.resolve()

    def _inside_workspace(self, path: Path) -> bool:
        return any(path == root or root in path.parents for root in self.trusted_roots)

    def _normalize_trusted_roots(self, roots: Sequence[Path] | None) -> tuple[Path, ...]:
        normalized: list[Path] = []
        for root in (self.root, *(roots or ())):
            resolved = root.expanduser().resolve()
            if resolved not in normalized:
                normalized.append(resolved)
        return tuple(normalized)

    def _path_from_command_argument(self, raw_path: str) -> Path:
        raw = raw_path.strip()
        if raw == "~" or raw.startswith("~/"):
            home = self.subprocess_env.get("HOME") if self.subprocess_env else None
            if home:
                suffix = raw[2:] if raw.startswith("~/") else ""
                return Path(home).expanduser() / suffix
        return Path(raw).expanduser()

    def _normalize_command(self, command: str) -> str:
        normalized = command.strip()
        if normalized.startswith("›"):
            return normalized.removeprefix("›").lstrip()
        return normalized

    def _has_pipe_operator(self, normalized: str) -> bool:
        return any(
            operator == PIPE_OPERATOR
            for _, _, operator in self._shell_operator_spans(normalized)
        )

    def _has_compound_operator(self, normalized: str) -> bool:
        return any(
            operator in {";", "&&", "||", "\n"}
            for _, _, operator in self._shell_operator_spans(normalized)
        )

    def _has_shell_syntax(self, normalized: str) -> bool:
        return bool(self._shell_operator_spans(normalized))

    def _contains_approval_only_shell_syntax(self, normalized: str) -> bool:
        policy_source = self._strip_heredoc_bodies(normalized)
        return any(
            operator in {"$", "`", "&"}
            for _, _, operator in self._shell_operator_spans(policy_source)
        )

    def _pipeline_segments(self, normalized: str) -> list[str]:
        return self._shell_segments(normalized, split_operators=frozenset({PIPE_OPERATOR}))

    def _shell_compound_segments(self, normalized: str) -> list[str]:
        return self._shell_segments(normalized, split_operators=frozenset({";", "&&", "||", "\n"}))

    def _shell_segments(
        self,
        normalized: str,
        *,
        split_operators: frozenset[str],
    ) -> list[str]:
        segments: list[str] = []
        cursor = 0
        for start, end, operator in self._shell_operator_spans(normalized):
            if operator not in split_operators:
                continue
            segment = normalized[cursor:start].strip()
            if segment:
                segments.append(segment)
            cursor = end
        final_segment = normalized[cursor:].strip()
        if final_segment:
            segments.append(final_segment)
        return segments

    def _shell_operator_spans(self, normalized: str) -> list[tuple[int, int, str]]:
        """Return shell operator spans outside quoted or escaped text."""

        spans: list[tuple[int, int, str]] = []
        single_quoted = False
        double_quoted = False
        escaped = False
        index = 0
        while index < len(normalized):
            character = normalized[index]
            if escaped:
                escaped = False
                index += 1
                continue
            if character == "\\" and not single_quoted:
                escaped = True
                index += 1
                continue
            if character == "'" and not double_quoted:
                single_quoted = not single_quoted
                index += 1
                continue
            if character == '"' and not single_quoted:
                double_quoted = not double_quoted
                index += 1
                continue
            if single_quoted or double_quoted:
                index += 1
                continue
            if character == "&":
                if normalized[index : index + 2] == "&&":
                    spans.append((index, index + 2, "&&"))
                    index += 2
                    continue
                spans.append((index, index + 1, "&"))
                index += 1
                continue
            if character == "|":
                if normalized[index : index + 2] == "||":
                    spans.append((index, index + 2, "||"))
                    index += 2
                    continue
                spans.append((index, index + 1, PIPE_OPERATOR))
                index += 1
                continue
            if character in {";", "\n", "`", "$"}:
                spans.append((index, index + 1, character))
                index += 1
                continue
            if character in {"<", ">"}:
                if normalized[index : index + 2] in {"<<", ">>"}:
                    spans.append((index, index + 2, character))
                    index += 2
                    continue
                spans.append((index, index + 1, character))
                index += 1
                continue
            index += 1
        return spans

    def _strip_null_redirections(self, normalized: str) -> str:
        null_device = r"(?:/dev/null|'\/dev\/null'|\"/dev/null\")"
        null_redirect = rf"\s*(?:\d*(?:<>|>>|>|<)|&>)\s*{null_device}(?=$|\s|[;&|])"
        fd_redirect = r"\s*\d*(?:>|<)&(?:\d+|-)\b"
        stripped = normalized
        previous = ""
        while stripped != previous:
            previous = stripped
            stripped = re.sub(null_redirect, "", stripped)
            stripped = re.sub(fd_redirect, "", stripped)
        return stripped.strip()

    def _execute_pipeline(
        self,
        normalized: str,
        *,
        long_running_callback: LongRunningCommandCallback | None = None,
    ) -> str:
        segments = self._pipeline_segments(normalized)
        if any(
            self._has_shell_syntax(segment) or shlex.split(segment)[:1] == ["cd"]
            for segment in segments
        ):
            # A segment carries a compound operator (&&, ;, redirection) or a shell
            # builtin such as `cd`. Manual stdout->stdin threading cannot represent
            # those, and running a builtin through subprocess.Popen(shell=False)
            # raises FileNotFoundError (e.g. 'cd'). The command already classified as
            # safe, so defer to the real shell instead.
            return self._execute_shell_command(
                normalized,
                long_running_callback=long_running_callback,
            )
        input_text: str | None = None
        stderr_parts: list[str] = []
        return_code = 0
        for segment in segments:
            result = self._execute_subprocess(
                shlex.split(segment),
                input=input_text,
                return_result=True,
                long_running_callback=long_running_callback,
            )
            if isinstance(result, str):
                return result
            input_text = result.stdout
            return_code = result.returncode
            if result.stderr.strip():
                stderr_parts.append(result.stderr.strip())
        output = "\n".join(part for part in ((input_text or "").strip(), *stderr_parts) if part)
        return self._abbreviate_command_output(
            output or f"Command exited with status {return_code}."
        )

    def _execute_shell_command(
        self,
        normalized: str,
        *,
        long_running_callback: LongRunningCommandCallback | None = None,
    ) -> str:
        output = self._execute_subprocess(
            normalized,
            shell=True,
            long_running_callback=long_running_callback,
        )
        assert isinstance(output, str)
        return output

    def _execute_subprocess(
        self,
        command: str | list[str],
        *,
        shell: bool = False,
        input: str | None = None,
        return_result: bool = False,
        long_running_callback: LongRunningCommandCallback | None = None,
    ) -> str | subprocess.CompletedProcess[str]:
        if self.cancel_event is not None and self.cancel_event.is_set():
            return "Command stopped because the agent was interrupted."

        try:
            process = self._open_subprocess(
                command,
                shell=shell,
                stdin=subprocess.PIPE if input is not None else None,
            )
        except OSError as error:
            return f"Command failed: {error}"
        deadline = time.monotonic() + COMMAND_TIMEOUT_SECONDS
        started_at = time.monotonic()
        reported_long_running = False
        pending_input = input
        while True:
            if self.cancel_event is not None and self.cancel_event.is_set():
                self._terminate_process(process)
                return "Command stopped because the agent was interrupted."
            if pending_input is None and process.poll() is not None:
                stdout, stderr = process.communicate()
                break
            if (
                not reported_long_running
                and long_running_callback is not None
                and time.monotonic() - started_at >= LONG_RUNNING_COMMAND_SECONDS
            ):
                reported_long_running = True
                handoff_output = long_running_callback(process)
                if handoff_output is not None:
                    return handoff_output
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate_process(process)
                return f"Command timed out after {COMMAND_TIMEOUT_SECONDS} seconds."
            if pending_input is None:
                time.sleep(min(0.1, remaining))
                continue
            try:
                stdout, stderr = process.communicate(
                    input=pending_input,
                    timeout=min(0.1, remaining),
                )
                break
            except subprocess.TimeoutExpired:
                pending_input = None

        completed = subprocess.CompletedProcess(
            command,
            process.returncode,
            stdout or "",
            stderr or "",
        )
        if return_result:
            return completed
        output = "\n".join(
            part for part in (completed.stdout.strip(), completed.stderr.strip()) if part
        )
        return self._abbreviate_command_output(
            output or f"Command exited with status {completed.returncode}."
        )

    def _abbreviate_command_output(self, output: str) -> str:
        lines = output.splitlines()
        if len(lines) <= MAX_COMMAND_OUTPUT_ROWS:
            return output
        if MAX_COMMAND_OUTPUT_ROWS <= 0:
            return (
                f"[... {len(lines)} More Rows omitted from the middle of this command output ...]"
            )

        head_count = (MAX_COMMAND_OUTPUT_ROWS + 1) // 2
        tail_count = MAX_COMMAND_OUTPUT_ROWS - head_count
        hidden_count = len(lines) - head_count - tail_count
        abbreviated_lines = [
            *lines[:head_count],
            (
                "[... "
                f"{hidden_count} More Rows omitted from the middle of this command output "
                "...]"
            ),
        ]
        if tail_count:
            abbreviated_lines.extend(lines[-tail_count:])
        return "\n".join(abbreviated_lines)

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        with suppress(ProcessLookupError, OSError):
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover - Windows-specific fallback
                process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError, OSError):
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:  # pragma: no cover - Windows-specific fallback
                    process.kill()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1)

    def terminate_process(self, process: subprocess.Popen[str]) -> None:
        """Terminate a process tree started by this manager."""

        self._terminate_process(process)

    def _start_subprocess(self, command: str) -> subprocess.Popen[str]:
        normalized = self._normalize_command(command)
        if not normalized:
            raise ValueError("empty command")
        if self._has_pipe_operator(normalized) or self._has_shell_syntax(normalized):
            return self._open_subprocess(normalized, shell=True)

        parts = shlex.split(normalized)
        if not parts:
            raise ValueError("empty command")
        if parts[0] == "cd":
            raise ValueError("cd cannot be started as an async process")
        return self._open_subprocess(parts)

    def _open_subprocess(
        self,
        command: str | list[str],
        *,
        shell: bool = False,
        stdin: int | None = None,
    ) -> subprocess.Popen[str]:
        return subprocess.Popen(
            command,
            cwd=self.current_dir,
            env=self.subprocess_env,
            shell=shell,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="backslashreplace",
            start_new_session=os.name != "nt",
        )
