"""Model backend and local tool runtime for the Anomx CLI agent."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping, MutableSet
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO
from uuid import uuid4

from anomx.agent.backends import backend_for_provider
from anomx.agent.agents.main import CONNECTED_PLATFORM_AGENT_PROMPT
from anomx.agent.base.backends import (
    AnthropicStreamResponse,
    AnthropicToolCall,
    BaseBackend,
    ImageAttachment,
    OllamaStreamResponse,
    OllamaToolCall,
    OpenAIStreamResponse,
    OpenAIToolCall,
    backend_supports_image_input,
    context_usage_percent,
    estimate_backend_context_tokens,
    image_mime_type,
    normalized_image_attachments,
)
from anomx.agent.base.interactions import QuestionOption, QuestionRequest, QuestionResponse
from anomx.agent.base.processes import AsyncProcessState
from anomx.agent.base.subagents import SubagentRuntimeState
from anomx.agent.base.tools import BaseTool, ToolExecutionContext
from anomx.agent.exceptions import ToolExecutionError
from anomx.agent.helpers.anomx_api import platform_api_base_url, platform_environment
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.platform_client import heartbeat_platform_connection
from anomx.agent.helpers.state import (
    PlanStep,
    latest_plan_steps,
    running_process_snapshots,
    running_subagent_snapshots,
    subagent_snapshots,
)
from anomx.agent.helpers.tool_manager import (
    ApprovalCallback,
    ApprovalChoice,
    CliToolManager,
    CommandApprovalRequest,
    CommandProcessResult,
    CommandResult,
    CommandRiskEvaluation,
    CommandSafety,
    discover_workspace_root,
)
from anomx.agent.helpers.utils import AgentKind, AgentSpec, agent_spec, session_id_from_path
from anomx.agent.memories import (
    MemoryKind,
    MemoryMetadata,
    fallback_memory_summary,
    fallback_memory_title,
    increment_memory_uses,
    load_memories,
)
from anomx.agent.skills import load_system_skills, sync_builtin_skills
from anomx.agent.store import (
    AnomxHome,
    model_context_window,
    normalize_thinking_intensity,
    utc_now_iso,
)
from anomx.agent.tools import command_control_tools, wait_tool

if TYPE_CHECKING:
    from anomx.agent.helpers.local_sandbox import LocalSandboxSession
    from anomx.agent.helpers.sandbox import SandboxSession

__all__ = [
    "AgentRuntime",
    "AgentRole",
    "AnthropicStreamResponse",
    "AnthropicToolCall",
    "AsyncProcessState",
    "ImageAttachment",
    "OllamaStreamResponse",
    "OllamaToolCall",
    "OpenAIStreamResponse",
    "OpenAIToolCall",
    "QuestionOption",
    "QuestionRequest",
    "QuestionResponse",
    "RuntimeCallbacks",
    "RuntimeCleanupResult",
    "backend_supports_image_input",
    "context_usage_percent",
    "image_mime_type",
]

StatusCallback = Callable[[str], None]
MessageCallback = Callable[[str], None]
ToolMessageCallback = Callable[[str], None]
DeltaCallback = Callable[[str], None]
SystemMessageCallback = Callable[[str, str], None]
CommandCallback = Callable[[str, str, str], None]
OutputResponseCallback = Callable[[dict[str, Any]], None]
SubagentCallback = Callable[[dict[str, Any]], None]
FinishCallback = Callable[[str], None]

MAX_PLAN_FINISH_REPROMPTS = 6
IMAGE_FILE_EXTENSIONS = (".gif", ".jpeg", ".jpg", ".png", ".webp")


class AgentRole(StrEnum):
    """Runtime role for a model-backed agent."""

    STANDARD = "standard"
    AUTOMATIC = "automatic"
    AUTONOMOUS = "autonomous"
    BUILD = "build"
    AUTO = "auto"
    PLAN = "plan"
    OPERATOR = "build"
    GENERAL = "general"
    EXPLORE = "explore"
    PLATFORM = "platform"
    WORKER = "general"
    SCOUT = "explore"


SUBAGENT_EVENT_TYPE = "subagent_event"


ProcessCallback = Callable[[AsyncProcessState], None]


@dataclass(frozen=True)
class RuntimeCleanupResult:
    """Summary of stale runtime state removed from a session transcript."""

    processes_ended: int = 0
    subagents_removed: int = 0
    workers_removed: int = 0


QuestionCallback = Callable[[QuestionRequest], QuestionResponse]


@dataclass(frozen=True)
class RuntimeCallbacks:
    """UI callbacks used while a model response is in progress."""

    status: StatusCallback | None = None
    message: MessageCallback | None = None
    tool_message: ToolMessageCallback | None = None
    command: CommandCallback | None = None
    output_response: OutputResponseCallback | None = None
    subagent: SubagentCallback | None = None
    delta: DeltaCallback | None = None
    approval: ApprovalCallback | None = None
    system_message: SystemMessageCallback | None = None
    question: QuestionCallback | None = None
    process: ProcessCallback | None = None
    finish: FinishCallback | None = None


class AgentRuntime:
    """Execute model requests and local tools for an agent session."""

    def __init__(
        self,
        home: AnomxHome,
        cwd: Path,
        session_allowed_commands: MutableSet[str] | None = None,
        session_rejected_commands: MutableSet[str] | None = None,
        mode: AgentMode = AgentMode.CONFIRM,
        role: AgentRole | str = AgentRole.STANDARD,
        cancel_event: threading.Event | None = None,
        workspace_root: Path | None = None,
        process_owner_id: str = "",
        process_owner_name: str = "",
        local_sandbox_enabled: bool = False,
        local_sandbox_home: Path | None = None,
        local_sandbox_allow_subprocess: bool = False,
    ) -> None:
        self.home = home
        self.cwd = cwd.expanduser().resolve()
        self.workspace_root = (
            discover_workspace_root(self.cwd)
            if workspace_root is None
            else workspace_root.expanduser().resolve()
        )
        self.cancel_event = threading.Event() if cancel_event is None else cancel_event
        self._local_sandbox_session: LocalSandboxSession | None = None
        if home.platform_connection() is not None and not bool(home.load_config().get("running_in_anomx_platform")):
            with suppress(Exception):
                heartbeat_platform_connection(home)
        self._platform_env = platform_environment(home)
        sync_builtin_skills(
            self.home.skills_dir,
            include_system=bool(self._platform_env),
        )
        if local_sandbox_enabled:
            self._local_sandbox_session = self._create_local_sandbox_session(
                home=local_sandbox_home,
                allow_subprocess=local_sandbox_allow_subprocess,
            )
        subprocess_env = self._runtime_subprocess_env()
        self.tool_manager = CliToolManager(
            self.workspace_root,
            session_allowed_commands,
            session_rejected_commands,
            mode,
            current_dir=self.cwd,
            cancel_event=self.cancel_event,
            subprocess_env=subprocess_env,
            strict_workspace=local_sandbox_enabled,
            trusted_roots=self.trusted_roots,
        )
        self.session_allowed_commands = session_allowed_commands
        self.session_rejected_commands = session_rejected_commands
        self.agent_spec: AgentSpec = agent_spec(role)
        self.role = AgentRole(self.agent_spec.kind.value)
        self._turn_abort_event = threading.Event()
        self._processes: dict[str, AsyncProcessState] = {}
        self._process_lock = threading.Lock()
        self._subagents: dict[str, SubagentRuntimeState] = {}
        self._subagent_lock = threading.Lock()
        self._debug_session_path: Path | None = None
        self._parent_session_id: str = ""
        self.process_owner_id = process_owner_id
        self.process_owner_name = process_owner_name
        self.backend: BaseBackend | None = None
        self._sandbox_session: SandboxSession | None = None

    @property
    def local_sandbox_session(self) -> LocalSandboxSession | None:
        return self._local_sandbox_session

    @property
    def sandbox_session(self) -> SandboxSession | None:
        return self._sandbox_session

    @property
    def trusted_roots(self) -> tuple[Path, ...]:
        """Return all paths tools may treat as trusted for this runtime."""

        return (
            self.workspace_root,
            self.home.skills_dir,
            self.home.responses_dir,
        )

    def has_platform_connection(self) -> bool:
        """Return whether this runtime has a configured platform API token."""

        connected = self.home.platform_connection() is not None
        next_env = platform_environment(self.home)
        if next_env != self._platform_env:
            self._platform_env = next_env
            self.tool_manager.subprocess_env = self._runtime_subprocess_env()
            self.tool_manager.set_trusted_roots(self.trusted_roots)
        return connected

    def can_output_response(self) -> bool:
        """Return whether rich platform response output is available."""

        config = self.home.load_config()
        return (
            self.has_platform_connection()
            and config.get("running_in_anomx_platform") is True
            and config.get("platform_output_response_enabled") is True
        )

    def _runtime_subprocess_env(self) -> dict[str, str] | None:
        if self._local_sandbox_session is not None:
            env = self._local_sandbox_session.env
            env.update(self._platform_env)
            return env
        if not self._platform_env:
            return None
        env = dict(os.environ)
        env.update(self._platform_env)
        return env

    def _create_local_sandbox_session(
        self,
        *,
        home: Path | None = None,
        allow_subprocess: bool = False,
    ) -> LocalSandboxSession:
        from anomx.agent.helpers.local_sandbox import LocalSandboxConfig, LocalSandboxSession

        return LocalSandboxSession(
            LocalSandboxConfig(
                root=self.workspace_root,
                home=(home or self.home.root.parent),
                current_dir=self.cwd,
                allow_subprocess=allow_subprocess,
                env=self._platform_env,
                trusted_roots=self.trusted_roots,
            )
        )

    def init_python_sandbox(
        self,
        *,
        home: Path | None = None,
        allow_subprocess: bool = True,
        status_callback: StatusCallback | None = None,
    ) -> bool:
        """Initialise the software sandbox rooted at the runtime workspace."""

        if status_callback:
            status_callback("Preparing Python sandbox")
        self._sandbox_session = None
        self._local_sandbox_session = self._create_local_sandbox_session(
            home=home,
            allow_subprocess=allow_subprocess,
        )
        self.tool_manager.current_dir = self._local_sandbox_session.current_dir
        self.tool_manager.subprocess_env = self._runtime_subprocess_env()
        self.tool_manager.strict_workspace = True
        self.tool_manager.set_trusted_roots(self.trusted_roots)
        if status_callback:
            status_callback("Python sandbox ready")
        return True

    def init_sandbox(
        self,
        config: Mapping[str, Any] | None = None,
        status_callback: StatusCallback | None = None,
    ) -> bool:
        """Initialise and start the configured sandbox runtime if enabled.

        Returns True when sandbox is active (either started or not needed).
        """
        cfg = self.home.load_config() if config is None else dict(config)
        if not cfg.get("sandbox_enabled"):
            return True

        from anomx.agent.helpers.local_sandbox import is_python_sandbox_system

        if is_python_sandbox_system(cfg.get("sandbox_system")):
            sandbox_home = cfg.get("sandbox_home")
            return self.init_python_sandbox(
                home=Path(str(sandbox_home)).expanduser() if sandbox_home else None,
                allow_subprocess=bool(cfg.get("sandbox_allow_subprocess", True)),
                status_callback=status_callback,
            )

        from anomx.agent.helpers.sandbox import (
            SandboxSession,
            sandbox_config_from_dict,
        )

        scfg = sandbox_config_from_dict(cfg)
        project_path = self.workspace_root or self.cwd
        sandbox_hash = self._load_sandbox_hash(project_path)
        self._sandbox_session = SandboxSession(
            scfg, project_path, sandbox_hash=sandbox_hash,
        )

        if status_callback:
            status_callback("Starting Sandbox")

        self._sandbox_session.start(status_callback=status_callback)
        return True

    def _load_sandbox_hash(self, project_path: Path) -> str:
        project = self.home.project_for_path(project_path)
        if project is not None and project.sandbox_hash:
            return project.sandbox_hash
        import hashlib
        raw = str(project_path.resolve()).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:6]

    def set_mode(self, mode: AgentMode) -> None:
        """Set the active command execution mode."""

        self.tool_manager.set_mode(mode)

    def set_agent(self, kind: AgentKind | str) -> None:
        """Set the active agent kind for future model turns."""

        self.agent_spec = agent_spec(kind)
        self.role = AgentRole(self.agent_spec.kind.value)
        self.tool_manager.set_mode(self.agent_spec.approval_mode)

    def abort_current_turn(
        self,
        session_path: Path | None = None,
    ) -> None:
        """Request cancellation of the active response loop."""

        self._turn_abort_event.set()

    def shutdown(self, session_path: Path | None = None) -> RuntimeCleanupResult:
        """Stop live runtime children before the CLI process exits."""

        self._turn_abort_event.set()
        self._sandbox_session = None
        processes_ended = self._end_all_process_states(session_path)
        subagents_removed = self._end_all_subagent_states(session_path)
        processes_ended = self._end_all_process_states(session_path)
        subagents_removed = self._end_all_subagent_states(session_path)
        return RuntimeCleanupResult(
            processes_ended=processes_ended,
            subagents_removed=subagents_removed,
            workers_removed=subagents_removed,
        )

    def cleanup_session_runtime_state(self, session_path: Path) -> RuntimeCleanupResult:
        """Clear stale process cards from a stored session transcript.

        Process handles are process-local. After a CLI restart, any
        running process shown by the transcript is stale even though the
        conversation history is still useful.
        """

        events = self.home.read_session_events(session_path)
        processes_ended = 0
        subagents_removed = 0
        for process in running_process_snapshots(events):
            if self._end_process_by_id_if_known(process.process_id, session_path):
                processes_ended += 1
                continue
            self.home.append_session_event(
                session_path,
                "process_event",
                {
                    "process_id": process.process_id,
                    "command": process.command,
                    "status": "ended",
                    "statement": process.statement,
                    "output": process.output,
                    "started_at": process.started_at,
                    "finished_at": process.finished_at or "",
                    "exit_code": process.exit_code,
                    "source": process.source,
                    "owner_id": process.owner_id,
                    "owner_name": process.owner_name,
                    "pid": process.pid,
                },
            )
        for subagent in running_subagent_snapshots(events):
            subagents_removed += 1
            self.home.append_session_event(
                session_path,
                SUBAGENT_EVENT_TYPE,
                {
                    "agent_id": subagent.agent_id,
                    "kind": subagent.kind,
                    "name": subagent.name,
                    "status": "interrupted",
                    "statement": "Subagent was interrupted because the runtime restarted.",
                    "prompt": subagent.prompt,
                    "response": subagent.response,
                    "error": subagent.error,
                    "session_path": subagent.session_path,
                    "started_at": subagent.started_at,
                    "finished_at": utc_now_iso(),
                    "context_tokens": subagent.context_tokens,
                    "context_percent": subagent.context_percent,
                },
            )
        return RuntimeCleanupResult(
            processes_ended=processes_ended,
            subagents_removed=subagents_removed,
            workers_removed=subagents_removed,
        )

    def _turn_aborted(self) -> bool:
        return self._turn_abort_event.is_set() or self.cancel_event.is_set()

    def backend_response(
        self,
        session_path: Path,
        callbacks: RuntimeCallbacks | None = None,
        *,
        debug_session_path: Path | None = None,
    ) -> str:
        """Generate a backend response for the current session."""

        self._turn_abort_event.clear()
        previous_debug_session_path = self._debug_session_path
        self._debug_session_path = debug_session_path or session_path
        try:
            active_callbacks = RuntimeCallbacks() if callbacks is None else callbacks
            config = self.home.load_config()
            provider = str(config.get("provider", ""))
            model = str(config.get("model", ""))
            if self.home.full_session_logs_enabled(config):
                self._ensure_debug_meta(session_path, config)
            thinking_intensity = normalize_thinking_intensity(config.get("thinking_intensity"))
            backend = backend_for_provider(provider, self)
            if backend is None:
                return f"{provider}/{model} backend is unavailable."
            self.backend = backend
            return backend.generate(
                session_path,
                model,
                active_callbacks,
                thinking_intensity=thinking_intensity,
            )
        finally:
            self._debug_session_path = previous_debug_session_path

    def openai_response(
        self,
        session_path: Path,
        model: str,
        callbacks: RuntimeCallbacks | None = None,
        *,
        thinking_intensity: str | None = None,
    ) -> str:
        """Compatibility wrapper for the OpenAI backend."""

        return self._backend_response_for_provider(
            "openai",
            session_path,
            model,
            RuntimeCallbacks() if callbacks is None else callbacks,
            thinking_intensity=thinking_intensity,
        )

    def anthropic_response(
        self,
        session_path: Path,
        model: str,
        callbacks: RuntimeCallbacks | None = None,
        *,
        thinking_intensity: str | None = None,
    ) -> str:
        """Compatibility wrapper for the Anthropic backend."""

        return self._backend_response_for_provider(
            "anthropic",
            session_path,
            model,
            RuntimeCallbacks() if callbacks is None else callbacks,
            thinking_intensity=thinking_intensity,
        )

    def desy_response(
        self,
        session_path: Path,
        model: str,
        callbacks: RuntimeCallbacks | None = None,
    ) -> str:
        """Compatibility wrapper for the DESY Assistant backend."""

        return self._backend_response_for_provider(
            "desy",
            session_path,
            model,
            RuntimeCallbacks() if callbacks is None else callbacks,
        )

    def ollama_response(
        self,
        session_path: Path,
        model: str,
        callbacks: RuntimeCallbacks | None = None,
    ) -> str:
        """Compatibility wrapper for the Ollama backend."""

        return self._backend_response_for_provider(
            "ollama",
            session_path,
            model,
            RuntimeCallbacks() if callbacks is None else callbacks,
        )

    def _backend_response_for_provider(
        self,
        provider: str,
        session_path: Path,
        model: str,
        callbacks: RuntimeCallbacks,
        *,
        thinking_intensity: str | None = None,
    ) -> str:
        backend = backend_for_provider(provider, self)
        if backend is None:
            return f"{provider}/{model} backend is unavailable."
        self.backend = backend
        return backend.generate(
            session_path,
            model,
            callbacks,
            thinking_intensity=thinking_intensity,
        )

    def backend_response_for_prompt(
        self,
        prompt: str,
        callbacks: RuntimeCallbacks | None = None,
        *,
        debug_session_path: Path | None = None,
        parent_session_path: Path | None = None,
    ) -> str:
        """Generate a response."""

        if self.agent_spec.can_spawn_subagents:
            session_path = parent_session_path or debug_session_path
            if session_path is None:
                session = self.home.create_session(
                    self.cwd,
                    provider=str(self.home.load_config().get("provider", "")),
                    model=str(self.home.load_config().get("model", "")),
                    mode=self.tool_manager.mode,
                )
                session_path = session.path
            self.home.append_session_event(session_path, "user_message", {"message": prompt})
        else:
            session_path = self.home.append_subagent_session_prompt(
                parent_session_path=parent_session_path or debug_session_path,
                subagent_name=self.process_owner_name or self.agent_spec.kind.value,
                subagent_id=self.process_owner_id or uuid4().hex[:8],
                subagent_kind=self.agent_spec.kind.value,
                prompt=prompt,
            )
        return self.backend_response(
            session_path,
            callbacks=callbacks,
            debug_session_path=debug_session_path,
        )

    def conversation_messages(self, session_path: Path) -> list[dict[str, Any]]:
        """Return stored user/assistant messages for a backend conversation."""

        messages: list[dict[str, Any]] = []
        for event in self.home.read_session_events(session_path):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = (
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            message = str(payload.get("message", "")).strip()
            backend_message = str(payload.get("backend_message", message)).strip()
            image_attachments = normalized_image_attachments(
                payload.get("image_attachments")
            )
            if event_type == "user_message" and (backend_message or image_attachments):
                user_message: dict[str, Any] = {
                    "role": "user",
                    "content": backend_message,
                }
                if image_attachments:
                    user_message["images"] = [
                        attachment.to_payload() for attachment in image_attachments
                    ]
                messages.append(user_message)
            elif event_type == "skill_invocation":
                prompt = str(payload.get("prompt", "")).strip()
                if prompt:
                    messages.append({"role": "user", "content": prompt})
            elif event_type == "agent_message" and message:
                messages.append({"role": "assistant", "content": message})
            elif event_type == "system_message" and message:
                messages.append({"role": "system", "content": message})
        return messages[-20:]

    def estimate_session_context_tokens(self, session_path: Path) -> int:
        """Estimate the current backend context for this runtime/session."""

        return estimate_backend_context_tokens(
            self._instructions(session_path),
            self.conversation_messages(session_path),
        )

    def suggest_session_title(self, session_path: Path) -> str | None:
        """Suggest a compact title for a session."""

        messages = self.conversation_messages(session_path)
        if not messages:
            return None

        config = self.home.load_config()
        provider = str(config.get("provider", ""))
        model = str(config.get("model", ""))
        backend = backend_for_provider(provider, self)
        if backend is not None:
            title = backend.suggest_session_title(messages, model)
            if title:
                return title
        return self._heuristic_session_title(messages)

    def suggest_project_name(self, project_path: Path, directory_outline: str) -> str | None:
        """Suggest a compact project name from a folder path and directory outline."""

        outline = directory_outline.strip() or "- empty directory"
        prompt = (
            f"Directory path:\n{project_path}\n\n"
            "Directory structure, first 3 levels:\n"
            f"{outline}"
        )
        config = self.home.load_config()
        provider = str(config.get("provider", ""))
        model = str(config.get("model", ""))
        backend = backend_for_provider(provider, self)
        if backend is not None:
            name = backend.suggest_project_name(prompt, model)
            if name:
                return name
        return None

    def suggest_session_continuation(self, session_path: Path, workspace_name: str) -> str:
        """Suggest the startup prompt copy for continuing an existing session."""

        messages = self.conversation_messages(session_path)
        fallback = self._heuristic_session_continuation(messages, workspace_name)
        if not messages:
            return fallback

        config = self.home.load_config()
        provider = str(config.get("provider", ""))
        model = str(config.get("model", ""))
        backend = backend_for_provider(provider, self)
        if backend is not None:
            statement = backend.suggest_session_continuation(messages, model)
            if statement:
                return statement
        return fallback

    def evaluate_command_request(
        self,
        session_path: Path,
        request: CommandApprovalRequest,
    ) -> CommandRiskEvaluation | None:
        """Evaluate a pending command approval request with the selected backend."""

        config = self.home.load_config()
        provider = str(config.get("provider", ""))
        model = str(config.get("model", ""))
        backend = backend_for_provider(provider, self)
        if backend is None:
            return None
        try:
            return backend.evaluate_command_request(
                command=request.command,
                statement=request.statement,
                user_message=self._latest_user_message(session_path),
                model=model,
            )
        except Exception:
            return None

    def suggest_memory_metadata(
        self,
        *,
        kind: MemoryKind | str,
        context: Mapping[str, Any],
        content: str,
    ) -> MemoryMetadata:
        """Suggest memory metadata with a deterministic fallback."""

        config = self.home.load_config()
        provider = str(config.get("provider", ""))
        model = str(config.get("model", ""))
        backend = backend_for_provider(provider, self)
        if backend is not None:
            try:
                metadata = backend.suggest_memory_metadata(
                    kind=kind,
                    context=context,
                    content=content,
                    model=model,
                )
                if metadata is not None:
                    return metadata
            except Exception:
                pass
        return MemoryMetadata(
            title=fallback_memory_title(content),
            summary=fallback_memory_summary(content),
        )

    def _latest_user_message(self, session_path: Path) -> str:
        for message in reversed(self.conversation_messages(session_path)):
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    def _ensure_debug_meta(
        self,
        session_path: Path,
        config: Mapping[str, Any],
    ) -> None:
        """Create or update the debug session *meta.json*."""
        is_subagent = not self.agent_spec.can_spawn_subagents
        session_id = (
            self._parent_session_id
            if is_subagent and self._parent_session_id
            else session_id_from_path(session_path)
        )
        project = self.home.project_for_path(self.cwd)
        meta: dict[str, Any] = {
            "session_id": session_id,
            "ai_model": str(config.get("model", "")),
            "provider": str(config.get("provider", "")),
            "mode": AgentMode.parse(config.get("agent_mode")).value,
            "location": str(self.cwd),
        }
        if project is not None:
            meta["name"] = project.name
        self.home.debug_logger.ensure_session(session_id, meta)

    def _debug_log_step(
        self,
        provider: str,
        payload: Mapping[str, Any],
        *,
        subagent_id: str | None = None,
    ) -> Path | None:
        """Write a debug step file when full session logs are enabled."""
        if not self.home.full_session_logs_enabled():
            return None
        session_path = self._debug_session_path
        if session_path is None:
            return None
        config = self.home.load_config()
        is_subagent = not self.agent_spec.can_spawn_subagents
        session_id = (
            self._parent_session_id
            if is_subagent and self._parent_session_id
            else session_id_from_path(session_path)
        )
        messages = self.home.debug_logger.normalize_payload_messages(payload)
        model = str(config.get("model", ""))
        agent_subagent_id = (
            subagent_id
            or (self.process_owner_id if is_subagent else None)
        )
        return self.home.debug_logger.write_step(
            session_id,
            messages,
            model=model,
            provider=provider,
            subagent_id=agent_subagent_id,
        )

    def extract_openai_text(self, data: dict[str, Any]) -> str:
        """Compatibility wrapper for OpenAI text extraction."""

        return BaseBackend(self).extract_openai_text(data)

    def extract_anthropic_text(self, data: dict[str, Any]) -> str:
        """Compatibility wrapper for Anthropic-compatible text extraction."""

        return BaseBackend(self).extract_anthropic_text(data)

    def _stream_openai_response(
        self,
        api_key: str,
        payload: dict[str, Any],
        delta_callback: DeltaCallback | None,
        status_callback: StatusCallback | None,
    ) -> OpenAIStreamResponse | str:
        backend = backend_for_provider("openai", self)
        if backend is None:
            return "OpenAI backend is unavailable."
        return backend._stream_openai_response(
            api_key,
            payload,
            delta_callback,
            status_callback,
        )

    def _stream_anthropic_response(
        self,
        api_key: str,
        payload: dict[str, Any],
        delta_callback: DeltaCallback | None,
        status_callback: StatusCallback | None,
    ) -> AnthropicStreamResponse | str:
        backend = backend_for_provider("anthropic", self)
        if backend is None:
            return "Anthropic backend is unavailable."
        return backend._stream_response(
            api_key,
            payload,
            delta_callback,
            status_callback,
        )

    def _stream_desy_response(
        self,
        api_key: str,
        payload: dict[str, Any],
        delta_callback: DeltaCallback | None,
        status_callback: StatusCallback | None,
    ) -> AnthropicStreamResponse | str:
        backend = backend_for_provider("desy", self)
        if backend is None:
            return "DESY Assistant backend is unavailable."
        return backend._stream_response(
            api_key,
            payload,
            delta_callback,
            status_callback,
        )

    def _stream_ollama_response(
        self,
        model: str,
        messages: list[dict[str, Any]],
        callbacks: RuntimeCallbacks,
    ) -> OllamaStreamResponse | str:
        backend = backend_for_provider("ollama", self)
        if backend is None:
            return "Ollama backend is unavailable."
        return backend._stream_ollama_response(model, messages, callbacks)

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [tool.definition() for tool in self._available_tools()]

    def _openai_tools(self) -> list[dict[str, Any]]:
        return BaseBackend(self)._openai_tools()

    def _anthropic_tools(self) -> list[dict[str, Any]]:
        return BaseBackend(self)._anthropic_tools()

    def _ollama_tools(self) -> list[dict[str, Any]]:
        return BaseBackend(self)._ollama_tools()

    def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        callbacks: RuntimeCallbacks,
        session_path: Path | None = None,
    ) -> str:
        if self._turn_aborted():
            return self._json_tool_result({"error": "Agent turn was aborted by user."})

        tool = self._tool_for_call(name)
        if tool is None:
            return self._json_tool_result({"error": f"Unknown tool: {name}"})

        try:
            return tool.execute(
                arguments,
                ToolExecutionContext(
                    runtime=self,
                    callbacks=callbacks,
                    session_path=session_path,
                ),
            )
        except ToolExecutionError as error:
            return self._json_tool_result({"error": str(error), "tool": name})

    def _tool_for_call(self, name: str) -> BaseTool | None:
        for tool in self._available_tools():
            if tool.handles(name):
                return tool
        return None

    def _available_tools(self) -> tuple[BaseTool, ...]:
        tools = [
            tool
            for tool in self.agent_spec.tools
            if (
                (tool.name != "use_anomx_api" or self.has_platform_connection())
                and (tool.name != "output_response" or self.can_output_response())
            )
        ]
        if self._running_command_states():
            tools.extend(command_control_tools())
        if self._running_command_states() or (
            self.agent_spec.can_spawn_subagents and self._running_subagent_states()
        ):
            target = (
                "active command tool calls or subagents"
                if self.agent_spec.can_spawn_subagents and self._running_subagent_states()
                else "active command tool calls"
            )
            tools.append(wait_tool(target))
        return tuple(tools)

    def _continuation_prompt_after_text(
        self,
        message: str,
        callbacks: RuntimeCallbacks,
        session_path: Path | None,
        plan_finish_attempts: int,
    ) -> tuple[str | None, bool]:
        if self._running_subagent_states():
            wait_output = self._wait_for_active_targets(callbacks)
            return self._subagent_continuation_prompt(wait_output), False
        if self._running_command_states():
            wait_output = self._wait_for_active_targets(callbacks)
            return self._command_continuation_prompt(wait_output), False

        if self.role != AgentRole.OPERATOR or not False:
            return self._plan_finish_continuation_prompt(
                session_path,
                plan_finish_attempts,
                callbacks,
            )
        wait_output = self._wait_for_active_targets(callbacks)
        return (None or wait_output), False

    def _wait_for_active_targets(self, callbacks: RuntimeCallbacks) -> str:
        return wait_tool("active command tool calls or subagents").execute(
            {},
            ToolExecutionContext(
                runtime=self,
                callbacks=callbacks,
                session_path=self._debug_session_path,
            ),
        )

    def _plan_finish_continuation_prompt(
        self,
        session_path: Path | None,
        plan_finish_attempts: int,
        callbacks: RuntimeCallbacks | None = None,
    ) -> tuple[str | None, bool]:
        if not self.agent_spec.can_use_plans or session_path is None:
            return None, False
        plan_steps = latest_plan_steps(self.home.read_session_events(session_path))
        if not plan_steps:
            return None, False
        if all(step.is_done for step in plan_steps):
            self.home.append_session_event(session_path, "plan_update", {"steps": []})
            return None, False
        if plan_finish_attempts >= MAX_PLAN_FINISH_REPROMPTS:
            return None, False

        validation_message = "Validating whether the plan is finished"
        if callbacks is not None and callbacks.tool_message is not None:
            callbacks.tool_message(validation_message)
        elif callbacks is not None and callbacks.status is not None:
            callbacks.status(validation_message)
        else:
            self.home.append_session_event(
                session_path,
                "work_message",
                {
                    "message": validation_message,
                    "role": "tool",
                },
            )
        return self._unfinished_plan_finish_prompt(plan_steps), True

    def _unfinished_plan_finish_warning(
        self,
        plan_steps: tuple[PlanStep, ...],
    ) -> str:
        open_titles = ", ".join(step.title for step in plan_steps if not step.is_done)
        if not open_titles:
            open_titles = "unknown open steps"
        return (
            "There is still an active plan that is not fully done. "
            "Are you sure you want to finish? Open plan steps: "
            f"{open_titles}."
        )

    def _unfinished_plan_finish_prompt(
        self,
        plan_steps: tuple[PlanStep, ...],
    ) -> str:
        lines = [
            "Your previous assistant message was not delivered because there is still "
            "an active plan with unfinished steps. Do not finish yet unless the plan is "
            "genuinely complete.",
            "",
            "Current plan:",
        ]
        for step in plan_steps:
            state = "done" if step.is_done else "open"
            lines.append(f"{step.position}. [{state}] {step.title}: {step.description}")
        lines.extend(
            [
                "",
                "Continue working, update the plan, or remove it if it is stale. "
                "If every item is complete, mark the plan done; the UI will remove it "
                "automatically when you finish. If you are sure the final answer should "
                "be delivered despite open plan steps, call finish_anyways(statement).",
            ]
        )
        return "\n".join(lines)

    def _command_continuation_prompt(self, wait_output: str) -> str:
        return (
            "A long-running command tool call is still active or just finished. Continue "
            "from the command wait result below. Do not produce a final answer while a "
            "required command is still running; use wait, check_command_status, or "
            "kill_command as appropriate.\n"
            f"{wait_output}"
        )

    def _subagent_continuation_prompt(self, wait_output: str) -> str:
        return (
            "One or more subagents were active while you were about to finish. Continue "
            "from the subagent wait result below. Integrate ready subagent outputs, prompt "
            "or remove subagents if more work is needed, and do not produce a final answer "
            "while required subagent work is still running.\n"
            f"{wait_output}"
        )

    def _subagent_display_name(
        self,
        requested_name: str,
        kind: AgentKind,
        agent_id: str,
    ) -> str:
        cleaned = " ".join(requested_name.split())
        if cleaned:
            return cleaned[:40]
        return f"{kind.value.title()} {agent_id[:4]}"

    def _start_subagent_worker(
        self,
        state: SubagentRuntimeState,
        prompt: str,
        parent_session_path: Path,
        parent_callbacks: RuntimeCallbacks,
    ) -> None:
        def run_subagent() -> None:
            self._run_subagent_turn(state, prompt, parent_session_path, parent_callbacks)

        worker = threading.Thread(target=run_subagent, daemon=True)
        state.worker = worker
        worker.start()

    def _run_subagent_turn(
        self,
        state: SubagentRuntimeState,
        prompt: str,
        parent_session_path: Path,
        parent_callbacks: RuntimeCallbacks,
    ) -> None:
        runtime = state.runtime
        if runtime is None:
            return

        def publish(
            statement: str = "",
            *,
            status: str = "working",
            message: str = "",
            command: dict[str, str] | None = None,
        ) -> None:
            text = statement.strip() or message.strip()
            with self._subagent_lock:
                if state.status == "removed":
                    return
                state.status = status
                if text:
                    state.statement = text
                self._refresh_subagent_context(state)
                if command is not None:
                    state.command_history.append(command)
            payload = self._publish_subagent_state(
                state,
                parent_session_path,
                message=message or text,
                command=command,
            )
            if payload is not None and parent_callbacks.subagent is not None:
                parent_callbacks.subagent(payload)

        def status_callback(message: str) -> None:
            normalized = message.strip()
            if not normalized or normalized in {"Thinking", "Loading model"}:
                return
            publish(normalized, status="working")

        def message_callback(message: str) -> None:
            self._append_subagent_agent_message(state, message, intermediate=True)
            publish(message, status="working", message=message)

        def tool_message_callback(message: str) -> None:
            self._append_subagent_work_message(state, message, role="tool")
            publish(message, status="working", message=message)

        def command_callback(statement: str, command: str, output: str) -> None:
            command_payload = {
                "statement": statement.strip(),
                "command": command.strip(),
                "output": output.strip(),
            }
            self._append_subagent_work_message(
                state,
                statement,
                role="tool",
                command=command,
            )
            publish(statement, status="working", command=command_payload)

        def delta_callback(_delta: str) -> None:
            return

        def system_message_callback(role: str, message: str) -> None:
            self._append_subagent_system_message(state, role, message)
            publish(message, status="working", message=message)

        def approval_callback(request: CommandApprovalRequest) -> ApprovalChoice:
            if parent_callbacks.approval is None:
                return ApprovalChoice.REJECT
            labelled = replace(
                request,
                agent_id=state.agent_id,
                agent_name=state.name,
            )
            return parent_callbacks.approval(labelled)

        try:
            session_path = self.home.append_subagent_session_prompt(
                parent_session_path=parent_session_path,
                subagent_name=state.name,
                subagent_id=state.agent_id,
                subagent_kind=state.kind.value,
                prompt=prompt,
            )
            state.session_path = session_path
            payload = self._publish_subagent_state(state, parent_session_path)
            if payload is not None and parent_callbacks.subagent is not None:
                parent_callbacks.subagent(payload)
            response = runtime.backend_response(
                session_path,
                callbacks=RuntimeCallbacks(
                    status=status_callback,
                    message=message_callback,
                    tool_message=tool_message_callback,
                    command=command_callback,
                    delta=delta_callback,
                    approval=approval_callback,
                    system_message=system_message_callback,
                    process=parent_callbacks.process,
                ),
                debug_session_path=session_path,
            )
            if state.cancel_event.is_set() or state.status == "removed":
                return
            state.response = response.strip()
            state.status = "ready"
            state.statement = ""
            state.finished_at = utc_now_iso()
            self._refresh_subagent_context(state)
            if state.response:
                self.home.append_session_event(
                    session_path,
                    "agent_message",
                    {"message": state.response},
                )
            payload = self._publish_subagent_state(
                state,
                parent_session_path,
                message=state.response,
            )
            if payload is not None and parent_callbacks.subagent is not None:
                parent_callbacks.subagent(payload)
        except Exception as error:  # pragma: no cover - defensive thread boundary
            state.error = f"{type(error).__name__}: {error}"
            state.status = "failed"
            state.statement = state.error
            state.finished_at = utc_now_iso()
            payload = self._publish_subagent_state(
                state,
                parent_session_path,
                message=state.error,
            )
            if payload is not None and parent_callbacks.subagent is not None:
                parent_callbacks.subagent(payload)

    def _append_subagent_agent_message(
        self,
        state: SubagentRuntimeState,
        message: str,
        *,
        intermediate: bool = False,
    ) -> None:
        if state.session_path is None or not message.strip():
            return
        payload: dict[str, object] = {"message": message.strip()}
        if intermediate:
            payload["intermediate"] = True
        self.home.append_session_event(state.session_path, "agent_message", payload)

    def _append_subagent_work_message(
        self,
        state: SubagentRuntimeState,
        message: str,
        *,
        role: str,
        command: str = "",
    ) -> None:
        if state.session_path is None or not message.strip():
            return
        payload: dict[str, object] = {"message": message.strip(), "role": role}
        if command:
            payload["command"] = command
        self.home.append_session_event(state.session_path, "work_message", payload)

    def _append_subagent_system_message(
        self,
        state: SubagentRuntimeState,
        role: str,
        message: str,
    ) -> None:
        if state.session_path is None or not message.strip():
            return
        self.home.append_session_event(
            state.session_path,
            "system_message",
            {"message": message.strip(), "role": role or "system"},
        )

    def _refresh_subagent_context(self, state: SubagentRuntimeState) -> None:
        if state.runtime is None or state.session_path is None:
            return
        with suppress(Exception):
            state.context_tokens = state.runtime.estimate_session_context_tokens(state.session_path)
        context_window = None
        with suppress(Exception):
            config = self.home.load_config()
            context_window = model_context_window(str(config.get("model", "")))
        state.context_percent = context_usage_percent(state.context_tokens, context_window)

    def _publish_subagent_state(
        self,
        state: SubagentRuntimeState,
        session_path: Path | None,
        *,
        message: str = "",
        command: dict[str, str] | None = None,
    ) -> dict[str, object] | None:
        if session_path is None:
            return None
        payload: dict[str, object] = {
            "agent_id": state.agent_id,
            "kind": state.kind.value,
            "name": state.name,
            "status": state.status,
            "statement": state.statement,
            "prompt": state.prompt,
            "response": state.response,
            "error": state.error,
            "session_path": str(state.session_path) if state.session_path is not None else "",
            "started_at": state.started_at,
            "updated_at": utc_now_iso(),
            "finished_at": state.finished_at,
            "context_tokens": state.context_tokens,
            "context_percent": state.context_percent,
        }
        if message:
            payload["message"] = message.strip()
        if command is not None:
            payload["command"] = command
        if state.command_history:
            payload["commands"] = list(state.command_history)
        self.home.append_session_event(session_path, SUBAGENT_EVENT_TYPE, payload)
        return payload

    def _running_subagent_states(self) -> tuple[SubagentRuntimeState, ...]:
        with self._subagent_lock:
            return tuple(
                state
                for state in self._subagents.values()
                if state.status in {"running", "working"}
                and state.worker is not None
                and state.worker.is_alive()
            )

    def _subagent_states(self) -> tuple[SubagentRuntimeState, ...]:
        with self._subagent_lock:
            return tuple(state for state in self._subagents.values() if state.status != "removed")

    def _end_all_subagent_states(self, session_path: Path | None = None) -> int:
        with self._subagent_lock:
            states = tuple(
                state
                for state in self._subagents.values()
                if state.status in {"running", "working"}
            )
        ended = 0
        for state in states:
            state.cancel_event.set()
            if state.runtime is not None:
                with suppress(Exception):
                    state.runtime.abort_current_turn(state.session_path)
                with suppress(Exception):
                    state.runtime.shutdown(state.session_path)
            state.status = "interrupted"
            state.statement = "Subagent was interrupted because Anomx was interrupted."
            state.finished_at = utc_now_iso()
            self._publish_subagent_state(state, session_path, message=state.statement)
            ended += 1
        return ended

    def end_process(self, process_id: str, session_path: Path | None = None) -> str:
        """End a running async process by id and append an updated process event."""

        return self._end_process(process_id, session_path)

    def _end_process_by_id_if_known(
        self,
        process_id: str,
        session_path: Path | None = None,
    ) -> bool:
        with self._process_lock:
            if process_id not in self._processes:
                return False
        self._end_process(process_id, session_path)
        return True

    def _end_all_process_states(self, session_path: Path | None = None) -> int:
        with self._process_lock:
            running_processes = tuple(
                process_state
                for process_state in self._processes.values()
                if process_state.status == "running"
            )

        ended_count = 0
        for process_state in running_processes:
            result = self._end_process(
                process_state.process_id,
                process_state.session_path or session_path,
            )
            with suppress(json.JSONDecodeError):
                if json.loads(result).get("ended") is True:
                    ended_count += 1
        return ended_count

    def _end_process(
        self,
        process_id: str,
        session_path: Path | None = None,
        callbacks: RuntimeCallbacks | None = None,
        *,
        allowed_sources: set[str] | None = None,
    ) -> str:
        with self._process_lock:
            process_state = self._processes.get(process_id)
            if process_state is None:
                return self._json_tool_result(
                    {"ended": False, "error": "Unknown process id."}
                )
            if allowed_sources is not None and process_state.source not in allowed_sources:
                return self._json_tool_result(
                    {"ended": False, "error": "Unknown command id."}
                )
            if process_state.status != "running":
                return self._json_tool_result(
                    {
                        "ended": True,
                        "process_id": process_id,
                        "status": process_state.status,
                    }
                )
            process_state.status = "ended"
            process_state.finished_at = utc_now_iso()
            self._publish_process_state(
                process_state,
                session_path or process_state.session_path,
                callbacks,
            )
            process = process_state.process

        self.tool_manager.terminate_process(process)
        with self._process_lock:
            process_state.exit_code = process.poll()
            self._publish_process_state(
                process_state,
                session_path or process_state.session_path,
                callbacks,
            )
        return self._json_tool_result(
            {"ended": True, "process_id": process_id, "status": process_state.status}
        )

    def _monitor_process(
        self,
        process_state: AsyncProcessState,
        session_path: Path | None,
        callbacks: RuntimeCallbacks | None = None,
    ) -> None:
        readers: list[threading.Thread] = []
        for stream in (process_state.process.stdout, process_state.process.stderr):
            if stream is None:
                continue
            reader = threading.Thread(
                target=self._read_process_stream,
                args=(process_state, stream, callbacks),
                daemon=True,
            )
            reader.start()
            readers.append(reader)

        return_code: int | None
        try:
            return_code = process_state.process.wait()
        except OSError as error:  # pragma: no cover - defensive process boundary
            self._append_process_output(process_state, str(error))
            return_code = process_state.process.poll()

        for reader in readers:
            reader.join(timeout=1)

        with self._process_lock:
            current = self._processes.get(process_state.process_id)
            if current is None:
                return
            current.exit_code = return_code
            if current.status == "running":
                current.status = "ended"
                current.finished_at = utc_now_iso()
                self._publish_process_state(current, session_path, callbacks)

    def _read_process_stream(
        self,
        process_state: AsyncProcessState,
        stream: TextIO,
        callbacks: RuntimeCallbacks | None = None,
    ) -> None:
        try:
            while True:
                chunk = stream.readline()
                if chunk == "":
                    break
                self._append_process_output(process_state, chunk, callbacks)
        except OSError as error:  # pragma: no cover - defensive process boundary
            self._append_process_output(process_state, str(error), callbacks)

    def _append_process_output(
        self,
        process_state: AsyncProcessState,
        chunk: str,
        callbacks: RuntimeCallbacks | None = None,
    ) -> None:
        if not chunk:
            return
        payload: dict[str, object] | None = None
        target_path: Path | None = None
        callback_state: AsyncProcessState | None = None
        with self._process_lock:
            current = self._processes.get(process_state.process_id)
            if current is None:
                return
            current.output_chunks.append(chunk)
            current.output = self._compact_process_output("".join(current.output_chunks))
            now = time.monotonic()
            if (
                current.status == "running"
                and current.session_path is not None
                and current.output != current.last_output_event_text
                and now - current.last_output_event_at >= 0.25
            ):
                current.last_output_event_at = now
                current.last_output_event_text = current.output
                target_path = current.session_path
                payload = self._process_state_payload(current)
                callback_state = current
        if payload is not None and target_path is not None:
            self.home.append_session_event(target_path, "process_event", payload)
            if (
                callbacks is not None
                and callbacks.process is not None
                and callback_state is not None
            ):
                callbacks.process(callback_state)

    def _publish_process_state(
        self,
        process_state: AsyncProcessState,
        session_path: Path | None,
        callbacks: RuntimeCallbacks | None = None,
    ) -> None:
        self._append_process_event(session_path or process_state.session_path, process_state)
        if callbacks is not None and callbacks.process is not None:
            callbacks.process(process_state)

    def _start_process_monitor(
        self,
        process_state: AsyncProcessState,
        session_path: Path | None,
        callbacks: RuntimeCallbacks | None = None,
    ) -> None:
        process_state.thread = threading.Thread(
            target=self._monitor_process,
            args=(process_state, session_path, callbacks),
            daemon=True,
        )
        process_state.thread.start()

    def _append_process_event(
        self,
        session_path: Path | None,
        process_state: AsyncProcessState,
    ) -> None:
        if session_path is None:
            return
        self.home.append_session_event(
            session_path,
            "process_event",
            self._process_state_payload(process_state),
        )

    def _append_command_event_snapshot(self, process_state: AsyncProcessState) -> None:
        if process_state.session_path is None:
            return
        self.home.append_session_event(
            process_state.session_path,
            "process_event",
            self._command_state_payload(process_state),
        )

    def _process_state_payload(self, process_state: AsyncProcessState) -> dict[str, object]:
        return {
            "process_id": process_state.process_id,
            "command": process_state.command,
            "statement": process_state.statement,
            "status": process_state.status,
            "output": process_state.output,
            "started_at": process_state.started_at,
            "finished_at": process_state.finished_at,
            "exit_code": process_state.exit_code,
            "source": process_state.source,
            "owner_id": process_state.owner_id,
            "owner_name": process_state.owner_name,
            "pid": process_state.process.pid,
        }

    def _command_state_payload(self, process_state: AsyncProcessState) -> dict[str, object]:
        payload = self._process_state_payload(process_state)
        payload["command_id"] = process_state.process_id
        payload["output"] = self._current_process_output(process_state)
        return payload

    def _current_process_output(self, process_state: AsyncProcessState) -> str:
        with self._process_lock:
            current = self._processes.get(process_state.process_id)
            if current is None:
                return process_state.output
            if current.output_chunks:
                return self._compact_process_output("".join(current.output_chunks))
            return current.output

    def _compact_process_output(self, output: str) -> str:
        compact = output.strip()
        if len(compact) <= 2_000:
            return compact
        return f"{compact[-1_997:]}"

    def _command_tool_sources(self) -> set[str]:
        return {"command"}

    def _command_states(self) -> tuple[AsyncProcessState, ...]:
        sources = self._command_tool_sources()
        with self._process_lock:
            return tuple(
                process_state
                for process_state in self._processes.values()
                if process_state.source in sources
            )

    def _running_command_states(self) -> tuple[AsyncProcessState, ...]:
        return tuple(
            process_state
            for process_state in self._command_states()
            if process_state.status == "running"
        )

    def _command_state(self, command_id: str) -> AsyncProcessState | None:
        sources = self._command_tool_sources()
        with self._process_lock:
            process_state = self._processes.get(command_id)
            if process_state is None or process_state.source not in sources:
                return None
            return process_state

    def _has_running_wait_targets(self) -> bool:
        return bool(self._running_command_states() or self._running_subagent_states())

    def _subagent_runtime_payload(self, state: SubagentRuntimeState) -> dict[str, object]:
        return {
            "agent_id": state.agent_id,
            "name": state.name,
            "agent_kind": state.kind.value,
            "status": state.status,
            "statement": state.statement,
            "response": state.response,
            "error": state.error,
            "session_path": str(state.session_path) if state.session_path is not None else "",
            "context_tokens": state.context_tokens,
            "context_percent": state.context_percent,
            "commands": list(state.command_history),
        }

    def _heuristic_session_continuation(
        self,
        messages: list[dict[str, Any]],
        workspace_name: str,
    ) -> str:
        topic = self._heuristic_session_title(messages)
        if topic:
            return f"Do you want to continue the previous session titled {topic}?"
        return f"Do you want to continue the previous Anomx session in {workspace_name}?"

    def _heuristic_session_title(self, messages: list[dict[str, Any]]) -> str | None:
        first_user_message = next(
            (
                str(message.get("content", ""))
                for message in messages
                if message.get("role") == "user" and message.get("content")
            ),
            "",
        )
        return self._sanitize_title(first_user_message)

    def _sanitize_title(self, title: str) -> str | None:
        cleaned = " ".join(title.strip().strip("\"'`").split())
        cleaned = cleaned.rstrip(".:;,-")
        if not cleaned:
            return None
        words = cleaned.split()
        if len(words) > 8:
            cleaned = " ".join(words[:8])
        return cleaned[:60] or None

    def _instructions(self, session_path: Path | None = None) -> str:
        tools = "\n".join(f"- {tool}" for tool in self._operator_tool_descriptions())
        runtime_context = self._operator_runtime_context(session_path)
        return "\n\n".join(
            [
                self.agent_spec.prompt,
                *self._instruction_environment_sections(),
                runtime_context,
                f"Available tools:\n{tools}",
            ]
        )

    def _instruction_environment_sections(self) -> list[str]:
        sections = [self.tool_manager.mode.system_prompt_statement]
        user_name = str(self.home.load_config().get("user_name") or "").strip()
        if user_name:
            sections.append(f"User profile:\n- Name: {user_name}")
        sections.append("\n".join(self.tool_manager.workspace_prompt_lines()))
        session_policy = self.tool_manager.session_policy_prompt_lines()
        if session_policy:
            sections.append("\n".join(session_policy))
        sandbox_section = self._sandbox_instruction_section()
        if sandbox_section:
            sections.append(sandbox_section)
        local_sandbox_section = self._local_sandbox_instruction_section()
        if local_sandbox_section:
            sections.append(local_sandbox_section)
        platform_section = self._platform_instruction_section()
        if platform_section:
            sections.append(platform_section)
        custom_section = self._custom_instructions_section()
        if custom_section:
            sections.append(custom_section)
        memory_section = self._memory_instruction_section()
        if memory_section:
            sections.append(memory_section)
        return sections

    def _sandbox_instruction_section(self) -> str | None:
        if self._sandbox_session is None:
            return None
        cfg = self._sandbox_session.config
        return cfg.sandbox_context_prompt()

    def _local_sandbox_instruction_section(self) -> str | None:
        if self._local_sandbox_session is None:
            return None
        return self._local_sandbox_session.config.sandbox_context_prompt()

    def _platform_instruction_section(self) -> str | None:
        connection = self.home.platform_connection()
        if connection is None:
            return None

        lines = [
            (
                CONNECTED_PLATFORM_AGENT_PROMPT.strip()
                if self.agent_spec.can_spawn_subagents
                else "## Connected Anomx Platform\n- A user-connected Anomx Platform is available for this session."
            ),
            f"- Platform API base URL: {platform_api_base_url(connection['url'])}",
            f"- Raw API responses are written to: {self.home.responses_dir}",
            "- Platform API environment variables are available to commands: "
            "ANOMX_PLATFORM_API_URL, ANOMX_PLATFORM_API_KEY, ANOMX_PLATFORM_TOKEN, "
            "ANOMX_API_KEY, and ANOMX_RESPONSES_DIR.",
            "- The helper folder is synced to ~/.anomx/skills/use-anomx-api and includes "
            "api.py for custom Python scripts.",
        ]
        if self.agent_spec.can_spawn_subagents:
            lines.append(
                "- Do not perform platform API discovery directly from the main agent. Use a "
                "`platform` subagent for platform data, objects, jobs, DAQ, anomaly detection, "
                "folders, pages, files, users, integrations, services, nodes, and endpoints."
            )
            if self.can_output_response():
                lines.append(
                    "- The `output_response` tool can render rich platform outputs such as text, "
                    "object cards, full objects, object grids/lists, and object forms. If the "
                    "user asks for a specific platform object, list, form, or database-style "
                    "result, use `output_response` at the very end."
                )
        else:
            lines.extend(
                [
                    "- The `use_anomx_api` tool is available. It returns metadata only and "
                    "stores the full response payload as a JSON file.",
                ]
            )
            for skill in load_system_skills():
                if skill.command != "use-anomx-api":
                    continue
                lines.extend(["", skill.body.strip()])
        platform_instructions = str(self.home.load_config().get("platform_instructions") or "").strip()
        if platform_instructions:
            lines.extend(
                [
                    "",
                    "## Platform Instructions",
                    platform_instructions,
                ]
            )
        return "\n".join(lines)

    def _custom_instructions_section(self) -> str | None:
        """Read custom instruction files and return a formatted section, or None."""
        instructions_dir = self.home.instructions_dir
        if not instructions_dir.is_dir():
            return None
        instruction_path = instructions_dir / "instruction.md"
        if not instruction_path.exists():
            return None
        content = instruction_path.read_text(encoding="utf-8").strip()
        if not content:
            return None
        return "## Custom Instructions\n\n" + content

    def _memory_instruction_section(self) -> str | None:
        """Return a compact memory context section, or None."""

        records = load_memories(self.home.brain_dir)[:12]
        if not records:
            return None
        increment_memory_uses(records)
        lines = [
            "## Memories",
            "",
            (
                "Durable local memories are stored as JSON files in ~/.anomx/brain "
                f"(current path: {self.home.brain_dir})."
            ),
            "Use these titles and summaries when they are relevant. You are encouraged "
            "to inspect that folder yourself when you need matching memory details.",
        ]
        for record in records:
            lines.extend(
                [
                    "",
                    f"- {record.title} ({record.kind.value}, uses {record.uses})",
                    f"  Summary: {record.summary}",
                ]
            )
        return "\n".join(lines)

    def _operator_tool_descriptions(self) -> tuple[str, ...]:
        descriptions = [
            f"{tool.name}: {tool.description}"
            for tool in self.agent_spec.tools
        ]
        if self._running_command_states():
            descriptions.extend(
                [
                    (
                        "check_command_status(command_id): inspect your own active "
                        "long-running command and read its current CLI output."
                    ),
                    "kill_command(command_id): kill your own active long-running command.",
                ]
            )
        if self._running_command_states() or self._running_subagent_states():
            descriptions.append(
                "wait(): wait 60 seconds for your active command tool calls or subagents."
            )
        return tuple(descriptions)

    def _operator_runtime_context(self, session_path: Path | None) -> str:
        if session_path is None:
            return "Runtime context:\n- No active session context."

        events = self.home.read_session_events(session_path)
        plan_steps = latest_plan_steps(events)
        processes = running_process_snapshots(events)
        subagents = subagent_snapshots(events)
        lines = ["Runtime context:"]
        if plan_steps and self.agent_spec.can_use_plans:
            lines.append("- Current plan:")
            for step in plan_steps:
                state = "done" if step.is_done else "open"
                lines.append(f"  {step.position}. [{state}] {step.title}: {step.description}")
            lines.append(
                "- Plan guidance: keep the current plan up to date as work changes. "
                "Ideally, every plan step should be marked done before ending operations; "
                "update or remove stale steps instead of leaving them open."
            )
        else:
            lines.append("- Current plan: none.")



        if processes:
            lines.append("- Async processes:")
            for process in processes:
                label = process.statement or process.command
                source_label = self._process_runtime_source_label(process)
                lines.append(
                    "  "
                    f"{process.process_id} · {source_label} · state=running · {label} · "
                    f"running for {self._runtime_duration(process.started_at)} · "
                    f"command: {process.command}"
                )
        else:
            lines.append("- Async processes: none.")
        if self.agent_spec.can_spawn_subagents:
            if subagents:
                lines.append("- Subagents:")
                for subagent in subagents:
                    latest = (
                        subagent.statement
                        or subagent.response
                        or subagent.error
                        or "No output yet"
                    )
                    context = (
                        f"{subagent.context_percent}% context"
                        if subagent.context_percent
                        else "context unknown"
                    )
                    lines.append(
                        "  "
                        f"{subagent.agent_id} · {subagent.kind} · {subagent.name} · "
                        f"status={subagent.status} · {context} · latest: {latest}"
                    )
            else:
                lines.append("- Subagents: none.")
        return "\n".join(lines)

    def _process_runtime_source_label(self, process: object) -> str:
        source = str(getattr(process, "source", "")).strip()
        if source == "command":
            return "operator command"
        return "process"

    def _runtime_duration(self, started_at: str) -> str:
        with suppress(ValueError):
            from datetime import UTC, datetime

            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            seconds = max(0, int((datetime.now(tz=UTC) - started).total_seconds()))
            minutes, remaining_seconds = divmod(seconds, 60)
            return f"{minutes:02d}:{remaining_seconds:02d}"
        return "unknown"

    def _json_tool_result(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _status(
        self,
        status_callback: StatusCallback | None,
        message: str = "Thinking",
    ) -> None:
        if status_callback is not None:
            status_callback(message)

    def _reasoning_status(
        self,
        status_callback: StatusCallback | None,
        reasoning_text: str,
    ) -> None:
        if status_callback is not None and reasoning_text.strip():
            status_callback("Thinking")

    def _compact_reasoning_text(self, reasoning_text: str) -> str:
        compact = " ".join(reasoning_text.strip().split())
        if not compact:
            return ""
        if len(compact) <= 120:
            return compact
        tail = compact[-117:]
        first_space = tail.find(" ")
        if first_space > 0:
            tail = tail[first_space + 1 :]
        return f"…{tail}"

    def _emit_command_system_message(
        self,
        callbacks: RuntimeCallbacks,
        result: CommandResult | CommandProcessResult,
        statement: str,
    ) -> None:
        if callbacks.system_message is None:
            return
        if result.blocked_by_mode:
            return
        if result.safety == CommandSafety.FORBIDDEN:
            display_statement = statement.strip() or result.command
            callbacks.system_message(
                "forbidden",
                f"Blocked: {display_statement}\nCommand: {result.command}\nReason: {result.reason}",
            )
