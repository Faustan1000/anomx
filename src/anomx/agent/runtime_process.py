"""Process-isolated client for the interface-agnostic agent runtime.

The terminal UI owns input and rendering.  A long-lived child process owns the
model backend, tools, sandboxes, subagents, and commands.  This module bridges
the two with a small request/event protocol so slow runtime work can never hold
the terminal UI's GIL.
"""

# ruff: noqa: ANN401

from __future__ import annotations

import multiprocessing
import queue
import threading
import traceback
from collections.abc import Callable, Mapping, MutableSet
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.utils import AgentKind
from anomx.agent.store import AnomxHome

if TYPE_CHECKING:
    from anomx.agent.base.interactions import QuestionResponse
    from anomx.agent.helpers.tool_manager import (
        ApprovalChoice,
        CommandApprovalRequest,
        CommandRiskEvaluation,
    )
    from anomx.agent.memories import MemoryKind, MemoryMetadata
    from anomx.agent.runtime import RuntimeCallbacks, RuntimeCleanupResult

Callback = Callable[..., Any]


class RuntimeProcessError(RuntimeError):
    """Raised when the isolated runtime cannot complete an RPC request."""


@dataclass(frozen=True, slots=True)
class RuntimeProcessConfig:
    """Serializable construction arguments for an isolated runtime."""

    home_root: str
    cwd: str
    allowed_commands: tuple[str, ...]
    rejected_commands: tuple[str, ...]
    mode: str
    agent_kind: str
    workspace_root: str
    process_owner_id: str = ""
    process_owner_name: str = ""
    local_sandbox_enabled: bool = False
    local_sandbox_home: str = ""
    local_sandbox_allow_subprocess: bool = False
    platform_chat_id: str = ""
    additional_instructions: str = ""


class RuntimeProcessClient:
    """Proxy an :class:`AgentRuntime` hosted in a dedicated process."""

    def __init__(
        self,
        home: AnomxHome,
        cwd: Path,
        session_allowed_commands: MutableSet[str] | None = None,
        session_rejected_commands: MutableSet[str] | None = None,
        mode: AgentMode = AgentMode.STANDARD,
        agent_kind: AgentKind | str = AgentKind.MAIN,
        workspace_root: Path | None = None,
        process_owner_id: str = "",
        process_owner_name: str = "",
        local_sandbox_enabled: bool = False,
        local_sandbox_home: Path | None = None,
        local_sandbox_allow_subprocess: bool = False,
        platform_chat_id: str = "",
        additional_instructions: str = "",
    ) -> None:
        resolved_cwd = cwd.expanduser().resolve()
        resolved_workspace = (workspace_root or resolved_cwd).expanduser().resolve()
        parsed_kind = AgentKind(str(getattr(agent_kind, "value", agent_kind)))
        self.config = RuntimeProcessConfig(
            home_root=str(home.root),
            cwd=str(resolved_cwd),
            allowed_commands=tuple(session_allowed_commands or ()),
            rejected_commands=tuple(session_rejected_commands or ()),
            mode=AgentMode.parse(mode).value,
            agent_kind=parsed_kind.value,
            workspace_root=str(resolved_workspace),
            process_owner_id=process_owner_id,
            process_owner_name=process_owner_name,
            local_sandbox_enabled=local_sandbox_enabled,
            local_sandbox_home=str(local_sandbox_home or ""),
            local_sandbox_allow_subprocess=local_sandbox_allow_subprocess,
            platform_chat_id=platform_chat_id,
            additional_instructions=additional_instructions,
        )
        self._context = multiprocessing.get_context("spawn")
        self._connection: Connection | None = None
        self._process: BaseProcess | None = None
        self._listener: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[str, queue.Queue[tuple[bool, Any]]] = {}
        self._callbacks: dict[str, dict[str, Callback]] = {}
        self._ready = threading.Event()
        self._startup_error = ""
        self._closed = False

    @property
    def process_id(self) -> int | None:
        """Return the runtime child PID after it has started."""

        return self._process.pid if self._process is not None else None

    @property
    def is_process_alive(self) -> bool:
        """Return whether the isolated runtime process is alive."""

        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        """Start the runtime process without waiting for its imports or setup."""

        self._ensure_started()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Wait for child initialization while another process renders the UI."""

        self._ensure_started()
        self._ready.wait(timeout)
        return self._ready.is_set() and not self._startup_error

    def set_mode(self, mode: AgentMode) -> None:
        self._call("set_mode", AgentMode.parse(mode))

    def set_agent(self, kind: AgentKind | str) -> None:
        self._call("set_agent", kind)

    def abort_current_turn(self, session_path: Path | None = None) -> None:
        self._call("abort_current_turn", session_path)

    def shutdown(self, session_path: Path | None = None) -> RuntimeCleanupResult:
        result = cast("RuntimeCleanupResult", self._call("shutdown", session_path))
        if session_path is None:
            self.close()
        return result

    def cleanup_session_runtime_state(self, session_path: Path) -> RuntimeCleanupResult:
        return cast(
            "RuntimeCleanupResult",
            self._call("cleanup_session_runtime_state", session_path),
        )

    def init_sandbox(
        self,
        config: Mapping[str, Any] | None = None,
        status_callback: Callback | None = None,
    ) -> bool:
        callbacks = {"status": status_callback} if status_callback is not None else {}
        return bool(self._call("init_sandbox", config, _callbacks=callbacks))

    def init_python_sandbox(
        self,
        *,
        home: Path | None = None,
        allow_subprocess: bool = True,
        status_callback: Callback | None = None,
    ) -> bool:
        callbacks = {"status": status_callback} if status_callback is not None else {}
        return bool(
            self._call(
                "init_python_sandbox",
                home=home,
                allow_subprocess=allow_subprocess,
                _callbacks=callbacks,
            )
        )

    def backend_response(
        self,
        session_path: Path,
        callbacks: RuntimeCallbacks | None = None,
        *,
        debug_session_path: Path | None = None,
    ) -> str:
        return str(
            self._call(
                "backend_response",
                session_path,
                debug_session_path=debug_session_path,
                _callbacks=self._callback_map(callbacks),
            )
        )

    def backend_response_for_prompt(
        self,
        prompt: str,
        callbacks: RuntimeCallbacks | None = None,
        *,
        debug_session_path: Path | None = None,
        parent_session_path: Path | None = None,
        prompt_message_id: str = "",
    ) -> str:
        return str(
            self._call(
                "backend_response_for_prompt",
                prompt,
                debug_session_path=debug_session_path,
                parent_session_path=parent_session_path,
                prompt_message_id=prompt_message_id,
                _callbacks=self._callback_map(callbacks),
            )
        )

    def openai_response(self, session_path: Path, model: str) -> str:
        return str(self._call("openai_response", session_path, model))

    def conversation_messages(self, session_path: Path) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self._call("conversation_messages", session_path))

    def backend_conversation_messages(self, session_path: Path) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            self._call("backend_conversation_messages", session_path),
        )

    def estimate_session_context_tokens(self, session_path: Path) -> int:
        return int(self._call("estimate_session_context_tokens", session_path))

    def suggest_session_title(self, session_path: Path) -> str | None:
        return cast(str | None, self._call("suggest_session_title", session_path))

    def suggest_project_name(self, project_path: Path, directory_outline: str) -> str | None:
        return cast(
            str | None,
            self._call("suggest_project_name", project_path, directory_outline),
        )

    def suggest_session_continuation(self, session_path: Path, workspace_name: str) -> str:
        return str(self._call("suggest_session_continuation", session_path, workspace_name))

    def evaluate_command_request(
        self,
        session_path: Path,
        request: CommandApprovalRequest,
    ) -> CommandRiskEvaluation | None:
        return cast(
            "CommandRiskEvaluation | None",
            self._call("evaluate_command_request", session_path, request),
        )

    def suggest_memory_metadata(
        self,
        *,
        kind: MemoryKind | str,
        context: Mapping[str, Any],
        content: str,
    ) -> MemoryMetadata:
        return cast(
            "MemoryMetadata",
            self._call(
                "suggest_memory_metadata",
                kind=kind,
                context=context,
                content=content,
            ),
        )

    def extract_openai_text(self, data: dict[str, object]) -> str:
        return str(self._call("extract_openai_text", data))

    def end_process(self, process_id: str, session_path: Path | None = None) -> str:
        return str(self._call("end_process", process_id, session_path))

    def is_sandbox_active(self) -> bool:
        return bool(self._call("is_sandbox_active"))

    def close(self) -> None:
        """Stop the child process and release IPC resources."""

        if self._closed:
            return
        self._closed = True
        connection = self._connection
        process = self._process
        if connection is not None and process is not None and process.is_alive():
            with suppress(BrokenPipeError, EOFError, OSError):
                self._send({"type": "close"})
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        if connection is not None:
            connection.close()
        self._fail_pending("The runtime process was closed.")

    def _ensure_started(self) -> None:
        if self._closed:
            raise RuntimeProcessError("The runtime process is closed.")
        if self._process is not None:
            return
        with self._start_lock:
            if self._process is not None:
                return
            parent_connection, child_connection = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=_runtime_process_main,
                args=(child_connection, self.config),
                name="anomx-agent-runtime",
                daemon=True,
            )
            process.start()
            child_connection.close()
            self._connection = parent_connection
            self._process = process
            self._listener = threading.Thread(
                target=self._listen,
                name="anomx-runtime-events",
                daemon=True,
            )
            self._listener.start()

    def _call(
        self,
        method: str,
        *args: Any,
        _callbacks: Mapping[str, Callback] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._ensure_started()
        request_id = uuid4().hex
        response: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        callback_map = dict(_callbacks or {})
        with self._pending_lock:
            self._pending[request_id] = response
            if callback_map:
                self._callbacks[request_id] = callback_map
        try:
            self._send(
                {
                    "type": "call",
                    "request_id": request_id,
                    "method": method,
                    "args": args,
                    "kwargs": kwargs,
                    "callback_names": tuple(callback_map),
                }
            )
            ok, value = response.get()
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)
                self._callbacks.pop(request_id, None)
        if ok:
            return value
        raise RuntimeProcessError(str(value))

    def _send(self, payload: dict[str, Any]) -> None:
        connection = self._connection
        if connection is None:
            raise RuntimeProcessError("The runtime process has not started.")
        with self._send_lock:
            connection.send(payload)

    def _listen(self) -> None:
        connection = self._connection
        if connection is None:
            return
        try:
            while True:
                payload = connection.recv()
                message_type = payload.get("type")
                if message_type == "result":
                    request_id = str(payload.get("request_id") or "")
                    with self._pending_lock:
                        response = self._pending.get(request_id)
                    if response is not None:
                        response.put((bool(payload.get("ok")), payload.get("value")))
                elif message_type == "event":
                    self._dispatch_event(payload)
                elif message_type == "ready":
                    self._ready.set()
                elif message_type == "fatal":
                    self._startup_error = str(
                        payload.get("error") or "Runtime process failed."
                    )
                    self._ready.set()
                    self._fail_pending(self._startup_error)
                    return
        except (EOFError, OSError):
            self._startup_error = "The runtime process exited unexpectedly."
            self._ready.set()
            self._fail_pending(self._startup_error)

    def _dispatch_event(self, payload: Mapping[str, Any]) -> None:
        request_id = str(payload.get("request_id") or "")
        callback_name = str(payload.get("callback") or "")
        with self._pending_lock:
            callback = self._callbacks.get(request_id, {}).get(callback_name)
        if callback is None:
            return
        args = tuple(payload.get("args") or ())
        interaction_id = str(payload.get("interaction_id") or "")
        if interaction_id:
            threading.Thread(
                target=self._answer_interaction,
                args=(callback, args, interaction_id),
                name=f"anomx-runtime-{callback_name}",
                daemon=True,
            ).start()
            return
        try:
            callback(*args)
        except Exception:
            return

    def _answer_interaction(
        self,
        callback: Callback,
        args: tuple[Any, ...],
        interaction_id: str,
    ) -> None:
        try:
            value = callback(*args)
        except Exception:
            value = self._fallback_interaction_value(callback)
        try:
            self._send(
                {
                    "type": "interaction_response",
                    "interaction_id": interaction_id,
                    "value": value,
                }
            )
        except (BrokenPipeError, EOFError, OSError, RuntimeProcessError):
            return

    @staticmethod
    def _fallback_interaction_value(callback: Callback) -> ApprovalChoice | QuestionResponse:
        from anomx.agent.base.interactions import QuestionResponse
        from anomx.agent.helpers.tool_manager import ApprovalChoice

        if "question" in getattr(callback, "__name__", ""):
            return QuestionResponse(answered=False, cancelled=True)
        return ApprovalChoice.REJECT

    @staticmethod
    def _callback_map(callbacks: RuntimeCallbacks | None) -> dict[str, Callback]:
        if callbacks is None:
            return {}
        names = (
            "status",
            "message",
            "tool_message",
            "thought",
            "command",
            "output_response",
            "subagent",
            "delta",
            "approval",
            "system_message",
            "question",
            "process",
            "finish",
            "usage",
        )
        return {
            name: callback
            for name in names
            if (callback := getattr(callbacks, name, None)) is not None
        }

    def _fail_pending(self, message: str) -> None:
        with self._pending_lock:
            pending = tuple(self._pending.values())
        for response in pending:
            try:
                response.put_nowait((False, message))
            except queue.Full:
                continue


def _runtime_process_main(connection: Connection, config: RuntimeProcessConfig) -> None:
    """Construct and serve one runtime inside the child process."""

    runtime: Any = None
    send_lock = threading.Lock()
    interactions: dict[str, queue.Queue[Any]] = {}
    interactions_lock = threading.Lock()

    def send(payload: dict[str, Any]) -> None:
        with send_lock:
            connection.send(payload)

    try:
        from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks

        runtime = AgentRuntime(
            AnomxHome(Path(config.home_root)),
            Path(config.cwd),
            set(config.allowed_commands),
            set(config.rejected_commands),
            AgentMode.parse(config.mode),
            agent_kind=config.agent_kind,
            workspace_root=Path(config.workspace_root),
            process_owner_id=config.process_owner_id,
            process_owner_name=config.process_owner_name,
            local_sandbox_enabled=config.local_sandbox_enabled,
            local_sandbox_home=(
                Path(config.local_sandbox_home) if config.local_sandbox_home else None
            ),
            local_sandbox_allow_subprocess=config.local_sandbox_allow_subprocess,
            platform_chat_id=config.platform_chat_id,
            additional_instructions=config.additional_instructions,
        )
        send({"type": "ready"})

        def callbacks_for(request: Mapping[str, Any]) -> RuntimeCallbacks:
            request_id = str(request.get("request_id") or "")
            names = set(request.get("callback_names") or ())

            def callback(name: str, *, interactive: bool = False) -> Callback | None:
                if name not in names:
                    return None

                def emit(*args: Any) -> Any:
                    interaction_id = uuid4().hex if interactive else ""
                    response: queue.Queue[Any] | None = None
                    if interaction_id:
                        response = queue.Queue(maxsize=1)
                        with interactions_lock:
                            interactions[interaction_id] = response
                    send(
                        {
                            "type": "event",
                            "request_id": request_id,
                            "callback": name,
                            "args": args,
                            "interaction_id": interaction_id,
                        }
                    )
                    if response is None:
                        return None
                    try:
                        return response.get()
                    finally:
                        with interactions_lock:
                            interactions.pop(interaction_id, None)

                return emit

            return RuntimeCallbacks(
                status=callback("status"),
                message=callback("message"),
                tool_message=callback("tool_message"),
                thought=callback("thought"),
                command=callback("command"),
                output_response=callback("output_response"),
                subagent=callback("subagent"),
                delta=callback("delta"),
                approval=callback("approval", interactive=True),
                system_message=callback("system_message"),
                question=callback("question", interactive=True),
                process=callback("process"),
                finish=callback("finish"),
                usage=callback("usage"),
            )

        def execute(request: dict[str, Any]) -> None:
            request_id = str(request.get("request_id") or "")
            try:
                method_name = str(request.get("method") or "")
                method = getattr(runtime, method_name)
                args = tuple(request.get("args") or ())
                kwargs = dict(request.get("kwargs") or {})
                callback_names = set(request.get("callback_names") or ())
                if method_name in {"backend_response", "backend_response_for_prompt"}:
                    kwargs["callbacks"] = callbacks_for(request)
                elif method_name in {"init_sandbox", "init_python_sandbox"}:
                    status = callbacks_for(request).status if "status" in callback_names else None
                    kwargs["status_callback"] = status
                value = method(*args, **kwargs)
                send({"type": "result", "request_id": request_id, "ok": True, "value": value})
            except BaseException as exc:
                error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                send({"type": "result", "request_id": request_id, "ok": False, "value": error})

        while True:
            request = connection.recv()
            message_type = request.get("type")
            if message_type == "close":
                runtime.shutdown()
                return
            if message_type == "interaction_response":
                interaction_id = str(request.get("interaction_id") or "")
                with interactions_lock:
                    response = interactions.get(interaction_id)
                if response is not None:
                    response.put(request.get("value"))
                continue
            if message_type != "call":
                continue
            threading.Thread(
                target=execute,
                args=(request,),
                name=f"anomx-runtime-{request.get('method', 'call')}",
                daemon=True,
            ).start()
    except BaseException as exc:
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        with suppress(BrokenPipeError, EOFError, OSError):
            send({"type": "fatal", "error": error})
    finally:
        if runtime is not None:
            with suppress(Exception):
                runtime.shutdown()
        connection.close()


__all__ = ["RuntimeProcessClient", "RuntimeProcessConfig", "RuntimeProcessError"]
