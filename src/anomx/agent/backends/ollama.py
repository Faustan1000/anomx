"""Ollama backend."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from anomx.agent.base.backends import (
    MAX_TOOL_ITERATIONS,
    BackendCallbacks,
    BaseBackend,
    OllamaStreamResponse,
    OllamaToolCall,
    ThinkingTagStreamFilter,
)
from anomx.agent.helpers.tool_manager import CommandRiskEvaluation
from anomx.agent.memories import MemoryKind, MemoryMetadata


class OllamaBackend(BaseBackend):
    """Local Ollama chat backend."""

    provider_key = "ollama"
    provider_label = "Ollama"
    env_var = ""

    def generate(
        self,
        session_path: Path,
        model: str,
        callbacks: BackendCallbacks,
        *,
        thinking_intensity: str | None = None,
    ) -> str:
        """Generate a response through a local Ollama server."""

        del thinking_intensity
        messages = [
            {"role": "system", "content": self.runtime._instructions(session_path)},
            *self._ollama_messages(self.runtime.conversation_messages(session_path), model),
        ]
        plan_finish_attempts = 0
        for _ in range(MAX_TOOL_ITERATIONS):
            if self.runtime._turn_aborted():
                return ""
            messages[0] = {
                "role": "system",
                "content": self.runtime._instructions(session_path),
            }
            response = self._stream_ollama_response(model, messages, callbacks)
            if isinstance(response, str):
                return response
            if self.runtime._turn_aborted():
                return ""

            if response.message:
                messages.append(response.message)
            if not response.tool_calls:
                text = response.text or "No response."
                continuation_prompt, used_plan_guard = (
                    self.runtime._continuation_prompt_after_text(
                        text,
                        callbacks,
                        session_path,
                        plan_finish_attempts,
                    )
                )
                if continuation_prompt is not None:
                    if used_plan_guard:
                        plan_finish_attempts += 1
                    messages.append({"role": "user", "content": continuation_prompt})
                    continue
                final_text = text
                if callbacks.finish is not None:
                    callbacks.finish(final_text)
                return final_text

            messages.extend(
                self._execute_ollama_requested_tools(
                    response,
                    callbacks,
                    session_path,
                )
            )

        return f"Ollama tool loop stopped after {MAX_TOOL_ITERATIONS} tool batches."

    def _stream_ollama_response(
        self,
        model: str,
        messages: list[dict[str, Any]],
        callbacks: BackendCallbacks,
    ) -> OllamaStreamResponse | str:
        payload = {
            "model": model,
            "messages": messages,
            "tools": self._ollama_tools(),
            "stream": True,
            "think": True,
        }
        self.runtime._status(callbacks.status, "Loading model")
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        def stream_once() -> OllamaStreamResponse:
            self.runtime._debug_log_step(self.provider_key, payload)
            thinking_parts: list[str] = []
            text_parts: list[str] = []
            text_filter = ThinkingTagStreamFilter()
            tool_calls: list[OllamaToolCall] = []
            with urllib.request.urlopen(request, timeout=120) as response:
                self.runtime._status(callbacks.status, "Thinking")
                for raw_line in response:
                    if self.runtime._turn_aborted():
                        return OllamaStreamResponse("", "", (), {"role": "assistant"})
                    stripped = raw_line.decode("utf-8", errors="replace").strip()
                    if not stripped:
                        continue
                    data = cast(dict[str, Any], json.loads(stripped))
                    stream_message = data.get("message")
                    if not isinstance(stream_message, dict):
                        continue
                    thinking = stream_message.get("thinking")
                    if isinstance(thinking, str) and thinking:
                        thinking_parts.append(thinking)
                        self.runtime._reasoning_status(
                            callbacks.status,
                            "".join(thinking_parts),
                        )
                    content = stream_message.get("content")
                    if isinstance(content, str) and content:
                        visible = self._visible_stream_text(
                            text_filter,
                            content,
                            callbacks.delta,
                            callbacks.status,
                            callbacks.thought,
                        )
                        if visible:
                            text_parts.append(visible)
                    raw_tool_calls = stream_message.get("tool_calls")
                    if isinstance(raw_tool_calls, list):
                        for item in raw_tool_calls:
                            tool_call = self._ollama_tool_call(item)
                            if tool_call is not None:
                                tool_calls.append(tool_call)

            trailing_text = self._finish_visible_stream_text(
                text_filter,
                callbacks.delta,
                callbacks.status,
                callbacks.thought,
            )
            if trailing_text:
                text_parts.append(trailing_text)

            thought = "".join(thinking_parts).strip()
            if thought:
                if callbacks.thought is not None:
                    callbacks.thought(thought)
                else:
                    self.runtime._status(callbacks.status, "Created a thought")

            assistant_message: dict[str, Any] = {"role": "assistant"}
            if thinking_parts:
                assistant_message["thinking"] = "".join(thinking_parts)
            if text_parts:
                assistant_message["content"] = "".join(text_parts)
            if tool_calls:
                assistant_message["tool_calls"] = [
                    self._ollama_tool_payload(tool_call) for tool_call in tool_calls
                ]
            return OllamaStreamResponse(
                "".join(text_parts).strip(),
                thought,
                tuple(tool_calls),
                assistant_message,
            )

        if self.runtime._turn_aborted():
            return ""
        response = self._model_request_with_retries(
            provider_key=self.provider_key,
            provider_label=self.provider_label,
            env_var=self.env_var,
            status_callback=callbacks.status,
            stream_once=stream_once,
        )
        if self.runtime._turn_aborted():
            return ""
        return cast(OllamaStreamResponse | str, response)

    def _ollama_tool_call(self, item: object) -> OllamaToolCall | None:
        if not isinstance(item, dict):
            return None
        function = item.get("function")
        if not isinstance(function, dict):
            return None
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str):
            return None
        if isinstance(arguments, dict):
            parsed_arguments = cast(dict[str, Any], arguments)
        elif isinstance(arguments, str):
            parsed_arguments = self._parse_tool_arguments(arguments)
        else:
            parsed_arguments = {}
        return OllamaToolCall(name=name, arguments=parsed_arguments)

    def _ollama_tool_payload(self, tool_call: OllamaToolCall) -> dict[str, Any]:
        return {
            "function": {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
        }

    def suggest_session_title(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str | None:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Name this CLI session in 3 to 6 words. "
                        "Return only the title. No quotes. No trailing punctuation."
                    ),
                },
                {"role": "user", "content": self._title_prompt(messages)},
            ],
            "stream": False,
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        message = data.get("message")
        if not isinstance(message, dict):
            return None
        return self._sanitize_title(str(message.get("content", "")))

    def evaluate_command_request(
        self,
        *,
        command: str,
        statement: str,
        user_message: str,
        model: str,
    ) -> CommandRiskEvaluation | None:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._command_evaluation_system_prompt()},
                {
                    "role": "user",
                    "content": self._command_evaluation_user_prompt(
                        command=command,
                        statement=statement,
                        user_message=user_message,
                    ),
                },
            ],
            "stream": False,
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        message = data.get("message")
        if not isinstance(message, dict):
            return None
        return self._sanitize_command_evaluation(str(message.get("content", "")))

    def suggest_memory_metadata(
        self,
        *,
        kind: MemoryKind | str,
        context: Mapping[str, Any],
        content: str,
        model: str,
    ) -> MemoryMetadata | None:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._memory_metadata_system_prompt()},
                {
                    "role": "user",
                    "content": self._memory_metadata_user_prompt(
                        kind=kind,
                        context=context,
                        content=content,
                    ),
                },
            ],
            "stream": False,
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        message = data.get("message")
        if not isinstance(message, dict):
            return None
        return self._sanitize_memory_metadata(str(message.get("content", "")))

    def suggest_project_name(self, prompt: str, model: str) -> str | None:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._project_name_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        message = data.get("message")
        if not isinstance(message, dict):
            return None
        return self._sanitize_project_name(str(message.get("content", "")))

    def suggest_session_continuation(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str | None:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._continuation_system_prompt()},
                {"role": "user", "content": self._title_prompt(messages)},
            ],
            "stream": False,
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        message = data.get("message")
        if not isinstance(message, dict):
            return None
        return self._sanitize_continuation_statement(str(message.get("content", "")))
