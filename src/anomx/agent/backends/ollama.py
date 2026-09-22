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
    ollama_token_usage,
)
from anomx.agent.context_management import (
    CONTINUE_AFTER_COMPRESSION_PROMPT,
    ContextMessage,
    projected_context_tokens,
    transient_context_message,
)
from anomx.agent.exceptions import BackendFailure
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
        context_entries = self.runtime.backend_conversation_entries(session_path)
        messages = [
            {"role": "system", "content": self.runtime._instructions(session_path)},
            *self._ollama_messages(
                [entry.payload for entry in context_entries],
                model,
            ),
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
                recovered_entries = self._recover_context_window(
                    response, session_path, context_entries, callbacks,
                )
                if recovered_entries is None:
                    return response
                context_entries = recovered_entries
                messages = [
                    {"role": "system", "content": self.runtime._instructions(session_path)},
                    *self._ollama_messages([entry.payload for entry in context_entries], model),
                ]
                continue
            if self.runtime._turn_aborted():
                return ""
            self._track_usage(response.usage, callbacks)

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
                    pending_entries = self._ollama_context_entries(response, ())
                    pending_entries.append(
                        transient_context_message("user", continuation_prompt)
                    )
                    context_entries.extend(pending_entries)
                    context_entries, compressed = self.runtime.compress_in_turn_context(
                        session_path,
                        context_entries,
                        current_context_tokens=projected_context_tokens(
                            response.usage.input_tokens if response.usage else 0,
                            response.usage.output_tokens if response.usage else 0,
                            pending_entries[1:],
                        ),
                        status_callback=callbacks.status,
                    )
                    if compressed:
                        context_entries.append(
                            transient_context_message(
                                "user",
                                CONTINUE_AFTER_COMPRESSION_PROMPT,
                            )
                        )
                        messages = [
                            {
                                "role": "system",
                                "content": self.runtime._instructions(session_path),
                            },
                            *self._ollama_messages(
                                [entry.payload for entry in context_entries],
                                model,
                            ),
                        ]
                    else:
                        messages.append(
                            {"role": "user", "content": continuation_prompt}
                        )
                    continue
                final_text = text
                if callbacks.finish is not None:
                    callbacks.finish(final_text)
                return final_text

            tool_messages = self._execute_ollama_requested_tools(
                response,
                callbacks,
                session_path,
            )
            messages.extend(tool_messages)
            pending_entries = self._ollama_context_entries(response, tool_messages)
            context_entries.extend(pending_entries)
            context_entries, compressed = self.runtime.compress_in_turn_context(
                session_path,
                context_entries,
                current_context_tokens=projected_context_tokens(
                    response.usage.input_tokens if response.usage else 0,
                    response.usage.output_tokens if response.usage else 0,
                    pending_entries[1:],
                ),
                status_callback=callbacks.status,
            )
            if compressed:
                context_entries.append(
                    transient_context_message(
                        "user",
                        CONTINUE_AFTER_COMPRESSION_PROMPT,
                    )
                )
                messages = [
                    {
                        "role": "system",
                        "content": self.runtime._instructions(session_path),
                    },
                    *self._ollama_messages(
                        [entry.payload for entry in context_entries],
                        model,
                    ),
                ]

        return BackendFailure(
            f"Ollama tool loop stopped after {MAX_TOOL_ITERATIONS} tool batches.",
            code="tool_limit_exceeded",
        )

    def _ollama_context_entries(
        self,
        response: OllamaStreamResponse,
        tool_outputs: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    ) -> list[ContextMessage]:
        assistant_parts = [response.text.strip()] if response.text.strip() else []
        assistant_parts.extend(
            (
                f"[Tool call: {tool_call.name}]\n"
                f"{json.dumps(tool_call.arguments, ensure_ascii=False, sort_keys=True)}"
            )
            for tool_call in response.tool_calls
        )
        entries = [
            transient_context_message("assistant", "\n\n".join(assistant_parts))
        ]
        if tool_outputs:
            results = "\n\n".join(
                (
                    f"[Tool result: {output.get('tool_name', '')}]\n"
                    f"{output.get('content', '')}"
                )
                for output in tool_outputs
            )
            entries.append(transient_context_message("user", results))
        return entries

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

        def stream_once() -> OllamaStreamResponse | str:
            self.runtime._debug_log_step(self.provider_key, payload)
            thinking_parts: list[str] = []
            text_parts: list[str] = []
            text_filter = ThinkingTagStreamFilter()
            tool_calls: list[OllamaToolCall] = []
            final_payload: dict[str, Any] | None = None
            with urllib.request.urlopen(request, timeout=120) as response:
                self.runtime._status(callbacks.status, "Thinking")
                for raw_line in response:
                    if self.runtime._turn_aborted():
                        return OllamaStreamResponse("", "", (), {"role": "assistant"})
                    stripped = raw_line.decode("utf-8", errors="replace").strip()
                    if not stripped:
                        continue
                    data = cast(dict[str, Any], json.loads(stripped))
                    if data.get("error"):
                        return self._api_error(
                            self.provider_key, self.provider_label, self.env_var,
                            400, stripped,
                        )
                    if data.get("done") is True:
                        final_payload = data
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
                usage=ollama_token_usage(final_payload),
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

    def summarize_conversation(
        self,
        messages: list[dict[str, Any]],
        previous_summary: str,
        model: str,
    ) -> str | None:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._context_summary_system_prompt()},
                {
                    "role": "user",
                    "content": self._context_summary_user_prompt(
                        messages,
                        previous_summary,
                    ),
                },
            ],
            "stream": False,
            "options": {"num_predict": 4096},
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        message = data.get("message")
        if not isinstance(message, dict):
            return None
        return str(message.get("content") or "").strip() or None
