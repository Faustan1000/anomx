"""Anthropic-compatible backends."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from anomx.agent.base.backends import (
    MAX_TOKENS_CONTINUATIONS,
    MAX_TOOL_ITERATIONS,
    AnthropicStreamResponse,
    AnthropicToolCall,
    BackendCallbacks,
    BackendTextCallback,
    BaseBackend,
    ThinkingTagStreamFilter,
    anthropic_token_usage,
)
from anomx.agent.context_management import (
    CONTINUE_AFTER_COMPRESSION_PROMPT,
    ContextMessage,
    projected_context_tokens,
    transient_context_message,
)
from anomx.agent.exceptions import BackendFailure
from anomx.agent.helpers.http_pool import pooled_https_post
from anomx.agent.helpers.tool_manager import CommandRiskEvaluation
from anomx.agent.memories import MemoryKind, MemoryMetadata


class AnthropicCompatibleBackend(BaseBackend):
    """Shared Messages API loop for Anthropic-compatible providers."""

    def generate(
        self,
        session_path: Path,
        model: str,
        callbacks: BackendCallbacks,
        *,
        thinking_intensity: str | None = None,
    ) -> str:
        return self._messages_api_response(
            session_path,
            model,
            callbacks,
            include_thinking=True,
            thinking_intensity=thinking_intensity,
        )

    def _messages_api_response(
        self,
        session_path: Path,
        model: str,
        callbacks: BackendCallbacks,
        *,
        include_thinking: bool,
        thinking_intensity: str | None = None,
    ) -> str:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return self._missing_api_key_message(self.provider_label, self.env_var)

        self.runtime._status(callbacks.status)
        context_entries = self.runtime.backend_conversation_entries(session_path)
        messages = self._anthropic_messages(
            [entry.payload for entry in context_entries],
            self.provider_key,
            model,
        )
        payload: dict[str, Any] = self._payload(
            session_path,
            model,
            messages,
            include_thinking=include_thinking,
            thinking_intensity=thinking_intensity,
        )

        plan_finish_attempts = 0
        max_tokens_continuations = 0
        truncated_prefix = ""
        for _ in range(MAX_TOOL_ITERATIONS):
            if self.runtime._turn_aborted():
                return ""
            self.runtime._status(callbacks.status)
            response = self._stream_response(
                api_key,
                payload,
                callbacks.delta,
                callbacks.status,
            )
            if isinstance(response, str):
                recovered_entries = self._recover_context_window(
                    response, session_path, context_entries, callbacks,
                )
                if recovered_entries is None:
                    return response
                context_entries = recovered_entries
                messages = self._anthropic_messages(
                    [entry.payload for entry in context_entries], self.provider_key, model,
                )
                payload = self._payload(
                    session_path, model, messages,
                    include_thinking=include_thinking, thinking_intensity=thinking_intensity,
                )
                continue
            if self.runtime._turn_aborted():
                return ""
            self._track_usage(response.usage, callbacks)

            tool_outputs = self._execute_anthropic_requested_tools(
                response,
                callbacks,
                session_path,
            )
            if not tool_outputs:
                text = response.text or self._extract_anthropic_text(response.content)
                if (
                    response.stop_reason == "max_tokens"
                    and max_tokens_continuations < MAX_TOKENS_CONTINUATIONS
                ):
                    max_tokens_continuations += 1
                    truncated_prefix += text
                    assistant_content = list(response.content) or [
                        {"type": "text", "text": text}
                    ]
                    messages.append({"role": "assistant", "content": assistant_content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous message was truncated by the output token "
                                "limit. Continue exactly where you left off; do not repeat "
                                "what you already wrote."
                            ),
                        }
                    )
                    payload = self._payload(
                        session_path,
                        model,
                        messages,
                        include_thinking=include_thinking,
                        thinking_intensity=thinking_intensity,
                    )
                    continue
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
                    assistant_content = list(response.content) or [
                        {"type": "text", "text": text}
                    ]
                    pending_entries = self._anthropic_context_entries(response, ())
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
                        messages = self._anthropic_messages(
                            [entry.payload for entry in context_entries],
                            self.provider_key,
                            model,
                        )
                    else:
                        messages.append(
                            {"role": "assistant", "content": assistant_content}
                        )
                        messages.append(
                            {"role": "user", "content": continuation_prompt}
                        )
                    payload = self._payload(
                        session_path,
                        model,
                        messages,
                        include_thinking=include_thinking,
                        thinking_intensity=thinking_intensity,
                    )
                    continue
                final_text = truncated_prefix + text
                if callbacks.finish is not None:
                    callbacks.finish(final_text)
                return final_text

            pending_entries = self._anthropic_context_entries(response, tool_outputs)
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
                messages = self._anthropic_messages(
                    [entry.payload for entry in context_entries],
                    self.provider_key,
                    model,
                )
            else:
                messages.append(
                    {"role": "assistant", "content": list(response.content)}
                )
                messages.append({"role": "user", "content": tool_outputs})
            payload = self._payload(
                session_path,
                model,
                messages,
                include_thinking=include_thinking,
                thinking_intensity=thinking_intensity,
            )

        return BackendFailure(
            f"{self.provider_label} tool loop stopped after {MAX_TOOL_ITERATIONS} tool batches.",
            code="tool_limit_exceeded",
        )

    def _anthropic_context_entries(
        self,
        response: AnthropicStreamResponse,
        tool_outputs: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    ) -> list[ContextMessage]:
        assistant_parts = [response.text.strip()] if response.text.strip() else []
        assistant_parts.extend(
            (
                f"[Tool call: {tool_call.name}]\n"
                f"{json.dumps(tool_call.input, ensure_ascii=False, sort_keys=True)}"
            )
            for tool_call in response.tool_calls
        )
        entries = [
            transient_context_message("assistant", "\n\n".join(assistant_parts))
        ]
        if tool_outputs:
            results = "\n\n".join(
                (
                    f"[Tool result: {output.get('tool_use_id', '')}]\n"
                    f"{output.get('content', '')}"
                )
                for output in tool_outputs
            )
            entries.append(transient_context_message("user", results))
        return entries

    def _payload(
        self,
        session_path: Path,
        model: str,
        messages: list[dict[str, Any]],
        *,
        include_thinking: bool,
        thinking_intensity: str | None,
    ) -> dict[str, Any]:
        configured = self.runtime.home.load_config().get("max_output_tokens")
        try:
            fallback = int(configured) if configured else 8_192
        except (TypeError, ValueError):
            fallback = 8_192
        payload: dict[str, Any] = {
            "model": model,
            "system": self._cached_system_blocks(self.runtime._instructions(session_path)),
            "messages": messages,
            "tools": self._cached_tools(self._anthropic_tools()),
            "max_tokens": self._max_output_tokens(model, fallback),
            "stream": True,
        }
        if include_thinking:
            payload["thinking"] = self._anthropic_thinking_config(model)
            output_config = self._anthropic_output_config(model, thinking_intensity)
            if output_config:
                payload["output_config"] = output_config
        return payload

    @staticmethod
    def _cached_system_blocks(instructions: str) -> list[dict[str, Any]]:
        """Wrap the system prompt as a cacheable content block.

        The system prompt (identity, workflow, tool descriptions, skills, memories) is
        rebuilt fresh every turn but is almost entirely static within a session. Without
        a cache breakpoint, every turn pays full input-token processing for it instead of
        a cheap cache read, which is the single largest source of avoidable per-turn
        latency in this loop.
        """

        return [
            {
                "type": "text",
                "text": instructions,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    @staticmethod
    def _cached_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Mark the tool schema list as cacheable via a breakpoint on its last entry."""

        if not tools:
            return tools
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        return tools

    def _stream_response(
        self,
        api_key: str,
        payload: dict[str, Any],
        delta_callback: BackendTextCallback | None,
        status_callback: BackendTextCallback | None,
    ) -> AnthropicStreamResponse | str:
        raise NotImplementedError

    def _stream_anthropic_compatible_response(
        self,
        endpoint: str,
        headers: dict[str, str],
        api_key: str,
        payload: dict[str, Any],
        delta_callback: BackendTextCallback | None,
        status_callback: BackendTextCallback | None,
    ) -> AnthropicStreamResponse | str:
        del api_key

        def stream_once() -> AnthropicStreamResponse | str:
            self.runtime._debug_log_step(self.provider_key, payload)
            request_body = json.dumps(payload).encode("utf-8")
            text_parts: list[str] = []
            text_filter = ThinkingTagStreamFilter()
            content_by_index: dict[int, dict[str, Any]] = {}
            tool_json_parts: dict[int, list[str]] = {}
            stop_reason: str | None = None
            usage_payload: dict[str, Any] = {}
            with pooled_https_post(
                endpoint,
                data=request_body,
                headers=headers,
                timeout=120,
            ) as response:
                for raw_line in response:
                    if self.runtime._turn_aborted():
                        return ""
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    event_data = line.removeprefix("data:").strip()
                    if not event_data:
                        continue
                    event = cast(dict[str, Any], json.loads(event_data))
                    event_type = str(event.get("type", ""))
                    if event_type == "message_start":
                        message = event.get("message")
                        if isinstance(message, dict):
                            message_usage = message.get("usage")
                            if isinstance(message_usage, dict):
                                usage_payload.update(message_usage)
                    elif event_type == "message_delta":
                        delta_usage = event.get("usage")
                        if isinstance(delta_usage, dict):
                            usage_payload.update(delta_usage)
                    elif event_type == "content_block_start":
                        index = event.get("index")
                        block = event.get("content_block")
                        if not isinstance(index, int) or not isinstance(block, dict):
                            continue
                        block_type = str(block.get("type", ""))
                        if block_type == "text":
                            text = str(block.get("text", ""))
                            visible = self._visible_stream_text(
                                text_filter,
                                text,
                                delta_callback,
                                status_callback,
                            )
                            content_by_index[index] = {"type": "text", "text": visible}
                            if visible:
                                text_parts.append(visible)
                        elif block_type == "tool_use":
                            content_by_index[index] = {
                                "type": "tool_use",
                                "id": str(block.get("id", "")),
                                "name": str(block.get("name", "")),
                                "input": {},
                            }
                            tool_json_parts[index] = []
                        elif block_type == "thinking":
                            content_by_index[index] = {
                                "type": "thinking",
                                "thinking": str(block.get("thinking", "")),
                                "signature": str(block.get("signature", "")),
                            }
                    elif event_type == "content_block_delta":
                        index = event.get("index")
                        delta = event.get("delta")
                        if not isinstance(index, int) or not isinstance(delta, dict):
                            continue
                        delta_type = str(delta.get("type", ""))
                        if delta_type == "text_delta":
                            text = str(delta.get("text", ""))
                            if not text:
                                continue
                            visible = self._visible_stream_text(
                                text_filter,
                                text,
                                delta_callback,
                                status_callback,
                            )
                            block = content_by_index.get(index)
                            if isinstance(block, dict) and block.get("type") == "text":
                                block["text"] = f"{block.get('text', '')}{visible}"
                            if visible:
                                text_parts.append(visible)
                        elif delta_type == "input_json_delta":
                            partial_json = str(delta.get("partial_json", ""))
                            if partial_json:
                                tool_json_parts.setdefault(index, []).append(partial_json)
                        elif delta_type == "thinking_delta":
                            thinking = str(delta.get("thinking", ""))
                            if not thinking:
                                continue
                            block = content_by_index.get(index)
                            if isinstance(block, dict) and block.get("type") == "thinking":
                                block["thinking"] = f"{block.get('thinking', '')}{thinking}"
                                self.runtime._reasoning_status(
                                    status_callback,
                                    str(block.get("thinking", "")),
                                )
                        elif delta_type == "signature_delta":
                            signature = str(delta.get("signature", ""))
                            block = content_by_index.get(index)
                            if (
                                signature
                                and isinstance(block, dict)
                                and block.get("type") == "thinking"
                            ):
                                block["signature"] = signature
                    elif event_type == "content_block_stop":
                        index = event.get("index")
                        if not isinstance(index, int):
                            continue
                        self._finalize_anthropic_tool_input(
                            content_by_index,
                            tool_json_parts,
                            index,
                        )
                    elif event_type == "message_delta":
                        delta = event.get("delta")
                        if isinstance(delta, dict):
                            reason = delta.get("stop_reason")
                            if isinstance(reason, str) and reason:
                                stop_reason = reason
                    elif event_type == "error":
                        error = event.get("error")
                        if isinstance(error, dict):
                            message = str(error.get("message", "")).strip()
                            if message:
                                return BackendFailure(
                                    f"{self.provider_label} request failed: {message}"
                                )

            for index in tuple(tool_json_parts):
                self._finalize_anthropic_tool_input(content_by_index, tool_json_parts, index)

            trailing_text = self._finish_visible_stream_text(text_filter, delta_callback)
            if trailing_text:
                text_parts.append(trailing_text)

            ordered_content = tuple(
                content_by_index[index] for index in sorted(content_by_index)
            )
            tool_calls = tuple(
                AnthropicToolCall(
                    name=str(block.get("name", "")),
                    tool_use_id=str(block.get("id", "")),
                    input=cast(dict[str, Any], block.get("input", {})),
                )
                for block in ordered_content
                if block.get("type") == "tool_use"
            )
            return AnthropicStreamResponse(
                "".join(text_parts).strip(),
                tool_calls,
                ordered_content,
                stop_reason,
                usage=anthropic_token_usage(usage_payload),
            )

        if self.runtime._turn_aborted():
            return ""
        response = self._model_request_with_retries(
            provider_key=self.provider_key,
            provider_label=self.provider_label,
            env_var=self.env_var,
            status_callback=status_callback,
            stream_once=stream_once,
        )
        if self.runtime._turn_aborted():
            return ""
        return cast(AnthropicStreamResponse | str, response)


class AnthropicBackend(AnthropicCompatibleBackend):
    """Anthropic Messages API backend."""

    provider_key = "anthropic"
    provider_label = "Anthropic"
    env_var = "ANTHROPIC_API_KEY"

    def _stream_response(
        self,
        api_key: str,
        payload: dict[str, Any],
        delta_callback: BackendTextCallback | None,
        status_callback: BackendTextCallback | None,
    ) -> AnthropicStreamResponse | str:
        return self._stream_anthropic_compatible_response(
            "https://api.anthropic.com/v1/messages",
            {
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "anthropic-version": "2023-06-01",
            },
            api_key,
            payload,
            delta_callback,
            status_callback,
        )

    def suggest_session_title(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str | None:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return None

        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": model,
                    "system": (
                        "Name this CLI session in 3 to 6 words. "
                        "Return only the title. No quotes. No trailing punctuation."
                    ),
                    "messages": [{"role": "user", "content": self._title_prompt(messages)}],
                    "max_tokens": 24,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_title(self.extract_anthropic_text(data))

    def evaluate_command_request(
        self,
        *,
        command: str,
        statement: str,
        user_message: str,
        model: str,
    ) -> CommandRiskEvaluation | None:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return None

        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": model,
                    "system": self._command_evaluation_system_prompt(),
                    "messages": [
                        {
                            "role": "user",
                            "content": self._command_evaluation_user_prompt(
                                command=command,
                                statement=statement,
                                user_message=user_message,
                            ),
                        }
                    ],
                    "max_tokens": 180,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_command_evaluation(self.extract_anthropic_text(data))

    def suggest_memory_metadata(
        self,
        *,
        kind: MemoryKind | str,
        context: Mapping[str, Any],
        content: str,
        model: str,
    ) -> MemoryMetadata | None:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return None

        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": model,
                    "system": self._memory_metadata_system_prompt(),
                    "messages": [
                        {
                            "role": "user",
                            "content": self._memory_metadata_user_prompt(
                                kind=kind,
                                context=context,
                                content=content,
                            ),
                        }
                    ],
                    "max_tokens": 120,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_memory_metadata(self.extract_anthropic_text(data))

    def suggest_project_name(self, prompt: str, model: str) -> str | None:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return None

        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": model,
                    "system": self._project_name_system_prompt(),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 16,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_project_name(self.extract_anthropic_text(data))

    def suggest_session_continuation(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str | None:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return None

        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": model,
                    "system": self._continuation_system_prompt(),
                    "messages": [{"role": "user", "content": self._title_prompt(messages)}],
                    "max_tokens": 48,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_continuation_statement(self.extract_anthropic_text(data))

    def summarize_conversation(
        self,
        messages: list[dict[str, Any]],
        previous_summary: str,
        model: str,
    ) -> str | None:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return None
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": model,
                    "system": self._context_summary_system_prompt(),
                    "messages": [
                        {
                            "role": "user",
                            "content": self._context_summary_user_prompt(
                                messages,
                                previous_summary,
                            ),
                        }
                    ],
                    "max_tokens": 4096,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self.extract_anthropic_text(data).strip() or None
