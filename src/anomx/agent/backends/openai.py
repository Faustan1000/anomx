"""OpenAI backend."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from anomx.agent.base.backends import (
    MAX_TOOL_ITERATIONS,
    OPENAI_MAX_TOOL_CALLS,
    BackendCallbacks,
    BackendTextCallback,
    BaseBackend,
    OpenAIStreamResponse,
    OpenAIToolCall,
    ThinkingTagStreamFilter,
    openai_token_usage,
)
from anomx.agent.context_management import (
    CONTINUE_AFTER_COMPRESSION_PROMPT,
    ContextMessage,
    projected_context_tokens,
    tool_call_payload,
    tool_exchange_entries,
    transient_context_message,
)
from anomx.agent.exceptions import BackendFailure
from anomx.agent.helpers.tool_manager import CommandRiskEvaluation
from anomx.agent.memories import MemoryKind, MemoryMetadata


class OpenAIBackend(BaseBackend):
    """OpenAI Responses API backend."""

    provider_key = "openai"
    provider_label = "OpenAI"
    env_var = "OPENAI_API_KEY"

    def generate(
        self,
        session_path: Path,
        model: str,
        callbacks: BackendCallbacks,
        *,
        thinking_intensity: str | None = None,
    ) -> str:
        """Generate a response through the OpenAI Responses API."""

        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return self._missing_api_key_message(self.provider_label, self.env_var)

        self.runtime._status(callbacks.status)
        reasoning = self._openai_reasoning_config(model, thinking_intensity)
        context_entries = self.runtime.backend_conversation_entries(session_path)
        payload: dict[str, Any] = {
            "model": model,
            "instructions": self.runtime._instructions(
                session_path, include_previous_conversation=False
            ),
            "input": self._openai_messages(
                [entry.payload for entry in context_entries], model, session_path=session_path
            ),
            "reasoning": reasoning,
            "tools": self._openai_tools(),
            "tool_choice": "auto",
            "max_tool_calls": OPENAI_MAX_TOOL_CALLS,
            "stream": True,
        }

        plan_finish_attempts = 0
        for _ in range(MAX_TOOL_ITERATIONS):
            if self.runtime._turn_aborted():
                return ""
            self.runtime._status(callbacks.status)
            response = self._stream_openai_response(
                api_key,
                payload,
                callbacks.delta,
                callbacks.status,
                callbacks.thought,
            )
            if isinstance(response, str):
                recovered_entries = self._recover_context_window(
                    response, session_path, context_entries, callbacks,
                )
                if recovered_entries is None:
                    return response
                context_entries = recovered_entries
                payload = {
                    **payload,
                    "instructions": self.runtime._instructions(
                        session_path, include_previous_conversation=False
                    ),
                    "input": self._openai_messages(
                        [entry.payload for entry in context_entries],
                        model,
                        session_path=session_path,
                    ),
                }
                payload.pop("previous_response_id", None)
                continue
            if self.runtime._turn_aborted():
                return ""
            self._track_usage(response.usage, callbacks)

            tool_outputs = self._execute_requested_tools(
                response,
                callbacks,
                session_path,
            )
            if getattr(self.runtime, "produced_output", None) is not None:
                return self.runtime.produced_output

            if not tool_outputs:
                continuation_prompt, used_plan_guard = (
                    self.runtime._continuation_prompt_after_text(
                        response.text,
                        callbacks,
                        session_path,
                        plan_finish_attempts,
                    )
                )
                if continuation_prompt is not None:
                    if used_plan_guard:
                        plan_finish_attempts += 1
                    if response.response_id is None:
                        return BackendFailure(
                            "OpenAI returned a continuation update without a response id."
                        )
                    pending_entries = self._openai_context_entries(response, ())
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
                    payload = {
                        "model": model,
                        "instructions": self.runtime._instructions(
                            session_path, include_previous_conversation=False
                        ),
                        "input": (
                            self._openai_messages(
                                [entry.payload for entry in context_entries],
                                model,
                                session_path=session_path,
                            )
                            if compressed
                            else [{"role": "user", "content": continuation_prompt}]
                        ),
                        "reasoning": reasoning,
                        "tools": self._openai_tools(),
                        "tool_choice": "auto",
                        "max_tool_calls": OPENAI_MAX_TOOL_CALLS,
                        "stream": True,
                    }
                    if not compressed:
                        payload["previous_response_id"] = response.response_id
                    continue
                final_text = response.text
                if callbacks.finish is not None:
                    callbacks.finish(final_text)
                return final_text

            if response.response_id is None:
                return BackendFailure("OpenAI requested tools but did not return a response id.")

            pending_entries = self._openai_context_entries(response, tool_outputs)
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

            payload = {
                "model": model,
                "instructions": self.runtime._instructions(
                    session_path, include_previous_conversation=False
                ),
                "input": (
                    self._openai_messages(
                        [entry.payload for entry in context_entries],
                        model,
                        session_path=session_path,
                    )
                    if compressed
                    else tool_outputs
                ),
                "reasoning": reasoning,
                "tools": self._openai_tools(),
                "tool_choice": "auto",
                "max_tool_calls": OPENAI_MAX_TOOL_CALLS,
                "stream": True,
            }
            if not compressed:
                payload["previous_response_id"] = response.response_id

        return BackendFailure(
            f"OpenAI tool loop stopped after {MAX_TOOL_ITERATIONS} tool batches.",
            code="tool_limit_exceeded",
        )

    def _openai_context_entries(
        self,
        response: OpenAIStreamResponse,
        tool_outputs: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    ) -> list[ContextMessage]:
        return tool_exchange_entries(
            response.text,
            [
                tool_call_payload(call.name, call.call_id, call.arguments)
                for call in response.tool_calls
            ],
            tool_outputs,
            content_key="output",
            native={"responses_reasoning": list(response.reasoning)},
        )

    def _stream_openai_response(
        self,
        api_key: str,
        payload: dict[str, Any],
        delta_callback: BackendTextCallback | None,
        status_callback: BackendTextCallback | None,
        thought_callback: BackendTextCallback | None = None,
    ) -> OpenAIStreamResponse | str:
        def stream_once() -> OpenAIStreamResponse | str:
            self.runtime._debug_log_step(self.provider_key, payload)
            request = urllib.request.Request(
                "https://api.openai.com/v1/responses",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            response_id: str | None = None
            text_parts: list[str] = []
            text_filter = ThinkingTagStreamFilter()
            reasoning_parts: dict[str, dict[int, str]] = {}
            reasoning_items: list[dict[str, Any]] = []
            emitted_reasoning: set[str] = set()
            tool_calls: list[OpenAIToolCall] = []
            usage_payload: dict[str, Any] | None = None
            with urllib.request.urlopen(request, timeout=120) as response:
                for raw_line in response:
                    if self.runtime._turn_aborted():
                        return OpenAIStreamResponse(None, "", ())
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    event_data = line.removeprefix("data:").strip()
                    if event_data == "[DONE]":
                        break
                    event = cast(dict[str, Any], json.loads(event_data))
                    event_type = str(event.get("type", ""))
                    if event_type in {"error", "response.failed", "response.incomplete"}:
                        failure = event.get("response") or event
                        return self._api_error(
                            self.provider_key, self.provider_label, self.env_var,
                            400, json.dumps(failure),
                        )
                    if event_type == "response.output_text.delta":
                        delta = str(event.get("delta", ""))
                        if delta:
                            visible = self._visible_stream_text(
                                text_filter,
                                delta,
                                delta_callback,
                                status_callback,
                                thought_callback,
                            )
                            if visible:
                                text_parts.append(visible)
                    elif event_type in {
                        "response.reasoning_summary_text.delta",
                        "response.reasoning_summary_text.done",
                    }:
                        item_id = str(event.get("item_id") or event.get("output_index", ""))
                        summary_index = int(event.get("summary_index", 0))
                        parts = reasoning_parts.setdefault(item_id, {})
                        if event_type.endswith(".done"):
                            parts[summary_index] = str(
                                event.get("text", parts.get(summary_index, ""))
                            )
                        else:
                            parts[summary_index] = parts.get(summary_index, "") + str(
                                event.get("delta", "")
                            )
                    elif event_type == "response.output_item.done":
                        item = event.get("item")
                        if isinstance(item, dict) and item.get("type") == "reasoning":
                            reasoning_items.append(item)
                            item_id = str(item.get("id") or event.get("output_index", ""))
                            parts = reasoning_parts.setdefault(item_id, {})
                            for index, part in enumerate(item.get("summary") or []):
                                if isinstance(part, dict) and part.get("type") == "summary_text":
                                    parts[index] = str(part.get("text", ""))
                            if item_id not in emitted_reasoning:
                                self._emit_thought(
                                    "\n\n".join(parts[index] for index in sorted(parts)),
                                    thought_callback, status_callback,
                                )
                                emitted_reasoning.add(item_id)
                        tool_call = self._tool_call_from_stream_item(item)
                        if tool_call is not None:
                            tool_calls.append(tool_call)
                    elif event_type == "response.completed":
                        response_payload = event.get("response")
                        if isinstance(response_payload, dict):
                            maybe_id = response_payload.get("id")
                            if isinstance(maybe_id, str):
                                response_id = maybe_id
                            maybe_usage = response_payload.get("usage")
                            if isinstance(maybe_usage, dict):
                                usage_payload = maybe_usage
            for item_id, parts in reasoning_parts.items():
                if item_id not in emitted_reasoning:
                    self._emit_thought(
                        "\n\n".join(parts[index] for index in sorted(parts)),
                        thought_callback, status_callback,
                    )
            trailing_text = self._finish_visible_stream_text(
                text_filter, delta_callback, status_callback, thought_callback,
            )
            if trailing_text:
                text_parts.append(trailing_text)
            return OpenAIStreamResponse(
                response_id,
                "".join(text_parts).strip(),
                tuple(tool_calls),
                usage=openai_token_usage(usage_payload),
                reasoning=tuple(reasoning_items),
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
        return cast(OpenAIStreamResponse | str, response)

    def _tool_call_from_stream_item(self, item: object) -> OpenAIToolCall | None:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            return None
        name = item.get("name")
        call_id = item.get("call_id")
        arguments = item.get("arguments")
        if isinstance(name, str) and isinstance(call_id, str) and isinstance(arguments, str):
            return OpenAIToolCall(name, call_id, arguments)
        return None

    def suggest_session_title(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str | None:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return None

        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(
                {
                    "model": model,
                    "instructions": (
                        "Name this CLI session in 3 to 6 words. "
                        "Return only the title. No quotes. No trailing punctuation."
                    ),
                    "input": [{"role": "user", "content": self._title_prompt(messages)}],
                    "max_output_tokens": 24,
                    "stream": False,
                    **self._background_effort_payload(model),
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_title(self.extract_openai_text(data))

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
            "https://api.openai.com/v1/responses",
            data=json.dumps(
                {
                    "model": model,
                    "instructions": self._command_evaluation_system_prompt(),
                    "input": [
                        {
                            "role": "user",
                            "content": self._command_evaluation_user_prompt(
                                command=command,
                                statement=statement,
                                user_message=user_message,
                            ),
                        }
                    ],
                    "max_output_tokens": 4096,
                    "stream": False,
                    **self._background_effort_payload(model),
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "command_risk_evaluation",
                            "schema": self._command_evaluation_schema(),
                            "strict": True,
                        }
                    },
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_command_evaluation(self.extract_openai_text(data))

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
            "https://api.openai.com/v1/responses",
            data=json.dumps(
                {
                    "model": model,
                    "instructions": self._memory_metadata_system_prompt(),
                    "input": [
                        {
                            "role": "user",
                            "content": self._memory_metadata_user_prompt(
                                kind=kind,
                                context=context,
                                content=content,
                            ),
                        }
                    ],
                    "max_output_tokens": 120,
                    "stream": False,
                    **self._background_effort_payload(model),
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "memory_metadata",
                            "schema": self._memory_metadata_schema(),
                            "strict": True,
                        }
                    },
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_memory_metadata(self.extract_openai_text(data))

    def suggest_project_name(self, prompt: str, model: str) -> str | None:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return None

        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(
                {
                    "model": model,
                    "instructions": self._project_name_system_prompt(),
                    "input": [{"role": "user", "content": prompt}],
                    "max_output_tokens": 16,
                    "stream": False,
                    **self._background_effort_payload(model),
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_project_name(self.extract_openai_text(data))

    def suggest_session_continuation(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str | None:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return None

        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(
                {
                    "model": model,
                    "instructions": self._continuation_system_prompt(),
                    "input": [{"role": "user", "content": self._title_prompt(messages)}],
                    "max_output_tokens": 48,
                    "stream": False,
                    **self._background_effort_payload(model),
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_continuation_statement(self.extract_openai_text(data))

    def summarize_conversation(
        self,
        messages: list[dict[str, Any]],
        previous_summary: str,
        model: str,
        *,
        system_prompt: str | None = None,
    ) -> str | None:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return None
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(
                {
                    "model": model,
                    "instructions": system_prompt or self._context_summary_system_prompt(),
                    "input": [
                        {
                            "role": "user",
                            "content": self._context_summary_user_prompt(
                                messages,
                                previous_summary,
                            ),
                        }
                    ],
                    "max_output_tokens": 4096,
                    "stream": False,
                    **self._background_effort_payload(model),
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self.extract_openai_text(data).strip() or None
