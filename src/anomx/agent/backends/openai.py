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
)
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
        payload: dict[str, Any] = {
            "model": model,
            "instructions": self.runtime._instructions(session_path),
            "input": self._openai_messages(
                self.runtime.conversation_messages(session_path),
                model,
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
            )
            if isinstance(response, str):
                return response
            if self.runtime._turn_aborted():
                return ""

            tool_outputs = self._execute_requested_tools(
                response,
                callbacks,
                session_path,
            )
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
                        return "OpenAI returned a continuation update without a response id."
                    payload = {
                        "model": model,
                        "instructions": self.runtime._instructions(session_path),
                        "previous_response_id": response.response_id,
                        "input": [
                            {
                                "role": "user",
                                "content": continuation_prompt,
                            }
                        ],
                        "reasoning": reasoning,
                        "tools": self._openai_tools(),
                        "tool_choice": "auto",
                        "max_tool_calls": OPENAI_MAX_TOOL_CALLS,
                        "stream": True,
                    }
                    continue
                final_text = response.text
                if callbacks.finish is not None:
                    callbacks.finish(final_text)
                return final_text

            if response.response_id is None:
                return "OpenAI requested tools but did not return a response id."

            payload = {
                "model": model,
                "instructions": self.runtime._instructions(session_path),
                "previous_response_id": response.response_id,
                "input": tool_outputs,
                "reasoning": reasoning,
                "tools": self._openai_tools(),
                "tool_choice": "auto",
                "max_tool_calls": OPENAI_MAX_TOOL_CALLS,
                "stream": True,
            }

        return f"OpenAI tool loop stopped after {MAX_TOOL_ITERATIONS} tool batches."

    def _stream_openai_response(
        self,
        api_key: str,
        payload: dict[str, Any],
        delta_callback: BackendTextCallback | None,
        status_callback: BackendTextCallback | None,
    ) -> OpenAIStreamResponse | str:
        def stream_once() -> OpenAIStreamResponse:
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
            reasoning_parts: list[str] = []
            tool_calls: list[OpenAIToolCall] = []
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
                    if event_type == "response.output_text.delta":
                        delta = str(event.get("delta", ""))
                        if delta:
                            visible = self._visible_stream_text(
                                text_filter,
                                delta,
                                delta_callback,
                                status_callback,
                            )
                            if visible:
                                text_parts.append(visible)
                    elif event_type == "response.reasoning_summary_text.delta":
                        delta = str(event.get("delta", ""))
                        if delta:
                            reasoning_parts.append(delta)
                            self.runtime._reasoning_status(
                                status_callback,
                                "".join(reasoning_parts),
                            )
                    elif event_type == "response.output_item.done":
                        tool_call = self._tool_call_from_stream_item(event.get("item"))
                        if tool_call is not None:
                            tool_calls.append(tool_call)
                    elif event_type == "response.completed":
                        response_payload = event.get("response")
                        if isinstance(response_payload, dict):
                            maybe_id = response_payload.get("id")
                            if isinstance(maybe_id, str):
                                response_id = maybe_id
            trailing_text = self._finish_visible_stream_text(text_filter, delta_callback)
            if trailing_text:
                text_parts.append(trailing_text)
            return OpenAIStreamResponse(
                response_id,
                "".join(text_parts).strip(),
                tuple(tool_calls),
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
                    "max_output_tokens": 180,
                    "stream": False,
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
            with urllib.request.urlopen(request, timeout=8) as response:
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
