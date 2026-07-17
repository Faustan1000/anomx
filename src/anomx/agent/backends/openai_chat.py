"""OpenAI-compatible Chat Completions backend primitives."""

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
    BackendTextCallback,
    BaseBackend,
    ImageAttachment,
    OpenAIChatCompletionStreamResponse,
    OpenAIToolCall,
    ThinkingTagStreamFilter,
    normalized_image_attachments,
)
from anomx.agent.helpers.tool_manager import CommandRiskEvaluation
from anomx.agent.memories import MemoryKind, MemoryMetadata


class OpenAICompatibleChatBackend(BaseBackend):
    """Shared agent loop for providers implementing Chat Completions."""

    chat_completions_endpoint = ""

    def generate(
        self,
        session_path: Path,
        model: str,
        callbacks: BackendCallbacks,
        *,
        thinking_intensity: str | None = None,
    ) -> str:
        """Generate a response through an OpenAI-compatible Chat Completions API."""

        del thinking_intensity
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return self._missing_api_key_message(self.provider_label, self.env_var)

        messages = self._chat_messages(session_path, model)
        plan_finish_attempts = 0
        thought_only_followups = 0
        for _ in range(MAX_TOOL_ITERATIONS):
            if self.runtime._turn_aborted():
                return ""
            self.runtime._status(callbacks.status)
            response = self._stream_chat_completion(
                api_key,
                self._chat_payload(model, messages, stream=True),
                callbacks.delta,
                callbacks.status,
                callbacks.thought,
            )
            if isinstance(response, str):
                return response
            if self.runtime._turn_aborted():
                return ""

            if response.assistant_message:
                messages.append(response.assistant_message)
            if not response.tool_calls:
                if not response.text and response.thoughts:
                    if thought_only_followups < 1:
                        thought_only_followups += 1
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Provide the final answer to the user's request now. "
                                    "Return only that answer; do not include reasoning or "
                                    "<think> tags."
                                ),
                            }
                        )
                        continue
                    final_text = "The model did not provide a final answer."
                    if callbacks.finish is not None:
                        callbacks.finish(final_text)
                    return final_text
                continuation_prompt, used_plan_guard = self.runtime._continuation_prompt_after_text(
                    response.text,
                    callbacks,
                    session_path,
                    plan_finish_attempts,
                )
                if continuation_prompt is not None:
                    if used_plan_guard:
                        plan_finish_attempts += 1
                    messages.append({"role": "user", "content": continuation_prompt})
                    continue
                if callbacks.finish is not None:
                    callbacks.finish(response.text)
                return response.text or "No response."

            messages.extend(
                self._execute_chat_completion_tools(response, callbacks, session_path)
            )

        return f"{self.provider_label} tool loop stopped after {MAX_TOOL_ITERATIONS} tool batches."

    def _chat_messages(self, session_path: Path, model: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.runtime._instructions(session_path)}
        ]
        for item in self.runtime.conversation_messages(session_path):
            role = str(item.get("role") or "user").strip()
            content = str(item.get("content") or "").strip()
            if role not in {"assistant", "system", "user"}:
                continue
            images = normalized_image_attachments(item.get("images")) if role == "user" else ()
            supports_images = (
                role == "user"
                and self.provider_key == "blablador"
                and model == "alias-code"
            )
            if supports_images and images:
                content_blocks: list[dict[str, Any]] = []
                text = self._content_with_image_labels(content, images)
                if text:
                    content_blocks.append({"type": "text", "text": text})
                content_blocks.extend(
                    block
                    for image in images
                    if (block := self._chat_image_block(image)) is not None
                )
                if content_blocks:
                    messages.append({"role": role, "content": content_blocks})
                continue
            if not content:
                continue
            messages.append({"role": role, "content": content})
        return messages

    def _chat_image_block(self, image: ImageAttachment) -> dict[str, Any] | None:
        encoded = self._image_base64(image)
        if encoded is None:
            return None
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{image.mime_type};base64,{encoded}",
            },
        }

    def _chat_payload(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            "tools": self._ollama_tools(),
            "tool_choice": "auto",
            "stream": stream,
        }

    def _stream_chat_completion(
        self,
        api_key: str,
        payload: dict[str, Any],
        delta_callback: BackendTextCallback | None,
        status_callback: BackendTextCallback | None,
        thought_callback: BackendTextCallback | None = None,
    ) -> OpenAIChatCompletionStreamResponse | str:
        def stream_once() -> OpenAIChatCompletionStreamResponse:
            self.runtime._debug_log_step(self.provider_key, payload)
            request = urllib.request.Request(
                self.chat_completions_endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            text_parts: list[str] = []
            text_filter = ThinkingTagStreamFilter()
            tagged_thoughts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls_by_index: dict[int, dict[str, Any]] = {}

            def record_thought(thought: str) -> None:
                normalized = self._normalized_thought(thought)
                if not normalized:
                    return
                tagged_thoughts.append(normalized)
                if thought_callback is not None:
                    thought_callback(normalized)
                else:
                    self.runtime._status(status_callback, "Created a thought")

            with urllib.request.urlopen(request, timeout=120) as response:
                for raw_line in response:
                    if self.runtime._turn_aborted():
                        return OpenAIChatCompletionStreamResponse("", (), {})
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    event_data = line.removeprefix("data:").strip()
                    if not event_data or event_data == "[DONE]":
                        continue
                    event = cast(dict[str, Any], json.loads(event_data))
                    choices = event.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    reasoning = self._reasoning_delta_text(delta)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        visible = self._visible_stream_text(
                            text_filter,
                            content,
                            delta_callback,
                            status_callback,
                            record_thought,
                        )
                        if visible:
                            text_parts.append(visible)
                    raw_tool_calls = delta.get("tool_calls")
                    if not isinstance(raw_tool_calls, list):
                        continue
                    for raw_call in raw_tool_calls:
                        if not isinstance(raw_call, dict):
                            continue
                        index = raw_call.get("index")
                        if not isinstance(index, int):
                            continue
                        tool_call = tool_calls_by_index.setdefault(
                            index,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        call_id = raw_call.get("id")
                        if isinstance(call_id, str) and call_id:
                            tool_call["id"] = call_id
                        function = raw_call.get("function")
                        if not isinstance(function, dict):
                            continue
                        name = function.get("name")
                        if isinstance(name, str) and name:
                            tool_call["name"] = name
                        arguments = function.get("arguments")
                        if isinstance(arguments, str) and arguments:
                            tool_call["arguments"] = f"{tool_call['arguments']}{arguments}"

            trailing_text = self._finish_visible_stream_text(
                text_filter,
                delta_callback,
                status_callback,
                record_thought,
            )
            if trailing_text:
                text_parts.append(trailing_text)
            structured_thought = self._normalized_thought("".join(reasoning_parts))
            if structured_thought:
                thought, recovered_final = ThinkingTagStreamFilter.split_unclosed_thought(
                    structured_thought
                )
                record_thought(thought)
                if recovered_final and not text_parts:
                    text_parts.append(recovered_final)
                    if delta_callback is not None:
                        delta_callback(recovered_final)

            tool_calls = tuple(
                OpenAIToolCall(
                    name=str(tool_call["name"]),
                    call_id=str(tool_call["id"]),
                    arguments=str(tool_call["arguments"]),
                )
                for _index, tool_call in sorted(tool_calls_by_index.items())
                if tool_call["id"] and tool_call["name"]
            )
            assistant_message: dict[str, Any] = {}
            if text_parts:
                assistant_message["role"] = "assistant"
                assistant_message["content"] = "".join(text_parts)
            if tool_calls:
                assistant_message["role"] = "assistant"
                assistant_message["tool_calls"] = [
                    {
                        "id": tool_call.call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ]
            return OpenAIChatCompletionStreamResponse(
                "".join(text_parts).strip(),
                tool_calls,
                assistant_message,
                tuple(tagged_thoughts),
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
        return cast(OpenAIChatCompletionStreamResponse | str, response)

    @staticmethod
    def _reasoning_delta_text(delta: Mapping[str, Any]) -> str:
        """Read common OpenAI-compatible private-reasoning fields."""
        for field in ("reasoning_content", "reasoning", "thinking"):
            value = delta.get(field)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, Mapping):
                for key in ("text", "content", "reasoning"):
                    candidate = value.get(key)
                    if isinstance(candidate, str) and candidate:
                        return candidate
            if isinstance(value, list):
                parts = [
                    item
                    for item in value
                    if isinstance(item, str) and item
                ]
                if parts:
                    return "".join(parts)
        return ""

    @staticmethod
    def _normalized_thought(value: str) -> str:
        """Keep provider reasoning out of final text without retaining tag markup."""
        return value.replace("<think>", "").replace("</think>", "").strip()

    def _execute_chat_completion_tools(
        self,
        response: OpenAIChatCompletionStreamResponse,
        callbacks: BackendCallbacks,
        session_path: Path,
    ) -> list[dict[str, str]]:
        outputs: list[dict[str, str]] = []
        for tool_call in response.tool_calls:
            output = self.runtime._execute_tool(
                tool_call.name,
                self._parse_tool_arguments(tool_call.arguments),
                callbacks,
                session_path,
            )
            outputs.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.call_id,
                    "content": output,
                }
            )
        return outputs

    def _chat_completion_content(
        self,
        api_key: str,
        payload: dict[str, Any],
        *,
        timeout: int,
    ) -> str | None:
        request = urllib.request.Request(
            self.chat_completions_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
        ):
            return None
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return None
        return self._extract_chat_content(message.get("content"))

    @staticmethod
    def _extract_chat_content(value: object) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "\n".join(
                str(item.get("text") or "").strip()
                for item in value
                if isinstance(item, dict) and item.get("text")
            ).strip()
        return ""

    def _simple_completion(self, system: str, user: str, model: str, *, timeout: int) -> str | None:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return None
        return self._chat_completion_content(
            api_key,
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            },
            timeout=timeout,
        )

    def suggest_session_title(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str | None:
        value = self._simple_completion(
            (
                "Name this CLI session in 3 to 6 words. Return only the title. "
                "No quotes. No trailing punctuation."
            ),
            self._title_prompt(messages),
            model,
            timeout=20,
        )
        return self._sanitize_title(value or "") if value else None

    def evaluate_command_request(
        self,
        *,
        command: str,
        statement: str,
        user_message: str,
        model: str,
    ) -> CommandRiskEvaluation | None:
        value = self._simple_completion(
            self._command_evaluation_system_prompt(),
            self._command_evaluation_user_prompt(
                command=command,
                statement=statement,
                user_message=user_message,
            ),
            model,
            timeout=8,
        )
        return self._sanitize_command_evaluation(value) if value else None

    def suggest_memory_metadata(
        self,
        *,
        kind: MemoryKind | str,
        context: Mapping[str, Any],
        content: str,
        model: str,
    ) -> MemoryMetadata | None:
        value = self._simple_completion(
            self._memory_metadata_system_prompt(),
            self._memory_metadata_user_prompt(kind=kind, context=context, content=content),
            model,
            timeout=8,
        )
        return self._sanitize_memory_metadata(value) if value else None

    def suggest_project_name(self, prompt: str, model: str) -> str | None:
        value = self._simple_completion(
            self._project_name_system_prompt(),
            prompt,
            model,
            timeout=20,
        )
        return self._sanitize_project_name(value or "") if value else None

    def suggest_session_continuation(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str | None:
        value = self._simple_completion(
            self._continuation_system_prompt(),
            self._title_prompt(messages),
            model,
            timeout=8,
        )
        return self._sanitize_continuation_statement(value or "") if value else None
