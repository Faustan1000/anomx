"""Base backend primitives and shared provider helpers."""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
import time
import urllib.error
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeAlias, cast

from anomx.agent.exceptions import BackendFailure
from anomx.agent.helpers.extract_json import extract_json_object
from anomx.agent.helpers.tool_manager import CommandRiskEvaluation
from anomx.agent.memories import MemoryKind, MemoryMetadata, sanitize_memory_metadata
from anomx.agent.store import (
    THINKING_INTENSITY_AUTO,
    model_output_token_budget,
    normalize_thinking_intensity,
    thinking_intensity_options,
)

if TYPE_CHECKING:
    from anomx.agent.context_management import ContextMessage

MAX_TOOL_ITERATIONS = 256
# How many times a response truncated by the output token limit (stop_reason
# "max_tokens") is automatically continued before the loop gives up.
MAX_TOKENS_CONTINUATIONS = 4
OPENAI_MAX_TOOL_CALLS = 256
# Only transient statuses are retried. 400 is a deterministic client error (bad
# request) that never succeeds on retry; retrying it turned real failures into an
# endless "Reconnecting" loop instead of surfacing the error. 404 is kept because
# the DESY backend returns it transiently while a route is warming up.
MODEL_REQUEST_RETRY_STATUS_CODES = frozenset({404, 408, 429, 500, 502, 503, 529})
MODEL_REQUEST_RETRY_COUNT = 10
MODEL_REQUEST_RETRY_INITIAL_DELAY_SECONDS = 1.0
MODEL_REQUEST_RETRY_MAX_DELAY_SECONDS = 60.0
MODEL_REQUEST_RETRY_BACKOFF_FACTOR = 2.0
MODEL_REQUEST_RETRY_SLEEP_SLICE_SECONDS = 0.25
CONTEXT_CHARACTERS_PER_TOKEN = 4
MESSAGE_CONTEXT_OVERHEAD_TOKENS = 4
INSTRUCTIONS_CONTEXT_OVERHEAD_TOKENS = 8
MESSAGE_IMAGE_CONTEXT_TOKENS = 2_000
SUPPORTED_IMAGE_MIME_TYPES = frozenset(
    {
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    }
)
OLLAMA_IMAGE_MODEL_MARKERS = frozenset(
    {
        "bakllava",
        "gemma3",
        "llama3.2-vision",
        "llava",
        "minicpm-v",
        "moondream",
        "qwen-vl",
        "qwen2-vl",
        "qwen2.5-vl",
        "qwen3-vl",
        "vision",
    }
)


@dataclass(frozen=True)
class TokenUsage:
    """Normalized token accounting reported by an AI backend for one API call.

    ``input_tokens`` is the full request context including any cached portions,
    so it directly reflects the context size occupied by the request. Provider
    cache details are kept separately in ``cached_tokens`` (read from cache) and
    ``cache_creation_tokens`` (written to cache).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    @classmethod
    def build(
        cls,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        cache_creation_tokens: int = 0,
        total_tokens: int = 0,
    ) -> TokenUsage | None:
        """Build a usage record, returning ``None`` when nothing was reported."""

        usage = cls(
            input_tokens=max(0, int(input_tokens)),
            output_tokens=max(0, int(output_tokens)),
            cached_tokens=max(0, int(cached_tokens)),
            cache_creation_tokens=max(0, int(cache_creation_tokens)),
            total_tokens=max(0, int(total_tokens))
            or max(0, int(input_tokens)) + max(0, int(output_tokens)),
        )
        return usage if usage.total_tokens > 0 else None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> TokenUsage | None:
        """Restore a usage record from its persisted ``to_dict`` payload."""

        if not isinstance(value, Mapping):
            return None
        return cls.build(
            input_tokens=_usage_int(value.get("input_tokens")),
            output_tokens=_usage_int(value.get("output_tokens")),
            cached_tokens=_usage_int(value.get("cached_tokens")),
            cache_creation_tokens=_usage_int(value.get("cache_creation_tokens")),
            total_tokens=_usage_int(value.get("total_tokens")),
        )

    def to_dict(self) -> dict[str, int]:
        """Serialize for persistence in session events or platform metadata."""

        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class UsageSnapshot:
    """Cumulative usage of a generation loop plus the latest context size."""

    total: TokenUsage
    context_tokens: int = 0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> UsageSnapshot | None:
        """Restore a snapshot from its persisted ``to_dict`` payload."""

        if not isinstance(value, Mapping):
            return None
        usage = TokenUsage.from_dict(value)
        if usage is None:
            return None
        return cls(total=usage, context_tokens=_usage_int(value.get("context_tokens")))

    def to_dict(self) -> dict[str, int]:
        """Serialize as a flat usage payload including the context size."""

        return {**self.total.to_dict(), "context_tokens": self.context_tokens}


UsageCallback: TypeAlias = Callable[[UsageSnapshot], None]


def _usage_int(value: object) -> int:
    """Coerce a provider usage value to a non-negative integer."""

    if isinstance(value, bool):
        return 0
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number):
        return 0
    return max(0, int(number))


def anthropic_token_usage(usage: Mapping[str, Any] | None) -> TokenUsage | None:
    """Build normalized usage from an Anthropic Messages API usage payload."""

    if not isinstance(usage, Mapping):
        return None
    cached_tokens = _usage_int(usage.get("cache_read_input_tokens"))
    cache_creation_tokens = _usage_int(usage.get("cache_creation_input_tokens"))
    return TokenUsage.build(
        input_tokens=(
            _usage_int(usage.get("input_tokens")) + cached_tokens + cache_creation_tokens
        ),
        output_tokens=_usage_int(usage.get("output_tokens")),
        cached_tokens=cached_tokens,
        cache_creation_tokens=cache_creation_tokens,
    )


def openai_token_usage(usage: Mapping[str, Any] | None) -> TokenUsage | None:
    """Build normalized usage from an OpenAI Responses API usage payload."""

    if not isinstance(usage, Mapping):
        return None
    details = usage.get("input_tokens_details")
    cached_tokens = _usage_int(details.get("cached_tokens")) if isinstance(details, Mapping) else 0
    return TokenUsage.build(
        input_tokens=_usage_int(usage.get("input_tokens")),
        output_tokens=_usage_int(usage.get("output_tokens")),
        cached_tokens=cached_tokens,
        total_tokens=_usage_int(usage.get("total_tokens")),
    )


def chat_completion_token_usage(usage: Mapping[str, Any] | None) -> TokenUsage | None:
    """Build normalized usage from a Chat Completions usage payload."""

    if not isinstance(usage, Mapping):
        return None
    details = usage.get("prompt_tokens_details")
    cached_tokens = _usage_int(details.get("cached_tokens")) if isinstance(details, Mapping) else 0
    return TokenUsage.build(
        input_tokens=_usage_int(usage.get("prompt_tokens")),
        output_tokens=_usage_int(usage.get("completion_tokens")),
        cached_tokens=cached_tokens,
        total_tokens=_usage_int(usage.get("total_tokens")),
    )


def ollama_token_usage(payload: Mapping[str, Any] | None) -> TokenUsage | None:
    """Build normalized usage from a final Ollama chat response payload."""

    if not isinstance(payload, Mapping):
        return None
    return TokenUsage.build(
        input_tokens=_usage_int(payload.get("prompt_eval_count")),
        output_tokens=_usage_int(payload.get("eval_count")),
    )


_TOKEN_COUNT_UNITS = ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k"))


def format_token_count(tokens: int) -> str:
    """Format a token count compactly, e.g. ``132``, ``12k``, ``324k``, ``1M``."""

    value = max(0, int(tokens))
    if value < 1_000:
        return str(value)
    for index, (divisor, suffix) in enumerate(_TOKEN_COUNT_UNITS):
        if value < divisor:
            continue
        scaled = value / divisor
        rounded = round(scaled) if scaled >= 100 else round(scaled, 1)
        if rounded >= 1000 and index > 0:
            divisor, suffix = _TOKEN_COUNT_UNITS[index - 1]
            scaled = value / divisor
            rounded = round(scaled) if scaled >= 100 else round(scaled, 1)
        return f"{rounded:g}{suffix}"
    return str(value)


@dataclass(frozen=True)
class OpenAIToolCall:
    """Function call emitted by the Responses API."""

    name: str
    call_id: str
    arguments: str


@dataclass(frozen=True)
class OpenAIStreamResponse:
    """Result collected from a streamed OpenAI response."""

    response_id: str | None
    text: str
    tool_calls: tuple[OpenAIToolCall, ...]
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class OpenAIChatCompletionStreamResponse:
    """Result collected from an OpenAI-compatible Chat Completions stream."""

    text: str
    tool_calls: tuple[OpenAIToolCall, ...]
    assistant_message: dict[str, Any]
    thoughts: tuple[str, ...] = ()
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class AnthropicToolCall:
    """Tool call emitted by the Anthropic Messages API."""

    name: str
    tool_use_id: str
    input: dict[str, Any]


@dataclass(frozen=True)
class AnthropicStreamResponse:
    """Result collected from a streamed Anthropic-compatible response."""

    text: str
    tool_calls: tuple[AnthropicToolCall, ...]
    content: tuple[dict[str, Any], ...]
    stop_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class ImageAttachment:
    """Image file attached to a user message."""

    label: str
    token: str
    path: Path
    mime_type: str

    def to_payload(self) -> dict[str, str]:
        """Serialize image metadata for session storage."""

        return {
            "label": self.label,
            "token": self.token,
            "path": self.path.as_posix(),
            "mime_type": self.mime_type,
        }


@dataclass(frozen=True)
class OllamaToolCall:
    """Function call emitted by Ollama chat responses."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class OllamaStreamResponse:
    """Result collected from a streamed Ollama response."""

    text: str
    thinking: str
    tool_calls: tuple[OllamaToolCall, ...]
    message: dict[str, Any]
    usage: TokenUsage | None = None


ModelRequestStreamResponse: TypeAlias = (
    OpenAIStreamResponse
    | OpenAIChatCompletionStreamResponse
    | AnthropicStreamResponse
    | OllamaStreamResponse
    | str
)
BackendTextCallback: TypeAlias = Callable[[str], None]


@dataclass
class ThinkingTagStreamFilter:
    """Hide provider-supplied ``<think>`` blocks from user-facing text streams."""

    _buffer: str = ""
    _inside_thinking: bool = False
    _active_thought_parts: list[str] = field(default_factory=list)
    _completed_thoughts: list[str] = field(default_factory=list)

    _OPEN_TAG: ClassVar[str] = "<think>"
    _CLOSE_TAG: ClassVar[str] = "</think>"

    def feed(self, text: str) -> tuple[str, bool]:
        """Return visible text and whether this chunk started a hidden thought."""
        self._buffer += text
        visible: list[str] = []
        thought_started = False

        while self._buffer:
            normalized = self._buffer.lower()
            if self._inside_thinking:
                closing_index = normalized.find(self._CLOSE_TAG)
                if closing_index < 0:
                    trailing_prefix = self._trailing_tag_prefix(self._buffer, self._CLOSE_TAG)
                    thought = (
                        self._buffer[: -len(trailing_prefix)]
                        if trailing_prefix
                        else self._buffer
                    )
                    if thought:
                        self._active_thought_parts.append(thought)
                    self._buffer = trailing_prefix
                    break
                if closing_index:
                    self._active_thought_parts.append(self._buffer[:closing_index])
                self._buffer = self._buffer[closing_index + len(self._CLOSE_TAG) :]
                self._inside_thinking = False
                self._complete_thought()
                continue

            opening_index = normalized.find(self._OPEN_TAG)
            closing_index = normalized.find(self._CLOSE_TAG)
            if closing_index >= 0 and (opening_index < 0 or closing_index < opening_index):
                visible.append(self._buffer[:closing_index])
                self._buffer = self._buffer[closing_index + len(self._CLOSE_TAG) :]
                continue
            if opening_index < 0:
                trailing_prefix = self._trailing_tag_prefix(self._buffer, self._OPEN_TAG)
                if trailing_prefix:
                    visible.append(self._buffer[: -len(trailing_prefix)])
                    self._buffer = trailing_prefix
                else:
                    visible.append(self._buffer)
                    self._buffer = ""
                break

            visible.append(self._buffer[:opening_index])
            self._buffer = self._buffer[opening_index + len(self._OPEN_TAG) :]
            self._inside_thinking = True
            thought_started = True

        return "".join(visible), thought_started

    def finish(self) -> str:
        """Flush ordinary trailing text while retaining unfinished thoughts separately."""
        if self._inside_thinking:
            thought, final_text = self.split_unclosed_thought(
                "".join(self._active_thought_parts)
            )
            self._active_thought_parts.clear()
            if thought:
                self._completed_thoughts.append(thought)
            self._buffer = ""
            return final_text
        if self._buffer and self._OPEN_TAG.startswith(self._buffer.lower()):
            self._buffer = ""
            return ""
        trailing = self._buffer
        self._buffer = ""
        return trailing

    def drain_completed_thoughts(self) -> tuple[str, ...]:
        """Return thought blocks completed since the last drain."""
        thoughts = tuple(self._completed_thoughts)
        self._completed_thoughts.clear()
        return thoughts

    def _complete_thought(self) -> None:
        thought = "".join(self._active_thought_parts).strip()
        self._active_thought_parts.clear()
        if thought:
            self._completed_thoughts.append(thought)

    @staticmethod
    def split_unclosed_thought(value: str) -> tuple[str, str]:
        """Recover a clearly separated final reply from malformed thought output.

        Some OpenAI-compatible providers emit a reasoning paragraph and then the
        user-facing answer in the same unclosed ``<think>`` block.  We only recover
        the last paragraph when the leading text unmistakably reads like reasoning;
        otherwise the entire value remains hidden.
        """
        text = value.strip()
        parts = [part.strip() for part in re.split(r"\n[\t ]*\n", text) if part.strip()]
        if len(parts) < 2:
            return text, ""

        thought = "\n\n".join(parts[:-1]).strip()
        final_text = parts[-1]
        normalized_thought = thought.lower()
        reasoning_markers = (
            "the user",
            "i should",
            "i need to",
            "we need",
            "let me",
            "should respond",
            "my response",
        )
        if not any(marker in normalized_thought for marker in reasoning_markers):
            return text, ""
        return thought, final_text

    @staticmethod
    def _trailing_tag_prefix(value: str, tag: str) -> str:
        lowered = value.lower()
        for length in range(min(len(value), len(tag) - 1), 0, -1):
            if lowered.endswith(tag[:length]):
                return value[-length:]
        return ""


def strip_thinking_tags(text: str) -> str:
    """Return complete persisted text without provider private-reasoning tags."""
    text_filter = ThinkingTagStreamFilter()
    visible, _thought_started = text_filter.feed(text)
    return f"{visible}{text_filter.finish()}".strip()


class BackendCallbacks(Protocol):
    """Callbacks used by backend generation loops."""

    status: BackendTextCallback | None
    delta: BackendTextCallback | None
    thought: BackendTextCallback | None
    finish: BackendTextCallback | None
    usage: UsageCallback | None


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens for display-only context accounting."""

    return (len(text) + CONTEXT_CHARACTERS_PER_TOKEN - 1) // CONTEXT_CHARACTERS_PER_TOKEN


def image_mime_type(path: Path) -> str | None:
    """Return a supported image MIME type inferred from a path."""

    mime_type, _encoding = mimetypes.guess_type(path.name)
    return mime_type if mime_type in SUPPORTED_IMAGE_MIME_TYPES else None


def backend_supports_image_input(provider_key: str, model: str) -> bool:
    """Return whether the selected backend/model can receive image input."""

    if provider_key in {"openai", "anthropic"}:
        return True
    if provider_key == "kimi":
        normalized = model.lower()
        return normalized in {"kimi-k3", "kimi-k2.5", "kimi-k2.6"} or (
            "vision" in normalized
        )
    if provider_key == "blablador":
        return model in {"alias-code", "alias-kimi-k3-1m", "alias-muse"}
    if provider_key == "ollama":
        normalized = model.lower()
        return any(marker in normalized for marker in OLLAMA_IMAGE_MODEL_MARKERS)
    return False


def estimate_backend_context_tokens(
    instructions: str,
    messages: Iterable[Mapping[str, Any]],
) -> int:
    """Estimate the context sent to a backend request."""

    tokens = estimate_text_tokens(instructions) + INSTRUCTIONS_CONTEXT_OVERHEAD_TOKENS
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        images = normalized_image_attachments(message.get("images"))
        if not content and not images:
            continue
        tokens += MESSAGE_CONTEXT_OVERHEAD_TOKENS
        tokens += estimate_text_tokens(role)
        tokens += estimate_text_tokens(content)
        tokens += len(images) * MESSAGE_IMAGE_CONTEXT_TOKENS
    return max(1, tokens)


def context_summary_system_prompt() -> str:
    """Return the shared instruction used for rolling conversation summaries."""

    return (
        "You are the assistant in this chat. Summarize this for you to quickly "
        "review what has happened before. Write it from an I-Perspective. Preserve "
        "the user's goals, decisions, constraints, important facts, file paths, "
        "commands, results, unresolved issues, and promised next steps. Preserve "
        "completed writes and their object references, verified working API paths, "
        "failed endpoints and reasons, pagination positions, and the next unfinished "
        "action. Distinguish successful changes from proposals and failed attempts. "
        "Replace superseded facts from the previous summary. Do not copy raw API "
        "responses, past-run metadata, or previous summaries verbatim; retain response "
        "file paths for detailed evidence. Keep the summary under 2000 words. "
        "Return only the summary."
    )


def context_summary_user_prompt(
    messages: list[dict[str, Any]],
    previous_summary: str,
) -> str:
    """Render role-labelled messages and an optional prior rolling summary."""

    sections: list[str] = []
    if previous_summary.strip():
        sections.extend(["Previous summary:", previous_summary.strip(), ""])
    sections.append("Conversation messages to incorporate:")
    for message in messages:
        role = str(message.get("role") or "unknown").strip().upper()
        content = str(message.get("content") or "").strip()
        images = normalized_image_attachments(message.get("images"))
        if images:
            image_labels = ", ".join(image.label or image.path.name for image in images)
            content = "\n".join(
                part
                for part in (content, f"[Image attachments: {image_labels}]")
                if part
            )
        if content:
            sections.extend(["", f"{role}:", content])
    return "\n".join(sections).strip()


def normalized_image_attachments(raw_images: object) -> tuple[ImageAttachment, ...]:
    """Build readable image attachments from persisted message metadata."""

    if not isinstance(raw_images, (list, tuple)):
        return ()

    attachments: list[ImageAttachment] = []
    for raw_image in raw_images:
        if not isinstance(raw_image, Mapping):
            continue
        raw_path = raw_image.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path).expanduser()
        mime_type = str(raw_image.get("mime_type") or image_mime_type(path) or "").strip()
        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            continue
        label = str(raw_image.get("label") or path.name).strip() or path.name
        token = str(raw_image.get("token") or f"[image: {label}]").strip()
        attachments.append(
            ImageAttachment(
                label=label,
                token=token,
                path=path,
                mime_type=mime_type,
            )
        )
    return tuple(attachments)


def context_usage_percent(used_tokens: int, context_window: int | None) -> int:
    """Return clamped percent of context window currently used."""

    if context_window is None or context_window <= 0 or used_tokens <= 0:
        return 0
    percent = round((used_tokens / context_window) * 100)
    return max(1, min(100, percent))


@dataclass
class BaseBackend:
    """Base class for model backends.

    Backend classes own provider request loops and protocol conversion. Runtime state
    stays on the runtime and is available through ``self.runtime``.
    """

    runtime: Any
    provider_key: ClassVar[str] = ""
    provider_label: ClassVar[str] = ""
    env_var: ClassVar[str] = ""
    _usage_total: TokenUsage = field(default_factory=TokenUsage, init=False)
    _latest_context_tokens: int = field(default=0, init=False)
    _context_recovery_attempted: bool = field(default=False, init=False)

    def _recover_context_window(
        self,
        response: str,
        session_path: Path,
        entries: list[ContextMessage],
        callbacks: BackendCallbacks,
    ) -> list[ContextMessage] | None:
        """Rebuild rejected context once without repeating any executed tools."""

        from anomx.agent.context_management import (
            CONTINUE_AFTER_COMPRESSION_PROMPT,
            transient_context_message,
        )

        if (
            not isinstance(response, BackendFailure)
            or response.code != "context_window_exceeded"
            or self._context_recovery_attempted
            or self.runtime._turn_aborted()
        ):
            return None
        self._context_recovery_attempted = True
        entries, compressed = self.runtime.compress_in_turn_context(
            session_path,
            entries,
            current_context_tokens=0,
            status_callback=callbacks.status,
            force=True,
        )
        if not compressed:
            return None
        entries.append(transient_context_message("user", CONTINUE_AFTER_COMPRESSION_PROMPT))
        return entries

    def __getattr__(self, name: str) -> object:
        """Delegate runtime-owned orchestration helpers to the active runtime."""

        return getattr(self.runtime, name)

    def _track_usage(self, usage: TokenUsage | None, callbacks: BackendCallbacks) -> None:
        """Accumulate provider usage and report the latest snapshot."""

        if usage is None:
            return
        self._usage_total = self._usage_total + usage
        self._latest_context_tokens = usage.input_tokens
        if callbacks.usage is not None:
            callbacks.usage(
                UsageSnapshot(
                    total=self._usage_total,
                    context_tokens=self._latest_context_tokens,
                )
            )

    def _visible_stream_text(
        self,
        text_filter: ThinkingTagStreamFilter,
        text: str,
        delta_callback: BackendTextCallback | None,
        status_callback: BackendTextCallback | None,
        thought_callback: BackendTextCallback | None = None,
    ) -> str:
        """Emit only final-answer text while retaining hidden reasoning separately."""
        visible, _thought_started = text_filter.feed(text)
        self._emit_completed_thoughts(text_filter, thought_callback, status_callback)
        if visible and delta_callback is not None:
            delta_callback(visible)
        return visible

    def _finish_visible_stream_text(
        self,
        text_filter: ThinkingTagStreamFilter,
        delta_callback: BackendTextCallback | None,
        status_callback: BackendTextCallback | None = None,
        thought_callback: BackendTextCallback | None = None,
    ) -> str:
        """Flush text buffered to detect a possible split thinking tag."""
        visible = text_filter.finish()
        self._emit_completed_thoughts(text_filter, thought_callback, status_callback)
        if visible and delta_callback is not None:
            delta_callback(visible)
        return visible

    def _emit_completed_thoughts(
        self,
        text_filter: ThinkingTagStreamFilter,
        thought_callback: BackendTextCallback | None,
        status_callback: BackendTextCallback | None,
    ) -> tuple[str, ...]:
        """Surface completed reasoning as an expandable work item when supported."""
        thoughts = text_filter.drain_completed_thoughts()
        for thought in thoughts:
            if thought_callback is not None:
                thought_callback(thought)
            else:
                self.runtime._status(status_callback, "Created a thought")
        return thoughts

    def generate(
        self,
        session_path: Path,
        model: str,
        callbacks: BackendCallbacks,
        *,
        thinking_intensity: str | None = None,
    ) -> str:
        """Generate a model response through this backend."""

        raise NotImplementedError

    def suggest_session_title(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str | None:
        """Suggest a compact session title."""

        del messages, model
        return None

    def evaluate_command_request(
        self,
        *,
        command: str,
        statement: str,
        user_message: str,
        model: str,
    ) -> CommandRiskEvaluation | None:
        """Evaluate the user-visible risk of a pending command."""

        del command, statement, user_message, model
        return None

    def suggest_memory_metadata(
        self,
        *,
        kind: MemoryKind | str,
        context: Mapping[str, Any],
        content: str,
        model: str,
    ) -> MemoryMetadata | None:
        """Suggest a compact memory title and summary."""

        del kind, context, content, model
        return None

    def suggest_project_name(self, prompt: str, model: str) -> str | None:
        """Suggest a compact project name."""

        del prompt, model
        return None

    def suggest_session_continuation(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str | None:
        """Suggest a continuation prompt for an existing session."""

        del messages, model
        return None

    def summarize_conversation(
        self,
        messages: list[dict[str, Any]],
        previous_summary: str,
        model: str,
    ) -> str | None:
        """Summarize a transcript prefix for rolling context compression."""

        del messages, previous_summary, model
        return None

    def _api_key(self, provider: str, env_var: str) -> str | None:
        env_key = os.environ.get(env_var)
        if env_key:
            return env_key
        api_keys = self.runtime.home.load_auth().get("api_keys")
        if not isinstance(api_keys, dict):
            return None
        configured_key = api_keys.get(provider)
        if isinstance(configured_key, str) and configured_key.strip():
            return configured_key.strip()
        return None

    def _api_error(
        self,
        provider_key: str,
        provider_label: str,
        env_var: str,
        status: int,
        body: str,
    ) -> str:
        detail, error_type = self._parse_api_error(body)
        if self._looks_like_invalid_api_key(provider_key, status, error_type, detail):
            return self._invalid_api_key_message(provider_label, env_var)
        context_error = any(marker in f"{error_type} {detail}".lower() for marker in (
            "context_length_exceeded", "maximum context length", "context window",
            "prompt is too long", "input is too long", "too many input tokens",
        ))
        return BackendFailure(
            f"{provider_label} request failed ({status}): {detail or 'No error detail.'}",
            code="context_window_exceeded" if context_error else "model_request_failed",
        )

    def _parse_api_error(self, body: str) -> tuple[str, str | None]:
        detail = body.strip()
        error_type: str | None = None
        with suppress(json.JSONDecodeError):
            payload = json.loads(body)
            if isinstance(payload, dict):
                if isinstance(payload.get("detail"), str):
                    detail = payload["detail"]
                error = payload.get("error")
                if isinstance(error, dict):
                    if isinstance(error.get("message"), str):
                        detail = error["message"]
                    if isinstance(error.get("type"), str):
                        error_type = error["type"]
                elif isinstance(error, str):
                    detail = error
        return detail or "No error detail.", error_type

    def _missing_api_key_message(self, provider_label: str, env_var: str) -> str:
        return BackendFailure(
            f"{provider_label} API key is not configured. "
            f"Add it during onboarding or set {env_var}.",
            code="model_authentication_failed",
        )

    def _invalid_api_key_message(self, provider_label: str, env_var: str) -> str:
        return BackendFailure(
            f"{provider_label} credentials were rejected. "
            f"Check {env_var} or update the saved {provider_label} API key in Anomx. "
            "The key may be invalid, expired, or revoked.",
            code="model_authentication_failed",
        )

    def _looks_like_invalid_api_key(
        self,
        provider_key: str,
        status: int,
        error_type: str | None,
        detail: str,
    ) -> bool:
        lowered = detail.lower()
        if provider_key == "openai":
            if error_type == "authentication_error" and "member of an organization" not in lowered:
                return True
            if status != 401:
                return False
            return any(
                needle in lowered
                for needle in (
                    "incorrect api key",
                    "invalid api key",
                    "invalid authentication",
                    "invalid_api_key",
                    "revoked",
                    "expired",
                )
            )
        if provider_key in {"anthropic", "desy", "blablador", "kimi"}:
            return status == 401 or error_type == "authentication_error"
        return False

    def _model_request_with_retries(
        self,
        *,
        provider_key: str,
        provider_label: str,
        env_var: str,
        status_callback: BackendTextCallback | None,
        stream_once: Callable[[], ModelRequestStreamResponse],
    ) -> ModelRequestStreamResponse:
        max_attempts = MODEL_REQUEST_RETRY_COUNT + 1
        for attempt in range(max_attempts):
            if getattr(self.runtime, "before_model_request", None) is not None:
                self.runtime.before_model_request()
            try:
                return stream_once()
            except urllib.error.HTTPError as error:
                error_body = error.read().decode("utf-8", errors="replace")
                message = self._api_error(
                    provider_key,
                    provider_label,
                    env_var,
                    error.code,
                    error_body,
                )
                if (
                    error.code not in MODEL_REQUEST_RETRY_STATUS_CODES
                    or attempt >= MODEL_REQUEST_RETRY_COUNT
                ):
                    return message
                delay = self._model_request_retry_delay(attempt)
                if not self._sleep_before_model_request_retry(
                    provider_label,
                    f"HTTP {error.code}",
                    delay,
                    attempt + 1,
                    MODEL_REQUEST_RETRY_COUNT,
                    status_callback,
                ):
                    return ""
            except (OSError, urllib.error.URLError, TimeoutError) as error:
                message = BackendFailure(f"{provider_label} request failed: {error}")
                if attempt >= MODEL_REQUEST_RETRY_COUNT:
                    return message
                delay = self._model_request_retry_delay(attempt)
                if not self._sleep_before_model_request_retry(
                    provider_label,
                    str(error),
                    delay,
                    attempt + 1,
                    MODEL_REQUEST_RETRY_COUNT,
                    status_callback,
                ):
                    return ""
        return BackendFailure(f"{provider_label} request failed.")

    @staticmethod
    def _model_request_retry_delay(attempt: int) -> float:
        return min(
            MODEL_REQUEST_RETRY_INITIAL_DELAY_SECONDS
            * (MODEL_REQUEST_RETRY_BACKOFF_FACTOR**attempt),
            MODEL_REQUEST_RETRY_MAX_DELAY_SECONDS,
        )

    def _sleep_before_model_request_retry(
        self,
        provider_label: str,
        failure: str,
        delay_seconds: float,
        retry_number: int,
        retry_count: int,
        status_callback: BackendTextCallback | None,
    ) -> bool:
        del provider_label, failure, retry_number, retry_count
        self.runtime._status(status_callback, "Reconnecting")
        deadline = time.monotonic() + delay_seconds
        while True:
            if self.runtime._turn_aborted():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(MODEL_REQUEST_RETRY_SLEEP_SLICE_SECONDS, remaining))

    def _openai_messages(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        supports_images = backend_supports_image_input("openai", model)
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            images = (
                normalized_image_attachments(message.get("images"))
                if role == "user" and supports_images
                else ()
            )
            image_blocks = [
                block
                for image in images
                if (block := self._openai_image_block(image)) is not None
            ]
            if not content and not image_blocks:
                continue
            if image_blocks and role == "user":
                content_blocks: list[dict[str, Any]] = []
                text = self._content_with_image_labels(content, images)
                if text:
                    content_blocks.append({"type": "input_text", "text": text})
                content_blocks.extend(image_blocks)
                converted.append({"role": role, "content": content_blocks})
            else:
                converted.append({"role": role, "content": content})
        return converted

    def _anthropic_messages(
        self,
        messages: list[dict[str, Any]],
        provider_key: str,
        model: str,
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        supports_images = backend_supports_image_input(provider_key, model)
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            images = (
                normalized_image_attachments(message.get("images"))
                if role == "user" and supports_images
                else ()
            )
            blocks = self._anthropic_content_blocks(content, images)
            if not blocks:
                continue
            if role == "assistant":
                self._append_anthropic_message(converted, "assistant", content)
            elif role == "system":
                self._append_anthropic_message(converted, "user", f"[System note]\n{content}")
            else:
                self._append_anthropic_blocks(converted, "user", blocks)
        return converted

    def _append_anthropic_message(
        self,
        messages: list[dict[str, Any]],
        role: str,
        text: str,
    ) -> None:
        self._append_anthropic_blocks(messages, role, ({"type": "text", "text": text},))

    def _append_anthropic_blocks(
        self,
        messages: list[dict[str, Any]],
        role: str,
        blocks: Iterable[dict[str, Any]],
    ) -> None:
        content_blocks = list(blocks)
        if not content_blocks:
            return
        if messages and messages[-1].get("role") == role:
            content = messages[-1].get("content")
            if isinstance(content, list):
                content.extend(content_blocks)
                return
        messages.append({"role": role, "content": content_blocks})

    def _anthropic_content_blocks(
        self,
        content: str,
        images: tuple[ImageAttachment, ...],
    ) -> tuple[dict[str, Any], ...]:
        blocks: list[dict[str, Any]] = []
        text = self._content_with_image_labels(content, images)
        if text:
            blocks.append({"type": "text", "text": text})
        for image in images:
            block = self._anthropic_image_block(image)
            if block is not None:
                blocks.append(block)
        return tuple(blocks)

    def _ollama_messages(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        supports_images = backend_supports_image_input("ollama", model)
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            images = (
                normalized_image_attachments(message.get("images"))
                if role == "user" and supports_images
                else ()
            )
            encoded_images = [
                encoded
                for image in images
                if (encoded := self._image_base64(image)) is not None
            ]
            if not content and not encoded_images:
                continue
            converted_message: dict[str, Any] = {
                "role": role,
                "content": self._content_with_image_labels(content, images),
            }
            if encoded_images and role == "user":
                converted_message["images"] = encoded_images
            converted.append(converted_message)
        return converted

    def _content_with_image_labels(
        self,
        content: str,
        images: tuple[ImageAttachment, ...],
    ) -> str:
        text = content.strip()
        if not images:
            return text
        image_lines = "\n".join(f"- {image.label}" for image in images)
        attachment_note = f"Attached images:\n{image_lines}"
        return f"{text}\n\n{attachment_note}" if text else attachment_note

    def _openai_image_block(self, image: ImageAttachment) -> dict[str, str] | None:
        encoded = self._image_base64(image)
        if encoded is None:
            return None
        return {
            "type": "input_image",
            "image_url": f"data:{image.mime_type};base64,{encoded}",
        }

    def _anthropic_image_block(self, image: ImageAttachment) -> dict[str, Any] | None:
        encoded = self._image_base64(image)
        if encoded is None:
            return None
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.mime_type,
                "data": encoded,
            },
        }

    def _image_base64(self, image: ImageAttachment) -> str | None:
        with suppress(OSError):
            return base64.b64encode(image.path.read_bytes()).decode("ascii")
        return None

    def _extract_anthropic_text(
        self,
        content: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    ) -> str:
        parts: list[str] = []
        for block in content:
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip() or "No response."

    def _finalize_anthropic_tool_input(
        self,
        content_by_index: dict[int, dict[str, Any]],
        tool_json_parts: dict[int, list[str]],
        index: int,
    ) -> None:
        block = content_by_index.get(index)
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            return
        raw_json = "".join(tool_json_parts.get(index, [])).strip()
        if not raw_json:
            block["input"] = {}
            return
        with suppress(json.JSONDecodeError):
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                block["input"] = parsed
                return
        block["input"] = {"raw_input": raw_json}

    def _max_output_tokens(self, model: str, fallback: int) -> int:
        return model_output_token_budget(model, fallback)

    def _openai_reasoning_config(
        self,
        model: str,
        thinking_intensity: str | None,
    ) -> dict[str, Any]:
        reasoning: dict[str, Any] = {"summary": "auto"}
        intensity = self._supported_thinking_intensity("openai", model, thinking_intensity)
        if intensity is not None:
            reasoning["effort"] = intensity
        return reasoning

    def _anthropic_output_config(
        self,
        model: str,
        thinking_intensity: str | None,
    ) -> dict[str, Any]:
        intensity = self._supported_thinking_intensity("anthropic", model, thinking_intensity)
        return {} if intensity is None else {"effort": intensity}

    def _supported_thinking_intensity(
        self,
        provider_key: str,
        model: str,
        thinking_intensity: str | None,
    ) -> str | None:
        intensity = normalize_thinking_intensity(thinking_intensity)
        if intensity == THINKING_INTENSITY_AUTO:
            return None
        supported = {option.value for option in thinking_intensity_options(provider_key, model)}
        return intensity if intensity in supported else None

    def _anthropic_thinking_config(self, model: str) -> dict[str, Any]:
        if model in {
            "claude-fable-5-1",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
        }:
            return {"type": "adaptive", "display": "summarized"}
        max_tokens = self._max_output_tokens(model, 4_096)
        budget_tokens = max(1_024, min(2_048, max_tokens - 1))
        return {
            "type": "enabled",
            "budget_tokens": budget_tokens,
            "display": "summarized",
        }

    def _anthropic_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            }
            for tool in self._tool_definitions()
        ]

    def _ollama_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in self._tool_definitions()
        ]

    def _openai_tools(self) -> list[dict[str, Any]]:
        # ``strict`` is intentionally omitted: strict mode requires every tool's
        # parameter schema (including nested objects) to list all properties in
        # ``required`` with ``additionalProperties: false``. Several tools use
        # optional/nested fields, so enabling strict makes the OpenAI Responses
        # API reject the whole request with HTTP 400. The Chat Completions path
        # (DESY/Blablador) omits strict for the same reason.
        return [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
            for tool in self._tool_definitions()
        ]

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [tool.definition() for tool in self.runtime._available_tools()]

    def _parse_tool_arguments(self, raw_arguments: str) -> dict[str, Any]:
        with suppress(json.JSONDecodeError):
            payload = json.loads(raw_arguments)
            if isinstance(payload, dict):
                return cast(dict[str, Any], payload)
        return {}

    def _execute_requested_tools(
        self,
        response: OpenAIStreamResponse,
        callbacks: BackendCallbacks,
        session_path: Path,
    ) -> list[dict[str, str]]:
        outputs: list[dict[str, str]] = []
        for tool_call in response.tool_calls:
            arguments = self._parse_tool_arguments(tool_call.arguments)
            output = self.runtime._execute_tool(
                tool_call.name,
                arguments,
                callbacks,
                session_path,
            )
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": output,
                }
            )
        return outputs

    def _execute_anthropic_requested_tools(
        self,
        response: AnthropicStreamResponse,
        callbacks: BackendCallbacks,
        session_path: Path,
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for tool_call in response.tool_calls:
            output = self.runtime._execute_tool(
                tool_call.name,
                tool_call.input,
                callbacks,
                session_path,
            )
            outputs.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.tool_use_id,
                    "content": output,
                }
            )
        return outputs

    def _execute_ollama_requested_tools(
        self,
        response: OllamaStreamResponse,
        callbacks: BackendCallbacks,
        session_path: Path,
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for tool_call in response.tool_calls:
            output = self.runtime._execute_tool(
                tool_call.name,
                tool_call.arguments,
                callbacks,
                session_path,
            )
            outputs.append(
                {
                    "role": "tool",
                    "tool_name": tool_call.name,
                    "content": output,
                }
            )
        return outputs

    def extract_openai_text(self, data: dict[str, Any]) -> str:
        """Extract text from an OpenAI Responses API payload."""

        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        parts: list[str] = []
        output = data.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for content_item in content:
                    if not isinstance(content_item, dict):
                        continue
                    text = content_item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
        return "\n".join(parts).strip() or "No response."

    def extract_anthropic_text(self, data: dict[str, Any]) -> str:
        """Extract text from an Anthropic Messages API payload."""

        content = data.get("content")
        if not isinstance(content, list):
            return "No response."
        return self._extract_anthropic_text(content)

    def _title_prompt(self, messages: list[dict[str, Any]]) -> str:
        conversation = "\n".join(
            f"{message.get('role', '')}: {message.get('content', '')}"
            for message in messages[-6:]
        )
        return f"Conversation:\n{conversation}"

    def _command_evaluation_system_prompt(self) -> str:
        return (
            "Explain what command is run and what it does to the user. Keep it short "
            "and factual, at most 2-3 sentences. What you output is shown directly "
            "to the user. Return only JSON with keys risk and description. risk must "
            'be one of "low", "medium", or "high". Classify purely read-only '
            "inspection commands as low risk. Classify commands that change the "
            "workspace, install packages, start or stop processes, write files, modify "
            "configuration, alter persistent state, or access paths outside the trusted "
            "workspace as medium risk unless they are clearly destructive or host-control "
            "operations. Classify severe destructive or host-control commands as high "
            "risk. Do not include markdown, code fences, or extra keys."
        )

    def _command_evaluation_user_prompt(
        self,
        *,
        command: str,
        statement: str,
        user_message: str,
    ) -> str:
        return (
            "Original user message:\n"
            f"{user_message.strip() or '(not available)'}\n\n"
            "Agent thought / stated intent:\n"
            f"{statement.strip() or '(not available)'}\n\n"
            "Command requested for approval:\n"
            f"{command.strip()}"
        )

    def _command_evaluation_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "risk": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "description": {
                    "type": "string",
                },
            },
            "required": ["risk", "description"],
            "additionalProperties": False,
        }

    def _sanitize_command_evaluation(self, text: str) -> CommandRiskEvaluation | None:
        payload = extract_json_object(text)
        if payload is None:
            return None
        risk = str(payload.get("risk") or "").strip().lower()
        if risk == "hight":
            risk = "high"
        if risk not in {"low", "medium", "high"}:
            return None
        description = " ".join(str(payload.get("description") or "").split())
        if not description:
            return None
        return CommandRiskEvaluation(risk=risk, description=description[:500])

    def _memory_metadata_system_prompt(self) -> str:
        return (
            "Create durable metadata for a local agent memory. Return only JSON with "
            "keys title and summary. The title must be specific, 3-8 words, and not "
            "generic. The summary must be one short precise sentence under 140 "
            "characters. Do not include markdown, code fences, or extra keys."
        )

    def _memory_metadata_user_prompt(
        self,
        *,
        kind: MemoryKind | str,
        context: Mapping[str, Any],
        content: str,
    ) -> str:
        return (
            f"Memory kind: {str(kind)}\n\n"
            "Context JSON:\n"
            f"{json.dumps(dict(context), indent=2, ensure_ascii=False, default=str)}\n\n"
            "Original memory content:\n"
            f"{content.strip()}"
        )

    def _memory_metadata_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["title", "summary"],
            "additionalProperties": False,
        }

    def _sanitize_memory_metadata(self, text: str) -> MemoryMetadata | None:
        return sanitize_memory_metadata(text)

    def _project_name_system_prompt(self) -> str:
        return (
            "Please only return a plain text name of 2-3 words for this directory "
            "in a project style. No quotes. No trailing punctuation."
        )

    def _context_summary_system_prompt(self) -> str:
        return context_summary_system_prompt()

    def _context_summary_user_prompt(
        self,
        messages: list[dict[str, Any]],
        previous_summary: str,
    ) -> str:
        return context_summary_user_prompt(messages, previous_summary)

    def _continuation_system_prompt(self) -> str:
        return (
            "Write one concise second-person question for an Anomx startup resume prompt. "
            "Ask whether to continue the previous CLI session and mention the concrete "
            "work if it is clear. Start with 'Do you want to continue'. Use 14 to 28 "
            "words. Return only the question. No quotes."
        )

    def _sanitize_continuation_statement(self, statement: str) -> str | None:
        cleaned = " ".join(statement.strip().strip("\"'`").split())
        if not cleaned:
            return None
        cleaned = cleaned.rstrip(".:;,-")
        if not cleaned.endswith("?"):
            cleaned = f"{cleaned}?"
        words = cleaned.split()
        if len(words) > 32:
            cleaned = " ".join(words[:32]).rstrip("?") + "?"
        return cleaned[:180] or None

    def _sanitize_title(self, title: str) -> str | None:
        cleaned = " ".join(title.strip().strip("\"'`").split())
        cleaned = cleaned.rstrip(".:;,-")
        if not cleaned:
            return None
        words = cleaned.split()
        if len(words) > 8:
            cleaned = " ".join(words[:8])
        return cleaned[:60] or None

    def _sanitize_project_name(self, name: str) -> str | None:
        cleaned = " ".join(name.strip().strip("\"'`").split())
        cleaned = cleaned.rstrip(".:;,-")
        if not cleaned:
            return None
        words = [
            word.strip(" .,:;!?()[]{}\"'`")
            for word in cleaned.replace("_", " ").replace("/", " ").split()
        ]
        words = [word for word in words if word]
        if len(words) < 2:
            return None
        return " ".join(words[:3])[:48] or None
