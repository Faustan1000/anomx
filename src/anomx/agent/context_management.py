"""Automatic context-window compression primitives for the agent runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from anomx.agent.base.backends import estimate_backend_context_tokens

CONTINUE_AFTER_COMPRESSION_PROMPT = (
    "Continue the current task from the compressed history in Previous "
    "Conversation. Preserve the established plan and use the recorded tool "
    "results without repeating completed work."
)


@dataclass(frozen=True)
class ContextMessage:
    """One backend-visible transcript message with a stable storage identifier."""

    message_id: str
    payload: dict[str, Any]

    @property
    def estimated_tokens(self) -> int:
        empty_context = estimate_backend_context_tokens("", ())
        return max(
            1,
            estimate_backend_context_tokens("", (self.payload,)) - empty_context,
        )


@dataclass(frozen=True)
class ContextCompressionState:
    """Rolling summary and transcript boundary persisted after a compression."""

    summary: str
    last_message_id: str
    compressed_message_count: int
    context_tokens_before: int
    maximum_context_tokens: int
    target_percent: int

    @classmethod
    def from_payload(cls, payload: object) -> ContextCompressionState | None:
        if not isinstance(payload, dict):
            return None
        summary = str(payload.get("summary") or "").strip()
        last_message_id = str(payload.get("last_message_id") or "").strip()
        if not summary or not last_message_id:
            return None
        return cls(
            summary=summary,
            last_message_id=last_message_id,
            compressed_message_count=max(
                0, int(payload.get("compressed_message_count") or 0)
            ),
            context_tokens_before=max(
                0, int(payload.get("context_tokens_before") or 0)
            ),
            maximum_context_tokens=max(
                0, int(payload.get("maximum_context_tokens") or 0)
            ),
            target_percent=max(0, int(payload.get("target_percent") or 0)),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "last_message_id": self.last_message_id,
            "compressed_message_count": self.compressed_message_count,
            "context_tokens_before": self.context_tokens_before,
            "maximum_context_tokens": self.maximum_context_tokens,
            "target_percent": self.target_percent,
        }


def transient_context_message(role: str, content: object) -> ContextMessage:
    """Build one provider-local message that can safely survive a chain reset."""

    if isinstance(content, str):
        text = content.strip()
    else:
        try:
            text = json.dumps(content, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            text = str(content).strip()
    return ContextMessage(
        message_id="",
        payload={"role": role, "content": text},
    )


def projected_context_tokens(
    input_tokens: int,
    output_tokens: int,
    pending_entries: list[ContextMessage],
) -> int:
    """Estimate the next request size from reported usage and new local data."""

    return max(0, input_tokens) + max(0, output_tokens) + sum(
        entry.estimated_tokens for entry in pending_entries
    )


def messages_after_compression(
    messages: list[ContextMessage],
    state: ContextCompressionState | None,
) -> list[ContextMessage]:
    """Return messages after the boundary represented by *state*."""

    if state is None:
        return messages
    for index, message in enumerate(messages):
        if message.message_id == state.last_message_id:
            return messages[index + 1 :]
    return messages


def compression_prefix(
    messages: list[ContextMessage],
    *,
    retained_message_tokens: int,
    minimum_retained_messages: int = 1,
) -> list[ContextMessage]:
    """Choose the oldest prefix so the remaining tail fits the token budget."""

    if not messages:
        return []
    minimum_retained_messages = max(
        0,
        min(minimum_retained_messages, len(messages)),
    )
    retained_tokens = 0
    retained_start = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        message_tokens = messages[index].estimated_tokens
        retained_count = len(messages) - retained_start
        if retained_count >= minimum_retained_messages and (
            retained_tokens + message_tokens > retained_message_tokens
        ):
            break
        retained_tokens += message_tokens
        retained_start = index
    return messages[:retained_start]


def context_summary_batches(
    messages: list[ContextMessage],
    *,
    maximum_batch_tokens: int,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Split summary input into bounded, role-preserving message batches."""

    maximum_batch_tokens = max(1, maximum_batch_tokens)
    batches: list[tuple[dict[str, Any], ...]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for message in messages:
        payloads = _split_large_message(message.payload, maximum_batch_tokens)
        for payload in payloads:
            payload_tokens = ContextMessage(message.message_id, payload).estimated_tokens
            if current and current_tokens + payload_tokens > maximum_batch_tokens:
                batches.append(tuple(current))
                current = []
                current_tokens = 0
            current.append(payload)
            current_tokens += payload_tokens
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _split_large_message(
    payload: dict[str, Any],
    maximum_tokens: int,
) -> tuple[dict[str, Any], ...]:
    content = str(payload.get("content") or "")
    if ContextMessage("", payload).estimated_tokens <= maximum_tokens or not content:
        return (payload,)
    chunk_size = max(1, maximum_tokens * 4 - 128)
    chunks = tuple(
        content[start : start + chunk_size]
        for start in range(0, len(content), chunk_size)
    )
    return tuple(
        {
            **payload,
            "content": f"[Part {index + 1} of {len(chunks)}]\n{chunk}",
        }
        for index, chunk in enumerate(chunks)
    )
