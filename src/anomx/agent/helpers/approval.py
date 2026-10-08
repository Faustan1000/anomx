"""Bounded, secret-redacted action details for approval evaluation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


def approval_user_context(messages: Sequence[Mapping[str, Any]]) -> str:
    """Keep the original request and recent user steering within a bounded context."""

    user_messages = [
        str(message.get("content") or "").strip()
        for message in messages
        if message.get("role") == "user" and str(message.get("content") or "").strip()
    ]
    if not user_messages:
        return ""

    def bounded(message: str) -> str:
        if len(message) <= 8000:
            return message
        return message[:4000] + "\n[Middle of user message omitted.]\n" + message[-4000:]

    original = bounded(user_messages[0])
    separator = "\n\nLater user message:\n"
    omitted = "\n\n[Earlier user messages omitted; do not infer additional permission.]"
    remaining = 32000 - len(original) - len(omitted)
    recent: list[str] = []
    for message in reversed(user_messages[1:]):
        entry = separator + bounded(message)
        if len(entry) > remaining:
            break
        recent.append(entry)
        remaining -= len(entry)
    return original + (omitted if len(recent) < len(user_messages) - 1 else "") + "".join(
        reversed(recent)
    )


def approval_action_details(value: object) -> str:
    """Retain targets and changes without forwarding credential fields."""

    def redact(item: object) -> object:
        if isinstance(item, dict):
            return {
                str(key): "[redacted]"
                if re.search(
                    r"password|secret|token|authorization|credential|api[_-]?key", str(key), re.I
                )
                else redact(entry)
                for key, entry in item.items()
            }
        if isinstance(item, list):
            return [redact(entry) for entry in item]
        return item

    serialized = json.dumps(redact(value), ensure_ascii=False, default=str)
    return (
        serialized
        if len(serialized) <= 16000
        else serialized[:16000]
        + "\n[Action details truncated; do not assume omitted effects are authorized.]"
    )
