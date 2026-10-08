"""Budgeted history compression preserves evidence and retries bounded summaries."""

import re

import pytest

from anomx.agent import AnomxHome
from anomx.agent.context_management import (
    ContextMessage,
    adaptive_context_target,
    budgeted_history_compression_prefix,
    tool_call_payload,
)
from anomx.agent.exceptions import AgentBackendError
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


def make_history(tmp_path, summarizer, *, large_index=None):
    home = AnomxHome(tmp_path / "home")
    home.save_config({**home.load_config(), "maximum_context_tokens": 32_000})
    session = home.create_session(tmp_path, provider="desy", model="coding")
    for index in range(50):
        home.append_session_event(
            session.path,
            "user_message" if index % 2 == 0 else "agent_message",
            {
                "message": "measurement " * (15_000 if index == large_index else 40),
                "message_id": f"m{index}",
            },
        )
    return AgentRuntime(home, tmp_path, context_summarizer=summarizer), session.path


def test_budgeted_tail_recovers_when_twenty_four_recent_messages_cannot_fit(tmp_path):
    requests, activities = [], []
    runtime, path = make_history(
        tmp_path,
        lambda system, user: requests.append((system, user)) or "Keep verified measurements.",
        large_index=30,
    )
    original = runtime.conversation_messages(path)
    entries = runtime.backend_conversation_entries(path)
    target = adaptive_context_target(runtime.maximum_context_tokens(), entries)

    runtime._prepare_context_compression(path, RuntimeCallbacks(context_activity=activities.append))

    state = runtime.context_compression_state(path)
    assert state.last_message_id == "m30"
    assert state.context_tokens_after <= target
    assert len(requests) == 1
    assert len(runtime.backend_conversation_entries(path)) == 19
    assert runtime.backend_conversation_entries(path)[-1] == entries[-1]
    assert runtime.conversation_messages(path) == original
    assert AgentRuntime(runtime.home, tmp_path).backend_conversation_entries(path) == entries[31:]
    assert activities[-1]["retained_message_count"] == 19
    assert activities[-1]["compression_attempts"] == 1
    assert activities[-1]["status"] == "completed"


def test_budgeted_boundary_preserves_parallel_tools_and_latest_exchange():
    entries = [ContextMessage(f"m{i}", {"role": "user", "content": "x"}) for i in range(48)]
    entries[30:33] = [
        ContextMessage("", {"role": "assistant", "tool_calls": [
            tool_call_payload("read", "a", {}), tool_call_payload("read", "b", {}),
        ]}),
        ContextMessage("t1", {"role": "tool", "tool_call_id": "a", "content": "x" * 20_000}),
        ContextMessage("t2", {"role": "tool", "tool_call_id": "b", "content": "done"}),
    ]
    prefix = budgeted_history_compression_prefix(
        entries, maximum_retained_tokens=500, minimum_retained_messages=1,
    )
    assert prefix == entries[:33]
    latest_exchange = entries[30:33]
    entries = entries[:30] + latest_exchange
    prefix = budgeted_history_compression_prefix(
        entries, maximum_retained_tokens=500, minimum_retained_messages=1,
    )
    assert entries[len(prefix):] == latest_exchange


def test_oversized_summary_retries_from_original_evidence_with_smaller_budget(tmp_path):
    requests, activities = [], []

    def summarize(system, user):
        requests.append((system, user))
        return "oversized " * 20_000 if len(requests) == 1 else "Verified results retained."

    runtime, path = make_history(tmp_path, summarize)
    runtime._prepare_context_compression(path, RuntimeCallbacks(context_activity=activities.append))

    assert len(requests) == 2
    budgets = [int(re.search(r"at most (\d+) tokens", system)[1]) for system, _ in requests]
    assert budgets[1] == budgets[0] // 2
    assert requests[0][1] == requests[1][1]
    assert "oversized" not in requests[1][1]
    assert runtime.context_compression_state(path).summary == "Verified results retained."
    assert activities[-1]["compression_attempts"] == 2
    assert activities[-1]["model_requests"] == 2
    assert len([
        event for event in runtime.home.read_session_events(path)
        if event.get("payload", {}).get("type", event["type"]) == "context_compression"
    ]) == 1


def test_failed_retries_preserve_previous_summary_and_transcript(tmp_path):
    requests, activities = [], []
    runtime, path = make_history(
        tmp_path, lambda *_: requests.append(True) or "oversized " * 20_000,
    )
    runtime.home.append_session_event(path, "context_compression", {
        "summary": "Previously verified constraints.", "last_message_id": "m0",
    })
    previous = runtime.context_compression_state(path)
    transcript = runtime.conversation_messages(path)
    runtime._context_activity_callback = activities.append

    with pytest.raises(AgentBackendError, match="after two attempts") as error:
        runtime.compress_in_turn_context(
            path, runtime.backend_conversation_entries(path),
            current_context_tokens=32_000, status_callback=None,
        )

    assert error.value.code == "context_compression_failed"
    assert len(requests) == 2
    assert runtime.context_compression_state(path) == previous
    assert runtime.conversation_messages(path) == transcript
    assert activities[-1]["status"] == "failed"
    assert activities[-1]["compression_attempts"] == 2
    assert activities[-1]["candidate_context_tokens"] > 32_000


def test_irreducible_latest_message_fails_before_requesting_a_summary(tmp_path):
    requests = []
    runtime, path = make_history(
        tmp_path, lambda *_: requests.append(True) or "Summary.", large_index=49,
    )
    original = runtime.conversation_messages(path)
    with pytest.raises(AgentBackendError, match="latest complete exchange"):
        runtime._prepare_context_compression(path, RuntimeCallbacks())
    assert requests == []
    assert runtime.context_compression_state(path) is None
    assert runtime.conversation_messages(path) == original


def test_cancellation_during_summary_never_commits_a_boundary(tmp_path):
    runtime, path = make_history(tmp_path, None)

    def summarize(*_):
        runtime._turn_abort_event.set()
        return "Partial work."

    runtime.context_summarizer = summarize
    entries = runtime.backend_conversation_entries(path)
    assert runtime.compress_in_turn_context(
        path, entries, current_context_tokens=20_000, status_callback=None,
    ) == (entries, False)
    assert runtime.context_compression_state(path) is None
