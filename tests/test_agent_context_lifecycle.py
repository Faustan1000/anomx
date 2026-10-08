"""Context activity and durable reductions across provider request boundaries."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from anomx.agent import AnomxHome
from anomx.agent.backends.desy_assistant import DesyAssistantBackend
from anomx.agent.backends.ollama import OllamaBackend
from anomx.agent.backends.openai import OpenAIBackend
from anomx.agent.backends.openai_chat import OpenAICompatibleChatBackend
from anomx.agent.context_management import ContextMessage, ContextToolResult
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


def make_runtime(tmp_path, **kwargs):
    home = AnomxHome(tmp_path / "home")
    home.save_config({**home.load_config(), "maximum_context_tokens": 32_000})
    session = home.create_session(tmp_path, provider="openai", model="gpt-5.5")
    home.append_session_event(
        session.path, "user_message", {"message": "Keep the constraints.", "message_id": "u1"}
    )
    return AgentRuntime(home, tmp_path, **kwargs), session


@pytest.mark.parametrize("reason", ["unavailable", "oversized", "cancelled"])
def test_preflight_does_not_emit_optimization_activity(tmp_path, monkeypatch, reason):
    calls, activities = [], []
    runtime, session = make_runtime(tmp_path)
    runtime._context_activity_callback = activities.append
    monkeypatch.setattr(runtime, "_background_work_backend", lambda _: None)
    if reason != "unavailable":
        runtime.context_optimizer = lambda *_: calls.append(True) or "Digest"
    if reason == "oversized":
        runtime.context_optimizer_context_window = 1_024
    if reason == "cancelled":
        runtime._turn_abort_event.set()
    entries = [
        ContextMessage(
            "t1",
            {
                "role": "tool",
                "content": "measurement " * 1_000,
                "context_kind": "tool",
            },
        )
    ]
    result, changed = runtime._optimize_tool_blocks(
        session.path, entries, maximum=32_000, current_context_tokens=20_000,
    )
    assert changed == (reason != "cancelled")
    assert calls == []
    if changed:
        assert json.loads(result[0].payload["content"])["result_mode"] == "file"
        assert activities[-1]["file_backed_results"] == 1
    else:
        assert result == entries and activities == []


@pytest.mark.parametrize("answer", ["KEEP", "Measurement x=3."])
def test_activity_begins_at_model_call_and_preserves_provider_token_baseline(tmp_path, answer):
    activities = []

    def optimize(*_):
        assert [activity["status"] for activity in activities] == ["running"]
        return answer

    runtime, session = make_runtime(tmp_path, context_optimizer=optimize)
    runtime._context_activity_callback = activities.append
    entries = [
        ContextMessage(
            "t1",
            {
                "role": "tool",
                "content": "measurement " * 1_000,
                "context_kind": "tool",
            },
        )
    ]
    result, changed = runtime._optimize_tool_blocks(
        session.path,
        entries,
        maximum=32_000,
        current_context_tokens=20_000,
    )
    saved = sum(entry.estimated_tokens for entry in entries) - sum(
        entry.estimated_tokens for entry in result
    )
    assert activities[-1]["context_tokens_after"] == 20_000 - saved
    assert activities[-1]["changed"] and changed
    if answer == "KEEP":
        assert json.loads(result[0].payload["content"])["result_mode"] == "file"
    assert activities[-1]["model_requests"] == 1


def test_unavailable_summary_model_emits_no_running_activity(tmp_path, monkeypatch):
    runtime, session = make_runtime(tmp_path)
    activities = []
    runtime._context_activity_callback = activities.append
    monkeypatch.setattr(runtime, "_background_work_backend", lambda _: None)
    entries = runtime.backend_conversation_entries(session.path)
    assert runtime.compress_in_turn_context(
        session.path,
        entries,
        current_context_tokens=26_000,
        status_callback=None,
    ) == (entries, False)
    assert [(activity["kind"], activity["status"]) for activity in activities] == [
        ("check", "running"),
        ("check", "completed"),
    ]
    assert activities[0]["model_requests"] == 0


def test_short_history_does_not_invoke_summary_model(tmp_path):
    calls, activities = [], []
    runtime, session = make_runtime(tmp_path, context_summarizer=lambda *_: calls.append(True))
    runtime._context_activity_callback = activities.append
    entries = runtime.backend_conversation_entries(session.path)
    assert runtime.compress_in_turn_context(
        session.path, entries, current_context_tokens=26_000, status_callback=None,
    ) == (entries, False)
    assert calls == []
    assert [(event["kind"], event["status"]) for event in activities] == [
        ("check", "running"),
        ("check", "completed"),
    ]
    assert len({event["id"] for event in activities}) == 1
    assert activities[-1]["context_tokens_after"] == 26_000


def test_policy_checks_only_emit_at_evaluation_checkpoints(tmp_path):
    calls, activities = [], []
    runtime, session = make_runtime(
        tmp_path, context_optimizer=lambda *_: calls.append(True) or "Digest",
    )
    runtime._context_activity_callback = activities.append
    entries = runtime.backend_conversation_entries(session.path)
    for tokens in (16_000, 17_000, 20_800, 21_000):
        assert runtime.compress_in_turn_context(
            session.path, entries, current_context_tokens=tokens, status_callback=None,
        ) == (entries, False)
    assert calls == []
    assert [activity["context_tokens_before"] for activity in activities] == [
        16_000,
        16_000,
        20_800,
        20_800,
    ]
    assert all(activity["kind"] == "check" for activity in activities)
    assert all(activity["model_requests"] == 0 for activity in activities)


@pytest.mark.parametrize("provider", ["openai", "chat", "anthropic", "ollama"])
def test_provider_reduction_preserves_protocol_and_survives_resume(tmp_path, monkeypatch, provider):
    from anomx.agent.base.backends import (
        AnthropicStreamResponse,
        AnthropicToolCall,
        OllamaStreamResponse,
        OllamaToolCall,
        OpenAIChatCompletionStreamResponse,
        OpenAIStreamResponse,
        OpenAIToolCall,
    )

    prompts = []
    runtime, session = make_runtime(
        tmp_path,
        context_optimizer=lambda _, prompt: (
            prompts.append(json.loads(prompt)) or '[{"id":"keep","value":42}]'
        ),
    )
    raw = json.dumps(
        [{"id": "keep", "value": 42}, *[{"id": f"noise-{i}", "value": i} for i in range(300)]]
    )
    # Avoid immediate reduction; exercise the backend rebuild with stored result IDs.
    monkeypatch.setattr(runtime, "maximum_context_tokens", lambda: 256_000)
    monkeypatch.setattr(
        runtime, "_tool_for_call", lambda _: SimpleNamespace(execute=lambda *_: raw)
    )
    result = runtime._execute_tool(
        "read",
        {"path": "measurements.json"},
        RuntimeCallbacks(),
        session.path,
        tool_call_id="call-1",
    )
    assert isinstance(result, ContextToolResult) and not result.optimized
    assert deepcopy(result).message_id == result.message_id
    args = '{"path":"measurements.json"}'
    call = OpenAIToolCall("read", "call-1", args)
    if provider == "openai":
        backend = OpenAIBackend(runtime)
        response = OpenAIStreamResponse(
            "r1",
            "Review evidence.",
            (call,),
            reasoning=({"type": "reasoning", "id": "reason-1", "summary": []},),
        )
        entries = backend._openai_context_entries(
            response, [{"call_id": "call-1", "output": result}]
        )
        def wire(records):
            return backend._openai_messages([e.payload for e in records], "gpt-5.5")
    elif provider == "chat":
        backend = OpenAICompatibleChatBackend(runtime)
        native = {
            "role": "assistant",
            "content": "Review evidence.",
            "reasoning_content": "Find the relevant measurement",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read", "arguments": args},
                }
            ],
        }
        response = OpenAIChatCompletionStreamResponse("Review evidence.", (call,), native)
        entries = backend._chat_context_entries(
            response, [{"tool_call_id": "call-1", "content": result}]
        )
        def wire(records):
            return backend._chat_messages_from_entries(session.path, "coding", records)[
                    1:
                ]
    elif provider == "anthropic":
        backend = DesyAssistantBackend(runtime)
        native = (
            {"type": "thinking", "thinking": "Find the measurement", "signature": "signed"},
            {"type": "text", "text": "Review evidence."},
            {
                "type": "tool_use",
                "id": "call-1",
                "name": "read",
                "input": {"path": "measurements.json"},
            },
        )
        response = AnthropicStreamResponse(
            "Review evidence.",
            (AnthropicToolCall("read", "call-1", {"path": "measurements.json"}),),
            native,
        )
        entries = backend._anthropic_context_entries(
            response, [{"tool_use_id": "call-1", "content": result}]
        )
        def wire(records):
            return backend._anthropic_messages(
                    [e.payload for e in records], "anthropic", "claude"
                )
    else:
        backend = OllamaBackend(runtime)
        native = {
            "role": "assistant",
            "content": "Review evidence.",
            "thinking": "Find the measurement",
            "tool_calls": [
                {"function": {"name": "read", "arguments": {"path": "measurements.json"}}}
            ],
        }
        response = OllamaStreamResponse(
            "Review evidence.",
            "Find the measurement",
            (OllamaToolCall("read", {"path": "measurements.json"}),),
            native,
        )
        entries = backend._ollama_context_entries(
            response, [{"tool_name": "read", "content": result}]
        )
        def wire(records):
            return backend._ollama_messages([e.payload for e in records], "qwen")
    original_call = deepcopy(entries[0].payload)
    before = wire(entries)
    optimized, changed = runtime._optimize_tool_blocks(
        session.path, entries, maximum=32_000, current_context_tokens=20_000
    )
    assert changed and optimized[0].payload == original_call
    assert len(optimized) == 2
    assert optimized[1].message_id == result.message_id
    assert optimized[1].payload["tool_call_id"] == entries[1].payload["tool_call_id"]
    assert json.loads(optimized[1].payload["content"]) == [{"id": "keep", "value": 42}]
    after = wire(optimized)
    assert before[:-1] == after[:-1]  # Includes native reasoning and tool arguments.
    assert "[Tool call:" not in json.dumps(after)
    assert "context_kind" not in json.dumps(after)
    restored = AgentRuntime(runtime.home, tmp_path).backend_conversation_entries(session.path)
    assert [e.payload["role"] for e in restored] == ["user", "assistant", "tool"]
    assert restored[-1].payload["tool_call_id"] == "call-1"
    assert restored[-1].payload["content"] == optimized[1].payload["content"]
    assert runtime.conversation_messages(session.path)[-1]["content"] == raw
    assert prompts[0]["tool"] == "read"
    assert json.loads(prompts[0]["arguments"])["path"] == "measurements.json"


def test_legacy_merged_replacements_restore_original_native_tool_pairs(tmp_path):
    runtime, session = make_runtime(tmp_path)
    for i in range(2):
        runtime.home.append_session_event(
            session.path,
            "tool_execution",
            {
                "message_id": f"t{i}",
                "tool": "read",
                "arguments": {"path": str(i)},
                "result": f"original-{i}",
            },
        )
    runtime.home.append_session_event(
        session.path,
        "context_optimization",
        {
            "replacements": {"t0": "[Optimized tool evidence] unsafe digest", "t1": ""},
        },
    )
    entries = runtime.backend_conversation_entries(session.path)
    assert [e.payload["role"] for e in entries] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert [e.payload["content"] for e in entries if e.payload["role"] == "tool"] == [
        "original-0",
        "original-1",
    ]
