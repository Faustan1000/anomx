import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anomx.agent import AnomxHome
from anomx.agent.app import AnomxCliApp
from anomx.agent.context_management import (
    ContextMessage,
    adaptive_context_target,
    context_evaluation_band,
    effective_context_limit,
    history_compression_prefix,
    should_optimize_tool_result,
)
from anomx.agent.exceptions import AgentBackendError
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


def runtime_session(tmp_path, **kwargs):
    home = AnomxHome(tmp_path / "home")
    home.save_config({**home.load_config(), "maximum_context_tokens": 32_000})
    session = home.create_session(tmp_path, provider="openai", model="gpt-5.5")
    runtime = AgentRuntime(home, tmp_path, **kwargs)
    return runtime, session


def append(runtime, session, kind, content, message_id):
    runtime.home.append_session_event(
        session.path,
        kind,
        {
            **(
                {"result": content, "tool": "read_file"}
                if kind == "tool_execution"
                else {"message": content}
            ),
            "message_id": message_id,
        },
    )


def test_legacy_percentage_is_discarded(tmp_path):
    runtime, _ = runtime_session(tmp_path)
    runtime.home.save_config(
        {**runtime.home.load_config(), "context_compression_target_percent": 75}
    )
    assert "context_compression_target_percent" not in runtime.home.load_config()
    assert "context_compression_target_percent" not in runtime.home.config_path.read_text()


def test_limit_and_adaptive_headroom():
    assert effective_context_limit(32_000, "unknown") == 32_000
    assert effective_context_limit(1_000_000, "alias-code") == 262_144 - 32_768
    assert effective_context_limit(32_000, "alias-code") == 32_000
    small = [ContextMessage("", {"role": "user", "content": "hello"})]
    large = [ContextMessage("", {"role": "user", "content": "data" * 10_000})]
    assert adaptive_context_target(32_000, small) == 20_800
    assert adaptive_context_target(32_000, large) == 16_000


@pytest.mark.parametrize(
    "percent,band", [(49, 0), (50, 50), (65, 65), (80, 80), (90, 90), (99, 99), (110, 99)]
)
def test_evaluation_thresholds(percent, band):
    assert context_evaluation_band(percent * 320, 32_000) == band


def test_history_prefix_does_not_split_parallel_tool_results():
    from anomx.agent.context_management import tool_call_payload

    entries = [
        ContextMessage("u", {"role": "user", "content": "Keep x"}),
        ContextMessage("", {"role": "assistant", "tool_calls": [
            tool_call_payload("read", "c1", {}), tool_call_payload("read", "c2", {}),
        ]}),
        ContextMessage("t1", {"role": "tool", "tool_call_id": "c1", "content": "x"}),
        ContextMessage("t2", {"role": "tool", "tool_call_id": "c2", "content": "y"}),
    ]
    assert history_compression_prefix(entries, 3) == entries[:1]
    assert history_compression_prefix(entries, 4) == entries


def test_followup_optimizes_tool_blocks_and_restores_them_without_losing_transcript(tmp_path):
    calls = []
    runtime, session = runtime_session(
        tmp_path,
        context_optimizer=lambda system, user: (
            calls.append((system, user)) or "Measurement x=3 at /data/result.csv; read completed."
        ),
    )
    append(runtime, session, "user_message", "Inspect the data", "u1")
    append(runtime, session, "tool_execution", "data " * 2_000, "t1")
    append(runtime, session, "tool_execution", "other data " * 1_000, "t2")
    append(runtime, session, "agent_message", "I found a measurement", "a1")
    append(runtime, session, "user_message", "Explain it", "u2")
    full_transcript = runtime.conversation_messages(session.path)
    activities = []
    runtime._prepare_context_compression(
        session.path, RuntimeCallbacks(context_activity=activities.append)
    )
    assert len(calls) == 2
    assert "untrusted data" in calls[0][0]
    assert runtime.conversation_messages(session.path) == full_transcript
    restored = AgentRuntime(runtime.home, tmp_path).backend_conversation_entries(session.path)
    assert [entry.message_id for entry in restored] == ["u1", "", "t1", "", "t2", "a1", "u2"]
    assert "/data/result.csv" in restored[2].payload["content"]
    assert [(activity["kind"], activity["status"]) for activity in activities] == [
        ("check", "running"),
        ("optimization", "running"),
        ("optimization", "running"),
        ("optimization", "completed"),
    ]
    assert activities[-1]["changed"]
    assert runtime.context_compression_state(session.path) is None


@pytest.mark.parametrize("answer", ["KEEP", "", "data " * 10_000], ids=["keep", "empty", "larger"])
def test_optimizer_never_replaces_with_empty_or_larger_output(tmp_path, answer):
    runtime, session = runtime_session(tmp_path, context_optimizer=lambda *_: answer)
    append(runtime, session, "tool_execution", "data " * 14_000, "t1")
    before = runtime.backend_conversation_entries(session.path)
    runtime._prepare_context_compression(session.path, RuntimeCallbacks())
    after = runtime.backend_conversation_entries(session.path)
    path = Path(json.loads(after[-1].payload["content"])["result_path"])
    assert path.read_text() == before[-1].payload["content"]


def test_large_tool_result_uses_file_and_keeps_original_evidence(tmp_path, monkeypatch):
    runtime, session = runtime_session(tmp_path)
    selection = []

    class Backend:
        def summarize_conversation(self, messages, previous, model, *, system_prompt):
            assert model == "easy-model"
            assert "Preserve the original format and structure" in system_prompt
            return "Read succeeded. x=3, file /data/result.csv."

    monkeypatch.setattr(
        runtime,
        "_background_work_backend",
        lambda key: selection.append(key) or (Backend(), "easy-model"),
    )
    result = "measurement " * 6_000
    monkeypatch.setattr(
        runtime, "_tool_for_call", lambda _: SimpleNamespace(execute=lambda *_: result)
    )
    optimized = runtime._execute_tool("read_file", {}, RuntimeCallbacks(), session.path)
    assert Path(json.loads(optimized)["result_path"]).read_text() == result
    assert optimized.optimized
    assert selection == []
    assert result in runtime.conversation_messages(session.path)[1]["content"]
    assert result not in runtime.backend_conversation_messages(session.path)[1]["content"]


def test_summary_uses_easy_model(tmp_path, monkeypatch):
    runtime, session = runtime_session(tmp_path)
    selections = []

    class Backend:
        def summarize_conversation(self, *args, system_prompt=None):
            assert "at most" in system_prompt
            return "Goals and measurements retained."

    monkeypatch.setattr(
        runtime,
        "_background_work_backend",
        lambda key: selections.append(key) or (Backend(), "easy-model"),
    )
    for i in range(50):
        append(runtime, session, "user_message", "Inspect measurements " * 80, f"u{i}")
    _, changed = runtime.compress_in_turn_context(
        session.path,
        runtime.backend_conversation_entries(session.path),
        current_context_tokens=26_000,
        status_callback=None,
    )
    assert changed
    assert selections == ["background_easy_work_model"]


def test_token_pressure_alone_never_summarizes_short_history(tmp_path):
    calls = []
    runtime, session = runtime_session(
        tmp_path, context_summarizer=lambda *_: calls.append(True) or None
    )
    append(runtime, session, "user_message", "Inspect all measurements", "u1")
    entries = runtime.backend_conversation_entries(session.path)
    for count in (25_600, 26_000, 28_800):
        result, changed = runtime.compress_in_turn_context(
            session.path, entries, current_context_tokens=count, status_callback=None
        )
        assert not changed and result == entries
    assert calls == []
    with pytest.raises(AgentBackendError, match="cannot fit safely"):
        runtime.compress_in_turn_context(
            session.path, entries, current_context_tokens=31_680, status_callback=None
        )
    assert runtime.context_compression_state(session.path) is None


def test_many_old_messages_trigger_compression_below_eighty_percent(tmp_path):
    runtime, session = runtime_session(
        tmp_path, context_summarizer=lambda *_: "Earlier constraints retained."
    )
    for index in range(50):
        append(
            runtime,
            session,
            "user_message" if index % 2 == 0 else "agent_message",
            "context " * 150,
            str(index),
        )
    tokens = runtime.estimate_session_context_tokens(session.path)
    assert 16_000 <= tokens < 25_600
    runtime._prepare_context_compression(session.path, RuntimeCallbacks())
    assert runtime.context_compression_state(session.path) is not None
    assert runtime.backend_conversation_entries(session.path)[-1].message_id == "49"
    assert len(runtime.conversation_messages(session.path)) == 50


def test_cli_context_activity_splits_tool_blocks_and_updates_in_place(tmp_path):
    runtime, session = runtime_session(tmp_path)
    for kind, payload in [
        ("work_message", {"message": "Read file", "role": "tool", "turn_id": "turn"}),
        ("context_activity", {"id": "opt", "kind": "optimization", "status": "running"}),
        (
            "context_activity",
            {
                "id": "opt",
                "kind": "optimization",
                "status": "completed",
                "changed": True,
                "tokens_before": 8_000,
                "tokens_after": 2_000,
            },
        ),
        ("work_message", {"message": "Continue analysis", "role": "tool", "turn_id": "turn"}),
    ]:
        runtime.home.append_session_event(session.path, kind, payload)
    app = AnomxCliApp(home=runtime.home, isolate_runtime=True)
    lines = app._read_message_lines(session.path)
    assert [line.role for line in lines] == ["work_summary", "context", "work_active"]
    assert "8,000 → 2,000" in lines[1].text


def test_in_turn_optimization_resets_openai_chain_without_summarizing_history(
    tmp_path, monkeypatch
):
    from anomx.agent.backends.openai import OpenAIBackend
    from anomx.agent.base.backends import OpenAIStreamResponse, OpenAIToolCall, TokenUsage

    def no_summary(*_):
        raise AssertionError("Tool optimization should leave history intact")

    runtime, session = runtime_session(
        tmp_path,
        context_optimizer=lambda *_: "Measurement x=3; completed read of /data/result.csv.",
        context_summarizer=no_summary,
    )
    runtime.home.set_api_key("openai", "test-key")
    append(runtime, session, "user_message", "Find x and keep the evidence.", "u1")
    responses = iter(
        [
            OpenAIStreamResponse(
                "r1", "", (OpenAIToolCall("read", "c1", "{}"),), TokenUsage(input_tokens=16_000)
            ),
            OpenAIStreamResponse("r2", "x=3", (), TokenUsage(input_tokens=2_000)),
        ]
    )
    payloads = []
    backend = OpenAIBackend(runtime)
    monkeypatch.setattr(
        backend,
        "_stream_openai_response",
        lambda key, payload, *_: payloads.append(payload) or next(responses),
    )
    monkeypatch.setattr(
        backend,
        "_execute_requested_tools",
        lambda response, *_: (
            [{"type": "function_call_output", "call_id": "c1", "output": "measurement " * 1_000}]
            if response.tool_calls
            else []
        ),
    )
    assert backend.generate(session.path, "gpt-5.5", RuntimeCallbacks()) == "x=3"
    assert "previous_response_id" not in payloads[1]
    assert any("Measurement x=3" in item.get("output", "") for item in payloads[1]["input"])
    assert any(
        "Find x and keep the evidence." in item.get("content", "") for item in payloads[1]["input"]
    )
    assert runtime.context_compression_state(session.path) is None


def test_cli_context_capacity_uses_configured_maximum_and_invalidates_on_change(tmp_path):
    runtime, session = runtime_session(tmp_path)
    append(runtime, session, "user_message", "Inspect the data", "u1")
    app = AnomxCliApp(home=runtime.home, isolate_runtime=True)
    assert "/32k" in app._context_status(session, "gpt-5.5")
    runtime.home.save_config({**runtime.home.load_config(), "maximum_context_tokens": 64_000})
    assert "/64k" in app._context_status(session, "gpt-5.5")


@pytest.mark.parametrize("changed", [True, False])
def test_cli_activity_uses_callback_order_when_runtime_writes_ahead(tmp_path, changed):
    runtime, session = runtime_session(tmp_path)
    activity = {"id": "opt", "kind": "optimization", "status": "running"}
    runtime.home.append_session_event(session.path, "context_activity", activity)
    runtime.home.append_session_event(
        session.path,
        "work_message",
        {
            "message": "Read file",
            "role": "tool",
            "turn_id": "turn",
        },
    )
    runtime.home.append_session_event(session.path, "context_activity_display", activity)
    runtime.home.append_session_event(
        session.path,
        "context_activity",
        {
            **activity,
            "status": "completed",
            "changed": changed,
        },
    )
    runtime.home.append_session_event(
        session.path,
        "context_activity_display",
        {
            **activity,
            "status": "completed",
            "changed": changed,
        },
    )
    runtime.home.append_session_event(
        session.path,
        "work_message",
        {
            "message": "Continue analysis",
            "role": "tool",
            "turn_id": "turn",
        },
    )
    app = AnomxCliApp(home=runtime.home, isolate_runtime=True)
    lines = app._read_message_lines(session.path)
    assert [line.role for line in lines] == (
        ["work_summary", "context", "work_active"] if changed else ["work_active"]
    )
    if not changed:
        app._expanded_work_turns.add("turn")
        expanded = app._read_message_lines(session.path)
        assert [line.text for line in expanded if line.role == "tool"] == [
            "Read file",
            "Continue analysis",
        ]


def test_cli_check_before_first_tool_joins_same_work_block(tmp_path):
    runtime, session = runtime_session(tmp_path)
    activity = {"id": "check", "kind": "check", "status": "completed", "model_requests": 0}
    runtime.home.append_session_event(session.path, "context_activity", activity)
    runtime.home.append_session_event(
        session.path, "context_activity_display", {**activity, "turn_id": "turn"},
    )
    runtime.home.append_session_event(
        session.path, "work_message",
        {"message": "Read file", "role": "tool", "turn_id": "turn"},
    )
    app = AnomxCliApp(home=runtime.home, isolate_runtime=True)
    assert [line.role for line in app._read_message_lines(session.path)] == ["work_active"]
    app._expanded_work_turns.add("turn")
    expanded = app._read_message_lines(session.path)
    assert [line.text for line in expanded if line.role == "tool"] == ["Read file"]


@pytest.mark.parametrize("context_tokens", [0, 18_000, 60_000])
def test_first_turn_externalizes_large_results_even_when_context_has_room(
    tmp_path, monkeypatch, context_tokens
):
    from anomx.agent.base.backends import TokenUsage, UsageSnapshot

    calls = []
    runtime, session = runtime_session(
        tmp_path, context_optimizer=lambda *_: calls.append(True) or "Digest"
    )
    runtime.home.save_config({**runtime.home.load_config(), "maximum_context_tokens": 256_000})
    usage = TokenUsage(input_tokens=context_tokens, output_tokens=100)
    runtime.last_usage_snapshot = UsageSnapshot(
        total=usage, context_tokens=context_tokens, latest=usage
    )
    append(runtime, session, "user_message", "Show the detuning.", "u1")
    result = "measurement " * 4_800  # About 14k tokens, like the reported chat.
    monkeypatch.setattr(
        runtime, "_tool_for_call", lambda _: SimpleNamespace(execute=lambda *_: result)
    )
    for _ in range(3):
        returned = runtime._execute_tool("read_file", {}, RuntimeCallbacks(), session.path)
        assert Path(json.loads(returned)["result_path"]).read_text() == result
    assert calls == []
    assert not any(
        (event.get("payload") or {}).get("type") == "context_activity"
        for event in runtime.home.read_session_events(session.path)
    )


def test_followup_and_message_count_checkpoints_do_not_optimize_small_context(tmp_path):
    calls = []
    runtime, session = runtime_session(
        tmp_path, context_optimizer=lambda *_: calls.append(True) or "Digest"
    )
    runtime.home.save_config({**runtime.home.load_config(), "maximum_context_tokens": 256_000})
    append(runtime, session, "user_message", "Show the detuning.", "u1")
    for index in range(25):
        append(runtime, session, "tool_execution", "measurement " * 100, f"t{index}")
    append(runtime, session, "user_message", "Explain it.", "u2")
    runtime._prepare_context_compression(session.path, RuntimeCallbacks())
    entries = runtime.backend_conversation_entries(session.path)
    assert runtime.compress_in_turn_context(
        session.path, entries, current_context_tokens=20_000, status_callback=None
    ) == (entries, False)
    assert calls == []
    assert runtime.context_compression_state(session.path) is None


def test_pending_tool_batch_counts_toward_pressure_and_resets_with_usage(tmp_path, monkeypatch):
    from anomx.agent.base.backends import TokenUsage, UsageSnapshot

    calls = []
    runtime, session = runtime_session(
        tmp_path,
        context_optimizer=lambda *_: calls.append(True) or "Verified measurements retained.",
    )
    runtime.home.save_config({**runtime.home.load_config(), "maximum_context_tokens": 256_000})
    callbacks = runtime._with_usage_tracking(RuntimeCallbacks())
    usage = TokenUsage(input_tokens=90_000, output_tokens=100)
    callbacks.usage(UsageSnapshot(total=usage, context_tokens=90_000, latest=usage))
    result = "measurement " * 4_800
    monkeypatch.setattr(
        runtime, "_tool_for_call", lambda _: SimpleNamespace(execute=lambda *_: result)
    )
    for _ in range(3):
        returned = runtime._execute_tool("read_file", {}, callbacks, session.path)
        assert returned.optimized
        assert Path(json.loads(returned)["result_path"]).read_text() == result
    assert calls == []
    assert 0 < runtime._pending_tool_context_tokens < 6000
    callbacks.usage(UsageSnapshot(total=usage, context_tokens=20_000, latest=usage))
    assert runtime._pending_tool_context_tokens == 0


def test_huge_first_result_still_gets_reduced_when_it_consumes_half_the_budget():
    assert should_optimize_tool_result(148_596, 168_000, 256_000)
    assert not should_optimize_tool_result(14_446, 34_000, 256_000)


def test_final_output_never_starts_another_optimization(tmp_path, monkeypatch):
    calls = []
    runtime, session = runtime_session(
        tmp_path, context_optimizer=lambda *_: calls.append(True) or "Digest"
    )
    result = "measurement " * 10_000

    def finish(*_):
        runtime.produced_output = "Delivered the chart"
        return result

    monkeypatch.setattr(runtime, "_tool_for_call", lambda _: SimpleNamespace(execute=finish))
    assert runtime._execute_tool("produce_output", {}, RuntimeCallbacks(), session.path) == result
    assert calls == []


@pytest.mark.parametrize(
    "candidate",
    [
        '{"summary":"id=1"}',
        "Selected object: id=1",
        '```json\n[{"id":1}]\n```',
        '[{"id":2,"value":42}]',
        '[{"id":1,"value":43}]',
        '[{"id":1}]',
        '[{"id":1,"value":"42"}]',
        '[{"id":1,"value":42,"extra":true}]',
    ],
)
def test_json_optimizer_rejects_changed_format_or_evidence(tmp_path, candidate):
    import json

    runtime, _ = runtime_session(tmp_path, context_optimizer=lambda *_: candidate)
    raw = json.dumps([{"id": 1, "value": 42}, *[{"id": 3, "value": n} for n in range(1000)]])
    entry = ContextMessage(
        "t1", {"role": "tool", "content": raw, "name": "search", "arguments": "{}"}
    )
    assert runtime._optimize_context_data([entry], target_tokens=1000) is None


def test_large_nested_json_is_batched_as_valid_envelopes_and_reassembled(tmp_path):
    import json

    calls = []

    def reduce(_, user):
        request = json.loads(user)
        result = json.loads(request["result"])
        calls.append(result)
        result["data"]["items"] = result["data"]["items"][:1]
        return json.dumps(result)

    runtime, _ = runtime_session(
        tmp_path, context_optimizer=reduce, context_optimizer_context_window=10_000
    )
    source = {
        "status": 200,
        "total": 200,
        "next": "/items?offset=200",
        "data": {"items": [{"id": i, "text": "a" * 300} for i in range(200)]},
    }
    entry = ContextMessage(
        "t1", {"role": "tool", "name": "search", "arguments": "{}", "content": json.dumps(source)}
    )
    result = runtime._optimize_context_data([entry], target_tokens=4000)
    assert result is not None and len(calls) > 1
    reduced = json.loads(result)
    assert (
        reduced["status"] == 200 and reduced["total"] == 200 and reduced["next"] == source["next"]
    )
    assert [item["id"] for item in reduced["data"]["items"]] == [
        batch["data"]["items"][0]["id"] for batch in calls
    ]


def test_optimizer_gets_original_request_followup_outline_and_call_arguments(tmp_path):
    import json

    calls = []
    runtime, session = runtime_session(
        tmp_path,
        context_optimizer=lambda _, user: calls.append(json.loads(user)) or "Relevant line",
    )
    runtime.home.save_config({**runtime.home.load_config(), "maximum_context_tokens": 256_000})
    append(runtime, session, "user_message", "Investigate the gun temperature", "u1")
    runtime.home.append_session_event(
        session.path,
        "tool_execution",
        {
            "message_id": "t1",
            "tool": "search",
            "arguments": {"query": "gun", "limit": 100},
            "result": "Relevant line\n" * 800,
        },
    )
    runtime.home.append_session_event(session.path, "work_message", {
        "role": "thought", "command": "The gun temperature is relevant",
    })
    append(runtime, session, "agent_message", "I found the gun channel", "a1")
    append(runtime, session, "user_message", "Focus on yesterday", "u2")
    runtime._prepare_context_compression(session.path, RuntimeCallbacks())
    assert len(calls) == 1
    request = calls[0]
    context = json.loads(request["task_context"])
    assert context["original_user_request"] == "Investigate the gun temperature"
    assert "Focus on yesterday" in context["recent_user_requests"]
    assert "Completed tools: search" in context["recent_work_outline"]
    assert context["recent_thoughts"] == ["The gun temperature is relevant"]
    assert json.loads(request["arguments"]) == {"query": "gun", "limit": 100}
    assert runtime.context_compression_state(session.path) is None


def test_huge_result_triggers_below_half_window():
    assert should_optimize_tool_result(40_000, 45_000, 256_000)


def test_long_identifiers_and_urls_cannot_be_shortened():
    import json

    from anomx.agent.context_management import valid_tool_reduction

    value = "https://example.test/" + "a" * 400
    assert not valid_tool_reduction(json.dumps({"url": value}), json.dumps({"url": value[:100]}))
    assert valid_tool_reduction(json.dumps({"text": value}), json.dumps({"text": value[:100]}))


def test_small_history_overflow_preserves_history_and_stops_without_summary(tmp_path):
    calls, activities = [], []
    runtime, session = runtime_session(tmp_path, context_summarizer=lambda *_: calls.append(True))
    runtime._context_activity_callback = activities.append
    append(runtime, session, "user_message", "Preserve this request", "u1")
    entries = runtime.backend_conversation_entries(session.path)
    with pytest.raises(AgentBackendError):
        runtime.compress_in_turn_context(
            session.path, entries, current_context_tokens=32_000, status_callback=None
        )
    assert calls == []
    assert runtime.backend_conversation_entries(session.path) == entries
    assert activities[-1]["status"] == "failed"
    assert activities[0]["id"] == activities[-1]["id"]
