import io
import json
import subprocess
import sys
from urllib.error import HTTPError

import pytest

from anomx.agent import AnomxHome
from anomx.agent.backends.blablador import BlabladorBackend
from anomx.agent.backends.desy_assistant import DesyAssistantBackend
from anomx.agent.backends.ollama import OllamaBackend
from anomx.agent.backends.openai import OpenAIBackend
from anomx.agent.base.backends import (
    MAX_TOOL_ITERATIONS,
    OPENAI_MAX_TOOL_CALLS,
    AnthropicStreamResponse,
    AnthropicToolCall,
    TokenUsage,
)
from anomx.agent.context_management import transient_context_message
from anomx.agent.exceptions import AgentBackendError, BackendFailure
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


def desy_runtime(tmp_path, summarizer):
    home = AnomxHome(tmp_path / "home")
    home.save_config(
        {
            **home.load_config(),
            "provider": "desy",
            "model": "coding",
            "maximum_context_tokens": 1_000_000,
        }
    )
    home.set_api_key("desy", "test-key")
    session = home.create_session(tmp_path, provider="desy", model="coding")
    home.append_session_event(
        session.path,
        "user_message",
        {"message": "Inspect this data. " * 500, "message_id": "user-1"},
    )
    runtime = AgentRuntime(home, tmp_path, context_summarizer=summarizer)
    return runtime, session, DesyAssistantBackend(runtime)


def tool_response(input_tokens=100):
    return AnthropicStreamResponse(
        text="",
        content=({"type": "tool_use", "id": "call-1", "name": "read", "input": {}},),
        tool_calls=(AnthropicToolCall("read", "call-1", {}),),
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=100),
    )


def tool_output(response=None, *_args):
    if response is not None and not response.tool_calls:
        return []
    return [
        {"type": "tool_result", "tool_use_id": "call-1", "content": "Important measurement result"}
    ]


def test_desy_compresses_before_output_reservation_overflows_context(tmp_path, monkeypatch):
    summaries = []
    runtime, session, backend = desy_runtime(
        tmp_path, lambda system, user: summaries.append(user) or "Keep the measurement result."
    )
    responses = iter([tool_response(990_000), AnthropicStreamResponse("Done.", (), ())])
    payloads = []
    monkeypatch.setattr(
        backend,
        "_stream_response",
        lambda key, payload, *_: payloads.append(payload) or next(responses),
    )
    monkeypatch.setattr(backend, "_execute_anthropic_requested_tools", tool_output)

    assert backend.generate(session.path, "coding", RuntimeCallbacks()) == "Done."
    assert len(summaries) == 1
    assert "Important measurement result" in summaries[0]
    assert "Keep the measurement result." in payloads[1]["system"]
    assert payloads[1]["messages"][-1]["content"][0]["text"].startswith("Continue the current task")
    assert payloads[0]["max_tokens"] == 32_768
    assert runtime.context_compression_state(session.path).last_message_id == "user-1"
    assert len(runtime.conversation_messages(session.path)) == 1


@pytest.mark.parametrize(
    "summary", [None, "", "unchanged context " * 500_000], ids=["none", "empty", "oversized"]
)
def test_failed_compression_does_not_send_another_oversized_request(tmp_path, monkeypatch, summary):
    runtime, session, backend = desy_runtime(tmp_path, lambda *_: summary)
    requests = []
    monkeypatch.setattr(
        backend, "_stream_response", lambda *_: requests.append(True) or tool_response(990_000)
    )
    monkeypatch.setattr(backend, "_execute_anthropic_requested_tools", tool_output)

    with pytest.raises(AgentBackendError) as error:
        backend.generate(session.path, "coding", RuntimeCallbacks())
    assert error.value.code == "context_compression_failed"
    assert len(requests) == 1
    assert runtime.context_compression_state(session.path) is None


def test_desy_context_rejection_compresses_once_and_retries_without_repeating_tools(
    tmp_path, monkeypatch
):
    summaries = []
    runtime, session, backend = desy_runtime(
        tmp_path, lambda *args: summaries.append(args) or "Retain the completed measurement."
    )
    failure = backend._api_error(
        "desy",
        "DESY Assistant",
        "DESY_ASSISTANT_API_KEY",
        400,
        json.dumps(
            {"error": {"message": "This model's maximum context length is 1048576 tokens."}}
        ),
    )
    responses = iter([tool_response(), failure, AnthropicStreamResponse("Done.", (), ())])
    payloads, tool_calls = [], []
    monkeypatch.setattr(
        backend,
        "_stream_response",
        lambda key, payload, *_: payloads.append(payload) or next(responses),
    )
    monkeypatch.setattr(
        backend,
        "_execute_anthropic_requested_tools",
        lambda response, *args: (
            (tool_calls.append(True) or tool_output(response)) if response.tool_calls else []
        ),
    )

    assert backend.generate(session.path, "coding", RuntimeCallbacks()) == "Done."
    assert len(tool_calls) == 1
    assert len(summaries) == 1
    assert "Retain the completed measurement." in payloads[2]["system"]
    assert runtime.context_compression_state(session.path) is not None


def test_repeated_context_rejection_is_a_terminal_failure(tmp_path, monkeypatch):
    _, session, backend = desy_runtime(tmp_path, lambda *_: "Retained history.")
    failure = BackendFailure("Context window exceeded", code="context_window_exceeded")
    requests = []
    monkeypatch.setattr(backend, "_stream_response", lambda *_: requests.append(True) or failure)
    assert backend.generate(session.path, "coding", RuntimeCallbacks()) is failure
    assert len(requests) == 2


def test_tool_limit_allows_256_batches_and_returns_typed_failure(tmp_path, monkeypatch):
    _, session, backend = desy_runtime(tmp_path, lambda *_: "Retained history.")
    requests = []
    monkeypatch.setattr(
        backend, "_stream_response", lambda *_: requests.append(True) or tool_response()
    )
    monkeypatch.setattr(backend, "_execute_anthropic_requested_tools", tool_output)
    result = backend.generate(session.path, "coding", RuntimeCallbacks())
    assert len(requests) == MAX_TOOL_ITERATIONS == OPENAI_MAX_TOOL_CALLS == 256
    assert isinstance(result, BackendFailure)
    assert result.code == "tool_limit_exceeded"
    with pytest.raises(AgentBackendError, match="256 tool batches"):
        result.raise_for_status()


def test_http_failure_survives_runtime_as_structured_error(tmp_path, monkeypatch):
    runtime, session, _ = desy_runtime(tmp_path, lambda *_: "Retained history.")

    def reject(*_args, **_kwargs):
        raise HTTPError(
            "https://example.invalid",
            400,
            "Bad Request",
            None,
            io.BytesIO(b'{"error":{"message":"Invalid model"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", reject)
    result = runtime.backend_response(session.path)
    assert isinstance(result, BackendFailure)
    assert result.code == "model_request_failed"
    assert "Invalid model" in result


def test_long_tool_loop_compresses_repeatedly_and_preserves_rolling_summary(tmp_path, monkeypatch):
    summaries = []
    runtime, session, backend = desy_runtime(
        tmp_path, lambda system, user: summaries.append(user) or f"Summary {len(summaries)}."
    )
    responses = iter(
        [
            tool_response(990_000),
            tool_response(990_000),
            AnthropicStreamResponse("Done.", (), ()),
        ]
    )
    monkeypatch.setattr(backend, "_stream_response", lambda *_: next(responses))
    monkeypatch.setattr(backend, "_execute_anthropic_requested_tools", tool_output)

    assert backend.generate(session.path, "coding", RuntimeCallbacks()) == "Done."
    assert len(summaries) == 2
    assert "Previous summary:\nSummary 1." in summaries[1]
    assert runtime.context_compression_state(session.path).summary == "Summary 2."
    assert len(runtime.conversation_messages(session.path)) == 1


def test_runtime_surfaces_empty_summary_as_failure(tmp_path, monkeypatch):
    runtime, session, _ = desy_runtime(tmp_path, lambda *_: None)
    monkeypatch.setattr(DesyAssistantBackend, "_stream_response", lambda *_: tool_response(990_000))
    monkeypatch.setattr(
        DesyAssistantBackend,
        "_execute_anthropic_requested_tools",
        lambda self, *args: tool_output(*args),
    )

    result = runtime.backend_response(session.path)
    assert isinstance(result, BackendFailure)
    assert result.code == "context_compression_failed"


@pytest.mark.parametrize(
    "backend_type,event",
    [
        (
            OpenAIBackend,
            {"type": "response.failed", "response": {"error": {"message": "Rejected"}}},
        ),
        (
            OpenAIBackend,
            {
                "type": "response.incomplete",
                "response": {"incomplete_details": {"reason": "max_output_tokens"}},
            },
        ),
        (BlabladorBackend, {"error": {"message": "Rejected"}}),
        (OllamaBackend, {"error": "Rejected"}),
    ],
)
def test_provider_stream_errors_are_not_successful_answers(
    tmp_path, monkeypatch, backend_type, event
):
    runtime, session, _ = desy_runtime(tmp_path, lambda *_: "Retained history.")
    backend = backend_type(runtime)
    runtime.home.set_api_key(backend.provider_key, "test-key")
    line = json.dumps(event) if backend_type is OllamaBackend else "data: " + json.dumps(event)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO((line + "\n").encode())
    )

    result = backend.generate(session.path, "coding", RuntimeCallbacks())
    assert isinstance(result, BackendFailure)


def test_backend_errors_can_be_imported_before_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from anomx.agent.exceptions import AgentBackendError, BackendFailure; "
            "from anomx.agent.base import BaseTool, BaseAgent",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_resuming_compressed_tool_loop_does_not_reinsert_saved_tool_results(tmp_path):
    runtime, session, _ = desy_runtime(tmp_path, lambda *_: "The measurement is complete.")
    entries = runtime.backend_conversation_entries(session.path)
    runtime.home.append_session_event(
        session.path,
        "tool_execution",
        {
            "message_id": "tool-1",
            "tool": "read",
            "result": "Measurement data",
        },
    )
    entries.append(transient_context_message("user", "Measurement data"))
    compacted, compressed = runtime.compress_in_turn_context(
        session.path,
        entries,
        current_context_tokens=990_000,
        status_callback=None,
    )

    assert compressed and compacted == []
    assert runtime.context_compression_state(session.path).last_message_id == "tool-1"
    resumed = AgentRuntime(runtime.home, tmp_path)
    assert resumed.backend_conversation_entries(session.path) == []
    assert len(resumed.conversation_messages(session.path)) == 2
    assert "The measurement is complete." in resumed._instructions(session.path)
