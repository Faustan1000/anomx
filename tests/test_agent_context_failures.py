import io
import json
import subprocess
import sys
from copy import deepcopy
from urllib.error import HTTPError

import pytest

from anomx.agent import AnomxHome
from anomx.agent.backends.blablador import BlabladorBackend
from anomx.agent.backends.desy_assistant import _DesyReasoningBackend
from anomx.agent.backends.ollama import OllamaBackend
from anomx.agent.backends.openai import OpenAIBackend
from anomx.agent.base.backends import (
    MAX_TOOL_ITERATIONS,
    OPENAI_MAX_TOOL_CALLS,
    OpenAIChatCompletionStreamResponse,
    OpenAIToolCall,
    TokenUsage,
)
from anomx.agent.exceptions import AgentBackendError, BackendFailure
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks


def desy_runtime(tmp_path, summarizer, history_count=48):
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
    for i in range(history_count):
        home.append_session_event(
            session.path,
            "agent_message",
            {
                "message": f"Earlier finding {i}. " * 100,
                "message_id": f"history-{i}",
            },
        )
    runtime = AgentRuntime(home, tmp_path, context_summarizer=summarizer)
    return runtime, session, _DesyReasoningBackend(runtime)


def tool_response(input_tokens=100):
    return OpenAIChatCompletionStreamResponse(
        text="",
        assistant_message={"role": "assistant", "tool_calls": [{
            "type": "function", "id": "call-1",
            "function": {"name": "read", "arguments": "{}"},
        }]},
        tool_calls=(OpenAIToolCall("read", "call-1", "{}"),),
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=100),
    )


def tool_output(response=None, *_args):
    if response is not None and not response.tool_calls:
        return []
    return [
        {"role": "tool", "tool_call_id": "call-1", "content": "Important measurement result"}
    ]


def test_desy_compresses_before_output_reservation_overflows_context(tmp_path, monkeypatch):
    summaries = []
    runtime, session, backend = desy_runtime(
        tmp_path, lambda system, user: summaries.append(user) or "Keep the measurement result."
    )
    responses = iter([
        tool_response(990_000),
        OpenAIChatCompletionStreamResponse("Done.", (), {"role": "assistant", "content": "Done."}),
    ])
    payloads = []
    monkeypatch.setattr(
        backend,
        "_stream_chat_completion",
        lambda key, payload, *_: payloads.append(deepcopy(payload)) or next(responses),
    )
    monkeypatch.setattr(backend, "_execute_chat_completion_tools", tool_output)

    assert backend.generate(session.path, "coding", RuntimeCallbacks()) == "Done."
    assert len(summaries) == 1
    assert "Earlier finding" in summaries[0]
    assert "Important measurement result" not in summaries[0]
    assert "Keep the measurement result." in payloads[1]["messages"][1]["content"]
    assert payloads[1]["messages"][-1]["content"].startswith("Continue the current task")
    assert payloads[0]["max_tokens"] == 32_768
    assert runtime.context_compression_state(session.path).last_message_id == "history-25"
    assert len(runtime.conversation_messages(session.path)) == 49


@pytest.mark.parametrize(
    "summary", [None, "", "unchanged context " * 500_000], ids=["none", "empty", "oversized"]
)
def test_failed_compression_does_not_send_another_oversized_request(tmp_path, monkeypatch, summary):
    runtime, session, backend = desy_runtime(tmp_path, lambda *_: summary)
    requests = []
    monkeypatch.setattr(
        backend, "_stream_chat_completion",
        lambda *_: requests.append(True) or tool_response(990_000),
    )
    monkeypatch.setattr(backend, "_execute_chat_completion_tools", tool_output)

    with pytest.raises(AgentBackendError) as error:
        backend.generate(session.path, "coding", RuntimeCallbacks())
    assert error.value.code == "context_compression_failed"
    assert len(requests) == 1
    assert runtime.context_compression_state(session.path) is None


def test_background_work_backend_resolves_with_env_var_only_api_key(tmp_path, monkeypatch):
    """A DESY key set only via the environment must still resolve a summary model.

    Reproduces "Context compression is required, but no summary model is
    available": chat requests succeed because they check the environment
    variable directly, but the background-work model lookup used for context
    compression used to only check the stored auth file.
    """

    home = AnomxHome(tmp_path / "home")
    home.save_config({**home.load_config(), "provider": "desy", "model": "coding"})
    monkeypatch.setenv("DESY_ASSISTANT_API_KEY", "sk-env-only")
    runtime = AgentRuntime(home, tmp_path)

    background = runtime._background_work_backend("background_medium_work_model")

    assert background is not None
    backend, model = background
    assert isinstance(backend, DesyAssistantBackend)
    assert model == "coding"


def test_desy_context_rejection_compresses_once_and_retries_without_repeating_tools(
    tmp_path, monkeypatch
):
    summaries = []
    runtime, session, backend = desy_runtime(
        tmp_path,
        lambda *args: summaries.append(args) or "Retain the completed measurement.",
        history_count=0,
    )
    runtime.context_optimizer = lambda *_: "Important measurement result"
    failure = backend._api_error(
        "desy",
        "DESY Assistant",
        "DESY_ASSISTANT_API_KEY",
        400,
        json.dumps(
            {"error": {"message": "This model's maximum context length is 1048576 tokens."}}
        ),
    )
    responses = iter([
        tool_response(), failure,
        OpenAIChatCompletionStreamResponse("Done.", (), {"role": "assistant", "content": "Done."}),
    ])
    payloads, tool_calls = [], []
    monkeypatch.setattr(
        backend,
        "_stream_chat_completion",
        lambda key, payload, *_: payloads.append(deepcopy(payload)) or next(responses),
    )
    monkeypatch.setattr(
        backend,
        "_execute_chat_completion_tools",
        lambda response, *args: (
            (
                tool_calls.append(True)
                or [
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "content": "Important measurement result\n" * 500,
                    }
                ]
            )
            if response.tool_calls
            else []
        ),
    )

    assert backend.generate(session.path, "coding", RuntimeCallbacks()) == "Done."
    assert len(tool_calls) == 1
    assert summaries == []
    assert payloads[2]["messages"][-2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "Important measurement result",
    }
    assert runtime.context_compression_state(session.path) is None


def test_repeated_context_rejection_is_a_terminal_failure(tmp_path, monkeypatch):
    _, session, backend = desy_runtime(tmp_path, lambda *_: "Retained history.")
    failure = BackendFailure("Context window exceeded", code="context_window_exceeded")
    requests = []
    monkeypatch.setattr(
        backend, "_stream_chat_completion", lambda *_: requests.append(True) or failure,
    )
    assert backend.generate(session.path, "coding", RuntimeCallbacks()) is failure
    assert len(requests) == 2


def test_tool_limit_allows_256_batches_and_returns_typed_failure(tmp_path, monkeypatch):
    _, session, backend = desy_runtime(tmp_path, lambda *_: "Retained history.")
    requests = []
    monkeypatch.setattr(
        backend, "_stream_chat_completion", lambda *_: requests.append(True) or tool_response()
    )
    monkeypatch.setattr(backend, "_execute_chat_completion_tools", tool_output)
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


def test_long_history_updates_memory_and_preserves_recent_tool_pairs(tmp_path):
    summaries = []
    runtime, session, _ = desy_runtime(
        tmp_path, lambda system, user: summaries.append(user) or f"Summary {len(summaries)}."
    )
    runtime._prepare_context_compression(session.path, RuntimeCallbacks())
    first = runtime.context_compression_state(session.path)
    for i in range(20):
        runtime.home.append_session_event(
            session.path,
            "tool_execution",
            {
                "message_id": f"tool-{i}",
                "tool": "read",
                "arguments": {"path": f"file-{i}"},
                "result": "Verified measurement. " * 20,
            },
        )
    runtime._prepare_context_compression(session.path, RuntimeCallbacks())
    state = runtime.context_compression_state(session.path)
    assert len(summaries) == 2
    assert "Previous summary:\nSummary 1." in summaries[1]
    assert state.summary == "Summary 2."
    assert state.compressed_message_count > first.compressed_message_count
    restored = AgentRuntime(runtime.home, tmp_path).backend_conversation_entries(session.path)
    assert restored[0].payload["role"] == "assistant"
    assert restored[-1].message_id == "tool-19"
    assert len(restored) == 24
    assert restored[0].payload["tool_calls"][0]["id"] == restored[1].payload["tool_call_id"]


def test_runtime_surfaces_empty_summary_as_failure(tmp_path, monkeypatch):
    runtime, session, _ = desy_runtime(tmp_path, lambda *_: None)
    monkeypatch.setattr(
        _DesyReasoningBackend, "_stream_chat_completion", lambda *_: tool_response(990_000),
    )
    monkeypatch.setattr(
        _DesyReasoningBackend,
        "_execute_chat_completion_tools",
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


def test_resuming_compressed_history_does_not_reinsert_summarized_tools(tmp_path):
    runtime, session, _ = desy_runtime(tmp_path, lambda *_: "Earlier work is complete.")
    for i in range(30):
        runtime.home.append_session_event(
            session.path,
            "tool_execution",
            {
                "message_id": f"tool-{i}",
                "tool": "read",
                "arguments": {},
                "result": "Measurement data",
            },
        )
    compacted, compressed = runtime.compress_in_turn_context(
        session.path,
        runtime.backend_conversation_entries(session.path),
        current_context_tokens=990_000,
        status_callback=None,
    )
    assert compressed and len(compacted) == 24
    assert runtime.context_compression_state(session.path).last_message_id == "tool-17"
    resumed = AgentRuntime(runtime.home, tmp_path)
    assert resumed.backend_conversation_entries(session.path) == compacted
    assert len(resumed.conversation_messages(session.path)) == 109
    assert (
        "Earlier work is complete." in resumed.context_system_messages(session.path)[1]["content"]
    )
