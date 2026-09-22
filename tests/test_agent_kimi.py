import json

from anomx.agent import AnomxHome
from anomx.agent.backends import backend_for_provider
from anomx.agent.backends.kimi import KimiBackend
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AI_PROVIDER_KEYS, provider_by_key


def test_kimi_is_the_last_cli_backend_and_is_registered(tmp_path):
    provider = provider_by_key("kimi")
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)

    assert provider is not None
    assert AI_PROVIDER_KEYS[-1] == "kimi"
    assert provider.label == "Kimi"
    assert provider.models == ("kimi-k3", "kimi-k2.7-code", "kimi-k2.6")
    assert isinstance(backend_for_provider("kimi", runtime), KimiBackend)


def test_kimi_model_discovery_uses_moonshot_api(monkeypatch):
    import anomx.agent.model_catalog as model_catalog

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"data": [{"id": "kimi-k3"}]}).encode("utf-8")

    captured = {}

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        captured["url"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr(model_catalog.urllib.request, "urlopen", fake_urlopen)

    assert model_catalog.discover_provider_models("kimi", "kimi-key") == ("kimi-k3",)
    assert captured == {
        "authorization": "Bearer kimi-key",
        "timeout": model_catalog.MODEL_DISCOVERY_TIMEOUT_SECONDS,
        "url": "https://api.moonshot.ai/v1/models",
    }


def test_kimi_stream_preserves_reasoning_across_tool_calls(tmp_path, monkeypatch):
    import anomx.agent.backends.openai_chat as openai_chat_module

    class FakeResponse:
        def __init__(self, lines):
            self.lines = lines

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(self.lines)

    responses = iter(
        (
            [
                b'data: {"choices":[{"delta":{"reasoning_content":"Inspect the workspace."}}]}\n',
                (
                    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
                    b'"function":{"name":"list","arguments":"{\\"path\\":\\".\\"}"}}]}}]}\n'
                ),
                b"data: [DONE]\n",
            ],
            [
                b'data: {"choices":[{"delta":{"content":"Workspace inspected."}}]}\n',
                b"data: [DONE]\n",
            ],
        )
    )
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("kimi", "kimi-key")
    session = home.create_session(tmp_path, provider="kimi", model="kimi-k3")
    home.append_session_event(session.path, "user_message", {"message": "Inspect this."})
    runtime = AgentRuntime(home, tmp_path)
    payloads = []
    thoughts = []

    def fake_urlopen(request, timeout):
        assert request.full_url == "https://api.moonshot.ai/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer kimi-key"
        assert timeout == 120
        payloads.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse(next(responses))

    monkeypatch.setattr(openai_chat_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        runtime,
        "_execute_tool",
        lambda name, arguments, callbacks, session_path: json.dumps(
            {"name": name, "arguments": arguments}
        ),
    )

    response = KimiBackend(runtime).generate(
        session.path,
        "kimi-k3",
        RuntimeCallbacks(thought=thoughts.append),
    )

    assert response == "Workspace inspected."
    assert thoughts == ["Inspect the workspace."]
    assert payloads[0]["model"] == "kimi-k3"
    assert payloads[0]["stream"] is True
    assert payloads[0]["tools"]
    assert all(
        tool["function"]["strict"] is False
        for tool in payloads[0]["tools"]
    )
    assert payloads[1]["messages"][-2] == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "list",
                    "arguments": '{"path":"."}',
                },
            }
        ],
        "reasoning_content": "Inspect the workspace.",
    }
    assert payloads[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"name": "list", "arguments": {"path": "."}}',
    }


def test_kimi_multimodal_messages_use_openai_compatible_image_blocks(tmp_path):
    image = tmp_path / "plot.png"
    image.write_bytes(b"png-bytes")
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="kimi", model="kimi-k2.6")
    home.append_session_event(
        session.path,
        "user_message",
        {
            "message": "Explain the plot.",
            "image_attachments": [
                {
                    "label": "plot.png",
                    "token": "[image: plot.png]",
                    "path": image.as_posix(),
                    "mime_type": "image/png",
                }
            ],
        },
    )
    backend = KimiBackend(AgentRuntime(home, tmp_path))

    message = backend._chat_messages(session.path, "kimi-k2.6")[-1]

    assert message["role"] == "user"
    assert message["content"][0] == {
        "type": "text",
        "text": "Explain the plot.\n\nAttached images:\n- plot.png",
    }
    assert message["content"][1]["type"] == "image_url"
    assert message["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
