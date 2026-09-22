import importlib.util
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from anomx.agent import AgentMode, AnomxHome
from anomx.agent.helpers.anomx_api import _build_url
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.skills import load_system_skills


@pytest.fixture
def runtime(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.ensure()
    home.set_platform_connection(url="https://platform.test/api", token="test-token")
    return AgentRuntime(home, tmp_path, mode=AgentMode.AUTONOMOUS)


def respond(monkeypatch, payloads):
    calls = []
    responses = iter(payloads)

    def urlopen(request, **kwargs):
        calls.append(request)
        payload = next(responses)
        if isinstance(payload, Exception):
            raise payload
        response = io.BytesIO(json.dumps(payload).encode())
        response.status = 200
        response.headers = {"content-type": "application/json"}
        return response

    monkeypatch.setattr("anomx.agent.helpers.anomx_api.urlopen", urlopen)
    return calls


def execute(runtime, name, **arguments):
    return json.loads(runtime._execute_tool(name, arguments, RuntimeCallbacks()))


def test_live_channel_search_uses_deployed_routes_and_retains_pagination(runtime, monkeypatch):
    channel = {"identifier": "XFEL.RF/LLRF.GUNTEMP/GUN.I1/DETUNING"}
    calls = respond(
        monkeypatch,
        [
            {"items": [channel], "has_more": True, "next_offset": 20, "offset": 10, "total": 25},
            [{"value": "XFEL.RF/LLRF.GUNTEMP/"}, {"value": "leaf", "channel": channel}],
        ],
    )
    result = execute(runtime, "search_anomx_data_channels", query="XFEL.RF", page=2)
    assert [call.full_url for call in calls] == [
        "https://platform.test/api/channels/live-search?query=XFEL.RF&limit=10&page=2",
        "https://platform.test/api/channels/live-hints?query=XFEL.RF&limit=10",
    ]
    assert result["ok"] is True
    assert result["channels"] == [channel]
    assert result["hints"] == [{"value": "XFEL.RF/LLRF.GUNTEMP/"}]
    assert result["pagination"] == {
        "has_more": True,
        "next_offset": 20,
        "offset": 10,
        "total": 25,
    }


def test_search_http_failure_is_not_a_successful_empty_result(runtime, monkeypatch):
    calls = respond(
        monkeypatch,
        [
            HTTPError(
                "https://platform.test/api/channels/live-search",
                404,
                "Not Found",
                {},
                io.BytesIO(b'{"detail":"Not found."}'),
            ),
        ],
    )
    result = execute(runtime, "search_anomx_data_channels", query="gun")
    assert result["ok"] is False
    assert "Changing search terms" in result["error"]
    assert result["request"]["status_code"] == 404
    assert result["request"]["response"] == {"detail": "Not found."}
    assert "channels" not in result
    assert len(calls) == 1


def test_hint_failure_preserves_discovered_channels(runtime, monkeypatch):
    respond(
        monkeypatch,
        [
            {"items": [{"id": "channel-1"}]},
            HTTPError(
                "https://platform.test/api/channels/live-hints",
                503,
                "Unavailable",
                {},
                io.BytesIO(b'{"detail":"Worker unavailable"}'),
            ),
        ],
    )
    result = execute(runtime, "search_anomx_data_channels", query="gun")
    assert result["ok"] is False and result["partial"] is True
    assert result["channels"] == [{"id": "channel-1"}]
    assert result["requests"][1]["status_code"] == 503


def test_channel_history_uses_unversioned_route(runtime, monkeypatch):
    calls = respond(monkeypatch, [{"samples": [[1, 2]]}])
    result = execute(
        runtime,
        "get_anomx_data_channel_history",
        object_reference="data_channel-example",
        range="1h",
        max_data_points=500,
    )
    assert calls[0].full_url == (
        "https://platform.test/api/channels/data_channel-example/history?range=1h&max_points=100"
    )
    assert result["history"] == {"samples": [[1, 2]]}


@pytest.mark.parametrize("paginated", [False, True])
def test_background_runs_do_not_reintroduce_compression_metadata(runtime, monkeypatch, paginated):
    runtime.set_mode(AgentMode.BACKGROUND)
    items = [
        {
            "id": str(i),
            "name": f"Run {i}",
            "status": "completed",
            "last_agent_message": "Done. " * 1000,
            "metadata": {"context_compression": {"summary": "old-summary" * 10000}},
        }
        for i in range(45)
    ]
    payload = {"items": items[20:40], "total": 45} if paginated else items
    respond(monkeypatch, [payload])
    result = execute(runtime, "get_background_runs", page=2)
    assert [run["id"] for run in result["runs"]] == [str(i) for i in range(20, 40)]
    assert result["next_page"] == 3
    assert result["total"] == 45
    assert result["runs"][0]["result_truncated"] is True
    assert "old-summary" not in json.dumps(result)
    assert len(json.dumps(result)) < 50000
    assert Path(result["request"]["response_path"]).exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("body", '{"name":"Updated"}'),
        ("query", "gun"),
        ("headers", []),
        ("body", None),
        ("body", {}),
    ],
)
def test_invalid_api_arguments_cannot_silently_send_empty_writes(
    runtime, monkeypatch, field, value
):
    monkeypatch.setattr(
        "anomx.agent.helpers.anomx_api.urlopen",
        lambda *_args, **_kwargs: pytest.fail("Must not send a malformed request"),
    )
    arguments = {"method": "PATCH", "path": "/systems/example", "body": {"name": "Updated"}}
    arguments[field] = value
    assert execute(runtime, "use_anomx_api", **arguments)["ok"] is False


def test_large_nested_patch_body_is_transmitted_without_loss(runtime, monkeypatch):
    body = {
        "properties": {
            "notes": "Measurement " * 1000,
            "linked_channels": [{"id": str(i)} for i in range(30)],
        }
    }
    calls = respond(monkeypatch, [body])
    result = execute(runtime, "use_anomx_api", method="PATCH", path="/systems/example", body=body)
    assert result["ok"] is True
    assert json.loads(calls[0].data) == body


def test_schema_and_skill_script_keep_reverse_proxy_prefix():
    script = Path(__file__).parents[1] / "src/anomx/agent/skills/use-anomx-api/api.py"
    spec = importlib.util.spec_from_file_location("api_skill", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for build_url in (_build_url, module._build_url):
        assert build_url("https://platform.test/prefix/api", "/openapi.json", None) == (
            "https://platform.test/prefix/api/openapi.json"
        )


def test_builtin_skills_teach_current_routes_and_system_hierarchy():
    skills = {skill.command: skill for skill in load_system_skills()}
    assert "source_object_reference" in skills["manage-systems"].body
    assert "/systems/explore" in skills["manage-systems"].body
    assert "kind=part_of" in skills["manage-systems"].body
    for name in ("retrieve-data", "manage-data", "manage-jobs"):
        assert "/data/channels" not in skills[name].body
        assert "/jobs/jobs" not in skills[name].body
