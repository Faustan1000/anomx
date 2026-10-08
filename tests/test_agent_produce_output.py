import json
from types import SimpleNamespace

import pytest

from anomx.agent import AgentMode
from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.runtime import AgentRuntime
from anomx.agent.store import AnomxHome
from anomx.agent.tools.produce_output import ProduceOutputTool


def output_context(*, enabled=True):
    emitted = []
    runtime = SimpleNamespace(can_output_response=lambda: enabled, produced_output=None)
    callbacks = SimpleNamespace(output_response=emitted.append)
    return ToolExecutionContext(runtime=runtime, callbacks=callbacks), emitted


def test_output_preserves_body_order_and_moves_sources_last():
    context, emitted = output_context()
    items = [
        {"kind": "reference", "content": {"url": "https://example.org/source"}},
        {"kind": "text", "content": "**Result**"},
        {"kind": "object", "content": {"object_reference": "content_page_example"}},
        {"kind": "objects", "content": ["data_channel_a", "data_channel_b"]},
        {
            "kind": "database",
            "content": {"model_reference": "data_channel", "search": "temperature"},
        },
        {"kind": "reference", "content": {"object_reference": "data_channel_a"}},
    ]
    result = json.loads(ProduceOutputTool().execute({"items": items}, context))
    assert result["ok"] is True
    assert emitted[0]["items"] == [items[1], items[2], items[3], items[4], items[0], items[5]]
    assert context.runtime.produced_output == "**Result**"


def proposition_item(**overrides):
    content = {
        "prompt": "Create a planned prompt that repeats this analysis every day at 08:00.",
        "label": "Run this every morning",
        "icon": "ClockFastForward",
        "description": "Schedule this analysis for 08:00 daily to track changes over time.",
    }
    return {"kind": "proposition", "content": {**content, **overrides}}


def test_output_places_the_proposition_between_body_and_sources():
    context, emitted = output_context()
    items = [
        proposition_item(),
        {"kind": "reference", "content": {"url": "https://example.org/source"}},
        {"kind": "text", "content": "**Result**"},
        {"kind": "object", "content": {"object_reference": "content_page_example"}},
    ]
    result = json.loads(ProduceOutputTool().execute({"items": items}, context))
    assert result["ok"] is True
    assert emitted[0]["items"] == [items[2], items[3], items[0], items[1]]
    assert context.runtime.produced_output == "**Result**"


def test_output_accepts_at_most_one_proposition():
    context, emitted = output_context()
    items = [
        {"kind": "text", "content": "Done"},
        proposition_item(),
        proposition_item(label="Run this every evening"),
    ]
    result = json.loads(ProduceOutputTool().execute({"items": items}, context))
    assert result["ok"] is False
    assert result["error"] == "items may contain at most one proposition."
    assert emitted == []
    assert context.runtime.produced_output is None


def test_output_decodes_encoded_array_without_losing_sources():
    context, emitted = output_context()
    items = [
        {"kind": "text", "content": 'No live data: HTTP 502 "illegal property".\nDetails.'},
        {"kind": "reference", "content": {"object_reference": "data_channel_a"}},
        {"kind": "reference", "content": {"url": "https://example.org"}},
    ]
    result = json.loads(ProduceOutputTool().execute({"items": json.dumps(items)}, context))
    assert result["ok"] is True
    assert emitted == [{"items": items, "end_turn": True}]
    assert context.runtime.produced_output == items[0]["content"]


@pytest.mark.parametrize("encoded", [False, True])
def test_invalid_output_after_valid_text_never_partially_publishes(encoded):
    context, emitted = output_context()
    items = [
        {"kind": "text", "content": "Complete answer"},
        {"kind": "reference", "content": {"url": "javascript:alert(1)"}},
    ]
    result = json.loads(ProduceOutputTool().execute(
        {"items": json.dumps(items) if encoded else items}, context
    ))
    assert result["ok"] is False
    assert result["error"].startswith("items[1]:")
    assert emitted == []
    assert context.runtime.produced_output is None


@pytest.mark.parametrize(
    ("arguments", "error_fragment"),
    [
        ({}, "received missing or null"),
        ({"items": {}}, "must be a JSON array"),
        ({"items": "{}"}, "must be a JSON array"),
        ({"items": []}, "received 0"),
        ({"items": [{"kind": "text", "content": "A"}] * 51}, "received 51"),
        ({"items": '[{"kind":"text","content":"HTTP 502 „illegal property""}]'},
         "string containing invalid JSON"),
        ({"items": [{"kind": "text", "content": "A", "content_type": "text/markdown"}]},
         "Remove unsupported fields: content_type"),
        ({"items": [{"kind": "text"}]}, "Missing fields: content"),
    ],
)
def test_output_errors_identify_the_actual_problem(arguments, error_fragment, caplog):
    context, emitted = output_context()
    result = json.loads(ProduceOutputTool().execute(arguments, context))
    assert result["error_code"] == "invalid_tool_arguments"
    assert error_fragment in result["error"]
    assert result["hint"]
    assert isinstance(result["example"]["items"], list)
    assert "produce_output_validation_failed" in caplog.text
    assert "illegal property" not in caplog.text
    assert emitted == []
    assert context.runtime.produced_output is None


@pytest.mark.parametrize(
    "item",
    [
        proposition_item(prompt=""),
        proposition_item(label=" "),
        proposition_item(icon=None),
        proposition_item(description=" "),
        proposition_item(description=None),
        proposition_item(description="x" * 301),
        {"kind": "proposition", "content": {"prompt": "Do more", "label": "More", "icon": "Plus"}},
        proposition_item(title="Unsupported"),
        {"kind": "proposition", "content": "Create a planned prompt."},
        {"kind": "object", "content": {}},
        {"kind": "objects", "content": [3]},
        {"kind": "database", "content": {"model_reference": "data_channel", "view": "unknown"}},
        {"kind": "reference", "content": {"url": "javascript:alert(1)"}},
        {"kind": "reference", "content": {"url": "https://[invalid"}},
        {"kind": "reference", "content": {"url": "https://user:secret@example.org"}},
        {"kind": "reference", "content": {"url": "https://example.org", "object_reference": "a"}},
        {"kind": "text", "content": ""},
        {"kind": "other", "content": {}},
    ],
)
def test_invalid_output_does_not_partially_publish(item):
    context, emitted = output_context()
    result = json.loads(
        ProduceOutputTool().execute(
            {"items": [{"kind": "text", "content": "Valid"}, item]}, context
        )
    )
    assert result["ok"] is False
    assert emitted == []
    assert context.runtime.produced_output is None


def test_output_cannot_publish_from_cli():
    context, emitted = output_context(enabled=False)
    result = json.loads(
        ProduceOutputTool().execute({"items": [{"kind": "text", "content": "Hello"}]}, context)
    )
    assert result["ok"] is False
    assert emitted == []


def test_missing_and_legacy_modes_use_automatic():
    assert AgentMode.parse(None) is AgentMode.AUTOMATIC
    assert AgentMode.parse("invalid") is AgentMode.AUTOMATIC
    assert AgentMode.parse("plan") is AgentMode.AUTOMATIC
    assert "plan" not in {mode.value for mode in AgentMode}


def test_connected_cli_does_not_advertise_platform_output(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    runtime = AgentRuntime(home, tmp_path)
    monkeypatch.setattr(runtime, "has_platform_connection", lambda: True)
    assert "produce_output" not in {tool.name for tool in runtime._available_tools()}
    config = home.load_config()
    config.update(running_in_anomx_platform=True, platform_output_response_enabled=True)
    monkeypatch.setattr(home, "load_config", lambda: config)
    assert "produce_output" not in {tool.name for tool in runtime._available_tools()}


def test_platform_chat_advertises_output_to_the_model(tmp_path, monkeypatch):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path, platform_chat_id="chat-id")
    monkeypatch.setattr(runtime, "has_platform_connection", lambda: False)
    assert "produce_output" not in {tool.name for tool in runtime._available_tools()}
    monkeypatch.setattr(runtime, "has_platform_connection", lambda: True)
    assert {"produce_output", "focus_object"} <= {tool["name"] for tool in runtime._openai_tools()}
    assert "# Output Contract" in runtime._instructions()
