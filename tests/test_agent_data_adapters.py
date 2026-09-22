import json
from types import SimpleNamespace
from unittest.mock import patch

from anomx.agent import MainAgent, SubAgent
from anomx.agent.tools.data_adapters import ManageDataAdaptersTool
from anomx.agent.tools.use_anomx_api import UseAnomxApiTool

INTEGRATION = "89f5f5e4-91c7-4f3d-8adf-9531c80462a1"


def test_adapter_tool_delegates_to_existing_api_authorization():
    context = SimpleNamespace(json_result=json.dumps)
    with patch.object(UseAnomxApiTool, "execute", return_value="approval required") as api:
        result = ManageDataAdaptersTool().execute(
            {
                "integration_id": INTEGRATION,
                "action": "activate",
                "id": "candidate",
                "create_channel": True,
            },
            context,
        )
    assert result == "approval required"
    request, actual_context = api.call_args.args
    assert actual_context is context
    assert request["method"] == "POST"
    assert request["path"] == f"/integrations/{INTEGRATION}/data-adapters"
    assert request["body"]["create_channel"] is True


def test_adapter_listing_preserves_pagination_and_invalid_ids_never_call_api():
    context = SimpleNamespace(json_result=json.dumps)
    with patch.object(UseAnomxApiTool, "execute", return_value="ok") as api:
        ManageDataAdaptersTool().execute(
            {
                "integration_id": INTEGRATION,
                "action": "list",
                "state": "needs_review",
                "offset": 50,
            },
            context,
        )
        assert api.call_args.args[0]["query"] == {"state": "needs_review", "offset": 50}
        api.reset_mock()
        response = ManageDataAdaptersTool().execute(
            {"integration_id": "../other", "action": "list"}, context
        )
        assert json.loads(response)["ok"] is False
        api.assert_not_called()


def test_main_and_subagents_can_manage_adapters():
    for agent in (MainAgent(), SubAgent()):
        assert "manage_data_adapters" in {tool.name for tool in agent.tools}
