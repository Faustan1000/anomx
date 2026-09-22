from anomx.agent import AgentKind, AgentMode, MainAgent, SubAgent, next_agent_mode


def test_agent_roles_and_modes_are_orthogonal():
    assert [kind.value for kind in AgentKind] == ["main", "sub"]
    assert [mode.value for mode in AgentMode] == [
        "plan",
        "background",
        "standard",
        "automatic",
        "autonomous",
    ]


def test_modes_have_one_central_policy_and_cycle():
    assert AgentMode.PLAN.policy.read_only is True
    assert AgentMode.RECOMMEND.policy.recommendations_only is True
    assert AgentMode.STANDARD.policy.requires_approval_for_unremembered is True
    assert AgentMode.AUTOMATIC.policy.auto_approves_risk("low") is True
    assert AgentMode.AUTOMATIC.policy.auto_approves_risk("medium") is False
    assert AgentMode.AUTONOMOUS.policy.bypass_command_policy is True

    assert next_agent_mode(AgentMode.PLAN) == AgentMode.STANDARD
    assert next_agent_mode(AgentMode.STANDARD) == AgentMode.AUTOMATIC
    assert next_agent_mode(AgentMode.AUTOMATIC) == AgentMode.AUTONOMOUS
    assert next_agent_mode(AgentMode.AUTONOMOUS) == AgentMode.PLAN
    assert next_agent_mode(AgentMode.PLAN, platform_connected=True) == AgentMode.STANDARD
    assert next_agent_mode(AgentMode.BACKGROUND, platform_connected=True) == AgentMode.PLAN


def test_subagent_tools_exclude_main_agent_coordination_tools():
    main_tools = {tool.name for tool in MainAgent().tools}
    subagent_tools = {tool.name for tool in SubAgent().tools}
    excluded = {
        "ask_question",
        "create_plan",
        "end_process",
        "finish_anyways",
        "get_subagent_info",
        "memorize",
        "output_response",
        "prompt_subagent",
        "remove_plan",
        "remove_subagent",
        "start_process",
        "start_subagent",
        "update_plan",
    }

    assert excluded <= main_tools
    assert excluded.isdisjoint(subagent_tools)


def test_main_agent_has_focused_anomx_read_tools():
    tools = {tool.name: tool for tool in MainAgent().tools}

    assert {
        "get_anomx_data_channel_history",
        "get_anomx_object_details",
        "search_anomx_data_channels",
        "search_anomx_objects",
    } <= tools.keys()
    assert "identifier segments" in tools["search_anomx_data_channels"].description
    assert "object references" in tools["search_anomx_objects"].description
