import pytest

from anomx.agent.app import AnomxCliApp
from anomx.agent.base.tools import statement_property
from anomx.agent.store import AnomxHome
from anomx.agent.ui.models import MessageLine


@pytest.fixture
def chat(tmp_path):
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=tmp_path, use_color=False)
    home.append_session_event(session.path, "user_message", {"message": "Inspect channels"})
    home.append_session_event(session.path, "work_message", {
        "message": "Reading channels", "role": "tool", "turn_id": "turn-1",
    })
    home.append_session_event(session.path, "agent_message", {
        "message": "I found the channel list.", "intermediate": True, "turn_id": "turn-1",
    })
    home.append_session_event(session.path, "work_message", {
        "message": "Counting channels", "role": "tool", "turn_id": "turn-1",
    })
    return home, session, app


def test_work_visualization_defaults_persists_and_normalizes(tmp_path):
    home = AnomxHome(tmp_path / "home")
    assert home.load_config()["work_visualization"] == "default"
    home.save_config({"work_visualization": "extended"})
    assert AnomxHome(home.root).load_config()["work_visualization"] == "extended"
    assert AnomxCliApp(home=home, cwd=tmp_path).work_visualization == "extended"
    home.save_config({"work_visualization": "invalid"})
    assert home.load_config()["work_visualization"] == "default"
    home.config_path.write_text('work_visualization = "invalid"\n')
    assert home.load_config()["work_visualization"] == "default"


def test_default_keeps_messages_visible_and_expands_each_tool_group(chat):
    home, session, app = chat
    messages = app._read_message_lines(session.path)
    assert messages == [
        MessageLine("user", "Inspect channels"),
        MessageLine("work_summary", "Reading channels", "turn-1"),
        MessageLine("agent_intermediate", "I found the channel list.", "turn-1"),
        MessageLine("work_active", "Counting channels", "turn-1:1"),
    ]
    assert app._messages_with_working_status(messages, "Thinking") == messages
    app._toggle_work_turn("turn-1")
    expanded = app._read_message_lines(session.path)
    assert [line.text for line in expanded] == [
        "Inspect channels", "Reading channels", "Reading channels · collapse",
        "I found the channel list.", "Counting channels",
    ]
    home.append_session_event(session.path, "work_message", {
        "message": "Inspecting metadata", "role": "tool", "turn_id": "turn-1",
    })
    app._toggle_work_turn("turn-1")
    assert app._read_message_lines(session.path)[-1].text == "Inspecting metadata"
    app._toggle_work_turn("turn-1:1")
    assert [line.text for line in app._read_message_lines(session.path)] == [
        "Inspect channels", "Reading channels", "I found the channel list.",
        "Counting channels", "Inspecting metadata", "Inspecting metadata · collapse",
    ]


def test_extended_invalidates_cache_and_keeps_intermediate_order(chat):
    _, session, app = chat
    app._read_message_lines(session.path)
    app.work_visualization = "extended"
    messages = app._read_message_lines(session.path)
    assert [line.role for line in messages] == ["user", "tool", "agent_intermediate", "tool"]
    assert [line.text for line in messages][1:] == [
        "Reading channels", "I found the channel list.", "Counting channels",
    ]


def test_new_activity_stays_after_the_user_message_that_triggered_it(chat):
    home, session, app = chat
    home.append_session_event(session.path, "user_message", {"message": "Also inspect metadata"})
    home.append_session_event(session.path, "work_message", {
        "message": "Reading metadata", "role": "tool", "turn_id": "turn-1",
    })
    messages = app._read_message_lines(session.path)
    assert [line.text for line in messages] == [
        "Inspect channels", "Reading channels", "I found the channel list.",
        "Counting channels",
        "Also inspect metadata", "Reading metadata",
    ]


def test_completed_summary_keeps_final_response_in_both_modes(chat):
    home, session, app = chat
    home.append_session_event(session.path, "work_summary", {
        "message": "Worked for 00:03", "turn_id": "turn-1",
    })
    home.append_session_event(session.path, "agent_message", {"message": "19 channels"})
    assert app._read_message_lines(session.path) == [
        MessageLine("user", "Inspect channels"),
        MessageLine("work_summary", "Worked for 00:03", "work:turn-1"),
        MessageLine("agent", "19 channels"),
    ]
    app._toggle_work_turn("work:turn-1")
    assert [line.text for line in app._read_message_lines(session.path)] == [
        "Inspect channels", "Worked for 00:03 · collapse", "Reading channels",
        "I found the channel list.", "Counting channels", "19 channels",
    ]
    app.work_visualization = "extended"
    messages = app._read_message_lines(session.path)
    assert len(messages) == 5
    assert messages[-1] == MessageLine("agent", "19 channels")


def test_default_keeps_errors_and_final_output_visible_without_summary(chat):
    home, session, app = chat
    home.append_session_event(session.path, "system_message", {
        "message": "Connection failed", "role": "system", "turn_id": "turn-1",
    })
    home.append_session_event(session.path, "agent_message", {
        "message": "The source is unavailable.", "turn_id": "turn-1",
    })
    messages = app._read_message_lines(session.path)
    assert MessageLine("system", "Connection failed", "turn-1") in messages
    assert MessageLine("agent", "The source is unavailable.", "turn-1") in messages


@pytest.mark.parametrize("completed", [False, True])
def test_intermediate_user_message_splits_tool_groups_and_survives_reload(chat, completed):
    home, session, app = chat
    home.append_session_event(session.path, "user_message", {
        "message": "Only count active channels", "intermediate": True, "turn_id": "turn-1",
    })
    for statement in ("Reading active flags", "Counting active channels"):
        home.append_session_event(session.path, "work_message", {
            "message": statement, "role": "tool", "turn_id": "turn-1",
        })
    if completed:
        home.append_session_event(session.path, "work_summary", {
            "message": "Worked for 00:03", "turn_id": "turn-1",
        })
        home.append_session_event(session.path, "agent_message", {
            "message": "12 active channels", "turn_id": "turn-1",
        })
    expected = [
        "Inspect channels", "Reading channels", "I found the channel list.",
        "Counting channels", "Only count active channels",
        "Counting active channels",
        *(["12 active channels"] if completed else []),
    ]
    if completed:
        expected = ["Inspect channels", "Worked for 00:03", "12 active channels"]
    assert [line.text for line in app._read_message_lines(session.path)] == expected
    reloaded = AnomxCliApp(home=home, cwd=session.path.parent, use_color=False)
    assert [line.text for line in reloaded._read_message_lines(session.path)] == expected
    if completed:
        app._toggle_work_turn("work:turn-1")
    app._toggle_work_turn("turn-1:2")
    expanded = app._read_message_lines(session.path)
    offset = 5 if completed else 4
    label = (
        "Reading active flags, Counting active channels"
        if completed else "Counting active channels"
    )
    assert [line.text for line in expanded][offset:offset + 4] == [
        "Only count active channels", "Reading active flags", "Counting active channels",
        label + " · collapse",
    ]


def test_completion_keeps_errors_visible_and_message_only_work_expandable(chat):
    home, session, app = chat
    for role, message in (("system", "Connection failed"), ("warning", "Retry unavailable")):
        home.append_session_event(session.path, "system_message", {
            "message": message, "role": role, "turn_id": "turn-1",
        })
    home.append_session_event(session.path, "work_summary", {
        "message": "Interrupted after 00:03", "turn_id": "turn-1",
    })
    home.append_session_event(session.path, "agent_message", {
        "message": "I can explain this without tools.", "intermediate": True, "turn_id": "turn-2",
    })
    home.append_session_event(session.path, "work_summary", {
        "message": "Worked for 00:01", "turn_id": "turn-2",
    })
    assert [line.text for line in app._read_message_lines(session.path)][-3:] == [
        "Connection failed", "Retry unavailable", "Worked for 00:01",
    ]
    app._toggle_work_turn("work:turn-2")
    assert app._read_message_lines(session.path)[-1].text == "I can explain this without tools."


def test_completion_collapses_work_and_remembers_nested_expansion(chat):
    home, session, app = chat
    app._toggle_work_turn("turn-1:1")
    before = app._read_message_lines(session.path)
    home.append_session_event(session.path, "work_summary", {
        "message": "Worked for 00:03", "turn_id": "turn-1",
    })
    home.append_session_event(session.path, "agent_message", {"message": "19 channels"})
    after = app._read_message_lines(session.path)
    assert [line.text for line in after] == ["Inspect channels", "Worked for 00:03", "19 channels"]
    app._toggle_work_turn("work:turn-1")
    reopened = app._read_message_lines(session.path)
    assert [line.text for line in reopened[2:-1]] == [line.text for line in before[1:]]
    assert reopened[-2].meta == before[-1].meta
    assert after[-1] == MessageLine("agent", "19 channels")


def test_manage_settings_saves_global_work_visualization(chat, monkeypatch):
    home, _, app = chat
    choices = iter(["work_visualization", "extended", None])
    menus = []

    def select(_window, title, _description, options):
        menus.append((title, options))
        return next(choices)

    monkeypatch.setattr(app, "_menu", select)
    app._run_manage_settings_panel(object())
    assert menus[1][0] == "Work Visualization"
    assert [option.value for option in menus[1][1]] == ["default", "extended"]
    assert home.load_config()["work_visualization"] == "extended"
    assert app.work_visualization == "extended"
    assert any(choice.label == "Work Visualization: Extended"
               for choice in app._manage_settings_choices())


@pytest.mark.parametrize("role", ["work_active", "work_summary"])
def test_working_fallback_and_long_activity_stay_on_one_line(chat, role):
    _, _, app = chat
    assert app._messages_with_working_status([], "Thinking")[0].text == "Thinking"
    lines = app._render_messages([
        MessageLine(role, "Inspecting " + "channel " * 100, "turn-1"),
    ], 40)
    assert len(lines) == 1
    assert len(lines[0].text) <= 40
    assert lines[0].meta == "turn-1"


def test_statement_schema_requests_short_action_labels():
    description = statement_property("Describe this tool call.")["description"]
    assert "3–7 words" in description
    assert "60 characters" in description


def test_finished_tool_block_summarizes_calls_and_thoughts_before_turn_completion(chat):
    home, session, app = chat
    for role, message, command in (
        ('tool', 'Read the first file', 'Tool: read_file\nParameters: {"path":"a"}'),
        ('tool', 'Read the second file', 'Tool: read_file\nParameters: {"path":"b"}'),
        ('thought', 'Created a thought', 'The two files are consistent.'),
    ):
        home.append_session_event(session.path, 'work_message', {
            'role': role, 'message': message, 'command': command, 'turn_id': 'turn-1',
        })
    assert app._read_message_lines(session.path)[-1].text == 'Created a thought'
    home.append_session_event(session.path, 'agent_message', {
        'message': 'Both files agree.', 'intermediate': True, 'turn_id': 'turn-1',
    })
    lines = app._read_message_lines(session.path)
    assert lines[-2] == MessageLine(
        'work_summary', 'Counting channels, 2× Read file, Thought', 'turn-1:1',
    )
    assert lines[-1].text == 'Both files agree.'
    home.append_session_event(session.path, 'work_message', {
        'role': 'tool', 'message': 'Checking the result', 'turn_id': 'turn-1',
    })
    lines = app._read_message_lines(session.path)
    assert lines[-1].role == 'work_active'
    assert lines[-3].text == 'Counting channels, 2× Read file, Thought'


def test_streaming_text_immediately_finishes_the_current_tool_label(chat):
    home, session, app = chat
    home.append_session_event(session.path, 'work_message', {
        'role': 'thought', 'message': 'Created a thought', 'turn_id': 'turn-1',
    })
    messages = app._read_message_lines(session.path)
    streaming = app._messages_with_transient_state(messages, 3, 'I found')
    assert streaming[-2] == MessageLine('work_summary', 'Counting channels, Thought', 'turn-1:1')
    assert streaming[-1] == MessageLine('agent', 'I found')
    assert messages[-1].role == 'work_active'  # The persisted cache is untouched.


def test_completed_turns_fold_independently_and_include_context_activity(chat):
    home, session, app = chat
    home.append_session_event(session.path, 'context_activity_display', {
        'id': 'context-1', 'turn_id': 'turn-1', 'kind': 'optimization',
        'status': 'completed', 'changed': True, 'model_requests': 1,
        'tokens_before': 4000, 'tokens_after': 1000,
    })
    home.append_session_event(session.path, 'work_summary', {
        'message': 'Worked for 00:03', 'turn_id': 'turn-1',
    })
    home.append_session_event(session.path, 'agent_message', {'message': '19 channels'})
    home.append_session_event(session.path, 'user_message', {'message': 'Inspect something else'})
    home.append_session_event(session.path, 'work_message', {
        'role': 'thought', 'message': 'Created a thought', 'turn_id': 'turn-2',
    })
    home.append_session_event(session.path, 'work_summary', {
        'message': 'Worked for 00:01', 'turn_id': 'turn-2',
    })
    home.append_session_event(session.path, 'agent_message', {'message': 'Second answer'})
    assert [line.text for line in app._read_message_lines(session.path)] == [
        'Inspect channels', 'Worked for 00:03', '19 channels',
        'Inspect something else', 'Worked for 00:01', 'Second answer',
    ]
    app._toggle_work_turn('work:turn-1')
    messages = app._read_message_lines(session.path)
    assert any(line.role == 'context' for line in messages)
    assert messages[-2] == MessageLine('work_summary', 'Worked for 00:01', 'work:turn-2')
    reloaded = AnomxCliApp(home=home, cwd=session.path.parent, use_color=False)
    assert len(reloaded._read_message_lines(session.path)) == 6


def test_completed_work_header_can_be_clicked_to_reopen_its_contents(chat):
    home, session, app = chat
    home.append_session_event(session.path, 'work_summary', {
        'message': 'Worked for 00:03', 'turn_id': 'turn-1',
    })
    home.append_session_event(session.path, 'agent_message', {'message': '19 channels'})

    class Window:
        def erase(self):
            pass

        def getmaxyx(self):
            return 40, 100

        def addnstr(self, *_args):
            pass

        def refresh(self):
            pass

    app._draw_session(Window(), session, app._read_message_lines(session.path), '', 0, 0)
    actions = [action for actions in app._click_targets.values() for action in actions]
    assert any(action.kind == 'toggle_work' and action.text == 'work:turn-1' for action in actions)
