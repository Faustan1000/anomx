"""Regression coverage for command approval routing and local data inspection."""

import shlex

import pytest

from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.tool_manager import (
    ApprovalChoice,
    CliToolManager,
    CommandResult,
    CommandSafety,
)


def python_command(source):
    return "python3 -c " + shlex.quote(source)


@pytest.mark.parametrize("mode", [AgentMode.STANDARD, AgentMode.AUTOMATIC, AgentMode.AUTONOMOUS])
def test_local_json_inspection_needs_no_approval(tmp_path, mode):
    response = tmp_path / "response.json"
    response.write_text('[{"name": "Example", "object_reference": "page-1"}]')
    manager = CliToolManager(tmp_path, mode=mode, strict_workspace=True)
    result = manager.run_command(
        python_command(
            f"import json\np={str(response)!r}\ndata=json.load(open(p))\n"
            "for x in data:\n    print(x.get('name'), '|', repr(x.get('object_reference')))"
        ),
        "Extract page references",
        None,
    )
    assert result.approved
    assert "Example | 'page-1'" in result.output


@pytest.mark.parametrize(
    "source",
    [
        "open('data.json', 'w').write('changed')",
        "import os; os.remove('data.json')",
        "eval('print(1)')",
        "import json; json.load(open('data.json'), object_hook=print)",
        "p = 'data.json'; p = input(); print(open(p).read())",
        "print((1).__class__.__bases__)",
        "read = open; read('data.json', 'w')",
        "import json; json = print; json('changed')",
        "for p in ['data.json']: print(open(p).read())",
        "p = 'data.json'\nfor x in [1,2]:\n print(open(p).read())\n p = '../outside.json'",
    ],
)
def test_complex_or_mutating_python_still_needs_assessment(tmp_path, source):
    manager = CliToolManager(tmp_path)
    assert manager.classify(python_command(source)).safety != CommandSafety.ALLOW


@pytest.mark.parametrize("mode", [AgentMode.STANDARD, AgentMode.AUTOMATIC])
@pytest.mark.parametrize("strict", [False, True])
def test_deterministic_failure_requests_explicit_approval(tmp_path, mode, strict):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"value": 42}')
    manager = CliToolManager(root, mode=mode, strict_workspace=strict)
    requests = []
    command = python_command(f"import json; print(json.load(open({str(outside)!r})))")
    result = manager.run_command(
        command,
        "Read outside file",
        lambda request: requests.append(request) or ApprovalChoice.ALLOW,
    )
    assert result.approved
    assert len(requests) == 1
    assert requests[0].requires_user_approval
    assert "42" in result.output


@pytest.mark.parametrize("strict", [False, True])
def test_autonomous_blocks_deterministic_failures(tmp_path, strict):
    manager = CliToolManager(tmp_path, mode=AgentMode.AUTONOMOUS, strict_workspace=strict)
    requests = []
    result = manager._authorize_command(
        "cat ../outside.json",
        "Read outside file",
        lambda request: requests.append(request) or ApprovalChoice.ALLOW,
    )
    assert isinstance(result, CommandResult)
    assert not result.approved
    assert result.blocked_by_mode
    assert result.safety == CommandSafety.FORBIDDEN
    assert not requests


@pytest.mark.parametrize("mode", [AgentMode.STANDARD, AgentMode.AUTOMATIC, AgentMode.AUTONOMOUS])
def test_explicit_user_rejections_remain_blocked(tmp_path, mode):
    manager = CliToolManager(tmp_path, mode=mode, session_rejected_commands={"cmd:cat"})
    result = manager._authorize_command(
        "cat file.json", "Read file", lambda _: pytest.fail("Must not ask again")
    )
    assert isinstance(result, CommandResult)
    assert not result.approved


def test_python_reads_cannot_follow_symlinks_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (workspace / "data.json").symlink_to(outside)
    manager = CliToolManager(workspace, mode=AgentMode.AUTONOMOUS)
    result = manager._authorize_command(
        python_command("print(open('data.json').read())"), "Read file", None
    )
    assert isinstance(result, CommandResult)
    assert not result.approved


@pytest.mark.parametrize(
    "command", ["cat ../outside.json", python_command("print(open('../outside.json').read())")]
)
def test_saved_command_allowance_does_not_bypass_path_failure(tmp_path, command):
    manager = CliToolManager(
        tmp_path, mode=AgentMode.AUTONOMOUS, session_allowed_commands={"cmd:cat", "cmd:python3 -c"}
    )
    result = manager._authorize_command(command, "Read file", None)
    assert isinstance(result, CommandResult)
    assert not result.approved


@pytest.mark.parametrize("mode", [AgentMode.STANDARD, AgentMode.AUTOMATIC])
def test_safe_commands_and_low_risk_assessments_do_not_prompt(tmp_path, mode):
    manager = CliToolManager(tmp_path, mode=mode)
    assert manager.run_command("pwd", "Show workspace", None).approved
    assert mode.policy.auto_approves_risk("low")
    assert not mode.policy.auto_approves_risk("high")
