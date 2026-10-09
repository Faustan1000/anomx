"""Editor context: the VS Code extension, its Python reader and note matching."""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from anomx.agent.cowork import (
    Note,
    NoteMessage,
    build_vsix,
    context_path,
    install_command,
    note_near_cursor,
    notes_in_view,
    read_editor_context,
)
from anomx.agent.cowork.extension import EXTENSION_DIR, extension_manifest, vsix_name

NOW = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)


def _write(path: Path, **overrides) -> Path:
    data = {
        "schema": 1,
        "updatedAt": NOW.isoformat().replace("+00:00", "Z"),
        "pid": 1,
        "workspaceFolders": ["/ws"],
        "activeFile": "/ws/anomx/examples/baseline_detection.py",
        "languageId": "python",
        "isDirty": False,
        "isActiveEditor": True,
        "cursor": {"line": 3, "character": 1},
        "selection": {
            "startLine": 3,
            "startCharacter": 1,
            "endLine": 3,
            "endCharacter": 1,
            "isEmpty": True,
        },
        "visibleRange": {"startLine": 1, "endLine": 40},
        "activeLineText": "from anomx import (",
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _note(
    note_id: str, line: int, file: str = "anomx/examples/baseline_detection.py", end_line: int = 0
) -> Note:
    return Note(
        id=note_id,
        kind="note",
        file=file,
        line=line,
        end_line=end_line or line,
        messages=(NoteMessage(f"m-{note_id}", note_id),),
    )


# -- Python reader --------------------------------------------------------------------


def test_reads_the_extension_output(tmp_path):
    context = read_editor_context(_write(tmp_path / "ctx.json"))

    assert context is not None
    assert context.file == Path("/ws/anomx/examples/baseline_detection.py")
    assert (context.line, context.character) == (3, 1)
    assert (context.visible_start_line, context.visible_end_line) == (1, 40)
    assert context.workspace_folders == (Path("/ws"),)
    assert context.active_line_text == "from anomx import (" and not context.is_dirty
    assert context.is_active_editor is True
    assert context_path(tmp_path).name == "editor-context.json"


def test_closed_editor_yields_context_without_a_file(tmp_path):
    path = _write(tmp_path / "ctx.json", activeFile=None, cursor=None, selection=None)

    context = read_editor_context(path)

    assert context is not None and context.file is None and context.line == 0


@pytest.mark.parametrize("content", ["", "{broken", "[]", '{"schema": 2}', '{"schema": 1}'])
def test_unusable_files_return_none(tmp_path, content):
    path = tmp_path / "ctx.json"
    path.write_text(content, encoding="utf-8")

    assert read_editor_context(path) is None
    assert read_editor_context(tmp_path / "missing.json") is None


def test_max_age_is_opt_in(tmp_path):
    path = _write(tmp_path / "ctx.json")
    later = NOW + timedelta(hours=2)

    assert read_editor_context(path, now=later) is not None  # idle editors stay valid
    assert read_editor_context(path, max_age_seconds=60, now=later) is None
    assert read_editor_context(path, max_age_seconds=60, now=NOW + timedelta(seconds=5))


# -- matching notes -------------------------------------------------------------------


def test_note_near_cursor_picks_the_closest_note_in_the_active_file(tmp_path):
    context = read_editor_context(_write(tmp_path / "ctx.json"))
    assert context is not None
    workspace = Path("/ws")
    notes = [
        _note("far", 31),
        _note("exact", 3),  # on the cursor line
        _note("elsewhere", 3, file="anomx/other.py"),
    ]

    assert note_near_cursor(context, notes, workspace).id == "exact"
    assert note_near_cursor(context, [_note("a", 5)], workspace) is None  # radius 0
    assert note_near_cursor(context, [_note("a", 5)], workspace, radius=2).id == "a"


def test_a_cursor_anywhere_inside_a_range_note_counts_as_on_the_note(tmp_path):
    workspace = Path("/ws")
    ranged = _note("block", 2, end_line=5)

    inside = read_editor_context(_write(tmp_path / "a.json"))  # cursor on line 3
    outside = read_editor_context(_write(tmp_path / "b.json", cursor={"line": 9, "character": 1}))

    assert inside is not None and outside is not None
    assert note_near_cursor(inside, [ranged], workspace).id == "block"
    assert note_near_cursor(outside, [ranged], workspace) is None


def test_notes_in_view_uses_the_visible_lines_of_the_active_file(tmp_path):
    context = read_editor_context(_write(tmp_path / "ctx.json"))
    assert context is not None
    notes = [_note("b", 21), _note("a", 5), _note("below", 81), _note("x", 6, file="anomx/x.py")]

    assert [note.id for note in notes_in_view(context, notes, Path("/ws"))] == ["a", "b"]


# -- packaging ------------------------------------------------------------------------


def test_vsix_contains_manifest_and_only_the_audited_files(tmp_path):
    vsix = build_vsix(tmp_path)

    assert vsix.name == vsix_name() == "anomx-local.anomx-editor-context-0.2.3.vsix"
    with zipfile.ZipFile(vsix) as archive:
        assert sorted(archive.namelist()) == [
            "[Content_Types].xml",
            "extension.vsixmanifest",
            "extension/README.md",
            "extension/comments.js",
            "extension/extension.js",
            "extension/media/anomx.svg",
            "extension/notes.js",
            "extension/package.json",
        ]
        manifest = archive.read("extension.vsixmanifest").decode("utf-8")
    assert 'Publisher="anomx-local"' in manifest and 'Version="0.2.3"' in manifest
    assert install_command(vsix) == f"code --install-extension {vsix}"
    assert extension_manifest()["main"] == "./extension.js"


def test_extension_declares_no_dependencies_or_scripts():
    manifest = extension_manifest()

    assert "dependencies" not in manifest and "scripts" not in manifest
    assert manifest["extensionKind"] == ["workspace"]  # runs where the files live (remote)


def test_extension_source_has_no_network_or_process_access():
    for name in ("extension.js", "notes.js", "comments.js"):
        source = (EXTENSION_DIR / name).read_text(encoding="utf-8")
        for forbidden in (
            "require('http",
            'require("http',
            "child_process",
            "net'",
            "fetch(",
            "https",
        ):
            assert forbidden not in source, (name, forbidden)


# -- the JavaScript itself (needs Node) -----------------------------------------------

FAKE_VSCODE_RUN = r"""
const Module = require('module');
const fs = require('fs');
const [,, extensionPath, outPath] = process.argv;

const handlers = {};
const register = (name) => (callback) => { handlers[name] = callback; return { dispose() {} }; };
const state = { focused: true };
const editor = {
  document: {
    uri: { scheme: 'file', fsPath: '/ws/app/a.py' }, languageId: 'python', isDirty: false,
    lineCount: 3, lineAt: (n) => ({ text: ['one', 'two', 'three'][n] }),
  },
  selection: {
    active: { line: 1, character: 4 }, start: { line: 1, character: 4 },
    end: { line: 1, character: 4 }, isEmpty: true,
  },
  visibleRanges: [{ start: { line: 0 }, end: { line: 2 } }],
};
const fake = {
  window: {
    state, activeTextEditor: editor, visibleTextEditors: [editor],
    onDidChangeActiveTextEditor: register('editor'),
    onDidChangeTextEditorSelection: register('selection'),
    onDidChangeTextEditorVisibleRanges: register('visible'),
    onDidChangeWindowState: register('window'),
  },
  workspace: {
    workspaceFolders: [{ uri: { fsPath: '/ws' } }],
    getConfiguration: () => ({ get: (key) => ({ enabled: true, path: outPath })[key] }),
    onDidChangeTextDocument: register('text'),
    onDidSaveTextDocument: register('save'),
    onDidChangeWorkspaceFolders: register('folders'),
  },
};
const originalLoad = Module._load;
Module._load = (request, ...rest) => (request === 'vscode' ? fake : originalLoad(request, ...rest));
const extension = require(extensionPath);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const read = () => JSON.parse(fs.readFileSync(outPath, 'utf8'));

(async () => {
  const results = {};
  const subscriptions = [];
  extension.activate({ subscriptions });
  await sleep(250);
  results.first = read();
  results.mode = (fs.statSync(outPath).mode & 0o777).toString(8);

  editor.selection.active = { line: 2, character: 0 };
  editor.selection.start = { line: 1, character: 0 };
  editor.selection.end = { line: 2, character: 0 };
  editor.selection.isEmpty = false;
  editor.document.isDirty = true;
  handlers.selection();
  await sleep(250);
  results.second = read();

  state.focused = false;
  editor.selection.active = { line: 0, character: 0 };
  handlers.selection();
  await sleep(250);
  results.unfocused = read().cursor;

  state.focused = true;
  fake.window.activeTextEditor = undefined;  // e.g. the terminal has focus, editor stays visible
  handlers.editor();
  await sleep(250);
  results.inactive = read();

  // The input box of a comment thread becomes the active "text editor" (scheme comment).
  const commentEditor = {
    document: { uri: { scheme: 'comment', fsPath: '/comment/1' }, languageId: 'markdown',
      isDirty: false,
      lineCount: 1, lineAt: () => ({ text: 'draft' }) },
    selection: { active: { line: 0, character: 5 }, start: { line: 0, character: 5 },
      end: { line: 0, character: 5 }, isEmpty: true },
    visibleRanges: [{ start: { line: 0 }, end: { line: 0 } }],
  };
  fake.window.activeTextEditor = commentEditor;
  fake.window.visibleTextEditors = [editor, commentEditor];
  handlers.editor();
  await sleep(250);
  results.commentBox = read();

  // Identical state again: the file must not be rewritten.
  const before = fs.statSync(outPath).mtimeMs;
  handlers.editor();
  await sleep(250);
  results.rewritten = fs.statSync(outPath).mtimeMs !== before;

  fake.window.activeTextEditor = undefined;
  fake.window.visibleTextEditors = [];  // the editor was closed
  handlers.editor();
  await sleep(250);
  results.noEditor = read();
  fake.window.visibleTextEditors = [editor];

  fake.window.activeTextEditor = editor;
  handlers.editor();
  await sleep(250);
  subscriptions.forEach((item) => item.dispose());
  results.cleared = read();
  const dir = require('path').dirname(outPath);
  results.leftover = fs.readdirSync(dir).filter((n) => n.endsWith('.tmp'));
  console.log(JSON.stringify(results));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_extension_writes_the_expected_context_with_a_simulated_vscode(tmp_path):
    runner = tmp_path / "run.js"
    runner.write_text(FAKE_VSCODE_RUN, encoding="utf-8")
    out = tmp_path / "home" / "editor-context.json"

    completed = subprocess.run(
        ["node", str(runner), str(EXTENSION_DIR / "extension.js"), str(out)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    results = json.loads(completed.stdout)
    first = results["first"]
    assert first["schema"] == 1 and first["activeFile"] == "/ws/app/a.py"
    assert first["cursor"] == {"line": 2, "character": 5}  # 1-based
    assert first["visibleRange"] == {"startLine": 1, "endLine": 3}
    assert first["activeLineText"] == "two" and first["isDirty"] is False
    assert first["workspaceFolders"] == ["/ws"]
    assert results["mode"] == "600"
    second = results["second"]
    assert second["selection"]["isEmpty"] is False and second["isDirty"] is True
    assert (second["selection"]["startLine"], second["selection"]["endLine"]) == (2, 3)
    assert results["unfocused"] == second["cursor"]  # an unfocused window does not write
    assert first["isActiveEditor"] is True
    inactive = results["inactive"]  # terminal focus: keep the last visible editor, flagged
    assert inactive["activeFile"] == "/ws/app/a.py" and inactive["isActiveEditor"] is False
    comment = results["commentBox"]  # writing in a comment box must not forget the file
    assert comment["activeFile"] == "/ws/app/a.py" and comment["isActiveEditor"] is False
    assert results["rewritten"] is False
    assert results["noEditor"]["activeFile"] is None  # closed editor: nothing stale is kept
    assert results["cleared"]["activeFile"] is None  # disposing clears stale state
    assert results["leftover"] == []
    assert stat.S_IMODE(out.parent.stat().st_mode) == 0o700


# -- installing the extension -----------------------------------------------------------


def test_the_hint_shows_only_inside_vscode_without_a_current_extension(tmp_path):
    from anomx.agent.cowork.extension import extension_manifest, extension_needs_install

    vscode = {"TERM_PROGRAM": "vscode"}
    context = tmp_path / "editor-context.json"
    assert extension_needs_install(tmp_path, vscode) == "missing"
    assert extension_needs_install(tmp_path, {"VSCODE_IPC_HOOK_CLI": "/tmp/x.sock"}) == "missing"
    assert extension_needs_install(tmp_path, {}) == ""  # a plain terminal does not need it
    context.write_text('{"schema": 1}', encoding="utf-8")  # 0.1.x wrote no version
    assert extension_needs_install(tmp_path, vscode) == "outdated"
    context.write_text('{"extensionVersion": "0.1.9"}', encoding="utf-8")
    assert extension_needs_install(tmp_path, vscode) == "outdated"
    version = extension_manifest()["version"]
    context.write_text(json.dumps({"extensionVersion": version}), encoding="utf-8")
    assert extension_needs_install(tmp_path, vscode) == ""


def test_install_builds_the_vsix_and_runs_code(tmp_path, monkeypatch):
    from anomx.agent.cowork import extension

    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(extension.shutil, "which", lambda _name: "/usr/bin/code")
    monkeypatch.setattr(extension.subprocess, "run", fake_run)

    ok, message = extension.install_extension(tmp_path)

    assert ok and "Reload Window" in message
    assert calls[0][:2] == ["/usr/bin/code", "--install-extension"]
    assert Path(calls[0][2]).name == extension.vsix_name() and Path(calls[0][2]).exists()


def test_install_explains_what_to_do_without_the_code_command(tmp_path, monkeypatch):
    from anomx.agent.cowork import extension

    monkeypatch.setattr(extension.shutil, "which", lambda _name: None)

    ok, message = extension.install_extension(tmp_path)

    assert not ok and "Install from VSIX" in message and ".vsix" in message


def test_cli_flag_installs_the_extension(tmp_path, monkeypatch, capsys):
    from anomx import cli
    from anomx.agent.cowork import extension

    monkeypatch.setattr(extension, "install_extension", lambda home: (True, "done"))

    assert cli.main(["--home", str(tmp_path), "--install-vscode"]) == 0
    assert "done" in capsys.readouterr().out


def test_copy_falls_back_to_the_terminal_clipboard_without_tools(monkeypatch):
    import base64
    import io

    from anomx.agent.ui.components import messages

    written = io.StringIO()
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(messages.shutil, "which", lambda _name: None)
    monkeypatch.setattr("builtins.open", lambda *_a, **_k: _Keep(written))

    class _Keep(io.StringIO):
        def __init__(self, target):
            super().__init__()
            self._target = target

        def write(self, text):
            return self._target.write(text)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    assert messages.MessagesComponentMixin()._copy_to_clipboard("anomx --install-vscode")  # noqa: SLF001
    sequence = written.getvalue()
    assert sequence.startswith("\x1b]52;c;") and sequence.endswith("\x07")
    payload = sequence.removeprefix("\x1b]52;c;").removesuffix("\x07")
    assert base64.b64decode(payload).decode() == "anomx --install-vscode"


def test_install_retries_with_a_live_socket_when_the_terminals_is_stale(tmp_path, monkeypatch):
    from anomx.agent.cowork import extension

    envs = []

    def fake_run(command, **kwargs):
        envs.append(kwargs["env"].get("VSCODE_IPC_HOOK_CLI"))
        if kwargs["env"].get("VSCODE_IPC_HOOK_CLI") == "/live.sock":
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 1, "", "Error: connect ENOENT /old.sock")

    monkeypatch.setenv("VSCODE_IPC_HOOK_CLI", "/old.sock")
    monkeypatch.setattr(extension.shutil, "which", lambda _name: "/usr/bin/code")
    monkeypatch.setattr(extension.subprocess, "run", fake_run)
    monkeypatch.setattr(extension, "_live_vscode_sockets", lambda _env: ["/live.sock"])

    ok, _message = extension.install_extension(tmp_path)

    assert ok and envs == ["/old.sock", "/live.sock"]


def test_install_failure_tells_to_open_a_new_terminal(tmp_path, monkeypatch):
    from anomx.agent.cowork import extension

    monkeypatch.setattr(extension.shutil, "which", lambda _name: "/usr/bin/code")
    monkeypatch.setattr(
        extension.subprocess,
        "run",
        lambda command, **_k: subprocess.CompletedProcess(command, 1, "", "Error: ENOENT"),
    )
    monkeypatch.setattr(extension, "_live_vscode_sockets", lambda _env: [])

    ok, message = extension.install_extension(tmp_path)

    assert not ok and "NEW terminal" in message
