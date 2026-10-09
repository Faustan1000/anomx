"""Pure logic of the notes part of the VS Code extension (``cowork/vscode/notes.js``)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from anomx.agent.cowork.extension import EXTENSION_DIR

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")

RUNNER = r"""
const notes = require(process.argv[2]);
const fs = require('fs');
const path = require('path');
const dir = process.argv[3];
const out = {};
const note = (line, endLine) => ({ kind: 'note', line, endLine, file: 'a.py', status: 'open' });
const shift = (n, c) => { const r = notes.shiftRange(n, c); return [r.line, r.endLine]; };

// note covers lines 10-12 (0-based 9-11)
out.above = shift(note(10, 12), { startLine: 2, endLine: 2, newLineCount: 3 });
out.aboveDelete = shift(note(10, 12), { startLine: 2, endLine: 4, newLineCount: 0 });
out.below = shift(note(10, 12), { startLine: 20, endLine: 20, newLineCount: 5 });
out.insideGrow = shift(note(10, 12), { startLine: 10, endLine: 10, newLineCount: 2 });
out.insideShrink = shift(note(10, 12), { startLine: 10, endLine: 11, newLineCount: 0 });
out.sameLineTyping = shift(note(10, 12), { startLine: 9, endLine: 9, newLineCount: 0 });
out.enterAtColumnZero = shift(note(10, 10), { startLine: 9, endLine: 9, startCharacter: 0, newLineCount: 1 });
out.enterInsideFirstLine = shift(note(10, 10), { startLine: 9, endLine: 9, startCharacter: 4, newLineCount: 1 });
out.spanningDelete = shift(note(10, 12), { startLine: 7, endLine: 14, newLineCount: 0 });
out.topOverlap = shift(note(10, 14), { startLine: 7, endLine: 10, newLineCount: 0 });
out.noteIsNotModified = (() => { const n = note(10, 12); notes.shiftRange(n, { startLine: 0, endLine: 0, newLineCount: 4 }); return [n.line, n.endLine]; })();

const list = [
  { ...note(10, 12), id: 'a' },
  { ...note(10, 12), id: 'other', file: 'b.py' },
  { ...note(30, 30), id: 'c' },
];
out.moved = notes.applyChanges(list, 'a.py', [
  { startLine: 2, endLine: 2, newLineCount: 1 },
  { startLine: 20, endLine: 20, newLineCount: 1 },
]);
out.applied = list.map((n) => [n.id, n.line, n.endLine]);

// files
const folder = path.join(dir, 'ws');
out.emptyRead = notes.readNotes(folder);
const created = notes.createNote({ kind: 'note', file: 'a.py', line: 3, anchorText: 'x = 1', text: 'make it a dataclass' }, new Date('2026-10-09T08:00:00Z'));
const question = notes.createNote({ kind: 'question', text: 'Where is the GUI built?', contextFile: '/ws/a.py', file: 'ignored', line: 9 });
notes.writeNotes(folder, [created, question]);
out.written = notes.readNotes(folder);
out.mode = (fs.statSync(notes.notesPath(folder)).mode & 0o777).toString(8);
out.ignore = fs.readFileSync(path.join(folder, '.anomx', '.gitignore'), 'utf8');
out.leftover = fs.readdirSync(path.join(folder, '.anomx')).filter((n) => n.endsWith('.tmp'));
fs.writeFileSync(notes.notesPath(folder), '{"notes": [');
out.corrupt = notes.readNotes(folder);
fs.writeFileSync(notes.notesPath(folder), JSON.stringify({ schema: 1, notes: [created, { id: 5 }, { id: 'x', kind: 'weird' }, 'junk'] }));
out.filtered = notes.readNotes(folder).map((n) => n.id);
out.noAgent = notes.readAgent(folder);
fs.writeFileSync(notes.agentPath(folder), 'not json');
out.badAgent = notes.readAgent(folder);

// view model
const m1 = created.messages[0];
const reply = { id: 'm-late', role: 'user', text: 'and add a test', at: '2026-10-09T09:00:00.000Z' };
const withReply = { ...created, messages: [m1, reply] };
const entry = {
  status: 'ready', detail: 'ignored when ready', pickedUp: [m1.id],
  messages: [{ id: 'a-1', text: 'Done: see app.py:12-20', at: '2026-10-09T08:30:00.000Z', locations: [{ file: 'app.py', line: 12, endLine: 20 }, { file: '', line: 0 }] }],
};
const view = (n, e) => { const v = notes.viewModel(n, e); return { status: v.status, pending: v.pending, roles: v.messages.map((m) => m.role), detail: v.detail, title: v.title }; };
out.viewReady = view(created, entry);
out.viewPending = view(withReply, entry);
out.viewNoEntry = view(created, undefined);
out.viewWorking = view(created, { status: 'working', detail: 'reading app.py', pickedUp: [m1.id] });
out.viewFailed = view(created, { status: 'failed', detail: 'boom', pickedUp: [m1.id] });
out.viewResolved = view({ ...created, status: 'resolved' }, entry);
out.locations = notes.viewModel(created, entry).messages[1].locations;
out.title = notes.titleOf({ messages: [{ text: '\n\n' + 'x'.repeat(200) }] }).length;
out.format = [notes.formatLocation({ file: 'a.py', line: 3 }), notes.formatLocation({ file: 'a.py', line: 3, endLine: 9 })];
out.found = notes.findLocations('see src/ui/app.py:12-30, and cfg.toml:4. Not v1.2 or a:0').map((f) => [f.file, f.line, f.endLine]);
out.rendered = notes.renderAnswer(notes.viewModel(created, entry));

// Margo import
const vault = { './anomx/src/x.py': [
  { id: '1', line: 4, title: 'rename', text: 'to foo', status: 'open' },
  { id: '2', line: 0, title: 'done', text: '', status: 'resolved' },
  { id: '3', line: -1, title: 'bad' }, { id: '4', line: 7, title: '', text: '' } ] };
const imported = notes.importMargoNotes([vault, null], []);
out.imported = imported.map((n) => [n.file, n.line, n.endLine, n.importedFrom, n.messages[0].text, n.kind]);
out.importedIds = new Set(imported.map((n) => n.id)).size;
out.again = notes.importMargoNotes([vault], imported).length;
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("notes")
    runner = tmp / "run.js"
    runner.write_text(RUNNER, encoding="utf-8")
    completed = subprocess.run(
        ["node", str(runner), str(EXTENSION_DIR / "notes.js"), str(tmp)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_shift_range_follows_edits_above_inside_and_below(result):
    assert result["above"] == [13, 15]  # three lines inserted above
    assert result["aboveDelete"] == [8, 10]  # two lines removed above
    assert result["below"] == [10, 12]
    assert result["insideGrow"] == [10, 14]
    assert result["insideShrink"] == [10, 11]
    assert result["sameLineTyping"] == [10, 12]
    assert result["noteIsNotModified"] == [10, 12]


def test_shift_range_distinguishes_enter_at_column_zero(result):
    assert result["enterAtColumnZero"] == [11, 11]  # the line moved down
    assert result["enterInsideFirstLine"] == [10, 11]  # the note now spans both halves


def test_shift_range_when_lines_of_the_note_are_deleted(result):
    assert result["spanningDelete"] == [8, 8]  # collapses onto the edit
    assert result["topOverlap"] == [8, 11]  # the top is gone, the rest moves up


def test_apply_changes_uses_original_coordinates_and_only_the_matching_file(result):
    assert result["moved"] == ["a", "c"]
    assert result["applied"] == [["a", 11, 13], ["other", 10, 12], ["c", 32, 32]]


def test_notes_file_roundtrip_is_private_atomic_and_tolerant(result):
    assert result["emptyRead"] == []
    written = {n["kind"]: n for n in result["written"]}
    assert written["note"]["file"] == "a.py" and written["note"]["line"] == 3
    assert written["note"]["endLine"] == 3 and written["note"]["anchorText"] == "x = 1"
    assert written["note"]["messages"][0]["text"] == "make it a dataclass"
    question = written["question"]  # a question never carries a code position
    assert question["file"] is None and question["line"] is None
    assert question["contextFile"] == "/ws/a.py"
    assert result["mode"] == "600" and result["ignore"] == "*\n" and result["leftover"] == []
    assert result["corrupt"] == []
    assert len(result["filtered"]) == 1
    assert result["noAgent"] == {} and result["badAgent"] == {}


def test_view_model_status_pending_and_order(result):
    assert result["viewReady"]["status"] == "ready" and result["viewReady"]["pending"] == []
    assert result["viewReady"]["roles"] == ["user", "agent"]
    assert result["viewReady"]["detail"] == "" and result["viewReady"]["title"] == "make it a dataclass"
    pending = result["viewPending"]  # a reply the agent has not picked up yet
    assert pending["status"] == "waiting" and pending["pending"] == ["m-late"]
    assert pending["roles"] == ["user", "agent", "user"]
    assert result["viewNoEntry"]["status"] == "waiting"
    assert result["viewWorking"]["status"] == "working"
    assert result["viewWorking"]["detail"] == "reading app.py"
    assert result["viewFailed"]["status"] == "failed" and result["viewFailed"]["detail"] == "boom"
    assert result["viewResolved"]["status"] == "resolved"
    assert result["locations"] == [{"file": "app.py", "line": 12, "endLine": 20}]  # bad one dropped
    assert result["title"] == 80


def test_locations_are_formatted_found_in_text_and_rendered(result):
    assert result["format"] == ["a.py:3", "a.py:3-9"]
    assert result["found"] == [["src/ui/app.py", 12, 30], ["cfg.toml", 4, None]]
    rendered = result["rendered"]
    assert "## Anomx agent" in rendered and "- app.py:12-20" in rendered


def test_margo_import_maps_open_notes_once(result):
    assert result["imported"] == [
        ["anomx/src/x.py", 5, 5, "margo:1", "rename\n\nto foo", "note"]  # line is 0-based in Margo
    ]
    assert result["importedIds"] == 1
    assert result["again"] == 0


def test_extension_files_stay_small():
    for name, limit in (("notes.js", 450), ("comments.js", 650)):
        lines = (Path(EXTENSION_DIR) / name).read_text(encoding="utf-8").count("\n")
        assert lines < limit, (name, lines)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_ask_view_html_is_locked_down_and_posts_only_the_text():
    script = (
        "const n = require(process.argv[1]);"
        "process.stdout.write(n.askViewHtml('abc123'));"
    )
    html = subprocess.run(
        ["node", "-e", script, str(Path(EXTENSION_DIR) / "notes.js")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert "default-src 'none'" in html and "script-src 'nonce-abc123'" in html
    assert '<script nonce="abc123">' in html
    assert "src=" not in html and "href=" not in html and "fetch(" not in html
    assert "postMessage({ type: 'ask', text })" in html
