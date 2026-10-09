'use strict';

// Anomx notes: pure logic, no `vscode` types, so it can be tested with Node.
//
// The extension owns `<workspace folder>/.anomx/notes.json`; Anomx (the CLI) owns
// `.anomx/agent.json`. Each file has one writer and is replaced atomically. The format is
// described in README.md.

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const SCHEMA = 1;
const DIR_NAME = '.anomx';
const NOTES_FILE = 'notes.json';
const AGENT_FILE = 'agent.json';
const KINDS = new Set(['note', 'question']);
const MAX_TITLE = 80;

// -- files --------------------------------------------------------------------------

function notesPath(folder) {
  return path.join(folder, DIR_NAME, NOTES_FILE);
}

function agentPath(folder) {
  return path.join(folder, DIR_NAME, AGENT_FILE);
}

function readJson(file) {
  try {
    const data = JSON.parse(fs.readFileSync(file, 'utf8'));
    return data && typeof data === 'object' && !Array.isArray(data) ? data : null;
  } catch (error) {
    return null; // missing, half-written or corrupt: treated as empty
  }
}

function cleanNote(raw) {
  if (!raw || typeof raw !== 'object' || typeof raw.id !== 'string' || !KINDS.has(raw.kind)) {
    return null;
  }
  const messages = Array.isArray(raw.messages)
    ? raw.messages.filter((m) => m && typeof m.id === 'string' && typeof m.text === 'string')
    : [];
  return { ...raw, status: raw.status === 'resolved' ? 'resolved' : 'open', messages };
}

function readNotes(folder) {
  const data = readJson(notesPath(folder));
  const raw = data && Array.isArray(data.notes) ? data.notes : [];
  return raw.map(cleanNote).filter(Boolean);
}

function readAgent(folder) {
  const data = readJson(agentPath(folder));
  const entries = data && data.notes && typeof data.notes === 'object' ? data.notes : {};
  return entries;
}

// Atomic replace so a reader never sees a half-written file.
function writeNotes(folder, notes) {
  const dir = path.join(folder, DIR_NAME);
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const ignore = path.join(dir, '.gitignore');
  if (!fs.existsSync(ignore)) {
    fs.writeFileSync(ignore, '*\n', { mode: 0o600 });
  }
  const target = notesPath(folder);
  const tmp = `${target}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify({ schema: SCHEMA, notes }, null, 2) + '\n', {
    mode: 0o600,
  });
  fs.renameSync(tmp, target);
}

// -- ids and notes ------------------------------------------------------------------

function newId(prefix, now = Date.now(), random = crypto.randomBytes(3).toString('hex')) {
  return `${prefix}-${now}-${random}`;
}

function newMessage(text, now = new Date()) {
  return { id: newId('m', now.getTime()), role: 'user', text, at: now.toISOString() };
}

function createNote(fields, now = new Date()) {
  const isQuestion = fields.kind === 'question';
  return {
    id: newId('n', now.getTime()),
    kind: isQuestion ? 'question' : 'note',
    file: isQuestion ? null : fields.file,
    line: isQuestion ? null : fields.line,
    endLine: isQuestion ? null : fields.endLine == null ? fields.line : fields.endLine,
    anchorText: isQuestion ? '' : String(fields.anchorText || ''),
    contextFile: isQuestion ? fields.contextFile || null : null,
    status: 'open',
    createdAt: now.toISOString(),
    messages: [newMessage(fields.text, now)],
    ...(fields.importedFrom ? { importedFrom: fields.importedFrom } : {}),
  };
}

// -- range tracking -----------------------------------------------------------------

// `change` describes one VS Code content change in 0-based lines of the document as it was
// before the edit: lines startLine..endLine were replaced by text that contains
// `newLineCount` line breaks. `startCharacter` is optional; it only tells "Enter at column
// 0" (the note moves down) from "Enter inside the first line" (the note stays).
// Returns the note's new { line, endLine } (1-based); the note itself is not modified.
function shiftRange(note, change) {
  const first = note.line - 1;
  const last = (note.endLine == null ? note.line : note.endLine) - 1;
  const removed = change.endLine - change.startLine;
  const delta = change.newLineCount - removed;
  const keep = { line: first + 1, endLine: last + 1 };
  if (change.startLine > last || (delta === 0 && change.startLine >= first)) {
    return keep; // below the note (or an edit inside the lines that adds/removes none)
  }
  if (change.startLine < first) {
    if (change.endLine < first) {
      return { line: first + delta + 1, endLine: last + delta + 1 }; // wholly above
    }
    if (change.endLine >= last) {
      // The note's lines were deleted or replaced: stay at the edit.
      const anchor = change.startLine + 1;
      return { line: anchor, endLine: anchor };
    }
    // The edit covers the top of the note: it starts after the replacement text.
    const top = change.startLine + change.newLineCount;
    return { line: top + 1, endLine: Math.max(top, last + delta) + 1 };
  }
  // The edit starts on a line of the note.
  if (
    change.startLine === first &&
    change.startCharacter === 0 &&
    removed === 0 &&
    change.newLineCount > 0
  ) {
    return { line: first + delta + 1, endLine: last + delta + 1 }; // Enter at column 0
  }
  return { line: first + 1, endLine: Math.max(first, last + delta) + 1 };
}

// Apply content changes of one document to the notes of that file. Changes of one VS Code
// event are expressed in the old document, so they are applied from the bottom up.
// Returns the ids of the notes that moved.
function applyChanges(notes, file, changes) {
  const ordered = [...changes].sort((a, b) => b.startLine - a.startLine);
  const moved = [];
  for (const note of notes) {
    if (note.kind !== 'note' || note.file !== file || note.status !== 'open') {
      continue;
    }
    const before = `${note.line}-${note.endLine}`;
    for (const change of ordered) {
      Object.assign(note, shiftRange(note, change));
    }
    if (`${note.line}-${note.endLine}` !== before) {
      moved.push(note.id);
    }
  }
  return moved;
}

// -- view model ---------------------------------------------------------------------

function titleOf(note) {
  const first = note.messages[0] ? note.messages[0].text : '';
  const line = first.split('\n').find((part) => part.trim()) || '(empty)';
  return line.length > MAX_TITLE ? `${line.slice(0, MAX_TITLE - 1)}…` : line.trim();
}

function time(message) {
  const parsed = Date.parse(message.at);
  return Number.isNaN(parsed) ? 0 : parsed;
}

const STATUS_TEXT = {
  waiting: 'waiting for the agent',
  working: 'agent is working',
  ready: 'answer ready',
  failed: 'failed',
  merging: 'change waiting in Source Control',
  merged: 'merged',
  discarded: 'discarded',
  resolved: 'resolved',
};

// One entry per note: status, pending messages and the thread messages in time order.
function viewModel(note, entry) {
  const agent = entry && typeof entry === 'object' ? entry : {};
  const pickedUp = new Set(Array.isArray(agent.pickedUp) ? agent.pickedUp : []);
  const pending = note.messages.filter((m) => !pickedUp.has(m.id)).map((m) => m.id);
  const replies = (Array.isArray(agent.messages) ? agent.messages : []).filter(
    (m) => m && typeof m.text === 'string',
  );
  let status;
  if (note.status === 'resolved') {
    status = 'resolved';
  } else if (agent.status === 'working') {
    status = 'working';
  } else if (pending.length > 0 || !agent.status || agent.status === 'idle') {
    status = 'waiting';
  } else {
    status = STATUS_TEXT[agent.status] ? agent.status : 'ready';
  }
  const messages = [
    ...note.messages.map((m) => ({ ...m, role: 'user' })),
    ...replies.map((m) => ({ ...m, role: 'agent', locations: normalizeLocations(m.locations) })),
  ]
    .map((m, index) => ({ m, index }))
    .sort((a, b) => time(a.m) - time(b.m) || a.index - b.index)
    .map(({ m }) => m);
  const detail = typeof agent.detail === 'string' ? agent.detail : '';
  return {
    id: note.id,
    kind: note.kind,
    file: note.file,
    line: note.line,
    endLine: note.endLine,
    title: titleOf(note),
    status,
    statusText: STATUS_TEXT[status],
    detail: status === 'working' || status === 'failed' ? detail : '',
    pending,
    messages,
  };
}

function normalizeLocations(raw) {
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw
    .filter((l) => l && typeof l.file === 'string' && Number.isInteger(l.line) && l.line > 0)
    .map((l) => ({
      file: l.file,
      line: l.line,
      endLine: Number.isInteger(l.endLine) && l.endLine > l.line ? l.endLine : null,
    }));
}

function formatLocation(location) {
  const end = location.endLine && location.endLine > location.line ? `-${location.endLine}` : '';
  return `${location.file}:${location.line}${end}`;
}

// `path/to/file.py:12` or `file.py:12-30` inside free text; offsets are for DocumentLinks.
const LOCATION_PATTERN = /([\w@.\-/]+\.[A-Za-z0-9]+):(\d+)(?:-(\d+))?/g;

function findLocations(text) {
  const found = [];
  for (const match of String(text).matchAll(LOCATION_PATTERN)) {
    const line = Number(match[2]);
    if (line < 1) {
      continue;
    }
    found.push({
      start: match.index,
      end: match.index + match[0].length,
      file: match[1],
      line,
      endLine: match[3] ? Number(match[3]) : null,
    });
  }
  return found;
}

// Markdown of the virtual "answer" document of a question.
function renderAnswer(model) {
  const lines = [`# ${model.title}`, '', `Status: ${model.statusText}${model.detail ? ` (${model.detail})` : ''}`, ''];
  const locations = [];
  for (const message of model.messages) {
    lines.push(message.role === 'agent' ? '## Anomx agent' : '## You', '', message.text.trim(), '');
    for (const location of message.locations || []) {
      if (!locations.some((l) => formatLocation(l) === formatLocation(location))) {
        locations.push(location);
      }
    }
  }
  if (locations.length > 0) {
    lines.push('## Locations', '', ...locations.map((l) => `- ${formatLocation(l)}`), '');
  }
  return lines.join('\n');
}

// -- Margo import -------------------------------------------------------------------

function normalizeMargoPath(file) {
  let cleaned = String(file).replace(/\\/g, '/');
  while (cleaned.startsWith('./')) {
    cleaned = cleaned.slice(2);
  }
  return cleaned;
}

// Notes to add for the given Margo vaults ({"./rel/path": [{id, line (0-based), title, text,
// status}]}). Resolved notes and notes imported earlier (`importedFrom`) are skipped.
function importMargoNotes(vaults, existing, now = new Date()) {
  const known = new Set(existing.map((n) => n.importedFrom).filter(Boolean));
  const created = [];
  let offset = 0;
  for (const vault of vaults) {
    if (!vault || typeof vault !== 'object') {
      continue;
    }
    for (const [file, entries] of Object.entries(vault)) {
      for (const entry of Array.isArray(entries) ? entries : []) {
        if (!entry || typeof entry.id !== 'string' || !Number.isInteger(entry.line)) {
          continue;
        }
        const source = `margo:${entry.id}`;
        if (entry.status === 'resolved' || entry.line < 0 || known.has(source)) {
          continue;
        }
        const text = `${entry.title || ''}\n\n${entry.text || ''}`.trim();
        if (!text) {
          continue;
        }
        known.add(source);
        const note = createNote(
          {
            kind: 'note',
            file: normalizeMargoPath(file),
            line: entry.line + 1,
            endLine: entry.line + 1,
            text,
            importedFrom: source,
          },
          new Date(now.getTime() + offset),
        );
        offset += 1; // distinct ids even within one millisecond
        created.push(note);
      }
    }
  }
  return created;
}

// HTML of the "Ask" field in the sidebar: a text box and a button, like the commit box of
// Source Control. It has no external resources; the only thing it does is post the text
// to the extension. Enter sends, Shift+Enter adds a line.
function askViewHtml(nonce) {
  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>
  body { padding: 6px 12px 8px; }
  textarea { box-sizing: border-box; width: 100%; min-height: 54px; resize: vertical;
    color: var(--vscode-input-foreground); background: var(--vscode-input-background);
    border: 1px solid var(--vscode-input-border, transparent); border-radius: 2px;
    padding: 4px 6px; font: inherit; }
  textarea:focus { outline: 1px solid var(--vscode-focusBorder); outline-offset: -1px; }
  button { box-sizing: border-box; width: 100%; margin-top: 6px; padding: 4px 8px; cursor: pointer;
    color: var(--vscode-button-foreground); background: var(--vscode-button-background);
    border: 0; border-radius: 2px; font: inherit; }
  button:hover { background: var(--vscode-button-hoverBackground); }
  button:disabled { opacity: .5; cursor: default; }
</style></head><body>
<textarea id="q" rows="2" placeholder="Ask about the code, e.g. Where is the GUI built? (Enter to send)"></textarea>
<button id="go" disabled>Ask the agent</button>
<script nonce="${nonce}">
  const vscode = acquireVsCodeApi();
  const box = document.getElementById('q');
  const go = document.getElementById('go');
  const send = () => {
    const text = box.value.trim();
    if (text) { vscode.postMessage({ type: 'ask', text }); box.value = ''; go.disabled = true; }
  };
  box.addEventListener('input', () => { go.disabled = box.value.trim() === ''; });
  box.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send(); }
  });
  go.addEventListener('click', send);
  window.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'focus') { box.focus(); }
  });
</script></body></html>`;
}

module.exports = {
  AGENT_FILE,
  askViewHtml,
  NOTES_FILE,
  agentPath,
  applyChanges,
  createNote,
  findLocations,
  formatLocation,
  importMargoNotes,
  newId,
  newMessage,
  normalizeLocations,
  notesPath,
  readAgent,
  readNotes,
  renderAnswer,
  shiftRange,
  titleOf,
  viewModel,
  writeNotes,
};
