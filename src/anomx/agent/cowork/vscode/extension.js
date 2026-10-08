'use strict';

// Anomx Editor Context
//
// Writes what the user is looking at to ONE local JSON file (mode 0600):
//   active file, cursor, selection range, visible lines, dirty flag, language and the text
//   of the cursor line. It does not read other files, open ports, use the network, call
//   other processes or send selected text. The Anomx CLI agent reads the file.
//
// The logic lives in plain functions (no `vscode` types) so it can be tested with Node.

const fs = require('fs');
const os = require('os');
const path = require('path');

const SCHEMA = 1;
const DEBOUNCE_MS = 100;
const MAX_LINE_TEXT = 300;

function defaultContextPath(env = process.env, home = os.homedir()) {
  const root = env.ANOMX_HOME ? env.ANOMX_HOME : path.join(home, '.anomx');
  return path.join(root, 'editor-context.json');
}

// Positions arrive 0-based from VS Code; the file uses 1-based lines and characters like
// editors display them.
function oneBased(position) {
  return { line: position.line + 1, character: position.character + 1 };
}

function buildContext(snapshot, now = new Date(), pid = process.pid) {
  const base = {
    schema: SCHEMA,
    updatedAt: now.toISOString(),
    pid,
    workspaceFolders: snapshot.workspaceFolders || [],
    activeFile: null,
    languageId: null,
    isDirty: false,
    cursor: null,
    selection: null,
    visibleRange: null,
    activeLineText: null,
  };
  const editor = snapshot.editor;
  if (!editor || editor.scheme !== 'file') {
    return base;
  }
  const active = oneBased(editor.selection.active);
  const start = oneBased(editor.selection.start);
  const end = oneBased(editor.selection.end);
  const ranges = editor.visibleRanges || [];
  return {
    ...base,
    activeFile: editor.file,
    languageId: editor.languageId,
    isDirty: Boolean(editor.isDirty),
    cursor: active,
    selection: {
      startLine: start.line,
      startCharacter: start.character,
      endLine: end.line,
      endCharacter: end.character,
      isEmpty: Boolean(editor.selection.isEmpty),
    },
    visibleRange: ranges.length
      ? { startLine: ranges[0].start + 1, endLine: ranges[ranges.length - 1].end + 1 }
      : null,
    activeLineText: String(editor.activeLineText || '').slice(0, MAX_LINE_TEXT),
  };
}

// Atomic replace so a reader never sees a half-written file.
function writeContextFile(filePath, context) {
  const dir = path.dirname(filePath);
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const tmp = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(context, null, 2) + '\n', { mode: 0o600 });
  fs.renameSync(tmp, filePath);
}

function takeSnapshot(vscode) {
  const editor = vscode.window.activeTextEditor;
  const folders = (vscode.workspace.workspaceFolders || []).map((folder) => folder.uri.fsPath);
  if (!editor) {
    return { workspaceFolders: folders, editor: null };
  }
  const { document, selection } = editor;
  const lineIndex = Math.min(selection.active.line, Math.max(document.lineCount - 1, 0));
  return {
    workspaceFolders: folders,
    editor: {
      scheme: document.uri.scheme,
      file: document.uri.fsPath,
      languageId: document.languageId,
      isDirty: document.isDirty,
      selection: {
        active: selection.active,
        start: selection.start,
        end: selection.end,
        isEmpty: selection.isEmpty,
      },
      visibleRanges: editor.visibleRanges.map((range) => ({
        start: range.start.line,
        end: range.end.line,
      })),
      activeLineText: document.lineCount ? document.lineAt(lineIndex).text : '',
    },
  };
}

function activate(extensionContext) {
  // Loaded only inside VS Code; kept out of the top level so tests can require this file.
  const vscode = require('vscode');
  let timer = null;
  let wroteLast = false;

  const settings = () => vscode.workspace.getConfiguration('anomxEditorContext');
  const targetPath = () => settings().get('path') || defaultContextPath();

  function write() {
    timer = null;
    // Several windows share one file: only the focused window writes, otherwise they
    // would overwrite each other with stale state.
    if (!settings().get('enabled') || !vscode.window.state.focused) {
      return;
    }
    try {
      writeContextFile(targetPath(), buildContext(takeSnapshot(vscode)));
      wroteLast = true;
    } catch (error) {
      console.error('anomx-editor-context: could not write context file', error);
    }
  }

  function schedule() {
    if (timer === null) {
      timer = setTimeout(write, DEBOUNCE_MS);
    }
  }

  extensionContext.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(schedule),
    vscode.window.onDidChangeTextEditorSelection(schedule),
    vscode.window.onDidChangeTextEditorVisibleRanges(schedule),
    vscode.window.onDidChangeWindowState(schedule),
    vscode.workspace.onDidChangeTextDocument(schedule),
    vscode.workspace.onDidSaveTextDocument(schedule),
    vscode.workspace.onDidChangeWorkspaceFolders(schedule),
    { dispose: () => timer !== null && clearTimeout(timer) },
    {
      // A closed window must not leave its last cursor behind as if it were current.
      dispose: () => {
        if (wroteLast) {
          try {
            writeContextFile(targetPath(), buildContext({ workspaceFolders: [], editor: null }));
          } catch (error) {
            console.error('anomx-editor-context: could not clear context file', error);
          }
        }
      },
    },
  );
  schedule();
}

function deactivate() {}

module.exports = { activate, deactivate, buildContext, defaultContextPath, writeContextFile };
