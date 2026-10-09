'use strict';

// Anomx notes in VS Code: comment threads on code, general questions, a sidebar list.
//
// Everything is stored in `<first workspace folder>/.anomx/notes.json` (written here) and
// answered through `.anomx/agent.json` (written by Anomx). No network, no ports, no child
// processes. Files other than those two are read only by the explicit "Import Margo notes"
// command and by "open location", which opens a file in the editor.

const fs = require('fs');
const path = require('path');
const notes = require('./notes');

const ANSWER_SCHEME = 'anomx-answer';
const OPEN_LOCATION = 'anomx.openLocation';
const PERSIST_DELAY_MS = 1000;
const POLL_MS = 5000;
const AGENT_NAME = 'Anomx agent';

const ICONS = {
  working: 'sync~spin',
  ready: 'check',
  failed: 'error',
  waiting: 'clock',
  merging: 'git-merge',
  merged: 'git-merge',
  discarded: 'trash',
};

function commandLink(location) {
  const args = encodeURIComponent(JSON.stringify([location]));
  return `command:${OPEN_LOCATION}?${args}`;
}

function activate(extensionContext, vscode) {
  if (!vscode.comments || !vscode.window.createTreeView) {
    return; // not a full VS Code (tests of the context writer use a reduced fake)
  }
  if (!vscode.workspace.getConfiguration('anomxNotes').get('enabled', true)) {
    return;
  }
  const folders = vscode.workspace.workspaceFolders || [];
  const folder = folders.length ? folders[0].uri.fsPath : null;
  const subscriptions = extensionContext.subscriptions;

  // -- state -------------------------------------------------------------------------
  let current = folder ? notes.readNotes(folder) : [];
  let agent = folder ? notes.readAgent(folder) : {};
  let signature = '';
  let persistTimer = null;
  const threads = new Map(); // note id -> CommentThread
  const noteOfThread = new Map(); // CommentThread -> note id
  const treeChanged = new vscode.EventEmitter();
  const answerChanged = new vscode.EventEmitter();

  const relative = (uri) => {
    if (!folder || uri.scheme !== 'file') {
      return null;
    }
    const rel = path.relative(folder, uri.fsPath);
    return rel && !rel.startsWith('..') && !path.isAbsolute(rel) ? rel.split(path.sep).join('/') : null;
  };
  const models = () =>
    current.filter((n) => n.status === 'open').map((n) => notes.viewModel(n, agent[n.id]));
  const modelOf = (id) => models().find((m) => m.id === id);

  // Read-modify-write on fresh data so edits by another window are not lost.
  function mutate(change) {
    if (!folder) {
      return;
    }
    flush();
    const fresh = notes.readNotes(folder);
    change(fresh);
    try {
      notes.writeNotes(folder, fresh);
    } catch (error) {
      vscode.window.showErrorMessage(`Anomx: could not save notes (${error.message})`);
      return;
    }
    refresh(true);
  }

  // Range changes are kept in memory first and written in one go.
  function schedulePersist() {
    if (persistTimer === null) {
      persistTimer = setTimeout(flush, PERSIST_DELAY_MS);
    }
  }
  function flush() {
    if (persistTimer !== null) {
      clearTimeout(persistTimer);
      persistTimer = null;
    }
    if (!folder) {
      return;
    }
    const fresh = notes.readNotes(folder);
    let changed = false;
    for (const note of fresh) {
      const mine = current.find((n) => n.id === note.id);
      if (mine && note.kind === 'note' && (mine.line !== note.line || mine.endLine !== note.endLine)) {
        note.line = mine.line;
        note.endLine = mine.endLine;
        changed = true;
      }
    }
    if (changed) {
      try {
        notes.writeNotes(folder, fresh);
      } catch (error) {
        console.error('anomx notes: could not save ranges', error);
      }
    }
  }

  // -- comment threads ---------------------------------------------------------------
  const controller = vscode.comments.createCommentController('anomx', 'Anomx');
  controller.commentingRangeProvider = {
    provideCommentingRanges(document) {
      if (document.uri.scheme !== 'file' || relative(document.uri) === null) {
        return [];
      }
      return [new vscode.Range(0, 0, Math.max(document.lineCount - 1, 0), 0)];
    },
  };
  subscriptions.push(controller);

  const rangeOf = (note) =>
    new vscode.Range(
      Math.max(note.line - 1, 0),
      0,
      Math.max((note.endLine || note.line) - 1, 0),
      0,
    );

  function commentsOf(model) {
    return model.messages.map((message) => {
      if (message.role === 'user') {
        return {
          body: message.text,
          mode: vscode.CommentMode.Preview,
          author: { name: 'You' },
          timestamp: new Date(message.at),
        };
      }
      const links = (message.locations || [])
        .map((location) => `[${notes.formatLocation(location)}](${commandLink(location)})`)
        .join('  \n');
      const body = new vscode.MarkdownString(links ? `${message.text}\n\n${links}` : message.text);
      body.isTrusted = { enabledCommands: [OPEN_LOCATION] };
      return {
        body,
        mode: vscode.CommentMode.Preview,
        author: { name: AGENT_NAME },
        timestamp: new Date(message.at),
      };
    });
  }

  function labelOf(model) {
    return `Anomx · ${model.statusText}${model.detail ? `: ${model.detail}` : ''}`;
  }

  function syncThreads() {
    const wanted = new Map();
    for (const model of models()) {
      if (model.kind === 'note') {
        wanted.set(model.id, model);
      }
    }
    for (const [id, thread] of threads) {
      if (!wanted.has(id)) {
        noteOfThread.delete(thread);
        thread.dispose();
        threads.delete(id);
      }
    }
    for (const note of current) {
      const model = wanted.get(note.id);
      if (!model || !note.file) {
        continue;
      }
      let thread = threads.get(note.id);
      if (!thread) {
        const uri = vscode.Uri.file(path.join(folder, note.file));
        thread = controller.createCommentThread(uri, rangeOf(note), []);
        thread.canReply = true;
        thread.collapsibleState = vscode.CommentThreadCollapsibleState.Collapsed;
        threads.set(note.id, thread);
        noteOfThread.set(thread, note.id);
      } else {
        thread.range = rangeOf(note);
      }
      thread.label = labelOf(model);
      thread.comments = commentsOf(model);
    }
  }

  // -- refresh -----------------------------------------------------------------------
  function fileSignature(file) {
    try {
      const stat = fs.statSync(file);
      return `${stat.mtimeMs}:${stat.size}`;
    } catch (error) {
      return 'none';
    }
  }

  function refresh(force) {
    if (!folder) {
      return;
    }
    const next = `${fileSignature(notes.notesPath(folder))}|${fileSignature(notes.agentPath(folder))}`;
    if (!force && next === signature) {
      return;
    }
    signature = next;
    flush();
    current = notes.readNotes(folder);
    agent = notes.readAgent(folder);
    syncThreads();
    treeChanged.fire();
    answerChanged.fire();
  }

  // -- creating and answering notes --------------------------------------------------
  function noteIdFrom(arg) {
    if (typeof arg === 'string') {
      return arg;
    }
    if (arg && typeof arg.noteId === 'string') {
      return arg.noteId; // tree item
    }
    if (arg && noteOfThread.has(arg)) {
      return noteOfThread.get(arg); // comment thread
    }
    if (arg && arg.thread && noteOfThread.has(arg.thread)) {
      return noteOfThread.get(arg.thread); // comment reply
    }
    return null;
  }

  // "Send" button of a comment box: the first comment creates the note, later ones reply.
  function submitComment(reply) {
    const text = String(reply && reply.text ? reply.text : '').trim();
    const thread = reply && reply.thread;
    if (!thread || !text) {
      return;
    }
    const existing = noteOfThread.get(thread);
    if (existing) {
      mutate((all) => {
        const note = all.find((n) => n.id === existing);
        if (note) {
          note.messages.push(notes.newMessage(text));
        }
      });
      return;
    }
    const file = relative(thread.uri);
    if (file === null || !thread.range) {
      vscode.window.showWarningMessage('Anomx notes only work on files inside the workspace folder.');
      thread.dispose();
      return;
    }
    const start = thread.range.start.line;
    let end = thread.range.end.line;
    if (end > start && thread.range.end.character === 0) {
      end -= 1; // a selection that ends at column 0 does not include that line
    }
    const document = vscode.workspace.textDocuments.find((d) => d.uri.toString() === thread.uri.toString());
    const created = notes.createNote({
      kind: 'note',
      file,
      line: start + 1,
      endLine: end + 1,
      anchorText: document && start < document.lineCount ? document.lineAt(start).text : '',
      text,
    });
    // The thread VS Code created is replaced by the one built from the stored note.
    thread.dispose();
    mutate((all) => all.push(created));
    const made = threads.get(created.id);
    if (made) {
      made.collapsibleState = vscode.CommentThreadCollapsibleState.Expanded;
    }
  }

  // The editor that was active before focus moved to the sidebar tells which repository a
  // question is about.
  let lastFileEditor = vscode.window.activeTextEditor;
  subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      if (editor && editor.document.uri.scheme === 'file') {
        lastFileEditor = editor;
      }
    }),
  );

  async function createQuestion(text) {
    if (!folder) {
      vscode.window.showWarningMessage('Anomx: open a folder first.');
      return;
    }
    const active = vscode.window.activeTextEditor || lastFileEditor;
    const contextFile =
      active && active.document.uri.scheme === 'file' ? active.document.uri.fsPath : null;
    const created = notes.createNote({ kind: 'question', text: text.trim(), contextFile });
    mutate((all) => all.push(created));
    await openAnswer(created.id);
  }

  // The "Ask" box in the sidebar. The command (palette, Ctrl+Alt+A) just moves focus there.
  let askView = null;
  subscriptions.push(
    vscode.window.registerWebviewViewProvider('anomx.ask', {
      resolveWebviewView(view) {
        askView = view;
        view.webview.options = { enableScripts: true };
        view.webview.html = notes.askViewHtml(notes.newId('nonce').replace(/[^a-z0-9]/gi, ''));
        view.webview.onDidReceiveMessage((message) => {
          if (message && message.type === 'ask' && typeof message.text === 'string' && message.text.trim()) {
            createQuestion(message.text);
          }
        });
        view.onDidDispose(() => {
          askView = null;
        });
      },
    }),
  );

  async function ask() {
    await vscode.commands.executeCommand('anomx.ask.focus');
    if (askView) {
      askView.webview.postMessage({ type: 'focus' });
    }
  }

  async function replyToQuestion(arg) {
    let id = noteIdFrom(arg);
    if (!id) {
      const editor = vscode.window.activeTextEditor;
      if (editor && editor.document.uri.scheme === ANSWER_SCHEME) {
        id = path.basename(editor.document.uri.path, '.md');
      }
    }
    if (!id || !modelOf(id)) {
      return;
    }
    const text = await vscode.window.showInputBox({ prompt: 'Follow-up question', ignoreFocusOut: true });
    if (!text || !text.trim()) {
      return;
    }
    mutate((all) => {
      const note = all.find((n) => n.id === id);
      if (note) {
        note.messages.push(notes.newMessage(text.trim()));
      }
    });
  }

  function setStatus(arg, status) {
    const id = noteIdFrom(arg);
    if (!id) {
      return;
    }
    mutate((all) => {
      const index = all.findIndex((n) => n.id === id);
      if (index < 0) {
        return;
      }
      if (status === 'deleted') {
        all.splice(index, 1);
      } else {
        all[index].status = status;
      }
    });
  }

  // -- opening things ----------------------------------------------------------------
  function insideFolder(file) {
    if (!folder) {
      return null;
    }
    const target = path.resolve(folder, file);
    const rel = path.relative(folder, target);
    return rel.startsWith('..') || path.isAbsolute(rel) ? null : target;
  }

  async function openLocation(location) {
    const target = location && typeof location.file === 'string' ? insideFolder(location.file) : null;
    if (!target) {
      return;
    }
    const line = Math.max((location.line || 1) - 1, 0);
    const end = Math.max((location.endLine || location.line || 1) - 1, line);
    try {
      const document = await vscode.workspace.openTextDocument(vscode.Uri.file(target));
      const range = new vscode.Range(line, 0, end, 0);
      const editor = await vscode.window.showTextDocument(document, { selection: range });
      editor.revealRange(range, vscode.TextEditorRevealType.InCenter);
    } catch (error) {
      vscode.window.showWarningMessage(`Anomx: cannot open ${location.file} (${error.message})`);
    }
  }

  async function openAnswer(id) {
    if (!modelOf(id)) {
      return;
    }
    const uri = vscode.Uri.parse(`${ANSWER_SCHEME}:/${id}.md`);
    const document = await vscode.workspace.openTextDocument(uri);
    await vscode.languages.setTextDocumentLanguage(document, 'markdown');
    await vscode.window.showTextDocument(document, { preview: true });
  }

  async function revealNote(id) {
    const note = current.find((n) => n.id === id);
    if (!note || !note.file) {
      return;
    }
    await openLocation({ file: note.file, line: note.line, endLine: note.endLine });
    const thread = threads.get(id);
    if (thread) {
      thread.collapsibleState = vscode.CommentThreadCollapsibleState.Expanded;
    }
  }

  // -- Margo import (the only place that reads other files, on explicit request) ----
  function importMargo() {
    if (!folder) {
      return;
    }
    const vaults = [path.join('.margo', 'local.notes.json'), '.notes.json'].map((name) => {
      try {
        return JSON.parse(fs.readFileSync(path.join(folder, name), 'utf8'));
      } catch (error) {
        return null;
      }
    });
    let added = 0;
    mutate((all) => {
      const created = notes.importMargoNotes(vaults, all);
      all.push(...created);
      added = created.length;
    });
    vscode.window.showInformationMessage(
      added ? `Anomx: imported ${added} Margo note(s).` : 'Anomx: no new Margo notes to import.',
    );
  }

  // -- sidebar -----------------------------------------------------------------------
  const tree = {
    onDidChangeTreeData: treeChanged.event,
    getTreeItem: (item) => item,
    getChildren() {
      if (!folder) {
        return []; // the welcome view (package.json) explains what to do
      }
      const all = models();
      if (all.length === 0) {
        return []; // the welcome view shows the "Ask a question" button
      }
      const order = (m) => (m.kind === 'question' ? 0 : 1);
      all.sort((a, b) => order(a) - order(b) || String(a.file).localeCompare(String(b.file)) || (a.line || 0) - (b.line || 0));
      return all.map((model) => {
        const item = new vscode.TreeItem(model.title, vscode.TreeItemCollapsibleState.None);
        item.noteId = model.id;
        item.description =
          model.kind === 'note' ? `${model.file}:${model.line} · ${model.statusText}` : model.statusText;
        item.tooltip = model.detail ? `${model.statusText}: ${model.detail}` : model.statusText;
        item.iconPath = new vscode.ThemeIcon(ICONS[model.status] || 'comment');
        item.contextValue = model.kind === 'question' ? 'anomx.question' : 'anomx.note';
        item.command = {
          command: model.kind === 'question' ? 'anomx.openAnswer' : 'anomx.revealNote',
          title: 'Open',
          arguments: [model.id],
        };
        return item;
      });
    },
  };
  subscriptions.push(vscode.window.createTreeView('anomx.notes', { treeDataProvider: tree }));

  subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(ANSWER_SCHEME, {
      onDidChange: answerChanged.event,
      provideTextDocumentContent(uri) {
        const model = modelOf(path.basename(uri.path, '.md'));
        return model ? notes.renderAnswer(model) : 'This question no longer exists.';
      },
    }),
    // `path:line` in an answer becomes a link that runs anomx.openLocation.
    vscode.languages.registerDocumentLinkProvider(
      { scheme: ANSWER_SCHEME },
      {
        provideDocumentLinks(document) {
          return notes.findLocations(document.getText()).map((found) => {
            const range = new vscode.Range(document.positionAt(found.start), document.positionAt(found.end));
            const link = new vscode.DocumentLink(
              range,
              vscode.Uri.parse(commandLink({ file: found.file, line: found.line, endLine: found.endLine })),
            );
            link.tooltip = 'Open in editor';
            return link;
          });
        },
      },
    ),
  );

  // -- keep ranges in step with edits -------------------------------------------------
  subscriptions.push(
    vscode.workspace.onDidChangeTextDocument((event) => {
      const file = relative(event.document.uri);
      if (file === null || event.contentChanges.length === 0) {
        return;
      }
      const changes = event.contentChanges.map((change) => ({
        startLine: change.range.start.line,
        endLine: change.range.end.line,
        startCharacter: change.range.start.character,
        newLineCount: (change.text.match(/\n/g) || []).length,
      }));
      const moved = notes.applyChanges(current, file, changes);
      for (const id of moved) {
        const thread = threads.get(id);
        const note = current.find((n) => n.id === id);
        if (thread && note) {
          thread.range = rangeOf(note);
        }
      }
      if (moved.length > 0) {
        schedulePersist();
        treeChanged.fire();
      }
    }),
    vscode.workspace.onDidSaveTextDocument((document) => {
      const file = relative(document.uri);
      if (file === null) {
        return;
      }
      // The anchor text follows the code, so the agent can tell when a note drifted.
      let changed = false;
      for (const note of current) {
        if (note.kind === 'note' && note.file === file && note.status === 'open' && note.line <= document.lineCount) {
          const text = document.lineAt(note.line - 1).text;
          if (text !== note.anchorText) {
            note.anchorText = text;
            changed = true;
          }
        }
      }
      if (changed) {
        mutate((all) => {
          for (const note of all) {
            const mine = current.find((n) => n.id === note.id);
            if (mine && note.kind === 'note') {
              note.anchorText = mine.anchorText;
              note.line = mine.line;
              note.endLine = mine.endLine;
            }
          }
        });
      }
    }),
  );

  // -- commands and watching ---------------------------------------------------------
  const register = (name, callback) =>
    subscriptions.push(vscode.commands.registerCommand(name, callback));
  register('anomx.submitComment', submitComment);
  register('anomx.resolveNote', (arg) => setStatus(arg, 'resolved'));
  register('anomx.deleteNote', (arg) => setStatus(arg, 'deleted'));
  register('anomx.ask', ask);
  register('anomx.replyToQuestion', replyToQuestion);
  register('anomx.openAnswer', openAnswer);
  register('anomx.revealNote', revealNote);
  register(OPEN_LOCATION, openLocation);
  register('anomx.importMargo', importMargo);

  if (folder) {
    const watcher = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(folder, '.anomx/*.json'));
    let watchTimer = null;
    const soon = () => {
      if (watchTimer === null) {
        watchTimer = setTimeout(() => {
          watchTimer = null;
          refresh(false);
        }, 150);
      }
    };
    subscriptions.push(
      watcher,
      watcher.onDidChange(soon),
      watcher.onDidCreate(soon),
      watcher.onDidDelete(soon),
    );
    // Safety net in case the file watcher misses an event (for example on remote hosts).
    const poll = setInterval(() => refresh(false), POLL_MS);
    subscriptions.push({ dispose: () => clearInterval(poll) });
    subscriptions.push({ dispose: () => watchTimer !== null && clearTimeout(watchTimer) });
  }
  subscriptions.push({ dispose: flush });
  refresh(true);
}

module.exports = { activate, commandLink };
