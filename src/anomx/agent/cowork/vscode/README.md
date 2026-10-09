# Anomx Editor Context

Tiny local extension for the Anomx CLI agent. It writes **one JSON file** describing what
you are looking at in VS Code, so the agent can tell which comment or code you mean.

```json
{
  "schema": 1,
  "updatedAt": "2026-10-08T10:15:00.000Z",
  "workspaceFolders": ["/home/me/projects"],
  "activeFile": "/home/me/projects/app/src/auth.py",
  "languageId": "python",
  "isDirty": false,
  "isActiveEditor": true,
  "cursor": { "line": 42, "character": 5 },
  "selection": { "startLine": 42, "startCharacter": 5, "endLine": 42, "endCharacter": 5, "isEmpty": true },
  "visibleRange": { "startLine": 30, "endLine": 70 },
  "activeLineText": "    await authenticate(token)"
}
```

Lines and characters are 1-based. The file is `$ANOMX_HOME/editor-context.json`
(default `~/.anomx/editor-context.json`), mode `0600`, replaced atomically.

## What it does not do

- no network access, no ports, no child processes, no dependencies
- the context writer never reads other files and never sends the selected text (only the
  cursor line); the notes part reads only `.anomx/*.json` and, on the explicit command
  "Anomx: Import Margo notes", the Margo vault files (`.margo/local.notes.json`,
  `.notes.json`) - the only place other files are read
- only the focused window writes; a closing window clears the file
- when no file editor is active (for example while you type in the terminal or write in a
  comment box) the last visible file editor is kept and `isActiveEditor` is `false`
- an unchanged state is not rewritten

The whole implementation is `extension.js` (context writer), `notes.js` (pure note logic) and
`comments.js` (VS Code bindings) and is meant to be read.

## Using it

- Comment on code (the `+` in the gutter): the first comment creates a note, later comments
  are replies. The agent's answers and its status appear in the same thread.
- `Anomx: Ask a question` (ctrl+alt+a): a general question without a line; the answer and
  clickable `path:line` locations show up in the Anomx sidebar.
- `Anomx: Import Margo notes`: one-time import of open Margo notes.

## Settings

- `anomxEditorContext.enabled` – turn writing off
- `anomxEditorContext.path` – write somewhere else
- `anomxNotes.enabled` – turn the comments, questions and sidebar off

## Notes (comments) and answers

Version 0.2 also owns the notes the agent works on. Two files live in `<first workspace
folder>/.anomx/` (the extension creates the folder with a `.gitignore` containing `*`).
Each file has exactly ONE writer and is replaced atomically, so there are no write races.

### `notes.json` - written only by the extension

```json
{
  "schema": 1,
  "notes": [
    {
      "id": "n-1791449926580-x93ibw",
      "kind": "note",
      "file": "anomx/src/ui/app.py",
      "line": 12,
      "endLine": 14,
      "anchorText": "def build_ui():",
      "contextFile": null,
      "status": "open",
      "createdAt": "2026-10-09T08:00:00.000Z",
      "messages": [
        { "id": "m-1791449926580-a1b2c3", "role": "user", "text": "make the dict a dataclass", "at": "2026-10-09T08:00:00.000Z" }
      ]
    }
  ]
}
```

- `kind`: `note` is a comment on code lines (the agent changes code in its own worktree);
  `question` is a general question without a line (the agent only reads and answers).
- `file`: POSIX path relative to the workspace folder; `null` for a question.
  `line`/`endLine`: 1-based, inclusive; `null` for a question. The extension keeps them
  up to date while the user edits (VS Code range tracking plus its own shifting).
- `anchorText`: text of the first line when the note was last saved (lets the agent notice
  that code moved); `""` for a question.
- `contextFile`: absolute path of the file that was active when a question was asked
  (tells anomx which repository to search); `null` for a note.
- `status`: `open` or `resolved` (resolved notes are ignored by the agent).
- `importedFrom` (optional): `margo:<id>` for notes created by "Anomx: Import Margo notes".
- `messages`: what the user wrote. The first one is the request; later ones are replies in
  the thread. Message ids are unique and never change.

### `agent.json` - written only by anomx

```json
{
  "schema": 1,
  "notes": {
    "n-1791449926580-x93ibw": {
      "status": "working",
      "detail": "reading app.py",
      "pickedUp": ["m-1791449926580-a1b2c3"],
      "updatedAt": "2026-10-09T08:00:05.000Z",
      "messages": [
        {
          "id": "a-1791449940000-d4e5f6",
          "role": "agent",
          "text": "The GUI is built in `build_ui` ...",
          "at": "2026-10-09T08:00:40.000Z",
          "afterMessage": "m-1791449926580-a1b2c3",
          "locations": [{ "file": "anomx/src/ui/app.py", "line": 12, "endLine": 40 }]
        }
      ]
    }
  }
}
```

- `status`: `idle` (not picked up), `working`, `ready` (answer or change available),
  `failed`, `merging`, `merged`, `discarded`.
- `detail`: short progress text, or the error when `failed`.
- `pickedUp`: ids of user messages anomx has started to handle. A user message that is not
  in this list is still waiting for the agent.
- `messages`: the agent's answers in order. `locations` are places in the code the answer
  refers to (path relative to the workspace folder, 1-based), already checked to exist.
