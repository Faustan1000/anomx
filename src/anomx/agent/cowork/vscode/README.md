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
- it never reads other files and never sends the selected text (only the cursor line)
- only the focused window writes; a closing window clears the file

The whole implementation is `extension.js` (under 200 lines) and is meant to be read.

## Settings

- `anomxEditorContext.enabled` – turn writing off
- `anomxEditorContext.path` – write somewhere else
