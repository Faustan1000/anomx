# Changelog

## 0.2.43

- Forward provider reasoning and OpenAI reasoning summaries as separate thought events.
- Preserve thoughts in subagent work history and retain signed/redacted Anthropic blocks for tool continuation.
- Keep structured reasoning and unfinished thinking blocks out of normal assistant text.
- Start agent turns with a brief text response and provide occasional progress updates during longer work.

## 0.2.8

- **Major codebase restructuring**: Tools and backends moved into dedicated classes for better object-oriented design.
- **Agent UI restructuring**: UI class refactored into a mixin-based architecture; subagent views now include a proper "back" navigation.
- **"@" referencing feature**: Added support for mentioning files and directories via "@" for agent context.
- **Paste support**: Added paste with pre/post handling.
- **Agent steering**: Implemented agent steering capability for better workflow control.
- **Global approvals**: Config now uses global approvals instead of session-scoped ones; improved /config command management.
- **Debug mode upgrade**: Full logs are now stored in `~/.anomx/debug` by default.
- **Removed deprecated shortcuts**: Cleaned up unused deprecated shortcut commands.
- **Various fixes**: Improved automatic restart logic, reconnection handling, and general stability.

## 0.2.6

- Agent sandbox updates: podman/docker modes, copy/mount project file modes, container stop/remove modes.
- Agent UI and store improvements for sub-process handling, session output, and formatting.
- Enhanced agent CLI with better visualization and on-brand UI.
- Various bug fixes and stability improvements.

## 0.1.0

- Initial package scaffold for datasets, scorers, detectors, models, and optional Darts integration.
