"""The agent that works on one cowork task inside its own git worktree.

It is deliberately small: it can read, search and edit files of its worktree and nothing
else (no shell, no web, no questions, no subagents), so what it does is fully visible in
the diff the user reviews before merging.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anomx.agent.base.agents import AgentKind, BaseAgent
from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.runtime import AgentRuntime
from anomx.agent.tools import SUBAGENT_STATEMENT_DESCRIPTION
from anomx.agent.tools.glob import GlobTool
from anomx.agent.tools.grep import GrepTool
from anomx.agent.tools.list_directory import ListDirectoryTool
from anomx.agent.tools.read_file import ReadFileTool

COWORK_PROMPT = """\
- You are the Anomx cowork agent. A user left one short note on a line of their code. You make
  exactly the change that note asks for, in your own git worktree, so the user can review it
  and merge it.
- Work like a careful colleague, not like a feature builder: keep the diff as small as the
  note allows. Do not refactor, rename, reformat or "improve" anything the note does not ask for.
- Keep the code consistent: when the note changes a signature or a data shape (for example a
  dict that becomes a dataclass), also update the definitions and call sites that would
  otherwise break. Find them with grep before you edit.
- The note refers to the code at the marked line and may mean the whole block that starts
  there. The line number can be slightly off; trust the quoted code.
- You cannot ask questions and nobody answers while you work. When the note is ambiguous, take
  the most conservative reading and name the assumption in your summary. If no change is needed
  or possible, change nothing and say why.
- You can read, list, glob, grep and edit files with `edit_file`. You cannot run commands, so
  say in your summary what you could not verify.
- Finish with a plain summary of at most six lines: what you changed (files), the assumptions
  you made and anything the user should check.
"""

EDIT_FILE_TOOL = "edit_file"


class EditError(ValueError):
    """An edit that must not be applied; the message is shown to the model."""


@dataclass(frozen=True)
class EditResult:
    path: str
    action: str  # "edited" or "created"
    replacements: int
    first_changed_line: int


def apply_edit(
    root: Path,
    raw_path: str,
    old_text: str,
    new_text: str,
    *,
    replace_all: bool = False,
) -> EditResult:
    """Replace ``old_text`` by ``new_text`` in one file below ``root``.

    ``old_text`` must match exactly once unless ``replace_all`` is set; an empty ``old_text``
    creates a new file. Paths outside ``root`` (also via symlinks) and anything inside
    ``.git`` are refused. Line endings and the file mode are preserved.
    """

    root = root.resolve()
    candidate = Path(raw_path)
    target = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        relative = target.relative_to(root)
    except ValueError:
        raise EditError(f"Path is outside the task worktree: {raw_path}") from None
    if ".git" in relative.parts:
        raise EditError("Files inside .git cannot be edited.")

    if not old_text:
        if target.exists():
            raise EditError("The file already exists; pass old_text to change it.")
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(target, new_text.encode("utf-8"), mode=None)
        return EditResult(relative.as_posix(), "created", 1, 1)

    if not target.is_file():
        raise EditError(f"File does not exist: {relative.as_posix()}")
    try:
        text = target.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        raise EditError("The file is not UTF-8 text.") from None

    if "\r\n" in text and "\r\n" not in old_text:
        old_text = old_text.replace("\n", "\r\n")
        new_text = new_text.replace("\n", "\r\n")
    count = text.count(old_text)
    if count == 0:
        raise EditError("old_text was not found; read the file again and copy it exactly.")
    if count > 1 and not replace_all:
        raise EditError(
            f"old_text matches {count} places; add surrounding lines to make it unique "
            "or set replace_all."
        )
    first_index = text.index(old_text)
    updated = text.replace(old_text, new_text) if replace_all else text.replace(
        old_text, new_text, 1
    )
    _write_atomic(target, updated.encode("utf-8"), mode=target.stat().st_mode)
    return EditResult(
        relative.as_posix(),
        "edited",
        count if replace_all else 1,
        text.count("\n", 0, first_index) + 1,
    )


def _write_atomic(target: Path, data: bytes, *, mode: int | None) -> None:
    handle, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".anomx-edit-")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        if mode is not None:
            os.chmod(tmp_name, mode & 0o7777)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class EditFileTool(BaseTool):
    """Edit or create a file inside the task worktree."""

    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name=EDIT_FILE_TOOL,
            description=(
                "Replace exact text in a file of the task worktree. old_text must match "
                "exactly one place (copy it from a fresh read, including indentation); "
                "an empty old_text creates a new file with new_text."
            ),
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "path": {"type": "string", "description": "File path in the worktree."},
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace; empty to create a new file.",
                    },
                    "new_text": {"type": "string", "description": "Replacement text."},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every match instead of requiring one.",
                    },
                },
                ["statement", "path", "old_text", "new_text"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        try:
            result = apply_edit(
                context.runtime.cwd,
                str(arguments.get("path", "")),
                str(arguments.get("old_text", "")),
                str(arguments.get("new_text", "")),
                replace_all=bool(arguments.get("replace_all", False)),
            )
        except (EditError, OSError) as error:
            return context.json_result({"error": str(error)})
        return context.json_result(
            {
                "path": result.path,
                "action": result.action,
                "replacements": result.replacements,
                "first_changed_line": result.first_changed_line,
            }
        )


def cowork_tools() -> tuple[BaseTool, ...]:
    statement = SUBAGENT_STATEMENT_DESCRIPTION
    return (
        ReadFileTool(statement_description=statement),
        ListDirectoryTool(statement_description=statement),
        GlobTool(statement_description=statement),
        GrepTool(statement_description=statement),
        EditFileTool(statement_description=statement),
    )


class CoworkAgent(BaseAgent):
    """Agent role used for every cowork task."""

    def __init__(self) -> None:
        super().__init__(
            kind=AgentKind.SUB,
            name="Cowork",
            system_prompt=COWORK_PROMPT,
            tools=cowork_tools(),
            color="subagent",
            symbol="C",
            can_spawn_subagents=False,
            can_ask_questions=False,
            can_use_plans=False,
            can_start_processes=False,
            can_use_web=False,
        )


class CoworkRuntime(AgentRuntime):
    """Runtime bound to one task worktree with the lean cowork prompt and tools."""

    def __init__(self, home: Any, worktree: Path, **kwargs: Any) -> None:
        kwargs.setdefault("agent_kind", AgentKind.SUB)
        super().__init__(home, worktree, workspace_root=worktree, **kwargs)
        self.agent_spec = CoworkAgent()
        self.agent_kind = self.agent_spec.kind

    def set_agent(self, kind: AgentKind | str) -> None:
        """The cowork role is fixed; callers cannot swap it for a general agent."""

    def _instructions(
        self,
        session_path: Path | None = None,
        *,
        include_previous_conversation: bool = True,
    ) -> str:
        del session_path, include_previous_conversation
        tools = "\n".join(f"- {tool}" for tool in self._tool_descriptions())
        workspace = (
            f"Your worktree (repository root for this task): {self.cwd}\n"
            "Paths may be relative to it. Read and edit only files inside it."
        )
        return "\n\n".join(
            (
                "# Identity\n\n" + self.agent_spec.prompt.strip(),
                "# Workspace\n\n" + workspace,
                "# Available Tools\n\n" + tools,
            )
        )
