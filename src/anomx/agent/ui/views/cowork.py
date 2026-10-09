"""The ``/cowork`` panel: editor notes, one agent per note, review and merge.

Everything the panel decides lives in :class:`anomx.agent.cowork.CoworkWorkspace`; this
module only draws it with the existing overlay components and forwards the user's choice.
"""

from __future__ import annotations

import curses
import os
import shutil
import subprocess
from contextlib import suppress
from pathlib import Path

from anomx.agent.cowork import (
    CoworkAction,
    CoworkFollower,
    CoworkRow,
    CoworkWorkspace,
    context_path,
)
from anomx.agent.store import SessionRecord
from anomx.agent.ui.models import CursesWindow, MenuChoice

STAY_IN_NOTE_MENU = frozenset({"diff", "compare", "follow"})
BACK_VALUE = "__back__"
FOLLOW_VALUE = "__follow__"
SERVE_VALUE = "__serve__"
STATUS_TAGS = {
    "new": "[new]",
    "queued": "[queued]",
    "running": "[running]",
    "answered": "[answered]",
    "ready": "[ready]",
    "failed": "[failed]",
    "merging": "[merging]",
    "merged": "[merged]",
    "orphan": "[note gone]",
    "no-repo": "[no repo]",
}
NO_EXTENSION_TEXT = (
    "Following needs the Anomx VS Code extension, which has not written {path} yet. "
    "Install it with `anomx --install-vscode` in a terminal and reload the VS Code window."
)
NO_NOTES_TEXT = (
    "No notes found. Add a comment in VS Code with the Anomx Editor Context extension "
    "(or ask a question with Ctrl+Alt+A); anomx looks for .anomx/notes.json in this folder, "
    "above it and in the folders of your open VS Code window."
)


def row_choice(row: CoworkRow) -> MenuChoice:
    """Menu line for one note: where it is and what its task is doing."""

    tag = STATUS_TAGS.get(row.status, f"[{row.status}]")
    detail = f"{tag} {row.detail}".strip()
    return MenuChoice(row.label, row.note_id, detail)


class CoworkViewMixin:
    """Overlay screens for cowork mode."""

    def _cowork_workspace(self) -> CoworkWorkspace | None:
        found = CoworkWorkspace.locate(self.home, self.cwd)
        if found is None:
            return None
        known = self._cowork_panels
        # Reuse the object of an earlier visit so background runs keep their progress.
        return known.setdefault(found.notes_workspace, found)

    def _run_cowork_panel(self, stdscr: CursesWindow) -> SessionRecord | None:
        """Show the panel; returns the chat session to switch to when one was opened."""

        panel = self._cowork_workspace()
        if panel is None:
            self._message(stdscr, "Cowork", NO_NOTES_TEXT)
            return None

        def refresh() -> tuple[str, tuple[MenuChoice, ...]]:
            rows = panel.rows()
            choices: list[MenuChoice] = []
            following = panel.following()
            if following is not None and following.task is not None:
                choices.append(
                    MenuChoice(
                        f"Back to {following.task.base_branch}",
                        BACK_VALUE,
                        f"Your working tree shows the task '{following.title}'",
                    )
                )
            choices.extend(row_choice(row) for row in rows)
            # Last, so the first note stays preselected and Enter never flips the mode by accident.
            choices.append(
                MenuChoice(
                    f"Follow my notes in the editor: {'on' if self._cowork_follower else 'off'}",
                    FOLLOW_VALUE,
                    "Put the cursor in a note: this chat and your working tree switch to it",
                )
            )
            choices.append(
                MenuChoice(
                    f"Work on notes from VS Code: {'on' if self._cowork_serving else 'off'}",
                    SERVE_VALUE,
                    "Agents start when you comment or ask in the editor; answers appear there",
                )
            )
            running = sum(1 for row in rows if row.running)
            subtitle = f"{len(rows)} notes" + (f" · {running} agent(s) working" if running else "")
            return subtitle, tuple(choices)

        selected: str | None = None
        while True:
            _subtitle, choices = refresh()
            if not choices:
                self._message(stdscr, "Cowork", "There are no open notes.")
                return None
            selected = self._run_overlay_menu(
                stdscr,
                "Cowork",
                footer="Esc Close · ↑↓ Navigate · Enter Open note",
                initial_value=selected,
                refresh=refresh,
            )
            if selected is None:
                return None
            if selected == BACK_VALUE:
                self._cowork_back(stdscr, panel)
                continue
            if selected == SERVE_VALUE:
                self._cowork_serving = not self._cowork_serving
                continue
            if selected == FOLLOW_VALUE:
                self._cowork_toggle_follow(stdscr, panel)
                continue
            row = next((item for item in panel.rows() if item.note_id == selected), None)
            if row is not None:
                chat = self._cowork_note_menu(stdscr, panel, row)
                if chat is not None:
                    return chat

    def _cowork_note_menu(
        self, stdscr: CursesWindow, panel: CoworkWorkspace, row: CoworkRow
    ) -> SessionRecord | None:
        """Actions of one note. Esc goes up one level (to the list of notes).

        Actions that only show something (changes, comparison, editor switch) return to
        this menu afterwards; actions that change the task return to the list.
        """

        initial: str | None = None
        while True:
            current = next((r for r in panel.rows() if r.note_id == row.note_id), row)
            actions = panel.actions(current)
            if not actions:
                detail = current.detail or current.status
                self._message(stdscr, "Cowork", f"{current.label}\n\n{detail}")
                return None
            choice = self._menu(
                stdscr,
                "Cowork",
                f"{current.label}\n{current.detail}".strip(),
                tuple(MenuChoice(item.label, item.value, item.detail) for item in actions),
                initial_value=initial,
            )
            if choice is None:
                return None
            record = self._cowork_apply(
                stdscr, panel, current, next(a for a in actions if a.value == choice)
            )
            if record is not None or choice not in STAY_IN_NOTE_MENU:
                return record
            initial = choice

    def _cowork_apply(
        self,
        stdscr: CursesWindow,
        panel: CoworkWorkspace,
        row: CoworkRow,
        action: CoworkAction,
    ) -> SessionRecord | None:
        message = self._message
        if action.value in {"chat", "discuss"}:
            record, problem = panel.open_chat(row, discuss=action.value == "discuss")
            if problem:
                message(stdscr, "Cannot open chat", problem)
            elif record is not None:
                # Opening the chat from the panel is an explicit request: answer the note now.
                self._cowork_send_pending(record)
            return record
        if action.value in {"start", "answer"}:
            problem = panel.start(row)
            if problem:
                message(stdscr, "Cannot start", problem)
        elif action.value == "stop":
            panel.stop(row)
        elif action.value == "diff":
            self._run_text_viewer(stdscr, f"Changes · {row.label}", panel.diff_lines(row))
        elif action.value == "compare":
            self._cowork_compare(stdscr, panel, row)
        elif action.value == "follow":
            result = panel.follow(row)
            if not result.ok:
                message(stdscr, "Cannot switch", result.reason)
            else:
                self._cowork_open_in_editor(row)
                message(
                    stdscr,
                    "Cowork",
                    "Your working tree now shows the agent's version. Open /cowork and choose "
                    f"'Back to {row.task.base_branch if row.task else 'base'}' when you are done.",
                )
        elif action.value in {"merge", "merge_now"}:
            self._cowork_merge(stdscr, panel, row, review=action.value == "merge")
        elif action.value == "abort_merge":
            panel.abort_merge(row)
            message(stdscr, "Cowork", "The pending merge was aborted; nothing is staged anymore.")
        elif action.value == "discard":
            if self._cowork_confirm(stdscr, "Discard this task and its branch?", "Discard"):
                panel.discard(row)
        return None

    def _cowork_merge(
        self, stdscr: CursesWindow, panel: CoworkWorkspace, row: CoworkRow, *, review: bool = True
    ) -> None:
        base = row.task.base_branch if row.task else "the base branch"
        diff_headers = [line for line in panel.diff_lines(row) if line.startswith("diff --git")]
        files = sorted({header.split(" b/")[-1] for header in diff_headers})
        changed = ", ".join(files) or "nothing"
        if review:
            question = (
                f"Stage the changes in {base} for review?  Changed: {changed}. You review them "
                "in VS Code's Source Control panel and commit there."
            )
            action = "Stage for review"
        else:
            question = f"Merge into {base} now?  Changed: {changed}"
            action = f"Merge into {base}"
        if not self._cowork_confirm(stdscr, question, action):
            return
        result = panel.merge(row, review=review)
        if result.ok and result.pending:
            self._message(
                stdscr,
                "Cowork",
                "The merge is staged. Open Source Control (Ctrl+Shift+G): the files are under "
                "'Staged Changes' with 'Merge in progress'. Review them and commit there to "
                "finish, or choose 'Abort the pending merge' in /cowork.",
            )
            return
        if result.ok:
            self._message(stdscr, "Cowork", f"Merged into {base}.")
            return
        detail = result.reason
        if result.conflicts:
            detail += "\nConflicting files: " + ", ".join(result.conflicts)
        self._message(stdscr, "Cannot merge", detail)

    def _cowork_send_pending(self, session: SessionRecord) -> bool:
        """Send the note of a cowork chat to the agent if nobody has answered it yet."""

        panel = self._cowork_workspace()
        if panel is None or self._active_turn_for_session(session) is not None:
            return False
        if not panel.awaiting_reply(session.path):
            return False
        self._start_session_turn(session)
        return True

    def _cowork_toggle_follow(self, stdscr: CursesWindow, panel: CoworkWorkspace) -> None:
        if self._cowork_follower is not None:
            self._cowork_follower = None
            return
        target = context_path(self.home.root)
        if not target.exists():
            self._message(stdscr, "Cowork", NO_EXTENSION_TEXT.format(path=target))
            return
        self._cowork_follower = CoworkFollower(panel, target)

    def _cowork_back(self, stdscr: CursesWindow, panel: CoworkWorkspace) -> None:
        following = panel.following()
        if following is None:
            return
        result = panel.return_to_base(following)
        if not result.ok:
            self._message(stdscr, "Cannot switch back", result.reason)

    def _cowork_confirm(self, stdscr: CursesWindow, question: str, action: str) -> bool:
        choice = self._menu(
            stdscr,
            "Cowork",
            question,
            (MenuChoice("Cancel", "cancel"), MenuChoice(action, "confirm")),
        )
        return choice == "confirm"

    def _vscode_cli(self) -> str | None:
        """The ``code`` command, but only when this terminal can steer its VS Code window."""

        code = shutil.which("code")
        return code if code is not None and os.environ.get("VSCODE_IPC_HOOK_CLI") else None

    def _cowork_open_in_editor(self, row: CoworkRow) -> None:
        """Best effort: bring the file to the front in the VS Code window of this terminal."""

        code = self._vscode_cli()
        if code is None or row.repo_root is None or row.task is None:
            return
        target = f"{row.repo_root / row.task.file}:{row.task.line}"
        with suppress(OSError, subprocess.SubprocessError):
            subprocess.run(  # noqa: S603 - fixed binary found on PATH, no shell
                [code, "--reuse-window", "--goto", target],
                capture_output=True,
                timeout=5,
                check=False,
            )

    def _open_diff_in_vscode(self, original: Path, agent: Path) -> bool:
        code = self._vscode_cli()
        if code is None:
            return False
        try:
            result = subprocess.run(  # noqa: S603 - fixed binary found on PATH, no shell
                [code, "--reuse-window", "--diff", str(original), str(agent)],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def _cowork_compare(
        self, stdscr: CursesWindow, panel: CoworkWorkspace, row: CoworkRow
    ) -> None:
        """Open VS Code's side-by-side diff: original on the left, the agent's file on the right."""

        files = panel.comparison_files(row)
        if not files:
            self._message(stdscr, "Cowork", "The agent changed no files.")
            return
        if self._vscode_cli() is None:
            first = files[0]
            self._message(
                stdscr,
                "Cannot open VS Code",
                "This terminal cannot steer a VS Code window. Run this in a VS Code terminal:\n"
                f"code --diff {first.original} {first.agent}",
            )
            return
        chosen = files
        if len(files) > 1:
            picked = self._menu(
                stdscr,
                "Compare in VS Code",
                f"{len(files)} files changed",
                (
                    MenuChoice(f"Open all {len(files)} comparisons", "__all__"),
                    *(MenuChoice(item.label, str(index)) for index, item in enumerate(files)),
                ),
            )
            if picked is None:
                return
            chosen = files if picked == "__all__" else [files[int(picked)]]
        failed = [
            item.file
            for item in chosen
            if not self._open_diff_in_vscode(item.original, item.agent)
        ]
        if failed:
            self._message(stdscr, "Cannot open VS Code", "Failed for: " + ", ".join(failed))
            return
        self._message(
            stdscr,
            "Cowork",
            f"Opened {len(chosen)} comparison(s) in VS Code. Left: the original, right: the "
            "agent's version in its own worktree (you can edit the right side).",
        )

    def _run_text_viewer(
        self, stdscr: CursesWindow, title: str, lines: list[str], *, subtitle: str = ""
    ) -> None:
        """Scrollable read-only text (used for diffs)."""

        text = lines or ["(nothing to show)"]
        top = 0
        page = 12
        while True:
            _, _, width, height = self._draw_overlay(
                stdscr,
                title=title,
                subtitle=subtitle,
                body_lines=tuple(
                    line[: max(10, 200)].replace("\t", "    ") for line in text[top : top + page]
                ),
                footer=f"Esc Back · ↑↓ PgUp PgDn Scroll · {top + 1}-{min(len(text), top + page)}"
                f" of {len(text)}",
            )
            page = max(5, height - 2)
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key) or self._is_enter(key):
                return
            last = max(0, len(text) - page)
            if key == curses.KEY_DOWN:
                top = min(last, top + 1)
            elif key == curses.KEY_UP:
                top = max(0, top - 1)
            elif key == curses.KEY_NPAGE:
                top = min(last, top + page)
            elif key == curses.KEY_PPAGE:
                top = max(0, top - page)
            elif key == curses.KEY_HOME:
                top = 0
            elif key == curses.KEY_END:
                top = last
