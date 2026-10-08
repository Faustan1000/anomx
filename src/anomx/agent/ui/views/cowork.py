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

from anomx.agent.cowork import CoworkAction, CoworkRow, CoworkWorkspace
from anomx.agent.ui.models import CursesWindow, MenuChoice

BACK_VALUE = "__back__"
STATUS_TAGS = {
    "new": "[new]",
    "queued": "[queued]",
    "running": "[running]",
    "ready": "[ready]",
    "failed": "[failed]",
    "merged": "[merged]",
    "orphan": "[note gone]",
    "no-repo": "[no repo]",
}
NO_MARGO_TEXT = (
    "No Margo notes found. Add a note in VS Code (Margo extension); anomx looks for "
    ".margo/local.notes.json or .notes.json in this folder and above it."
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
        return known.setdefault(found.margo_workspace, found)

    def _run_cowork_panel(self, stdscr: CursesWindow) -> None:
        panel = self._cowork_workspace()
        if panel is None:
            self._message(stdscr, "Cowork", NO_MARGO_TEXT)
            return

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
            running = sum(1 for row in rows if row.running)
            subtitle = f"{len(rows)} notes" + (f" · {running} agent(s) working" if running else "")
            return subtitle, tuple(choices)

        selected: str | None = None
        while True:
            _subtitle, choices = refresh()
            if not choices:
                self._message(stdscr, "Cowork", "There are no open notes.")
                return
            selected = self._run_overlay_menu(
                stdscr,
                "Cowork",
                footer="Esc Close · ↑↓ Navigate · Enter Open note",
                initial_value=selected,
                refresh=refresh,
            )
            if selected is None:
                return
            if selected == BACK_VALUE:
                self._cowork_back(stdscr, panel)
                continue
            row = next((item for item in panel.rows() if item.note_id == selected), None)
            if row is not None:
                self._cowork_note_menu(stdscr, panel, row)

    def _cowork_note_menu(
        self, stdscr: CursesWindow, panel: CoworkWorkspace, row: CoworkRow
    ) -> None:
        actions = panel.actions(row)
        subtitle = f"{row.label}\n{row.detail}".strip()
        if not actions:
            self._message(stdscr, "Cowork", f"{row.label}\n\n{row.detail or row.status}")
            return
        choice = self._menu(
            stdscr,
            "Cowork",
            subtitle,
            tuple(MenuChoice(item.label, item.value, item.detail) for item in actions),
        )
        if choice is not None:
            self._cowork_apply(stdscr, panel, row, next(a for a in actions if a.value == choice))

    def _cowork_apply(
        self,
        stdscr: CursesWindow,
        panel: CoworkWorkspace,
        row: CoworkRow,
        action: CoworkAction,
    ) -> None:
        message = self._message
        if action.value == "start":
            problem = panel.start(row)
            if problem:
                message(stdscr, "Cannot start", problem)
        elif action.value == "stop":
            panel.stop(row)
        elif action.value == "diff":
            self._run_text_viewer(stdscr, f"Changes · {row.label}", panel.diff_lines(row))
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
        elif action.value == "merge":
            self._cowork_merge(stdscr, panel, row)
        elif action.value == "discard":
            if self._cowork_confirm(stdscr, "Discard this task and its branch?", "Discard"):
                panel.discard(row)

    def _cowork_merge(
        self, stdscr: CursesWindow, panel: CoworkWorkspace, row: CoworkRow
    ) -> None:
        base = row.task.base_branch if row.task else "the base branch"
        diff_headers = [line for line in panel.diff_lines(row) if line.startswith("diff --git")]
        files = sorted({header.split(" b/")[-1] for header in diff_headers})
        question = f"Merge into {base}?  Changed: {', '.join(files) or 'nothing'}"
        if not self._cowork_confirm(stdscr, question, f"Merge into {base}"):
            return
        result = panel.merge(row)
        if result.ok:
            self._message(stdscr, "Cowork", f"Merged into {base}.")
            return
        detail = result.reason
        if result.conflicts:
            detail += "\nConflicting files: " + ", ".join(result.conflicts)
        self._message(stdscr, "Cannot merge", detail)

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

    def _cowork_open_in_editor(self, row: CoworkRow) -> None:
        """Best effort: bring the file to the front in the VS Code window of this terminal."""

        if row.repo_root is None or row.task is None:
            return
        code = shutil.which("code")
        if code is None or not os.environ.get("VSCODE_IPC_HOOK_CLI"):
            return
        target = f"{row.repo_root / row.task.file}:{row.task.line}"
        with suppress(OSError, subprocess.SubprocessError):
            subprocess.run(  # noqa: S603 - fixed binary found on PATH, no shell
                [code, "--reuse-window", "--goto", target],
                capture_output=True,
                timeout=5,
                check=False,
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
