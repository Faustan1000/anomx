"""Version update approval view."""

from __future__ import annotations

import curses
import os
import shutil
import subprocess
import sys
import textwrap

from anomx import __version__
from anomx.agent.startup import check_latest_version
from anomx.agent.ui.models import (
    AgentState,
    CursesWindow,
)


class ApproveUpdateViewMixin:
    """Version update approval view."""

    @staticmethod
    def _check_latest_version() -> str | None:
        """Fetch the latest anomx version from PyPI.
        Returns the version string if newer, or None if up-to-date / unreachable.
        """
        return check_latest_version(__version__)

    def _run_version_check(self, stdscr: CursesWindow) -> bool:
        """Check PyPI for a newer version and prompt the user if one exists."""
        self.state = AgentState.VERSION_CHECK
        config = self.home.load_config()
        skipped = str(config.get("skipped_version", "") or "")
        startup_tasks = getattr(self, "_startup_tasks", None)
        if startup_tasks is not None:
            startup_tasks.poll()
            if not startup_tasks.version_ready:
                return True
            latest = startup_tasks.latest_version
        else:
            latest = self._check_latest_version()
        if latest is None or latest == skipped:
            return True
        selected = 0
        while True:
            self._draw_version_check(stdscr, selected, latest)
            key = stdscr.get_wch()
            if self._is_escape(key) or self._is_ctrl_c(key):
                return True
            if key == curses.KEY_UP:
                selected = max(0, selected - 1)
            elif key == curses.KEY_DOWN:
                selected = min(2, selected + 1)
            elif self._is_enter(key):
                if selected == 0:
                    stdscr.erase()
                    stdscr.refresh()
                    curses.endwin()
                    print(f"Updating anomx from {__version__} to {latest}...")
                    try:
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", "--upgrade", "anomx"],
                        )
                        print("Update successful. Restarting...")
                        import anomx as _anomx_pkg

                        _pkg_dir = os.path.dirname(_anomx_pkg.__file__)
                        _pycache = os.path.join(_pkg_dir, "__pycache__")
                        if os.path.isdir(_pycache):
                            shutil.rmtree(_pycache, ignore_errors=True)
                        for _mod in list(sys.modules):
                            if _mod == "anomx" or _mod.startswith("anomx."):
                                del sys.modules[_mod]
                    except subprocess.CalledProcessError:
                        input("Update failed. Press Enter to continue...")
                    os.execve(
                        sys.executable,
                        [sys.executable, "-m", "anomx"] + sys.argv[1:],
                        {**os.environ, "ANOMX_JUST_UPDATED": "1"},
                    )
                elif selected == 1:
                    return True
                else:
                    config = self.home.load_config()
                    config["skipped_version"] = latest
                    self.home.save_config(config)
                    return True
        return True

    def _draw_version_check(
        self,
        stdscr: CursesWindow,
        selected: int,
        latest: str,
    ) -> None:
        height, width = self._draw_shell(
            stdscr,
            "Version Check",
            f"v{__version__}",
        )
        y = 8
        self._add(
            stdscr,
            y,
            4,
            f"Local version:  {__version__}",
            width - 8,
            self._attr("bold"),
        )
        y += 1
        self._add(
            stdscr,
            y,
            4,
            f"Latest version: {latest}",
            width - 8,
            self._attr("accent"),
        )
        y += 3
        copy = "There is a new version of anomx available. Would you like to update now?"
        for line in textwrap.wrap(copy, width=max(24, width - 8)):
            self._add(stdscr, y, 4, line, width - 8)
            y += 1
        y += 2
        choices = ("Update now", "Update later", "Skip this version")
        for index, choice in enumerate(choices):
            marker = "›" if index == selected else "•"
            attr = self._attr("accent") if index == selected else curses.A_NORMAL
            self._add(stdscr, y + index, 4, f"{marker} {index + 1}. {choice}", width - 8, attr)
        self._add(
            stdscr,
            min(height - 2, y + len(choices) + 2),
            4,
            "Enter to confirm · Esc to skip",
            width - 8,
            self._attr("light"),
        )
        stdscr.refresh()
