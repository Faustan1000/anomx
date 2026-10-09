"""Keep tests independent of the terminal they are started from."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _not_inside_vscode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running pytest in VS Code's terminal must not trigger the extension hint."""

    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.delenv("VSCODE_IPC_HOOK_CLI", raising=False)
