"""Git operations behind cowork tasks.

Every task works in its own ``git worktree`` on its own branch, so agents never edit the
user's files and never each other's. The user's working tree can *follow* a task (be
switched to the task's current commit) so the editor shows what the agent produced, and
finished branches are merged back only when the user approves.

Switching the user's tree is the risky part: it rewrites files that VS Code has open. It
is therefore refused whenever tracked files have uncommitted changes (``FollowResult``
carries the reason) and never uses stash, reset or force.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

BRANCH_PREFIX = "anomx/cowork/"
AGENT_IDENTITY = ("-c", "user.name=Anomx Agent", "-c", "user.email=agent@anomx.local")


class GitError(RuntimeError):
    """A git command failed in a way the caller cannot recover from."""


@dataclass(frozen=True)
class TaskWorkspace:
    """Branch and worktree created for one task."""

    branch: str
    path: Path
    base_branch: str
    base_commit: str


@dataclass(frozen=True)
class FollowResult:
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class MergeResult:
    ok: bool
    conflicts: tuple[str, ...] = ()
    reason: str = ""


def find_repo_root(path: Path) -> Path | None:
    """Git repository that contains ``path`` (a file or folder), if any.

    A VS Code workspace often spans several repositories, so each note is resolved to the
    repository of its own file instead of assuming one repository per workspace.
    """

    folder = path if path.is_dir() else path.parent
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    result = subprocess.run(  # noqa: S603 - fixed git binary, no shell
        ["git", "-C", str(folder), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"},
    )
    top = result.stdout.strip()
    return Path(top) if result.returncode == 0 and top else None


def branch_for(note_id: str) -> str:
    """Branch name for a note; Margo ids are already close to valid ref names."""

    return BRANCH_PREFIX + re.sub(r"[^A-Za-z0-9._-]", "-", note_id).strip(".-")


class CoworkGit:
    """Git helper bound to the user's repository."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    # -- inspection -------------------------------------------------------------------

    def current_branch(self) -> str | None:
        """Checked-out branch of the user's tree, ``None`` when HEAD is detached."""

        result = self._git("symbolic-ref", "--quiet", "--short", "HEAD", check=False)
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def head(self, cwd: Path | None = None) -> str:
        return self._git("rev-parse", "HEAD", cwd=cwd).stdout.strip()

    def is_clean(self) -> bool:
        """No uncommitted changes to tracked files (untracked files are ignored)."""

        return not self._git("status", "--porcelain", "--untracked-files=no").stdout.strip()

    def tip(self, branch: str) -> str:
        """Commit a branch currently points to (empty string if it does not exist)."""

        result = self._git("rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", check=False)
        return result.stdout.strip() if result.returncode == 0 else ""

    def is_file_committed(self, relative: str) -> bool:
        """The file is tracked and has no uncommitted changes (what an agent would see)."""

        tracked = self._git("ls-files", "--error-unmatch", "--", relative, check=False)
        if tracked.returncode != 0:
            return False
        return not self._git("status", "--porcelain", "--", relative).stdout.strip()

    def branch_exists(self, branch: str) -> bool:
        ref = f"refs/heads/{branch}"
        return self._git("rev-parse", "--verify", "--quiet", ref, check=False).returncode == 0

    def changed_files(self, branch: str, base: str) -> list[str]:
        out = self._git("diff", "--name-only", f"{base}...{branch}").stdout
        return [line for line in out.splitlines() if line]

    def diff(self, branch: str, base: str) -> str:
        return self._git("diff", f"{base}...{branch}").stdout

    def commits_ahead(self, branch: str, base: str) -> int:
        return int(self._git("rev-list", "--count", f"{base}..{branch}").stdout.strip() or 0)

    # -- task branches ----------------------------------------------------------------

    def create_task(
        self, note_id: str, worktree: Path, *, base_ref: str | None = None
    ) -> TaskWorkspace:
        """Create ``anomx/cowork/<id>`` from the current base and check it out in ``worktree``."""

        base_branch = self.current_branch()
        if base_branch is None:
            raise GitError("Check out a branch first; cowork needs a base branch to merge into.")
        start = base_ref or "HEAD"
        base_commit = self._git("rev-parse", start).stdout.strip()
        branch = branch_for(note_id)
        if self.branch_exists(branch):
            raise GitError(f"branch {branch} already exists")
        worktree.parent.mkdir(parents=True, exist_ok=True)
        self._git("worktree", "add", "-b", branch, str(worktree), base_commit)
        return TaskWorkspace(branch, worktree, base_branch, base_commit)

    def commit_all(self, worktree: Path, message: str) -> str | None:
        """Commit everything in a task worktree; returns the new commit or ``None``."""

        self._git("add", "-A", cwd=worktree)
        if not self._git("status", "--porcelain", cwd=worktree).stdout.strip():
            return None
        self._git(*AGENT_IDENTITY, "commit", "-m", message, cwd=worktree)
        return self.head(worktree)

    def remove_task(self, branch: str, worktree: Path, *, force: bool = False) -> None:
        """Delete a task's worktree and branch (``force`` also drops unmerged work)."""

        if worktree.exists():
            self._git("worktree", "remove", "--force", str(worktree), check=False)
        self._git("worktree", "prune", check=False)
        if self.branch_exists(branch):
            self._git("branch", "-D" if force else "-d", branch, check=False)

    # -- following and merging --------------------------------------------------------

    def follow(self, branch: str) -> FollowResult:
        """Show a task's current state in the user's tree (detached, nothing is moved)."""

        if not self.branch_exists(branch):
            return FollowResult(False, f"branch {branch} does not exist")
        if not self.is_clean():
            return FollowResult(
                False, "You have uncommitted changes in tracked files; commit or stash them first."
            )
        result = self._git("switch", "--detach", branch, check=False)
        if result.returncode != 0:
            return FollowResult(False, result.stderr.strip() or "git switch failed")
        return FollowResult(True)

    def return_to(self, base_branch: str) -> FollowResult:
        """Switch the user's tree back to the branch it was on."""

        if not self.is_clean():
            return FollowResult(False, "You have uncommitted changes in tracked files.")
        result = self._git("switch", base_branch, check=False)
        if result.returncode != 0:
            return FollowResult(False, result.stderr.strip() or "git switch failed")
        return FollowResult(True)

    def merge(self, branch: str, base_branch: str, message: str) -> MergeResult:
        """Merge a task branch into the base branch with a merge commit.

        Runs in the user's tree (the base branch is checked out there). On conflicts the
        merge is aborted so nothing half-merged is left behind.
        """

        back = FollowResult(True)
        if self.current_branch() != base_branch:
            back = self.return_to(base_branch)
        if not back.ok:
            return MergeResult(False, reason=back.reason)
        if not self.is_clean():
            return MergeResult(False, reason="You have uncommitted changes in tracked files.")
        result = self._git(*AGENT_IDENTITY, "merge", "--no-ff", "-m", message, branch, check=False)
        if result.returncode == 0:
            return MergeResult(True)
        unmerged = self._git("diff", "--name-only", "--diff-filter=U", check=False)
        conflicts = tuple(line for line in unmerged.stdout.splitlines() if line)
        self._git("merge", "--abort", check=False)
        reason = result.stderr.strip()
        if conflicts:
            reason = "Merge conflicts; the branch was not merged."
        return MergeResult(False, conflicts, reason)

    # -- plumbing ---------------------------------------------------------------------

    def _git(
        self, *args: str, cwd: Path | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
        result = subprocess.run(  # noqa: S603 - fixed git binary, no shell
            ["git", "-C", str(cwd or self.repo_root), *args],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if check and result.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result
