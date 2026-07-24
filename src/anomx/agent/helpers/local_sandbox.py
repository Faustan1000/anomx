"""Local Python sandbox execution for agent runtimes."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from shlex import split as shell_split

from anomx.agent.helpers.tool_manager import (
    COMMAND_TIMEOUT_SECONDS,
    command_timeout_output,
)

MAX_OUTPUT_CHARS = 80_000
PYTHON_SANDBOX_SYSTEMS = frozenset({"python", "local", "local-python", "software"})


def is_python_sandbox_system(value: object) -> bool:
    """Return whether a stored sandbox runtime means the software sandbox."""

    return str(value or "").strip().lower().replace("_", "-") in PYTHON_SANDBOX_SYSTEMS


@dataclass(frozen=True)
class LocalSandboxConfig:
    """Configuration for a local per-user, per-chat sandbox."""

    root: Path
    home: Path
    current_dir: Path | None = None
    allow_subprocess: bool = False
    env: dict[str, str] = field(default_factory=dict)
    trusted_roots: tuple[Path, ...] = field(default_factory=tuple)

    def sandbox_context_prompt(self) -> str:
        trusted_roots = [
            root.expanduser().resolve()
            for root in self.trusted_roots
            if root.expanduser().resolve() != self.root.expanduser().resolve()
        ]
        extra = ""
        if trusted_roots:
            extra = (
                "- Additional trusted agent data roots:\n"
                + "\n".join(f"  - {root}" for root in trusted_roots)
                + "\n"
            )
        return (
            "## Python Sandbox\n"
            "- Commands run in a per-chat software sandbox rooted at the chat workspace.\n"
            f"- The workspace root is: {self.root.expanduser().resolve()}.\n"
            f"- The sandbox HOME is: {self.home.expanduser().resolve()}.\n"
            f"{extra}"
            "- Work only with files inside the workspace root or trusted agent data roots. "
            "Commands that target parent folders or absolute host paths are blocked by "
            "command policy."
        )


class LocalSandboxSession:
    """Strict local sandbox that executes a constrained command set in Python."""

    def __init__(self, config: LocalSandboxConfig) -> None:
        self.config = config
        self.root = config.root.expanduser().resolve()
        self.home = config.home.expanduser().resolve()
        self.trusted_roots = self._normalize_trusted_roots(config.trusted_roots)
        self.current_dir = (
            self.root
            if config.current_dir is None
            else config.current_dir.expanduser().resolve()
        )
        if not self._inside_root(self.current_dir):
            self.current_dir = self.root
        self.ensure()

    @property
    def is_running(self) -> bool:
        return True

    @property
    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.config.env)
        env["HOME"] = str(self.home)
        env["ANOMX_HOME"] = str(self.home / ".anomx")
        env["ANOMX_SANDBOX_ROOT"] = str(self.root)
        return env

    def ensure(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / ".anomx").mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.current_dir.mkdir(parents=True, exist_ok=True)

    def exec_command(
        self,
        command: str,
        *,
        timeout: float = COMMAND_TIMEOUT_SECONDS,
    ) -> str:
        normalized = command.strip()
        if not normalized:
            return "Command is empty."
        try:
            if self._has_shell_syntax(normalized):
                if self.config.allow_subprocess:
                    return self._run_shell_subprocess(normalized, timeout=timeout)
                return (
                    "Local sandbox blocked this command: shell operators, environment "
                    "expansion, and redirection are not available in strict sandbox mode."
                )
            parts = shell_split(normalized)
        except OSError as error:
            return f"Command failed: {error}"
        except subprocess.TimeoutExpired as error:
            return self._limit_output(command_timeout_output(error, timeout))
        except ValueError as error:
            return f"Local sandbox blocked this command: {error}"
        if not parts:
            return "Command is empty."

        executable = Path(parts[0]).name
        args = parts[1:]
        try:
            if executable == "pwd":
                return self._cmd_pwd(args)
            if executable == "cd":
                return self._cmd_cd(args)
            if executable == "ls":
                return self._cmd_ls(args)
            if executable == "cat":
                return self._cmd_cat(args)
            if executable == "head":
                return self._cmd_head_tail(args, tail=False)
            if executable == "tail":
                return self._cmd_head_tail(args, tail=True)
            if executable == "wc":
                return self._cmd_wc(args)
            if executable == "mkdir":
                return self._cmd_mkdir(args)
            if executable == "touch":
                return self._cmd_touch(args)
            if executable in {"rm", "unlink"}:
                return self._cmd_rm(args)
            if executable == "rmdir":
                return self._cmd_rmdir(args)
            if executable == "cp":
                return self._cmd_cp(args)
            if executable == "mv":
                return self._cmd_mv(args)
            if executable == "echo":
                return " ".join(args)
            if executable in {"grep", "rg"}:
                return self._cmd_grep(args)
            if executable == "find":
                return self._cmd_find(args)
            if executable == "whoami":
                return "anomx-agent"
            if executable == "which":
                return self._cmd_which(args)
            if executable == "true":
                return "Command exited with status 0."
            if executable == "false":
                return "Command exited with status 1."
            if self.config.allow_subprocess:
                return self._run_subprocess(parts, timeout=timeout)
            return (
                "Local sandbox blocked unsupported command "
                f"`{executable}`. Use the built-in file and inspection commands."
            )
        except OSError as error:
            return f"Command failed: {error}"
        except subprocess.TimeoutExpired as error:
            return self._limit_output(command_timeout_output(error, timeout))
        except ValueError as error:
            return f"Local sandbox blocked this command: {error}"

    def _cmd_pwd(self, args: list[str]) -> str:
        if args:
            raise ValueError("pwd does not accept arguments.")
        return str(self.current_dir)

    def _cmd_cd(self, args: list[str]) -> str:
        if len(args) > 1:
            raise ValueError("cd accepts at most one path.")
        target = self._resolve_path(args[0] if args else ".")
        if not target.is_dir():
            raise ValueError("cd target is not a directory.")
        self.current_dir = target
        return str(self.current_dir)

    def _cmd_ls(self, args: list[str]) -> str:
        show_all = False
        long = False
        paths: list[str] = []
        for arg in args:
            if arg.startswith("-"):
                show_all = show_all or "a" in arg
                long = long or "l" in arg
                continue
            paths.append(arg)
        targets = [self._resolve_path(path) for path in (paths or ["."])]
        chunks: list[str] = []
        for target in targets:
            if not target.exists():
                raise ValueError(f"Path does not exist: {target}")
            if target.is_file():
                chunks.append(self._format_ls_entry(target, long))
                continue
            entries = sorted(
                entry for entry in target.iterdir()
                if show_all or not entry.name.startswith(".")
            )
            if len(targets) > 1:
                chunks.append(f"{target}:")
            chunks.extend(self._format_ls_entry(entry, long) for entry in entries)
        return self._limit_output("\n".join(chunks))

    def _cmd_cat(self, args: list[str]) -> str:
        if not args:
            raise ValueError("cat requires at least one file.")
        parts = [
            self._read_file(self._resolve_path(path))
            for path in args
            if not path.startswith("-")
        ]
        return self._limit_output("\n".join(parts))

    def _cmd_head_tail(self, args: list[str], *, tail: bool) -> str:
        count = 10
        paths: list[str] = []
        index = 0
        while index < len(args):
            arg = args[index]
            if arg == "-n" and index + 1 < len(args):
                count = max(0, int(args[index + 1]))
                index += 2
                continue
            if arg.startswith("-") and arg[1:].isdigit():
                count = max(0, int(arg[1:]))
                index += 1
                continue
            if arg.startswith("-"):
                index += 1
                continue
            paths.append(arg)
            index += 1
        if not paths:
            raise ValueError("head/tail requires at least one file.")
        output: list[str] = []
        for raw_path in paths:
            lines = self._read_file(self._resolve_path(raw_path)).splitlines()
            selected = lines[-count:] if tail else lines[:count]
            output.extend(selected)
        return self._limit_output("\n".join(output))

    def _cmd_wc(self, args: list[str]) -> str:
        flags = {arg for arg in args if arg.startswith("-")}
        paths = [arg for arg in args if not arg.startswith("-")]
        if not paths:
            raise ValueError("wc requires at least one file.")
        show_lines = not flags or any("l" in flag for flag in flags)
        show_words = not flags or any("w" in flag for flag in flags)
        show_bytes = not flags or any("c" in flag for flag in flags)
        rows: list[str] = []
        for raw_path in paths:
            text = self._read_file(self._resolve_path(raw_path))
            values: list[str] = []
            if show_lines:
                values.append(str(len(text.splitlines())))
            if show_words:
                values.append(str(len(text.split())))
            if show_bytes:
                values.append(str(len(text.encode())))
            rows.append(" ".join([*values, raw_path]))
        return "\n".join(rows)

    def _cmd_mkdir(self, args: list[str]) -> str:
        parents = "-p" in args
        paths = [arg for arg in args if not arg.startswith("-")]
        if not paths:
            raise ValueError("mkdir requires at least one path.")
        for raw_path in paths:
            self._resolve_path(raw_path, for_write=True).mkdir(
                parents=parents,
                exist_ok=parents,
            )
        return "Directories created."

    def _cmd_touch(self, args: list[str]) -> str:
        paths = [arg for arg in args if not arg.startswith("-")]
        if not paths:
            raise ValueError("touch requires at least one file.")
        for raw_path in paths:
            path = self._resolve_path(raw_path, for_write=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        return "Files touched."

    def _cmd_rm(self, args: list[str]) -> str:
        recursive = any("r" in arg or "R" in arg for arg in args if arg.startswith("-"))
        force = any("f" in arg for arg in args if arg.startswith("-"))
        paths = [arg for arg in args if not arg.startswith("-")]
        if not paths:
            raise ValueError("rm requires at least one path.")
        for raw_path in paths:
            path = self._resolve_path(raw_path)
            if self._is_trusted_root(path):
                raise ValueError("refusing to remove sandbox trust root.")
            if not path.exists():
                if force:
                    continue
                raise ValueError(f"Path does not exist: {raw_path}")
            if path.is_dir():
                if not recursive:
                    raise ValueError(f"Path is a directory: {raw_path}")
                shutil.rmtree(path)
            else:
                path.unlink()
        return "Paths removed."

    def _cmd_rmdir(self, args: list[str]) -> str:
        paths = [arg for arg in args if not arg.startswith("-")]
        if not paths:
            raise ValueError("rmdir requires at least one directory.")
        for raw_path in paths:
            path = self._resolve_path(raw_path)
            if self._is_trusted_root(path):
                raise ValueError("refusing to remove sandbox trust root.")
            path.rmdir()
        return "Directories removed."

    def _cmd_cp(self, args: list[str]) -> str:
        recursive = any("r" in arg or "R" in arg for arg in args if arg.startswith("-"))
        paths = [arg for arg in args if not arg.startswith("-")]
        if len(paths) < 2:
            raise ValueError("cp requires source and destination.")
        sources = [self._resolve_path(path) for path in paths[:-1]]
        destination = self._resolve_path(paths[-1], for_write=True)
        if len(sources) > 1 and not destination.is_dir():
            raise ValueError("cp destination must be a directory for multiple sources.")
        for source in sources:
            target = destination / source.name if destination.is_dir() else destination
            if source.is_dir():
                if not recursive:
                    raise ValueError(f"Source is a directory: {source.name}")
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        return "Paths copied."

    def _cmd_mv(self, args: list[str]) -> str:
        paths = [arg for arg in args if not arg.startswith("-")]
        if len(paths) < 2:
            raise ValueError("mv requires source and destination.")
        sources = [self._resolve_path(path) for path in paths[:-1]]
        destination = self._resolve_path(paths[-1], for_write=True)
        if len(sources) > 1 and not destination.is_dir():
            raise ValueError("mv destination must be a directory for multiple sources.")
        for source in sources:
            target = destination / source.name if destination.is_dir() else destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
        return "Paths moved."

    def _cmd_grep(self, args: list[str]) -> str:
        flags = [arg for arg in args if arg.startswith("-")]
        positional = [arg for arg in args if not arg.startswith("-")]
        if not positional:
            raise ValueError("grep requires a pattern.")
        pattern = positional[0]
        paths = positional[1:] or ["."]
        ignore_case = any("i" in flag for flag in flags)
        regex_flags = re.IGNORECASE if ignore_case else 0
        expression = re.compile(pattern, regex_flags)
        rows: list[str] = []
        for raw_path in paths:
            path = self._resolve_path(raw_path)
            for file_path in self._iter_files(path):
                with file_path.open(encoding="utf-8", errors="replace") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if expression.search(line):
                            rows.append(f"{file_path}:{line_number}:{line.rstrip()}")
        return self._limit_output("\n".join(rows) or "No matches.")

    def _cmd_find(self, args: list[str]) -> str:
        if any(arg in {"-exec", "-execdir", "-delete"} for arg in args):
            raise ValueError("find action is not available in local sandbox.")
        roots: list[str] = []
        max_depth: int | None = None
        type_filter = ""
        name_pattern = ""
        index = 0
        while index < len(args):
            arg = args[index]
            if arg == "-maxdepth" and index + 1 < len(args):
                max_depth = max(0, int(args[index + 1]))
                index += 2
                continue
            if arg == "-type" and index + 1 < len(args):
                type_filter = args[index + 1]
                index += 2
                continue
            if arg == "-name" and index + 1 < len(args):
                name_pattern = args[index + 1]
                index += 2
                continue
            if arg.startswith("-"):
                index += 1
                continue
            roots.append(arg)
            index += 1
        resolved_roots = [self._resolve_path(path) for path in (roots or ["."])]
        rows: list[str] = []
        for root in resolved_roots:
            if root.is_file():
                if self._find_matches(root, root, max_depth, type_filter, name_pattern):
                    rows.append(str(root))
                continue
            for path in root.rglob("*"):
                self._assert_inside_root(path.resolve())
                if self._find_matches(path, root, max_depth, type_filter, name_pattern):
                    rows.append(str(path))
        return self._limit_output("\n".join(rows))

    def _find_matches(
        self,
        path: Path,
        root: Path,
        max_depth: int | None,
        type_filter: str,
        name_pattern: str,
    ) -> bool:
        if max_depth is not None:
            depth = len(path.relative_to(root).parts)
            if depth > max_depth:
                return False
        if type_filter == "f" and not path.is_file():
            return False
        if type_filter == "d" and not path.is_dir():
            return False
        return not (name_pattern and not path.match(name_pattern))

    def _cmd_which(self, args: list[str]) -> str:
        if not args:
            raise ValueError("which requires a command name.")
        builtins = {
            "cat", "cd", "cp", "echo", "find", "grep", "head", "ls", "mkdir",
            "mv", "pwd", "rg", "rm", "rmdir", "tail", "touch", "wc", "which",
            "whoami",
        }
        rows: list[str] = []
        for arg in args:
            if arg in builtins:
                rows.append(f"{arg}: local-sandbox builtin")
                continue
            if self.config.allow_subprocess:
                resolved = shutil.which(arg, path=self.env.get("PATH"))
                if resolved:
                    rows.append(resolved)
                    continue
            rows.append(f"{arg} not found")
        return "\n".join(rows)

    def _run_shell_subprocess(self, command: str, *, timeout: float) -> str:
        shell = shutil.which("bash") or shutil.which("sh") or "/bin/sh"
        result = subprocess.run(
            command,
            cwd=self.current_dir,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="backslashreplace",
            shell=True,
            executable=shell,
        )
        output = "\n".join(
            part for part in (result.stdout.strip(), result.stderr.strip()) if part
        )
        if result.returncode != 0:
            output = (
                f"[exit {result.returncode}]\n{output}"
                if output
                else f"[exit {result.returncode}]"
            )
        return self._limit_output(
            output or f"Command exited with status {result.returncode}."
        )

    def _run_subprocess(self, parts: list[str], *, timeout: float) -> str:
        result = subprocess.run(
            parts,
            cwd=self.current_dir,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="backslashreplace",
        )
        output = "\n".join(
            part for part in (result.stdout.strip(), result.stderr.strip()) if part
        )
        if result.returncode != 0:
            output = (
                f"[exit {result.returncode}]\n{output}"
                if output
                else f"[exit {result.returncode}]"
            )
        return self._limit_output(
            output or f"Command exited with status {result.returncode}."
        )

    def _read_file(self, path: Path) -> str:
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path.name}")
        return path.read_text(encoding="utf-8", errors="replace")

    def _iter_files(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        if not path.is_dir():
            raise ValueError(f"Path is not a file or directory: {path.name}")
        return [
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and self._inside_root(candidate.resolve())
        ]

    def _resolve_path(self, raw_path: str, *, for_write: bool = False) -> Path:
        raw = raw_path.strip()
        if raw == "~" or raw.startswith("~/"):
            suffix = raw[2:] if raw.startswith("~/") else ""
            candidate = self.home / suffix
        else:
            candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.current_dir / candidate
        if for_write and not candidate.exists():
            parent = candidate.parent.resolve()
            self._assert_inside_root(parent)
            return parent / candidate.name
        resolved = candidate.resolve()
        self._assert_inside_root(resolved)
        return resolved

    def _assert_inside_root(self, path: Path) -> None:
        if not self._inside_root(path):
            raise ValueError(f"path is outside the sandbox root: {path}")

    def _inside_root(self, path: Path) -> bool:
        return any(path == root or root in path.parents for root in self.trusted_roots)

    def _is_trusted_root(self, path: Path) -> bool:
        return any(path == root for root in self.trusted_roots)

    def _normalize_trusted_roots(self, roots: tuple[Path, ...]) -> tuple[Path, ...]:
        normalized: list[Path] = []
        for root in (self.root, *roots):
            resolved = root.expanduser().resolve()
            if resolved not in normalized:
                normalized.append(resolved)
        return tuple(normalized)

    def _format_ls_entry(self, path: Path, long: bool) -> str:
        if not long:
            return path.name + ("/" if path.is_dir() else "")
        stat = path.stat()
        kind = "d" if path.is_dir() else "-"
        return f"{kind} {stat.st_size:>10} {path.name}"

    def _has_shell_syntax(self, command: str) -> bool:
        return any(
            token in command
            for token in ("|", ";", "&&", "||", ">", "<", "`", "$", "\n")
        )

    def _limit_output(self, output: str) -> str:
        if len(output) <= MAX_OUTPUT_CHARS:
            return output
        return output[:MAX_OUTPUT_CHARS] + "\n[... output truncated by local sandbox ...]"
