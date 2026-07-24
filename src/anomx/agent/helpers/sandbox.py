"""Sandbox mode: containerised execution via Docker or Podman."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anomx.agent.helpers.tool_manager import (
    COMMAND_TIMEOUT_SECONDS,
    command_timeout_output,
)

SANDBOX_IMAGE = "ubuntu:24.04"
StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class SandboxConfig:
    """Sandbox configuration from the user's config.toml."""

    enabled: bool = False
    system: str = "docker"
    method: str = "mount"
    cpu_limit: str = "2"
    ram_limit: str = "4g"
    hd_limit: str = "10g"
    copy_threshold_bytes: int = 2_000_000_000
    strategy: str = "stop"

    @property
    def method_label(self) -> str:
        return "mounted" if self.method == "mount" else "copied"

    def sandbox_context_prompt(self) -> str:
        return (
            "## Sandbox\n"
            "- Commands run inside a disposable container. "
            "The project directory is located at /project.\n"
            f"- Project files are {self.method_label} into the container.\n"
            "- You can install packages, create files, and run commands normally."
        )


SANDBOX_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".next",
        ".turbo",
        "dist",
        "build",
        "target",
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        ".nox",
    }
)


def sandbox_config_from_dict(d: Mapping[str, Any]) -> SandboxConfig:
    return SandboxConfig(
        enabled=bool(d.get("sandbox_enabled", False)),
        system=str(d.get("sandbox_system", "docker")),
        method=str(d.get("sandbox_method", "mount")),
        cpu_limit=str(d.get("sandbox_cpu_limit", "2")),
        ram_limit=str(d.get("sandbox_ram_limit", "4g")),
        hd_limit=str(d.get("sandbox_hd_limit", "10g")),
        strategy=str(d.get("sandbox_strategy", "stop")),
    )


def detect_container_runtime() -> str | None:
    for candidate in ("docker", "podman"):
        if shutil.which(candidate) is not None:
            return candidate
    return None


def project_size_bytes(path: Path) -> int:
    total = 0
    abs_path = path.resolve()
    ignored = frozenset(SANDBOX_IGNORED_DIRS)

    for dirpath_str, dirnames, filenames in os.walk(abs_path):
        dirpath = Path(dirpath_str)
        dirnames[:] = [d for d in dirnames if d not in ignored and not d.startswith(".")]

        for name in filenames:
            if name in ignored or name.endswith((".pyc", ".pyo", ".swp", ".swo")):
                continue
            try:
                fp = dirpath / name
                total += fp.stat(follow_symlinks=False).st_size
            except OSError:
                pass
    return total


class SandboxCopySizeException(Exception):
    def __init__(self, size_bytes: int, threshold_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.threshold_bytes = threshold_bytes
        super().__init__(f"project size {size_bytes:,d} B exceeds {threshold_bytes:,d} B threshold")


class SandboxSession:
    """A persistent sandbox container session tied to a project hash.

    Usage:
        session = SandboxSession(config, project_path, sandbox_hash="abc123")
        session.start(status_callback=print)
        session.exec_command("ls -la")
        session.exec_command("python script.py")
        session.stop()
    """

    CONTAINER_NAME_PREFIX = "anomx-sandbox-"

    def __init__(
        self,
        config: SandboxConfig,
        project_path: Path,
        sandbox_hash: str = "",
        runtime: str | None = None,
    ) -> None:
        self.config = config
        self.project_path = project_path.resolve()
        self._runtime = (runtime or config.system)
        self._container_id: str | None = None
        self._project_dir = Path("/project")
        self._container_name = f"{self.CONTAINER_NAME_PREFIX}{sandbox_hash}"

    @property
    def container_id(self) -> str | None:
        return self._container_id

    @property
    def container_name(self) -> str:
        return self._container_name

    @property
    def is_running(self) -> bool:
        if self._container_id is None:
            return False
        result = subprocess.run(
            [self._runtime, "ps", "--filter", f"id={self._container_id}",
             "--format", "{{.ID}}"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0 and result.stdout.strip() == self._container_id[:12]

    def _run(
        self,
        args: list[str],
        *,
        timeout: float = 120,
        capture: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [self._runtime] + args
        return subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout,
            errors="backslashreplace",
        )

    def _run_captured(
        self,
        args: list[str],
        *,
        timeout: float = 120,
    ) -> str:
        result = self._run(args, timeout=timeout, capture=True)
        output = "\n".join(
            part for part in (result.stdout.strip(), result.stderr.strip()) if part
        )
        if result.returncode != 0:
            if not output:
                output = f"[exit {result.returncode}]"
            else:
                output = f"[exit {result.returncode}]\n{output}"
        return output

    # --- Lifecycle ---

    def start(
        self,
        *,
        status_callback: StatusCallback | None = None,
    ) -> bool:
        self._ensure_runtime()

        existing_id = self._find_existing_container()
        if existing_id is not None:
            self._container_id = existing_id
            if not self._is_container_running(existing_id):
                if status_callback:
                    status_callback("Starting sandbox container")
                self._run(["start", existing_id], timeout=30)
            if status_callback:
                status_callback("Sandbox startup completed")
            return True

        if status_callback:
            status_callback("Pulling sandbox image")

        pull = self._run(["pull", SANDBOX_IMAGE], timeout=300)
        if pull.returncode != 0:
            raise RuntimeError(
                f"Failed to pull sandbox image {SANDBOX_IMAGE}: {pull.stderr.strip()}"
            )

        if status_callback:
            status_callback("Starting sandbox container")

        if status_callback and self.config.method == "copy":
            status_callback("Evaluating project size")

        project_dir, mount_arg = self._prepare_project()

        run_args: list[str] = [
            "run", "-d",
            "--name", self._container_name,
            "--cpus", self.config.cpu_limit,
            "--memory", self.config.ram_limit,
            "-v", mount_arg,
            "-w", str(project_dir),
            "--init",
        ]
        run_args.extend([SANDBOX_IMAGE, "tail", "-f", "/dev/null"])

        result = self._run(run_args, timeout=60, capture=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start sandbox container: {result.stderr.strip()}"
            )

        self._container_id = result.stdout.strip()

        if status_callback:
            status_callback("Sandbox startup completed")

        return True

    def stop(self) -> None:
        if self._container_id is None:
            return
        with suppress(Exception):
            self._run(["stop", self._container_id], timeout=30)
        self._container_id = None

    def remove(self) -> None:
        if self._container_id is None:
            return
        with suppress(Exception):
            self._run(["rm", "-f", self._container_id], timeout=30)
        self._container_id = None

    def ensure_started(
        self,
        *,
        status_callback: StatusCallback | None = None,
    ) -> bool:
        return self.start(status_callback=status_callback)

    @staticmethod
    def stop_all(runtime: str = "docker") -> int:
        result = subprocess.run(
            [runtime, "ps", "-q", "--filter",
             f"name={SandboxSession.CONTAINER_NAME_PREFIX}"],
            capture_output=True, text=True, timeout=30,
        )
        ids = [cid for cid in result.stdout.strip().splitlines() if cid.strip()]
        if not ids:
            return 0
        subprocess.run(
            [runtime, "stop"] + ids,
            capture_output=True, text=True, timeout=60,
        )
        return len(ids)

    @staticmethod
    def remove_all(runtime: str = "docker") -> int:
        result = subprocess.run(
            [runtime, "ps", "-aq", "--filter",
             f"name={SandboxSession.CONTAINER_NAME_PREFIX}"],
            capture_output=True, text=True, timeout=30,
        )
        ids = [cid for cid in result.stdout.strip().splitlines() if cid.strip()]
        if not ids:
            return 0
        subprocess.run(
            [runtime, "rm", "-f"] + ids,
            capture_output=True, text=True, timeout=30,
        )
        return len(ids)

    def exec_command(
        self,
        command: str,
        *,
        timeout: float = COMMAND_TIMEOUT_SECONDS,
    ) -> str:
        if self._container_id is None:
            return "[sandbox: container not running]"

        exec_args: list[str] = [
            "exec", "-i",
            self._container_id,
        ]
        exec_args.extend(["/bin/bash", "-c", command])

        try:
            return self._run_captured(exec_args, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            return command_timeout_output(error, timeout)

    # --- Internals ---

    def _find_existing_container(self) -> str | None:
        result = self._run(
            ["ps", "-a", "--filter", f"name=^{self._container_name}$",
             "--format", "{{.ID}}"],
            timeout=15,
        )
        cid = result.stdout.strip()
        return cid if cid else None

    def _is_container_running(self, container_id: str) -> bool:
        result = self._run(
            ["ps", "--filter", f"id={container_id}", "--format", "{{.ID}}"],
            timeout=15,
        )
        return result.stdout.strip() == container_id[:12]

    def _ensure_runtime(self) -> None:
        if shutil.which(self._runtime) is None:
            raise RuntimeError(
                f"configured container runtime '{self._runtime}' not found. "
                f"Install {self._runtime} or change the sandbox setting to an available runtime."
            )

    def _prepare_project(self) -> tuple[Path, str]:
        project_dir = Path("/project")
        if self.config.method == "mount":
            mount_arg = f"{self.project_path}:{project_dir}"
            return project_dir, mount_arg

        size = project_size_bytes(self.project_path)
        if size > self.config.copy_threshold_bytes:
            raise SandboxCopySizeException(size, self.config.copy_threshold_bytes)

        temp_root = Path(tempfile.mkdtemp(prefix="anomx-sandbox-"))
        dest = temp_root / "project"
        shutil.copytree(
            str(self.project_path),
            str(dest),
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc",
                "node_modules", ".venv", "venv",
                ".next", ".turbo", "dist", "build", "target",
            ),
            symlinks=True,
        )
        mount_arg = f"{dest}:{project_dir}"
        return project_dir, mount_arg

    def build_run_command(
        self,
        image: str,
        command: list[str],
        *,
        interactive: bool = True,
        rm: bool = True,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        cmd = [self._runtime, "run"]

        if interactive:
            cmd.append("-it")
        if rm:
            cmd.append("--rm")

        cmd.extend(["--cpus", self.config.cpu_limit])
        cmd.extend(["--memory", self.config.ram_limit])

        if extra_args:
            cmd.extend(extra_args)

        project_dir, mount_arg = self._prepare_project()
        cmd.extend(["-v", mount_arg])

        if workdir is not None:
            cmd.extend(["-w", workdir])
        else:
            cmd.extend(["-w", str(project_dir)])

        if env:
            for k, v in env.items():
                cmd.extend(["-e", f"{k}={v}"])

        cmd.append(image)
        cmd.extend(command)
        return cmd

    def launch(self, command: list[str]) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
