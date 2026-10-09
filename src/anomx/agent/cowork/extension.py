"""Package and locate the bundled ``anomx-editor-context`` VS Code extension.

The extension is plain JavaScript without dependencies (``cowork/vscode``). A ``.vsix`` is a
zip file, so it is built here with the standard library; no Node tooling is needed to
install it. Installing runs code inside VS Code, so the source is meant to be read first.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast
from xml.sax.saxutils import escape

INSTALL_CLI_COMMAND = "anomx --install-vscode"
"""The one command that builds and installs the extension (shown to the user)."""

EXTENSION_DIR = Path(__file__).parent / "vscode"
PACKAGED_FILES = (
    "package.json",
    "extension.js",
    "notes.js",
    "comments.js",
    "README.md",
    "media/anomx.svg",
)

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/Content-Types">'
    '<Default Extension=".json" ContentType="application/json"/>'
    '<Default Extension=".js" ContentType="application/javascript"/>'
    '<Default Extension=".md" ContentType="text/markdown"/>'
    '<Default Extension=".svg" ContentType="image/svg+xml"/>'
    '<Default Extension=".vsixmanifest" ContentType="text/xml"/>'
    "</Types>\n"
)


def extension_manifest() -> dict[str, object]:
    manifest = json.loads((EXTENSION_DIR / "package.json").read_text(encoding="utf-8"))
    return cast("dict[str, object]", manifest)


def vsix_name() -> str:
    manifest = extension_manifest()
    return f"{manifest['publisher']}.{manifest['name']}-{manifest['version']}.vsix"


def _vsix_manifest() -> str:
    manifest = extension_manifest()
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<PackageManifest Version="2.0.0" '
        'xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011" '
        'xmlns:d="http://schemas.microsoft.com/developer/vsx-schema-design/2011">'
        "<Metadata>"
        f'<Identity Language="en-US" Id="{escape(str(manifest["name"]))}" '
        f'Version="{escape(str(manifest["version"]))}" '
        f'Publisher="{escape(str(manifest["publisher"]))}"/>'
        f"<DisplayName>{escape(str(manifest['displayName']))}</DisplayName>"
        f'<Description xml:space="preserve">{escape(str(manifest["description"]))}</Description>'
        "</Metadata>"
        '<Installation><InstallationTarget Id="Microsoft.VisualStudio.Code"/></Installation>'
        "<Dependencies/>"
        "<Assets>"
        '<Asset Type="Microsoft.VisualStudio.Code.Manifest" '
        'Path="extension/package.json" Addressable="true"/>'
        "</Assets>"
        "</PackageManifest>\n"
    )


def build_vsix(destination: Path) -> Path:
    """Write the extension as ``<publisher>.<name>-<version>.vsix`` into ``destination``."""

    destination.mkdir(parents=True, exist_ok=True)
    target = destination / vsix_name()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
        archive.writestr("extension.vsixmanifest", _vsix_manifest())
        for name in PACKAGED_FILES:
            archive.write(EXTENSION_DIR / name, f"extension/{name}")
    return target


def install_command(vsix: Path) -> str:
    """Shell command that installs the package into the VS Code the terminal belongs to."""

    return f"code --install-extension {vsix}"


def in_vscode_terminal(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this terminal belongs to a VS Code window (also a remote one)."""

    env = os.environ if environ is None else environ
    return env.get("TERM_PROGRAM") == "vscode" or bool(env.get("VSCODE_IPC_HOOK_CLI"))


def _version_tuple(text: object) -> tuple[int, ...]:
    parts = str(text or "").split(".")
    return tuple(int(part) if part.isdigit() else 0 for part in parts) or (0,)


def extension_needs_install(home_root: Path, environ: Mapping[str, str] | None = None) -> str:
    """Why the extension should be (re)installed in this VS Code, or ``""`` if it is fine.

    Returns ``"missing"`` when it never wrote its context file and ``"outdated"`` when the
    installed version is older than the bundled one (versions before 0.2 wrote none).
    """

    from anomx.agent.cowork.editor import context_path

    if not in_vscode_terminal(environ):
        return ""
    try:
        data = json.loads(context_path(home_root).read_text(encoding="utf-8"))
    except OSError:
        return "missing"
    except ValueError:
        return ""
    installed = data.get("extensionVersion") if isinstance(data, dict) else None
    bundled = extension_manifest()["version"]
    return "outdated" if _version_tuple(installed) < _version_tuple(bundled) else ""


def _live_vscode_sockets(environ: Mapping[str, str]) -> list[str]:
    """Sockets of running VS Code windows, newest first (a terminal opened before a reload
    keeps pointing at a socket that no longer exists)."""

    import socket

    runtime = environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    candidates = sorted(
        Path(runtime).glob("vscode-ipc-*.sock"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    live: list[str] = []
    for path in candidates:
        probe = socket.socket(socket.AF_UNIX)
        probe.settimeout(1)
        try:
            probe.connect(str(path))
        except OSError:
            continue
        finally:
            probe.close()
        live.append(str(path))
    return live


def _run_code(code: str, vsix: Path, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed binary found on PATH, no shell
        [code, "--install-extension", str(vsix), "--force"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=dict(env),
    )


def install_extension(home_root: Path) -> tuple[bool, str]:
    """Build the ``.vsix`` and install it with the ``code`` command of this terminal.

    When the terminal's VS Code socket is stale, running VS Code windows are tried instead.
    Returns ``(ok, message)``; the message tells what to do next, also when it failed.
    """

    vsix = build_vsix(home_root / "vscode")
    code = shutil.which("code")
    manual = f"Install it by hand: VS Code > Extensions > ... > Install from VSIX > {vsix}"
    if code is None:
        return False, f"The `code` command was not found in PATH. {manual}"
    environment = dict(os.environ)
    try:
        result = _run_code(code, vsix, environment)
        if result.returncode != 0 and "ENOENT" in (result.stderr + result.stdout):
            for socket_path in _live_vscode_sockets(environment):
                environment["VSCODE_IPC_HOOK_CLI"] = socket_path
                result = _run_code(code, vsix, environment)
                if result.returncode == 0:
                    break
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"Could not run `code`: {error}. {manual}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        reason = detail[0] if detail else "unknown error"
        return False, (
            f"`code --install-extension` failed: {reason}. Open a NEW terminal in VS Code "
            f"(this one points at a closed window) and run it again. {manual}"
        )
    return True, (
        f"Installed {vsix.name}. Now reload the VS Code window "
        "(Ctrl+Shift+P > Developer: Reload Window)."
    )
