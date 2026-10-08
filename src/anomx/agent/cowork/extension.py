"""Package and locate the bundled ``anomx-editor-context`` VS Code extension.

The extension is plain JavaScript without dependencies (``cowork/vscode``). A ``.vsix`` is a
zip file, so it is built here with the standard library; no Node tooling is needed to
install it. Installing runs code inside VS Code, so the source is meant to be read first.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import cast
from xml.sax.saxutils import escape

EXTENSION_DIR = Path(__file__).parent / "vscode"
PACKAGED_FILES = ("package.json", "extension.js", "README.md")

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/Content-Types">'
    '<Default Extension=".json" ContentType="application/json"/>'
    '<Default Extension=".js" ContentType="application/javascript"/>'
    '<Default Extension=".md" ContentType="text/markdown"/>'
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
