"""Portable skill resources shared by the runtime and connected platforms."""

from __future__ import annotations

import base64
import binascii
import re
from typing import Literal, TypedDict

MAX_SKILL_FILES = 512
MAX_SKILL_FILE_BYTES = 8 * 1024 * 1024
MAX_SKILL_BYTES = 32 * 1024 * 1024
RESERVED_SKILL_PATHS = {"skill.md", "readme.md", ".anomx_builtin", ".anomx_platform"}


class SkillFile(TypedDict):
    """JSON representation of a UTF-8 resource or base64-encoded binary asset."""

    content: str
    encoding: Literal["utf-8", "base64"]


def validate_skill_path(path: object) -> str:
    """Reject paths that cannot be materialized safely on supported hosts."""

    if not isinstance(path, str) or not path or len(path) > 240:
        raise ValueError("Skill files need a relative path of at most 240 characters.")
    parts = path.split("/")
    if any(
        not part
        or part in {".", ".."}
        or part != part.strip()
        or part.endswith(".")
        or re.search(r'[\x00-\x1f\x7f\\:<>"|?*]', part)
        or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part)
        for part in parts
    ):
        raise ValueError(f"Invalid relative skill path: {path!r}.")
    if parts[0].lower() in RESERVED_SKILL_PATHS:
        raise ValueError("SKILL.md and runtime metadata are managed by the skill editor.")
    return path


def skill_file_bytes(entry: SkillFile) -> bytes:
    """Decode a validated resource without executing its contents."""

    if entry["encoding"] == "base64":
        return base64.b64decode(entry["content"], validate=True)
    return entry["content"].encode("utf-8")


def normalize_skill_files(value: object) -> dict[str, SkillFile]:
    """Validate resource paths, encodings, collisions, and decoded size limits.

    Plain string values are accepted as UTF-8 for convenient API authoring.
    Paths are compared case-insensitively to support Linux, macOS and Windows.
    """

    if not isinstance(value, dict):
        raise ValueError("Skill files must be an object keyed by relative path.")
    if len(value) > MAX_SKILL_FILES:
        raise ValueError(f"A skill supports at most {MAX_SKILL_FILES} supporting files.")
    result: dict[str, SkillFile] = {}
    occupied: set[str] = set()
    directories: set[str] = set()
    total = 0
    for key, raw in value.items():
        path = validate_skill_path(key)
        folded = path.casefold()
        parents = {"/".join(folded.split("/")[:i]) for i in range(1, len(path.split("/")))}
        if folded in occupied or folded in directories or parents & occupied:
            raise ValueError(f"Conflicting skill file path: {path}.")
        if isinstance(raw, str):
            raw = {"content": raw, "encoding": "utf-8"}
        if not isinstance(raw, dict) or not isinstance(raw.get("content"), str):
            raise ValueError(f"{path}: expected content and encoding.")
        encoding = raw.get("encoding", "utf-8")
        if encoding not in ("utf-8", "base64"):
            raise ValueError(f"{path}: encoding must be utf-8 or base64.")
        entry: SkillFile = {"content": raw["content"], "encoding": encoding}
        if len(entry["content"]) > MAX_SKILL_FILE_BYTES * 4 // 3 + 4:
            raise ValueError(f"{path}: file exceeds 8 MiB.")
        try:
            size = len(skill_file_bytes(entry))
        except (ValueError, binascii.Error, UnicodeError) as error:
            raise ValueError(f"{path}: invalid {encoding} content.") from error
        total += size
        if size > MAX_SKILL_FILE_BYTES or total > MAX_SKILL_BYTES:
            raise ValueError("Skill resources exceed the 8 MiB file or 32 MiB total limit.")
        result[path] = entry
        occupied.add(folded)
        directories.update(parents)
    return result
