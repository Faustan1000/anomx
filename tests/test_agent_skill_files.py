"""Portable skill bundles and non-destructive synchronization."""

import base64

import pytest

from anomx.agent.skills import (
    BUILTIN_MARKER_NAME,
    DEFAULT_PLATFORM_SKILL_COMMANDS,
    PLATFORM_MARKER_NAME,
    Skill,
    load_builtin_skills,
    load_user_skills,
    parse_skill_markdown,
    skill_to_markdown,
    sync_builtin_skills,
    sync_platform_skills,
    write_user_skill,
)
from anomx.agent.skills.files import normalize_skill_files


def test_sync_materializes_text_and_binary_and_removes_obsolete_resources(tmp_path):
    payload = [
        {
            "command": "inspect-data",
            "name": "Inspect data",
            "instructions": "Read [schema](references/schema.md).",
            "files": {
                "references/schema.md": "# Schema\n",
                "scripts/read.py": {"encoding": "utf-8", "content": "print('ready')\n"},
                "assets/sample.bin": {
                    "encoding": "base64",
                    "content": base64.b64encode(b"\x00\xff").decode(),
                },
            },
        }
    ]
    sync_platform_skills(tmp_path, payload)
    directory = tmp_path / "inspect-data"
    assert (directory / "SKILL.md").read_text() == (directory / "README.md").read_text()
    assert (directory / "references/schema.md").read_text() == "# Schema\n"
    assert (directory / "assets/sample.bin").read_bytes() == b"\x00\xff"
    assert (directory / PLATFORM_MARKER_NAME).is_file()
    assert load_user_skills(tmp_path)[0].source == "platform"
    payload[0]["files"] = {"references/new.md": "Updated"}
    sync_platform_skills(tmp_path, payload)
    assert not (directory / "scripts/read.py").exists()
    assert not (directory / "references/schema.md").exists()
    assert (directory / "references/new.md").read_text() == "Updated"


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/absolute",
        "C:/absolute",
        "a\\b",
        "a/../b",
        "a//b",
        "a/./b",
        "SKILL.md",
        "skill.md",
        "README.md",
        ".anomx_platform",
        "SKILL.md/child",
        "a\x00b",
        "a/CON.txt",
        "a/b.",
        " a",
        "a ",
    ],
)
def test_unsafe_or_reserved_paths_are_rejected(path):
    with pytest.raises(ValueError):
        normalize_skill_files({path: "text"})


@pytest.mark.parametrize(
    "files",
    [
        {"references/a.md": "a", "REFERENCES/A.md": "b"},
        {"references": "a", "references/a.md": "b"},
        {"references/a.md": "a", "references": "b"},
        {"assets/a": {"content": "???", "encoding": "base64"}},
        {"assets/a": {"content": "a", "encoding": "latin-1"}},
        {"assets/a": {"content": 123}},
        {"assets/a": {"content": "a", "encoding": []}},
        ["not-a-file-map"],
    ],
)
def test_invalid_resources_are_rejected(files):
    with pytest.raises(ValueError):
        normalize_skill_files(files)


def test_invalid_update_keeps_previous_bundle(tmp_path):
    sync_platform_skills(
        tmp_path,
        [
            {
                "command": "sample",
                "instructions": "old",
                "files": {"ref.md": "old"},
            }
        ],
    )
    sync_platform_skills(
        tmp_path,
        [
            {
                "command": "sample",
                "instructions": "new",
                "files": {"../escape": "bad"},
            }
        ],
    )
    assert (tmp_path / "sample/ref.md").read_text() == "old"
    assert "old" in (tmp_path / "sample/SKILL.md").read_text()
    assert not (tmp_path.parent / "escape").exists()


def test_local_skill_and_symlink_are_never_replaced(tmp_path):
    skills = tmp_path / "skills"
    local = skills / "custom"
    local.mkdir(parents=True)
    (local / "README.md").write_text("local")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / PLATFORM_MARKER_NAME).write_text("marker")
    (outside / "keep").write_text("keep")
    (skills / "linked").symlink_to(outside, target_is_directory=True)
    sync_platform_skills(skills, [{"command": "custom"}, {"command": "linked"}])
    assert (local / "README.md").read_text() == "local"
    assert (skills / "linked").is_symlink()
    assert (outside / "keep").read_text() == "keep"
    sync_platform_skills(skills, [])
    assert (skills / "linked").is_symlink()


def test_entrypoint_precedence_and_local_resources_survive_edit(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "README.md").write_text("old entrypoint")
    assert load_user_skills(tmp_path)[0].body == "old entrypoint"
    (legacy / "SKILL.md").write_text("new entrypoint")
    assert load_user_skills(tmp_path)[0].body == "new entrypoint"
    (legacy / "reference.txt").write_text("preserve")
    write_user_skill(
        tmp_path,
        Skill(
            command="legacy",
            title="Legacy",
            description="local",
            body="edit",
            source="user",
        ),
    )
    assert (legacy / "reference.txt").read_text() == "preserve"
    assert load_user_skills(tmp_path)[0].body == "edit"


def test_frontmatter_quotes_round_trip():
    skill = Skill(
        command="sample",
        title='Say "hello"',
        description='A colon: and "quotes"',
        body="body",
        source="platform",
    )
    parsed = parse_skill_markdown(
        skill_to_markdown(skill),
        default_command="sample",
        source="platform",
        path=None,
    )
    assert parsed.title == skill.title
    assert parsed.description == skill.description


def test_builtin_entrypoints_and_resources_are_available(tmp_path):
    skills = load_builtin_skills(include_system=True)
    assert {skill.command for skill in skills} == set(DEFAULT_PLATFORM_SKILL_COMMANDS)
    assert all(skill.system and skill.hidden for skill in skills)
    sync_builtin_skills(tmp_path, include_system=True)
    for command in DEFAULT_PLATFORM_SKILL_COMMANDS:
        assert (tmp_path / command / "SKILL.md").is_file()
        assert (tmp_path / command / "README.md").is_file()
        assert (tmp_path / command / BUILTIN_MARKER_NAME).is_file()
    assert (tmp_path / "manage-jobs/references/jobs.md").is_file()


def test_decoded_limits(monkeypatch):
    import anomx.agent.skills.files as resources

    monkeypatch.setattr(resources, "MAX_SKILL_FILES", 2)
    monkeypatch.setattr(resources, "MAX_SKILL_FILE_BYTES", 5)
    monkeypatch.setattr(resources, "MAX_SKILL_BYTES", 8)
    assert normalize_skill_files({"a": "1234", "b": "1234"})
    examples = ({"a": "123456"}, {"a": "12345", "b": "1234"}, {"a": "", "b": "", "c": ""})
    for files in examples:
        with pytest.raises(ValueError):
            normalize_skill_files(files)


def test_unchanged_platform_skills_keep_their_files_and_timestamps(tmp_path):
    payload = [{"command": "inspect", "instructions": "Inspect data", "files": {"ref.txt": "data"}}]
    sync_platform_skills(tmp_path, payload)
    paths = tuple((tmp_path / "inspect").rglob("*"))
    before = {path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in paths}
    sync_platform_skills(tmp_path, payload)
    assert {path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in paths} == before
    (tmp_path / "inspect/ref.txt").write_text("Changed locally")
    sync_platform_skills(tmp_path, payload)
    assert (tmp_path / "inspect/ref.txt").read_text() == "data"
