"""Cell-width aware text helpers used by the terminal renderer."""

from __future__ import annotations

from anomx.agent.helpers.terminal import (
    CODE_END,
    CODE_START,
    cell_width,
    markdown_to_terminal_rendered_lines,
    pad_cells,
    split_overwide_lines,
    strip_code_markers,
    truncate_cells,
)


def test_cell_width_counts_wide_and_combining_characters():
    assert cell_width("abc") == 3
    assert cell_width("日本語") == 6
    assert cell_width("é") == 1
    assert cell_width(f"{CODE_START}code{CODE_END}") == 4


def test_truncate_cells_never_splits_a_wide_glyph():
    assert truncate_cells("日本語", 5) == "日本"
    assert truncate_cells("日本語", 6) == "日本語"
    assert truncate_cells("abc", 0) == ""


def test_pad_cells_pads_to_exact_cell_width():
    assert pad_cells("日本", 7) == "日本   "
    assert cell_width(pad_cells("日本語", 5)) <= 5
    assert pad_cells("ab", 4) == "ab  "


def test_split_overwide_lines_respects_cell_width():
    lines = split_overwide_lines(["日本語日本語"], 5)
    assert all(cell_width(line) <= 5 for line in lines)
    assert "".join(lines) == "日本語日本語"


def test_strip_code_markers_removes_only_markers():
    assert strip_code_markers(f"a {CODE_START}b{CODE_END} c") == "a b c"


def test_markdown_rendering_keeps_wide_text_within_width():
    markdown = (
        "日本語のテキスト、これは長い文章です。" * 3
        + "\n\n| 列 | b |\n|---|---|\n| 日本語日本語 | x |"
    )
    for width in (20, 24, 30):
        lines = markdown_to_terminal_rendered_lines(markdown, width)
        assert max(cell_width(strip_code_markers(line.text)) for line in lines) <= width
