"""Terminal rendering helpers for Anomx agent transcripts."""

from __future__ import annotations

import re
import textwrap
import unicodedata
from dataclasses import dataclass

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
TABLE_DELIMITER_RE = re.compile(r":?-{3,}:?")
ESCAPED_PIPE_PLACEHOLDER = "\0PIPE\0"
CODE_START = "\x01"
CODE_END = "\x02"


def terminal_cell_spans(text: str, width: int) -> list[tuple[int, str, int]]:
    """Return visible glyphs with column offsets, keeping combining marks attached."""
    spans: list[tuple[int, str, int]] = []
    column = 0
    for character in text:
        if spans and (unicodedata.category(character) in {"Mn", "Me"}):
            offset, glyph, cells = spans[-1]
            spans[-1] = (offset, glyph + character, cells)
            continue
        # Joined emoji need a grapheme-aware terminal engine; leave such labels
        # untouched instead of overwriting half of a composed glyph.
        if unicodedata.category(character).startswith("C"):
            return []
        cells = 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        if column + cells > width:
            break
        spans.append((column, character, cells))
        column += cells
    return spans


@dataclass(frozen=True)
class TerminalLine:
    """A rendered terminal line with optional transcript styling metadata."""

    text: str
    style: str = "normal"


@dataclass(frozen=True)
class _MarkdownTable:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


def markdown_to_terminal_lines(markdown: str, width: int) -> list[str]:
    """Convert a Markdown fragment into wrapped terminal transcript lines."""

    return [
        line.text.replace(CODE_START, "").replace(CODE_END, "")
        for line in markdown_to_terminal_rendered_lines(markdown, width)
    ]


def markdown_to_terminal_rendered_lines(markdown: str, width: int) -> list[TerminalLine]:
    """Convert Markdown into wrapped terminal lines with lightweight styles."""

    lines: list[TerminalLine] = []
    in_code_block = False
    safe_width = max(20, width)
    raw_lines = markdown.splitlines() or [""]
    index = 0

    while index < len(raw_lines):
        raw_line = raw_lines[index]
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            index += 1
            continue

        if in_code_block:
            lines.extend(
                TerminalLine(text)
                for text in _wrap_preserving_indent(f"  {line}", safe_width)
            )
            index += 1
            continue

        if not stripped:
            lines.append(TerminalLine(""))
            index += 1
            continue

        table_result = _parse_markdown_table(raw_lines, index)
        if table_result is not None:
            table, index = table_result
            lines.extend(_render_markdown_table(table, safe_width))
            continue

        normalized = _normalize_markdown_line(stripped)
        is_heading = stripped.startswith("#")
        style = "bold" if is_heading else "normal"
        lines.extend(
            TerminalLine(text, style)
            for text in _wrap_preserving_indent(normalized, safe_width)
        )
        index += 1

    while lines and lines[-1].text == "":
        lines.pop()
    return lines or [TerminalLine("")]


def _parse_markdown_table(
    raw_lines: list[str],
    start: int,
) -> tuple[_MarkdownTable, int] | None:
    if start + 1 >= len(raw_lines):
        return None

    header = _parse_pipe_cells(raw_lines[start])
    delimiter = _parse_pipe_cells(raw_lines[start + 1])
    if (
        header is None
        or delimiter is None
        or len(header) < 2
        or len(delimiter) != len(header)
        or not _is_table_delimiter(delimiter)
    ):
        return None

    column_count = len(header)
    rows: list[tuple[str, ...]] = []
    index = start + 2
    while index < len(raw_lines):
        cells = _parse_pipe_cells(raw_lines[index])
        if cells is None or not any(cell.strip() for cell in cells):
            break
        rows.append(_normalize_table_row(cells, column_count))
        index += 1

    return _MarkdownTable(_normalize_table_row(header, column_count), tuple(rows)), index


def _parse_pipe_cells(line: str) -> tuple[str, ...] | None:
    stripped = line.strip()
    if "|" not in stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    working = stripped.replace(r"\|", ESCAPED_PIPE_PLACEHOLDER)
    cells = tuple(
        cell.replace(ESCAPED_PIPE_PLACEHOLDER, "|").strip()
        for cell in working.split("|")
    )
    if len(cells) < 2:
        return None
    return cells


def _is_table_delimiter(cells: tuple[str, ...]) -> bool:
    return bool(cells) and all(
        TABLE_DELIMITER_RE.fullmatch(cell.replace(" ", "")) is not None
        for cell in cells
    )


def _normalize_table_row(cells: tuple[str, ...], column_count: int) -> tuple[str, ...]:
    if len(cells) > column_count:
        cells = (*cells[: column_count - 1], " | ".join(cells[column_count - 1 :]))
    padded = (*cells, *([""] * max(0, column_count - len(cells))))
    return tuple(_normalize_table_cell(cell) for cell in padded[:column_count])


def _normalize_table_cell(cell: str) -> str:
    return " ".join(_normalize_markdown_line(cell.strip()).split())


def _render_markdown_table(table: _MarkdownTable, width: int) -> list[TerminalLine]:
    column_count = len(table.header)
    content_budget = width - ((column_count * 3) + 1)
    if content_budget < column_count:
        return _render_table_fallback(table, width)

    rows = (table.header, *table.rows)
    widths = _table_column_widths(rows, content_budget)
    rendered: list[TerminalLine] = [
        TerminalLine(_table_border(widths, "┌", "┬", "┐"), "table_border")
    ]
    rendered.extend(_table_row_lines(table.header, widths, "table_header"))
    rendered.append(TerminalLine(_table_border(widths, "├", "┼", "┤"), "table_border"))
    for row in table.rows:
        rendered.extend(_table_row_lines(row, widths, "table_row"))
    rendered.append(TerminalLine(_table_border(widths, "└", "┴", "┘"), "table_border"))
    return rendered


def _render_table_fallback(table: _MarkdownTable, width: int) -> list[TerminalLine]:
    rendered: list[TerminalLine] = []
    for row in (table.header, *table.rows):
        rendered.extend(
            TerminalLine(text)
            for text in _wrap_preserving_indent(" | ".join(row), width)
        )
    return rendered


def _table_column_widths(rows: tuple[tuple[str, ...], ...], content_budget: int) -> list[int]:
    column_count = len(rows[0])
    desired = [
        max(1, max(len(row[column]) for row in rows))
        for column in range(column_count)
    ]
    if sum(desired) <= content_budget:
        return desired

    minimum = [
        min(desired[column], max(3, min(_longest_word_width(rows, column), 18)))
        for column in range(column_count)
    ]
    if sum(minimum) > content_budget:
        return _squeezed_column_widths(desired, content_budget)

    widths = minimum[:]
    remaining = content_budget - sum(widths)
    while remaining > 0:
        candidates = [
            (desired[column] - widths[column], column)
            for column in range(column_count)
            if desired[column] > widths[column]
        ]
        if not candidates:
            break
        _, column = max(candidates, key=lambda candidate: candidate[0])
        widths[column] += 1
        remaining -= 1
    return widths


def _longest_word_width(rows: tuple[tuple[str, ...], ...], column: int) -> int:
    return max(
        (
            len(word)
            for row in rows
            for word in row[column].split()
        ),
        default=1,
    )


def _squeezed_column_widths(desired: list[int], content_budget: int) -> list[int]:
    column_count = len(desired)
    widths = [1] * column_count
    remaining = max(0, content_budget - column_count)
    while remaining > 0 and any(
        widths[column] < desired[column] for column in range(column_count)
    ):
        column = max(
            range(column_count),
            key=lambda candidate: desired[candidate] - widths[candidate],
        )
        widths[column] += 1
        remaining -= 1
    return widths


def _table_border(widths: list[int], left: str, separator: str, right: str) -> str:
    return left + separator.join("─" * (width + 2) for width in widths) + right


def _table_row_lines(
    row: tuple[str, ...],
    widths: list[int],
    style: str,
) -> list[TerminalLine]:
    wrapped_cells = [
        _wrap_table_cell(cell, widths[column])
        for column, cell in enumerate(row)
    ]
    row_height = max(len(cell_lines) for cell_lines in wrapped_cells)
    lines: list[TerminalLine] = []
    for line_index in range(row_height):
        cells = [
            (
                wrapped_cells[column][line_index]
                if line_index < len(wrapped_cells[column])
                else ""
            ).ljust(widths[column])
            for column in range(len(widths))
        ]
        lines.append(TerminalLine(f"│ {' │ '.join(cells)} │", style))
    return lines


def _wrap_table_cell(cell: str, width: int) -> list[str]:
    if not cell:
        return [""]
    return textwrap.wrap(
        cell,
        width=max(1, width),
        replace_whitespace=False,
        drop_whitespace=True,
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]


def _normalize_markdown_line(line: str) -> str:
    heading = line.lstrip("#").strip() if line.startswith("#") else line
    linked = MARKDOWN_LINK_RE.sub(r"\1 (\2)", heading)
    code = INLINE_CODE_RE.sub(lambda m: f"{CODE_START}{m.group(1)}{CODE_END}", linked)
    without_emphasis = code.replace("**", "").replace("__", "").replace("*", "")
    return without_emphasis.replace("–", "-")


def _wrap_preserving_indent(line: str, width: int) -> list[str]:
    indent_width = len(line) - len(line.lstrip())
    indent = " " * indent_width
    content = line.lstrip()
    bullet_prefix = _bullet_prefix(content)
    subsequent_indent = indent + (" " * len(bullet_prefix) if bullet_prefix else "")
    return textwrap.wrap(
        indent + content,
        width=width,
        subsequent_indent=subsequent_indent,
        replace_whitespace=False,
        drop_whitespace=True,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [indent]


def _bullet_prefix(content: str) -> str:
    for prefix in ("- ", "* ", "+ "):
        if content.startswith(prefix):
            return prefix
    if len(content) >= 3 and content[0].isdigit() and content[1:3] == ". ":
        return content[:3]
    return ""
