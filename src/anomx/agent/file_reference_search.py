"""Process-isolated, cached workspace file-reference search."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

from anomx.agent.ui.constants import (
    FILE_REFERENCE_FIRST_LEVEL_LIMIT,
    FILE_REFERENCE_INDEX_REFRESH_SECONDS,
    FILE_REFERENCE_LIMIT,
    IGNORED_FILE_REFERENCE_DIRS,
)

_QUERY_CACHE_LIMIT = 96
_INDEX_CACHE_VERSION = 2


@dataclass(frozen=True, slots=True)
class FileReferenceSearchResult:
    """A serializable suggestion returned by the search worker."""

    label: str
    value: str
    query: str
    highlight_spans: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class _SearchResponse:
    query: str
    generation: int
    results: tuple[FileReferenceSearchResult, ...]
    refreshing: bool


class FileReferenceSearchClient:
    """Non-blocking client for a long-lived workspace search process."""

    def __init__(self, workspace_root: Path, cache_path: Path) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        self.cache_path = cache_path.expanduser()
        self._context = multiprocessing.get_context("spawn")
        self._requests: Queue[dict[str, Any]] | None = None
        self._responses: Queue[_SearchResponse] | None = None
        self._process: BaseProcess | None = None
        self._start_lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[FileReferenceSearchResult, ...]] = OrderedDict()
        self._submitted_query: str | None = None
        self._generation = -1
        self._refreshing = False
        self._closed = False

    @property
    def process_id(self) -> int | None:
        """Return the search worker PID once it has started."""

        return self._process.pid if self._process is not None else None

    def start(self) -> None:
        """Start the search worker and return without waiting for its index."""

        if self._closed or self._process is not None:
            return
        with self._start_lock:
            if self._closed or self._process is not None:
                return
            requests: Queue[dict[str, Any]] = self._context.Queue(maxsize=1)
            responses: Queue[_SearchResponse] = self._context.Queue()
            process = self._context.Process(
                target=_search_process_main,
                args=(
                    str(self.workspace_root),
                    str(self.cache_path),
                    requests,
                    responses,
                ),
                name="anomx-file-search",
                daemon=True,
            )
            process.start()
            self._requests = requests
            self._responses = responses
            self._process = process

    def suggestions(self, query: str) -> tuple[FileReferenceSearchResult, ...]:
        """Return cached suggestions immediately and enqueue missing work."""

        normalized = _normalize_query(query)
        self._poll()
        cached = self._cache.get(normalized)
        if cached is not None:
            self._cache.move_to_end(normalized)
            return cached
        self._submit(normalized)
        return self._prefix_fallback(normalized)

    def is_searching(self, query: str) -> bool:
        """Return whether the current query is still being indexed or ranked."""

        normalized = _normalize_query(query)
        self._poll()
        if normalized not in self._cache:
            self._submit(normalized)
            return True
        return self._refreshing and normalized == self._submitted_query

    def close(self) -> None:
        """Stop the worker and release queue resources."""

        if self._closed:
            return
        self._closed = True
        process = self._process
        requests = self._requests
        if process is not None and process.is_alive() and requests is not None:
            try:
                requests.put_nowait({"type": "close"})
            except queue.Full:
                process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        if requests is not None:
            requests.close()
        if self._responses is not None:
            self._responses.close()
        self._process = None
        self._requests = None
        self._responses = None

    def _submit(self, query_text: str) -> None:
        if self._closed:
            return
        self.start()
        if query_text == self._submitted_query:
            return
        requests = self._requests
        if requests is None:
            return
        try:
            requests.put_nowait({"type": "search", "query": query_text})
        except queue.Full:
            return
        self._submitted_query = query_text

    def _poll(self) -> None:
        responses = self._responses
        if responses is None:
            return
        while True:
            try:
                response = responses.get_nowait()
            except queue.Empty:
                return
            if response.generation > self._generation:
                self._cache.clear()
                self._generation = response.generation
            self._cache[response.query] = response.results
            self._cache.move_to_end(response.query)
            while len(self._cache) > _QUERY_CACHE_LIMIT:
                self._cache.popitem(last=False)
            if response.query == self._submitted_query:
                self._refreshing = response.refreshing

    def _prefix_fallback(self, query_text: str) -> tuple[FileReferenceSearchResult, ...]:
        """Narrow a previous small result set while the exact query is pending."""

        if not query_text:
            return ()
        for cached_query in reversed(self._cache):
            if not query_text.startswith(cached_query):
                continue
            narrowed = tuple(
                result
                for result in self._cache[cached_query]
                if _quick_result_match(query_text, result.label)
            )
            if narrowed:
                return narrowed
        return ()


class _WorkspaceFileIndex:
    def __init__(self, workspace_root: Path, cache_path: Path) -> None:
        self.workspace_root = workspace_root
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._paths: tuple[str, ...] = ()
        self._first_level: tuple[str, ...] = ()
        self._generation = 0
        self._refreshing = False
        self._generated_at = 0.0

    def load(self) -> None:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        paths: tuple[str, ...] = ()
        if (
            isinstance(payload, dict)
            and int(payload.get("version", 0) or 0) == _INDEX_CACHE_VERSION
            and str(payload.get("project_path", "")) == str(self.workspace_root)
            and isinstance(payload.get("paths"), list)
        ):
            paths = tuple(
                normalized
                for item in payload["paths"]
                if (normalized := _normalized_path(item))
            )
            generated_at = payload.get("generated_at")
            self._generated_at = (
                float(generated_at) if isinstance(generated_at, int | float) else 0.0
            )
        first_level = _first_level_paths(paths)
        if not first_level:
            first_level = self._read_first_level()
        with self._lock:
            self._paths = paths
            self._first_level = first_level
            self._generation += 1

    def start_refresh(self) -> None:
        with self._lock:
            if self._refreshing:
                return
            age = max(0.0, time.time() - self._generated_at)
            if self._paths and age < FILE_REFERENCE_INDEX_REFRESH_SECONDS:
                return
            self._refreshing = True
        threading.Thread(
            target=self._refresh,
            name="anomx-file-index-scan",
            daemon=True,
        ).start()

    def snapshot(self) -> tuple[tuple[str, ...], tuple[str, ...], int, bool]:
        with self._lock:
            return self._paths, self._first_level, self._generation, self._refreshing

    def _refresh(self) -> None:
        paths = self._scan()
        generated_at = time.time()
        payload = {
            "version": _INDEX_CACHE_VERSION,
            "project_path": str(self.workspace_root),
            "generated_at": generated_at,
            "paths": list(paths),
        }
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_suffix(f"{self.cache_path.suffix}.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            temporary.replace(self.cache_path)
        except OSError:
            pass
        with self._lock:
            self._paths = paths
            self._first_level = _first_level_paths(paths) or self._first_level
            self._generated_at = generated_at
            self._generation += 1
            self._refreshing = False

    def _scan(self) -> tuple[str, ...]:
        paths: list[str] = []
        for root, dirnames, filenames in os.walk(self.workspace_root):
            dirnames[:] = sorted(
                (
                    name
                    for name in dirnames
                    if name not in IGNORED_FILE_REFERENCE_DIRS and not name.startswith(".")
                ),
                key=str.lower,
            )
            filenames.sort(key=str.lower)
            root_path = Path(root)
            try:
                relative_root = root_path.relative_to(self.workspace_root).as_posix()
            except ValueError:
                relative_root = ""
            prefix = f"{relative_root}/" if relative_root and relative_root != "." else ""
            for dirname in dirnames:
                paths.append(f"{prefix}{dirname}/")
            for filename in filenames:
                if filename.endswith((".pyc", ".pyo", ".DS_Store")):
                    continue
                paths.append(f"{prefix}{filename}")
        return tuple(sorted(set(paths), key=str.lower))

    def _read_first_level(self) -> tuple[str, ...]:
        paths: list[str] = []
        try:
            entries = sorted(
                self.workspace_root.iterdir(),
                key=lambda entry: (not entry.is_dir(), entry.name.lower()),
            )
        except OSError:
            return ()
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                if entry.name in IGNORED_FILE_REFERENCE_DIRS:
                    continue
                paths.append(f"{entry.name}/")
            elif entry.is_file() and not entry.name.endswith((".pyc", ".pyo", ".DS_Store")):
                paths.append(entry.name)
            if len(paths) >= FILE_REFERENCE_FIRST_LEVEL_LIMIT:
                break
        return tuple(paths)


def _search_process_main(
    workspace_root: str,
    cache_path: str,
    requests: Queue[dict[str, Any]],
    responses: Queue[_SearchResponse],
) -> None:
    index = _WorkspaceFileIndex(Path(workspace_root), Path(cache_path))
    index.load()
    index.start_refresh()
    result_cache: OrderedDict[
        tuple[int, str], tuple[FileReferenceSearchResult, ...]
    ] = OrderedDict()
    last_query: str | None = None
    last_generation = -1

    while True:
        request: dict[str, Any] | None = None
        try:
            request = requests.get(timeout=0.04)
            while True:
                request = requests.get_nowait()
        except queue.Empty:
            pass

        if request is not None:
            if request.get("type") == "close":
                return
            if request.get("type") == "search":
                last_query = _normalize_query(str(request.get("query") or ""))
                index.start_refresh()

        paths, first_level, generation, refreshing = index.snapshot()
        should_publish = last_query is not None and (
            request is not None or generation != last_generation
        )
        if not should_publish:
            continue
        assert last_query is not None
        cache_key = (generation, last_query)
        results = result_cache.get(cache_key)
        if results is None:
            results = _search_paths(last_query, paths, first_level)
            result_cache[cache_key] = results
            while len(result_cache) > _QUERY_CACHE_LIMIT:
                result_cache.popitem(last=False)
        responses.put(
            _SearchResponse(
                query=last_query,
                generation=generation,
                results=results,
                refreshing=refreshing,
            )
        )
        last_generation = generation


def _search_paths(
    query_text: str,
    paths: tuple[str, ...],
    first_level: tuple[str, ...],
) -> tuple[FileReferenceSearchResult, ...]:
    if not query_text:
        return tuple(
            FileReferenceSearchResult(path, path, "")
            for path in first_level[:FILE_REFERENCE_LIMIT]
        )
    if query_text.endswith("/"):
        children = _directory_children(query_text, paths)
        if children:
            parent_end = min(len(query_text), len(children[0]))
            spans = ((0, parent_end),) if parent_end else ()
            return tuple(
                FileReferenceSearchResult(path, path, query_text, spans)
                for path in children[:FILE_REFERENCE_LIMIT]
            )

    matches: list[tuple[int, int, int, str, FileReferenceSearchResult]] = []
    for path in paths:
        match = _match_path(query_text, path)
        if match is None:
            continue
        rank, score, spans = match
        matches.append(
            (
                rank,
                score,
                len(path),
                path.lower(),
                FileReferenceSearchResult(path, path, query_text, spans),
            )
        )
    matches.sort(key=lambda item: item[:4])
    return tuple(item[4] for item in matches[:FILE_REFERENCE_LIMIT])


def _directory_children(query_text: str, paths: tuple[str, ...]) -> tuple[str, ...]:
    query_path = query_text.strip().lstrip("/")
    directories = tuple(path for path in paths if path.endswith("/"))
    parent = query_path if query_path in directories else ""
    if not parent:
        ranked: list[tuple[int, int, str]] = []
        for directory in directories:
            match = _match_path(query_path.rstrip("/"), directory)
            if match is not None:
                ranked.append((match[0], match[1], directory))
        if ranked:
            parent = min(ranked)[2]
    if not parent:
        return ()
    children: list[str] = []
    for path in paths:
        if not path.startswith(parent) or path == parent:
            continue
        remainder = path[len(parent) :].rstrip("/")
        if remainder and "/" not in remainder:
            children.append(path)
    return tuple(sorted(children, key=lambda path: (not path.endswith("/"), path.lower())))


def _match_path(
    query_text: str,
    path: str,
) -> tuple[int, int, tuple[tuple[int, int], ...]] | None:
    query_lower = query_text.strip().lstrip("/").rstrip("/").lower()
    path_lower = path.rstrip("/").lower()
    if not query_lower:
        return 0, 0, ()
    query_segments = tuple(segment for segment in query_lower.split("/") if segment)
    path_segments = tuple(segment for segment in path_lower.split("/") if segment)
    if not query_segments or len(query_segments) > len(path_segments):
        return None

    if len(query_segments) > 1:
        for start in range(len(path_segments) - len(query_segments) + 1):
            window = path_segments[start : start + len(query_segments)]
            if not all(
                target == query or (index == len(query_segments) - 1 and target.startswith(query))
                for index, (query, target) in enumerate(zip(query_segments, window, strict=True))
            ):
                continue
            if start + len(query_segments) != len(path_segments):
                continue
            literal = "/".join(query_segments)
            span_start = path_lower.find(literal)
            if span_start < 0:
                parent_literal = "/".join(path_segments[start : start + len(query_segments) - 1])
                span_start = path_lower.find(parent_literal)
                span_end = path_lower.find("/", span_start + len(parent_literal)) + 1
                span_end += len(query_segments[-1])
            else:
                span_end = span_start + len(literal)
            return 0 if window[-1] == query_segments[-1] else 1, start, (
                (span_start, max(span_start, span_end)),
            )
        return None

    query = query_segments[0]
    basename = path_segments[-1]
    basename_start = path_lower.rfind(basename)
    if basename == query:
        return 0, 0, ((basename_start, basename_start + len(query)),)
    if basename.startswith(query):
        return 1, len(basename) - len(query), ((basename_start, basename_start + len(query)),)
    for segment in reversed(path_segments[:-1]):
        segment_start = path_lower.find(segment)
        if segment == query:
            return 2, segment_start, ((segment_start, segment_start + len(query)),)
        if segment.startswith(query):
            return 3, segment_start, ((segment_start, segment_start + len(query)),)
    contains_at = basename.find(query)
    if contains_at >= 0:
        start = basename_start + contains_at
        return 4, contains_at, ((start, start + len(query)),)
    contains_at = path_lower.find(query)
    if contains_at >= 0:
        return 5, contains_at, ((contains_at, contains_at + len(query)),)
    subsequence = _subsequence_span(query, basename)
    if subsequence is not None:
        start, end = subsequence
        return 6, end - start, ((basename_start + start, basename_start + end),)
    return None


def _subsequence_span(query_text: str, value: str) -> tuple[int, int] | None:
    positions: list[int] = []
    search_from = 0
    for character in query_text:
        position = value.find(character, search_from)
        if position < 0:
            return None
        positions.append(position)
        search_from = position + 1
    return (positions[0], positions[-1] + 1) if positions else None


def _first_level_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        path
        for path in paths
        if "/" not in path.rstrip("/")
    )[:FILE_REFERENCE_FIRST_LEVEL_LIMIT]


def _normalized_path(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lstrip("/")
    if not normalized:
        return ""
    return f"{normalized.strip('/')}/" if normalized.endswith("/") else normalized.rstrip("/")


def _normalize_query(query_text: str) -> str:
    return query_text.strip().lstrip("/")


def _quick_result_match(query_text: str, label: str) -> bool:
    compact_query = query_text.lower().replace("/", "")
    compact_label = label.lower().replace("/", "")
    iterator = iter(compact_label)
    return all(character in iterator for character in compact_query)


__all__ = ["FileReferenceSearchClient", "FileReferenceSearchResult"]
