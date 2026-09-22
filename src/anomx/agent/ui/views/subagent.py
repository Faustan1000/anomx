"""Subagent view and subagent activity rendering helpers."""

from __future__ import annotations

import curses
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path

from anomx.agent.helpers.state import (
    AsyncProcessSnapshot,
    SubagentSnapshot,
    subagent_snapshots,
)
from anomx.agent.helpers.utils import parse_agent_kind
from anomx.agent.runtime import (
    format_token_count,
)
from anomx.agent.store import (
    SessionRecord,
)
from anomx.agent.ui.constants import ACTIVITY_FRAME_SECONDS
from anomx.agent.ui.models import (
    ActivityDetailEntry,
    ActivityItem,
    CursesWindow,
)


class SubagentViewMixin:
    """Subagent view and subagent activity rendering helpers."""

    def _open_subagent_session(
        self,
        stdscr: CursesWindow,
        parent_session: SessionRecord,
        agent_id: str,
    ) -> None:
        scroll = 0
        frame = 0
        with suppress(curses.error):
            stdscr.nodelay(True)
        previous_accent = self._accent_attr_name
        self._accent_attr_name = "subagent"
        try:
            while True:
                parent_turn = self._active_turn_for_session(parent_session)
                if parent_turn is not None and parent_turn.worker is not None:
                    self._drain_session_turn_events(stdscr, parent_turn, render_events=False)

                snapshot = self._subagent_snapshot_for_session(parent_session.path, agent_id)
                if snapshot is None or not snapshot.session_path:
                    self._draw_subagent_placeholder(stdscr, parent_session, agent_id)
                else:
                    subagent_session = self._subagent_session_record(parent_session, snapshot)
                    viewport = self._draw_session(
                        stdscr,
                        subagent_session,
                        self._read_message_lines(subagent_session.path),
                        "",
                        0,
                        scroll,
                        working_text=(
                            self._subagent_activity_statement(snapshot)
                            if snapshot.status in {"running", "working"}
                            else None
                        ),
                        working_frame=frame,
                        show_prompt_bar=False,
                        hide_plan=True,
                        title_override=self._subagent_session_title(parent_session, snapshot),
                        back_link_text="Back to Session",
                    )
                    scroll = viewport.scroll
                key = self._read_nonblocking_key(stdscr)
                if key is None:
                    time.sleep(ACTIVITY_FRAME_SECONDS)
                    frame += 1
                    continue
                if self._is_escape(key) or self._is_ctrl_c(key):
                    return
                if key == curses.KEY_UP:
                    scroll += 1
                elif key == curses.KEY_DOWN:
                    scroll -= 1
                elif key == curses.KEY_PPAGE:
                    scroll += 5
                elif key == curses.KEY_NPAGE:
                    scroll -= 5
                frame += 1
        finally:
            self._accent_attr_name = previous_accent
            with suppress(curses.error):
                stdscr.nodelay(False)

    def _subagent_snapshot_for_session(
        self,
        session_path: Path,
        agent_id: str,
    ) -> SubagentSnapshot | None:
        for snapshot in subagent_snapshots(
            self._session_events(session_path),
            include_removed=True,
        ):
            if snapshot.agent_id == agent_id:
                return snapshot
        return None

    def _subagent_session_record(
        self,
        parent_session: SessionRecord,
        snapshot: SubagentSnapshot,
    ) -> SessionRecord:
        path = Path(snapshot.session_path).expanduser()
        return SessionRecord(
            session_id=f"{parent_session.session_id}:{snapshot.agent_id}",
            path=path,
            created_at=snapshot.started_at or parent_session.created_at,
            updated_at=snapshot.finished_at or snapshot.updated_at or parent_session.updated_at,
            cwd=parent_session.cwd,
            provider=parent_session.provider,
            model=parent_session.model,
            title=snapshot.name,
            message_count=0,
            unread=False,
            last_user_at=snapshot.started_at,
            mode=parent_session.mode,
            agent_kind=parse_agent_kind(snapshot.kind, parent_session.agent_kind),
        )

    def _subagent_session_title(
        self,
        parent_session: SessionRecord,
        snapshot: SubagentSnapshot,
    ) -> str:
        project_name = self._current_project_name()
        parent_title = parent_session.title.strip() or "New session"
        pieces = [piece for piece in (project_name, parent_title, snapshot.name) if piece]
        return " › ".join(pieces)

    def _draw_subagent_placeholder(
        self,
        stdscr: CursesWindow,
        parent_session: SessionRecord,
        agent_id: str,
    ) -> None:
        height, width = self._draw_shell(
            stdscr,
            f"{self._session_project_title(parent_session)} › Subagent",
            str(parent_session.cwd or self.cwd),
        )
        del height
        self._add(
            stdscr,
            self._session_body_top((), subtitle_line_count=1),
            4,
            f"Subagent {agent_id} is starting.",
            width - 8,
            self._attr("light"),
        )
        stdscr.refresh()

    def _activity_items(
        self,
        subagents: tuple[SubagentSnapshot, ...],
        processes: tuple[AsyncProcessSnapshot, ...],
        events: Sequence[Mapping[str, object]],
        frame: int,
    ) -> tuple[ActivityItem, ...]:
        del events
        items: list[ActivityItem] = []
        items.extend(self._subagent_activity_item(subagent, frame) for subagent in subagents)
        items.extend(self._process_activity_item(process, frame) for process in processes)
        return tuple(items)

    def _subagent_activity_item(
        self,
        subagent: SubagentSnapshot,
        frame: int,
    ) -> ActivityItem:
        active = subagent.status in {"running", "working"}
        latest = self._subagent_activity_statement(subagent)
        suffix = f" › {latest}" if latest else ""
        if active and suffix:
            suffix = f"{suffix}{self._activity_dots(frame)}"
        return ActivityItem(
            key=f"subagent:{subagent.agent_id}",
            title=subagent.name,
            right_text=self._subagent_right_text(subagent),
            details=self._subagent_activity_details(subagent),
            active=active,
            marker=self._activity_marker(active, frame),
            kind="subagent",
            badge=subagent.name,
            title_suffix=suffix,
            accent="subagent",
            open_agent_id=subagent.agent_id,
        )

    def _subagent_activity_statement(self, subagent: SubagentSnapshot) -> str:
        if subagent.status not in {"running", "working"}:
            return ""
        statement = subagent.statement.strip()
        if not statement or statement.lower() == "thinking":
            return "Thinking"
        return self._single_line_work_text(statement)

    def _subagent_right_text(self, subagent: SubagentSnapshot) -> str:
        parts: list[str] = []
        if subagent.context_tokens:
            parts.append(format_token_count(subagent.context_tokens))
        state = self._subagent_state_label(subagent)
        if state:
            parts.append(state)
        return " · ".join(parts)

    def _subagent_state_label(self, subagent: SubagentSnapshot) -> str:
        if subagent.status in {"running", "working"}:
            return self._process_runtime_duration(subagent.started_at)
        if subagent.status == "ready":
            return "Ready"
        if subagent.status == "removed":
            return "Removed"
        if subagent.status in {"interrupted", "cancelled", "canceled"}:
            return "Interrupted"
        if subagent.status == "failed":
            return "Failed"
        return subagent.status.title() if subagent.status else ""

    def _subagent_activity_details(
        self,
        subagent: SubagentSnapshot,
    ) -> tuple[ActivityDetailEntry, ...]:
        entries: list[ActivityDetailEntry] = []
        for index, entry in enumerate(subagent.history):
            entries.append(
                ActivityDetailEntry(
                    self._activity_entry_key(
                        subagent.agent_id,
                        entry.kind,
                        index,
                        entry.text,
                    ),
                    entry.text,
                    entry.text,
                )
            )
        if subagent.response:
            entries.append(
                ActivityDetailEntry(
                    self._activity_entry_key(
                        subagent.agent_id,
                        "response",
                        len(entries),
                        subagent.response,
                    ),
                    "Final response",
                    subagent.response,
                )
            )
        if subagent.error:
            entries.append(
                ActivityDetailEntry(
                    self._activity_entry_key(
                        subagent.agent_id,
                        "error",
                        len(entries),
                        subagent.error,
                    ),
                    "Error",
                    subagent.error,
                )
            )
        return tuple(entries)
