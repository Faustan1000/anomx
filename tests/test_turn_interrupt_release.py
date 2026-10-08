"""Interrupting a turn must unblock a worker that waits for an approval or question."""

from __future__ import annotations

import queue
import threading

from anomx.agent.app import AnomxCliApp
from anomx.agent.helpers.tool_manager import ApprovalChoice
from anomx.agent.ui.models import RuntimeUiEvent


def _turn():
    class Turn:
        events: queue.SimpleQueue = queue.SimpleQueue()
        pending_events: list = []

    turn = Turn()
    turn.events = queue.SimpleQueue()
    turn.pending_events = []
    return turn


def test_blocked_approval_and_question_are_answered_on_abort():
    turn = _turn()
    approval: queue.SimpleQueue = queue.SimpleQueue()
    question: queue.SimpleQueue = queue.SimpleQueue()
    status = RuntimeUiEvent("status", text="Thinking")
    turn.events.put(RuntimeUiEvent("approval", approval_response=approval))
    turn.events.put(status)
    turn.pending_events.append(RuntimeUiEvent("question", question_response=question))

    AnomxCliApp._release_blocked_turn_callbacks(turn)

    assert approval.get_nowait() == ApprovalChoice.REJECT
    answer = question.get_nowait()
    assert answer.cancelled and not answer.answered
    assert turn.pending_events == []
    # Non-interactive events stay queued for the regular drain.
    assert turn.events.get_nowait() is status
    assert turn.events.empty()


def test_worker_waiting_on_approval_finishes_after_release():
    turn = _turn()
    response: queue.SimpleQueue = queue.SimpleQueue()
    turn.events.put(RuntimeUiEvent("approval", approval_response=response))
    outcome: list[ApprovalChoice] = []
    worker = threading.Thread(target=lambda: outcome.append(response.get()), daemon=True)
    worker.start()

    AnomxCliApp._release_blocked_turn_callbacks(turn)
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert outcome == [ApprovalChoice.REJECT]
