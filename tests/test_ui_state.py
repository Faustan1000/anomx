"""State holders that replace the long list of local variables in the screen loops."""

from __future__ import annotations

from anomx.agent.ui.models import PromptPasteSpan, PromptState, SessionViewState


def test_prompt_clear_resets_everything_with_fresh_containers():
    prompt = PromptState(
        input_text="hello @a.txt",
        cursor=5,
        pasted_spans=[PromptPasteSpan(0, 2)],
        command_selected=2,
        file_selected=1,
        file_references={"@a.txt": "/tmp/a.txt"},
        image_attachments={"[Image #1]": {"path": "/tmp/x.png"}},
    )
    old_spans, old_refs, old_images = (
        prompt.pasted_spans,
        prompt.file_references,
        prompt.image_attachments,
    )

    prompt.clear()

    assert prompt == PromptState()
    # A running turn may still hold the old containers; they must not be wiped.
    assert old_spans and old_refs and old_images


def test_view_state_jump_to_bottom_unpins_and_resets_scroll():
    view = SessionViewState(scroll=12, pinned_anchor=40)

    view.jump_to_bottom()

    assert (view.scroll, view.pinned_anchor) == (0, None)


def test_view_state_reset_running_notice_forgets_abort_confirmation():
    view = SessionViewState(
        running_notice="Press Ctrl+C again",
        running_notice_role="danger",
        running_abort_key="ctrl+c",
        running_abort_deadline=123.0,
    )

    view.reset_running_notice("Agent is working")

    assert view.running_notice == "Agent is working"
    assert view.running_notice_role == "light"
    assert view.running_abort_key == ""
    assert view.running_abort_deadline == 0.0
