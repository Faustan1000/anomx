from __future__ import annotations

import curses

from anomx.agent.app import AnomxCliApp
from anomx.agent.backends.anthropic import AnthropicBackend
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import (
    AnomxHome,
    model_detail,
    model_metadata,
    provider_by_key,
    thinking_intensity_options,
)


def test_model_menu_uses_only_curated_models_with_specific_labels(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    for provider_key in ("openai", "anthropic", "desy", "blablador", "kimi"):
        home.set_api_key(provider_key, "test-key")
    home.set_backend_connected("ollama", True)
    app = AnomxCliApp(home=home, isolate_runtime=True)

    monkeypatch.setattr(
        app,
        "_provider_with_discovered_models",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the /model menu must not use provider discovery")
        ),
    )

    choices = app._connected_model_menu_choices()

    assert [(choice.label, choice.value) for choice in choices] == [
        ("6 Astra", "openai::gpt-6-astra"),
        ("5.6 Sol", "openai::gpt-5.6-sol"),
        ("5.6 Terra", "openai::gpt-5.6-terra"),
        ("5.6 Luna", "openai::gpt-5.6-luna"),
        ("5.5", "openai::gpt-5.5"),
        ("5.4", "openai::gpt-5.4"),
        ("5.4 Mini", "openai::gpt-5.4-mini"),
        ("Fable 5.1", "anthropic::claude-fable-5-1"),
        ("Opus 5", "anthropic::claude-opus-5"),
        ("Sonnet 5", "anthropic::claude-sonnet-5"),
        ("Haiku 4.5", "anthropic::claude-haiku-4-5-20251001"),
        ("Coding", "desy::coding"),
        ("Assistant", "desy::desy-assistant"),
        ("Reasoning", "desy::reasoning"),
        ("Kimi K3", "blablador::alias-kimi-k3-1m"),
        ("GLM 5.2 (AWQ INT4)", "blablador::alias-glm-huge"),
        ("DeepSeek V4 Flash", "blablador::alias-deepseek-v4-flash-0731"),
        ("Muse Glimmer (30B)", "blablador::alias-muse"),
        ("GPT OSS (120B)", "blablador::alias-fast"),
        ("Qwen 3.5 (122B)", "blablador::alias-large"),
        ("Qwen 3.8 (27B)", "blablador::alias-code"),
        ("MiniMax M2.7", "blablador::alias-huge"),
    ]
    assert all(not choice.label.startswith("Custom ") for choice in choices)


def test_direct_model_override_remains_available_outside_curated_menu(tmp_path):
    home = AnomxHome(tmp_path / "home")
    app = AnomxCliApp(
        home=home,
        startup_provider="openai",
        startup_model="gpt-experimental",
        isolate_runtime=True,
    )

    app.prepare_startup_config()

    config = home.load_config()
    assert config["provider"] == "openai"
    assert config["model"] == "gpt-experimental"


def test_curated_claude_5_models_use_adaptive_thinking_and_all_effort_levels(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    backend = AnthropicBackend(runtime)

    for model in ("claude-fable-5-1", "claude-opus-5", "claude-sonnet-5"):
        assert backend._anthropic_thinking_config(model) == {
            "type": "adaptive",
            "display": "summarized",
        }
        assert [
            option.value for option in thinking_intensity_options("anthropic", model)
        ] == ["auto", "low", "medium", "high", "xhigh", "max"]


def test_curated_openai_models_expose_their_supported_effort_levels():
    expected = {
        "gpt-6-astra": ["auto", "low", "medium", "high", "xhigh", "max"],
        "gpt-5.6-sol": ["auto", "none", "low", "medium", "high", "xhigh", "max"],
        "gpt-5.6-terra": ["auto", "none", "low", "medium", "high", "xhigh", "max"],
        "gpt-5.6-luna": ["auto", "none", "low", "medium", "high", "xhigh", "max"],
        "gpt-5.5": ["auto", "none", "low", "medium", "high", "xhigh"],
        "gpt-5.4": ["auto", "none", "low", "medium", "high", "xhigh"],
        "gpt-5.4-mini": ["auto", "none", "low", "medium", "high", "xhigh"],
    }

    for model, values in expected.items():
        assert [
            option.value for option in thinking_intensity_options("openai", model)
        ] == values


def test_curated_models_have_context_and_output_metadata():
    for model in (
        "coding",
        "desy-assistant",
        "reasoning",
        "alias-fast",
        "alias-large",
        "alias-code",
        "alias-huge",
    ):
        detail = model_detail(model)
        assert "context" in detail
        assert "max output" in detail
        assert "custom context" not in detail


def test_blablador_catalog_includes_current_frontier_and_agentic_models():
    provider = provider_by_key("blablador")

    assert provider is not None
    assert provider.models[:4] == (
        "alias-kimi-k3-1m",
        "alias-glm-huge",
        "alias-deepseek-v4-flash-0731",
        "alias-muse",
    )
    assert model_metadata("alias-kimi-k3-1m").context_window == 1_048_576
    assert model_metadata("alias-glm-huge").context_window == 1_048_576
    assert model_metadata("alias-deepseek-v4-flash-0731").context_window == 1_048_576
    assert model_metadata("alias-muse").context_window is None
    assert model_metadata("alias-huge").display_name == "MiniMax M2.7"
    assert model_metadata("alias-huge").context_window == 131_072


def test_manage_settings_is_last_config_item_with_background_work_defaults(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), isolate_runtime=True)

    assert app._config_menu_choices()[-1].label == "Manage Settings"
    assert [(choice.label, choice.detail) for choice in app._background_work_setting_choices()] == [
        ("Hard Work: Selection", "e.g. computation of recommended next steps"),
        ("Medium Work: Selection", "e.g. risk assessment of agent commands"),
        ("Easy Work: Selection", "e.g. automatic naming of chats"),
    ]
    manage_choices = app._manage_settings_choices()
    assert [(choice.label, choice.selectable) for choice in manage_choices] == [
        ("Background Work", False),
        ("Hard Work: Selection", True),
        ("Medium Work: Selection", True),
        ("Easy Work: Selection", True),
        ("", False),
        ("Context Management", False),
        ("Maximum Context: 256k", True),
        ("Compression Target: 50%", True),
    ]


def test_manage_settings_can_select_model_and_restore_current_selection(
    tmp_path,
    monkeypatch,
):
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("openai", "test-key")
    app = AnomxCliApp(home=home, isolate_runtime=True)
    selections = iter(
        [
            "background_hard_work_model",
            "openai::gpt-5.4",
            "background_easy_work_model",
            "current",
            "maximum_context_tokens",
            "64000",
            "context_compression_target_percent",
            "75",
            None,
        ]
    )
    seen_choices: list[list[tuple[str, str]]] = []

    def select_menu(_stdscr, _title, _subtitle, choices, **_kwargs):
        seen_choices.append([(choice.label, choice.detail) for choice in choices])
        return next(selections)

    monkeypatch.setattr(app, "_menu", select_menu)

    app._run_manage_settings_panel(None)

    config = home.load_config()
    assert config["background_hard_work_model"] == "openai::gpt-5.4"
    assert config["background_easy_work_model"] == "current"
    assert config["maximum_context_tokens"] == 64_000
    assert config["context_compression_target_percent"] == 75
    assert seen_choices[1][0] == (
        "Current Model",
        "Use the currently selected model for background work",
    )
    assert seen_choices[5] == [
        ("32k Tokens", "For focused tasks and everyday conversations"),
        ("64k Tokens", "For detailed tasks and longer conversations"),
        ("128k Tokens", "For complex tasks and multi-file projects"),
        ("256k Tokens", "For large projects and extensive codebases"),
        ("512k Tokens", "For expansive projects and sustained workflows"),
        ("1M Tokens", "For extremely challenging tasks and long-running work"),
    ]
    assert seen_choices[7] == [
        ("25%", "For a good experience with simple conversations"),
        ("50%", "For a good balance between performance and cost"),
        ("75%", "For complex longer-running agentic tasks"),
    ]
    assert app._background_work_setting_choices()[2].label == "Easy Work: Selection"


def test_manage_settings_section_rows_are_skipped_by_keyboard_navigation(
    tmp_path,
    monkeypatch,
):
    class Window:
        def __init__(self):
            self.keys = iter((curses.KEY_DOWN, "\n"))

        def get_wch(self):
            return next(self.keys)

    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), isolate_runtime=True)
    monkeypatch.setattr(app, "_draw_overlay", lambda *_args, **_kwargs: None)
    choices = app._manage_settings_choices()

    assert app._run_overlay_menu(
        Window(),
        "Manage Settings",
        choices=choices,
    ) == "background_medium_work_model"


def test_background_work_runtime_uses_current_or_configured_model(
    tmp_path,
    monkeypatch,
):
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("openai", "test-key")
    runtime = AgentRuntime(home, tmp_path)
    backend = object()
    monkeypatch.setattr(
        "anomx.agent.runtime.backend_for_provider",
        lambda provider, _runtime: backend if provider == "openai" else None,
    )

    assert runtime._background_work_backend("background_easy_work_model") == (
        backend,
        "gpt-5.5",
    )

    config = home.load_config()
    config["background_easy_work_model"] = "openai::gpt-5.6-luna"
    home.save_config(config)

    assert runtime._background_work_backend("background_easy_work_model") == (
        backend,
        "gpt-5.6-luna",
    )


def test_automatic_context_compression_preserves_full_transcript_and_rolls_summary(
    tmp_path,
):
    home = AnomxHome(tmp_path / "home")
    config = home.load_config()
    config["maximum_context_tokens"] = 32_000
    config["context_compression_target_percent"] = 50
    home.save_config(config)
    session = home.create_session(tmp_path, provider="openai", model="gpt-5.5")
    summary_prompts: list[tuple[str, str]] = []

    def summarize(system_prompt: str, user_prompt: str) -> str:
        summary_prompts.append((system_prompt, user_prompt))
        return f"I retained rolling summary {len(summary_prompts)}."

    runtime = AgentRuntime(home, tmp_path, context_summarizer=summarize)
    statuses: list[str] = []
    for index in range(20):
        role = "user_message" if index % 2 == 0 else "agent_message"
        home.append_session_event(
            session.path,
            role,
            {
                "message": f"message {index} " + ("context " * 1_000),
                "message_id": f"message-{index}",
            },
        )

    first_state = runtime._prepare_context_compression(
        session.path,
        RuntimeCallbacks(status=statuses.append),
    )

    assert first_state is not None
    assert statuses == ["Automatic Context Compression"]
    assert len(runtime.conversation_messages(session.path)) == 20
    assert len(runtime.backend_conversation_messages(session.path)) < 20
    assert "USER:" in summary_prompts[0][1]
    assert "ASSISTANT:" in summary_prompts[0][1]
    assert runtime._instructions(session.path).endswith(
        "## Previous Conversation\n\nI retained rolling summary 1."
    )

    for index in range(20, 40):
        role = "user_message" if index % 2 == 0 else "agent_message"
        home.append_session_event(
            session.path,
            role,
            {
                "message": f"message {index} " + ("new context " * 1_000),
                "message_id": f"message-{index}",
            },
        )
    second_state = runtime._prepare_context_compression(
        session.path,
        RuntimeCallbacks(status=statuses.append),
    )

    assert second_state is not None
    assert second_state.compressed_message_count > first_state.compressed_message_count
    assert "Previous summary:\nI retained rolling summary 1." in summary_prompts[-1][1]
    assert len(runtime.conversation_messages(session.path)) == 40
