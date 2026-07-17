"""Initial onboarding flow."""

from __future__ import annotations

from anomx.agent.ui.models import (
    AgentState,
    CursesWindow,
)


class OnboardingViewMixin:
    """Initial onboarding flow."""

    def _run_onboarding(self, stdscr: CursesWindow) -> bool:
        self.state = AgentState.ONBOARDING
        provider = self._select_provider(stdscr)
        if provider is None:
            return False

        if provider.key in {"openai", "anthropic", "blablador", "desy"}:
            api_key = self._prompt_text(
                stdscr,
                title=provider.label,
                label="API key",
                mask=True,
                optional=True,
            )
            if api_key:
                self.home.set_api_key(provider.key, api_key)
                provider = self._provider_with_discovered_models(provider, refresh=True)

        model = self._select_model(stdscr, provider)
        if model is None:
            return False

        thinking_intensity = self._select_thinking_intensity(stdscr, provider, model)
        if thinking_intensity is None:
            return False

        user_name = self._prompt_text(
            stdscr,
            title="Your Name",
            label="Name",
            optional=False,
        )
        if not user_name:
            return False

        config = self.home.load_config()
        config["onboarding_complete"] = True
        config["provider"] = provider.key
        config["model"] = model
        config["thinking_intensity"] = thinking_intensity
        config["user_name"] = user_name.strip()
        self.home.save_config(config)
        return True
