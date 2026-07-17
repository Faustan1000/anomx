"""JSC Blablador backend."""

from __future__ import annotations

from anomx.agent.backends.openai_chat import OpenAICompatibleChatBackend


class BlabladorBackend(OpenAICompatibleChatBackend):
    """JSC Blablador's OpenAI-compatible Chat Completions backend."""

    provider_key = "blablador"
    provider_label = "JSC Blablador"
    env_var = "BLABLADOR_API_KEY"
    chat_completions_endpoint = "https://api.blablador.fz-juelich.de/v1/chat/completions"
