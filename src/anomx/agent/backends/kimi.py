"""Kimi backend."""

from __future__ import annotations

from typing import Any

from anomx.agent.backends.openai_chat import OpenAICompatibleChatBackend


class KimiBackend(OpenAICompatibleChatBackend):
    """Moonshot AI's OpenAI-compatible Kimi backend."""

    provider_key = "kimi"
    provider_label = "Kimi"
    env_var = "MOONSHOT_API_KEY"
    chat_completions_endpoint = "https://api.moonshot.ai/v1/chat/completions"
    preserve_reasoning_content = True

    def _ollama_tools(self) -> list[dict[str, Any]]:
        tools = super()._ollama_tools()
        for tool in tools:
            function = tool.get("function")
            if isinstance(function, dict):
                function["strict"] = False
        return tools
