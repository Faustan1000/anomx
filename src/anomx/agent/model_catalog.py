"""Model discovery for supported Anomx agent providers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterable
from typing import Any, cast

MODEL_DISCOVERY_TIMEOUT_SECONDS = 8
OPENAI_MODELS_ENDPOINT = "https://api.openai.com/v1/models"
ANTHROPIC_MODELS_ENDPOINT = "https://api.anthropic.com/v1/models"
BLABLADOR_MODELS_ENDPOINT = "https://api.blablador.fz-juelich.de/v1/models"
OLLAMA_MODELS_ENDPOINT = "http://127.0.0.1:11434/api/tags"


def discover_provider_models(provider_key: str, api_key: str | None = None) -> tuple[str, ...]:
    """Return model identifiers currently advertised by a provider.

    The function intentionally degrades to an empty tuple. Callers retain their
    provider defaults when discovery is unavailable, such as before credentials
    have been configured or while a local Ollama service is offline.
    """

    normalized_provider = provider_key.strip().lower()
    request = _model_request(normalized_provider, api_key)
    if request is None:
        return ()
    try:
        with urllib.request.urlopen(request, timeout=MODEL_DISCOVERY_TIMEOUT_SECONDS) as response:
            payload = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
    except (
        OSError,
        TimeoutError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ):
        return ()
    return _model_ids(normalized_provider, payload)


def merge_provider_models(
    fallback_models: Iterable[str],
    discovered_models: Iterable[str],
) -> tuple[str, ...]:
    """Merge configured fallbacks and discovered models without duplicates."""

    models: list[str] = []
    for model in (*fallback_models, *discovered_models):
        value = str(model or "").strip()
        if value and value not in models:
            models.append(value)
    return tuple(models)


def _model_request(provider_key: str, api_key: str | None) -> urllib.request.Request | None:
    if provider_key == "ollama":
        return urllib.request.Request(OLLAMA_MODELS_ENDPOINT, method="GET")

    token = str(api_key or "").strip()
    if not token:
        return None
    if provider_key == "openai":
        return urllib.request.Request(
            OPENAI_MODELS_ENDPOINT,
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
    if provider_key == "anthropic":
        return urllib.request.Request(
            ANTHROPIC_MODELS_ENDPOINT,
            headers={
                "anthropic-version": "2023-06-01",
                "x-api-key": token,
            },
            method="GET",
        )
    if provider_key == "blablador":
        return urllib.request.Request(
            BLABLADOR_MODELS_ENDPOINT,
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
    return None


def _model_ids(provider_key: str, payload: dict[str, Any]) -> tuple[str, ...]:
    entries = payload.get("models") if provider_key == "ollama" else payload.get("data")
    if not isinstance(entries, list):
        return ()

    models: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidate = entry.get("name") if provider_key == "ollama" else entry.get("id")
        model = str(candidate or "").strip()
        if model and model not in models:
            models.append(model)
    return tuple(models)
