import io
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from anomx.agent.base import backends
from anomx.agent.base.backends import BaseBackend
from anomx.agent.exceptions import BackendFailure


@pytest.mark.parametrize("provider", ["blablador", "openai", "anthropic", "kimi"])
def test_missing_model_fails_without_reconnecting(provider):
    statuses = []
    backend = BaseBackend(SimpleNamespace(_status=lambda callback, text: callback(text)))

    def request():
        raise HTTPError("https://example.com", 404, "Not Found", None, io.BytesIO(
            b'{"error":{"message":"Model not found"}}'
        ))

    result = backend._model_request_with_retries(
        provider_key=provider, provider_label=provider, env_var="TEST_KEY",
        status_callback=statuses.append, stream_once=request,
    )

    assert isinstance(result, BackendFailure)
    assert "Model not found" in result
    assert statuses == []


def test_transient_failure_reports_all_ten_retries_then_fails(monkeypatch):
    monkeypatch.setattr(backends, "MODEL_REQUEST_RETRY_INITIAL_DELAY_SECONDS", 0)
    statuses = []
    attempts = []
    backend = BaseBackend(SimpleNamespace(
        _status=lambda callback, text: callback(text), _turn_aborted=lambda: False,
    ))

    def request():
        attempts.append(1)
        raise HTTPError("https://example.com", 503, "Unavailable", None, io.BytesIO(
            b'{"error":{"message":"Temporarily unavailable"}}'
        ))

    result = backend._model_request_with_retries(
        provider_key="blablador", provider_label="JSC", env_var="TEST_KEY",
        status_callback=statuses.append, stream_once=request,
    )

    assert isinstance(result, BackendFailure)
    assert len(attempts) == 11
    assert statuses == [f"Reconnecting {attempt}/10" for attempt in range(1, 11)]
