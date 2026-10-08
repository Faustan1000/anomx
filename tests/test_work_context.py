import json
from unittest.mock import Mock, patch

import pytest

from anomx import WorkContext, get_work_context
from anomx.integrations.platform import PlatformClient, PlatformError

CHANNEL_REFERENCE = "data_channel-12345678-1234-5678-1234-567812345678"

def test_work_context_routes_events_and_restores_nested_context():
    callback = Mock(return_value={"reference": "finding:1"})
    outer = WorkContext(callbacks={"detection": callback})
    inner = WorkContext()
    with outer.activate():
        assert get_work_context() is outer
        with inner.activate():
            assert get_work_context() is inner
        assert get_work_context() is outer
        assert outer.detection("Drift", score=2.5, channel=CHANNEL_REFERENCE) == {
            "reference": "finding:1"
        }
    callback.assert_called_once_with(title="Drift", score=2.5, channel=CHANNEL_REFERENCE)
    assert outer.events[0]["kind"] == "detection"
    with pytest.raises(RuntimeError, match="No work context"):
        get_work_context()


def test_work_metrics_and_publication_are_validated(tmp_path):
    work = WorkContext()
    work.metric("loss", 0.2, step=1)
    assert work.events[-1]["value"] == 0.2
    with pytest.raises(ValueError, match="finite"):
        work.metric("loss", float("nan"))
    with pytest.raises(RuntimeError, match="compute"):
        work.compute("result = 1")
    with pytest.raises(SyntaxError):
        work.compute("result = (")
    with pytest.raises(ValueError, match="existing"):
        work.publish_model(tmp_path / "missing.onnx", name="baseline")
    artifact = tmp_path / "baseline.onnx"
    artifact.write_bytes(b"model bytes")
    publish = Mock(return_value={"id": "model:1"})
    WorkContext({"publish_model": publish}).publish_model(artifact, name="baseline")
    publish.assert_called_once_with(
        path=str(artifact.resolve()), name="baseline", metrics={}, metadata={}
    )


def test_missing_callbacks_do_not_pretend_a_model_was_published(tmp_path):
    artifact = tmp_path / "baseline.onnx"
    artifact.write_bytes(b"model bytes")
    work = WorkContext()
    with pytest.raises(RuntimeError, match="publish_model"):
        work.publish_model(artifact, name="baseline")
    assert work.events == []


def test_platform_client_posts_references_without_loading_agent():
    response = Mock()
    response.read.return_value = json.dumps(
        {"records": [{"timestamp": "2026-01-01", "x": 1}]}
    ).encode()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    client = PlatformClient("https://example.test/api", "secret")
    with patch("anomx.integrations.platform.urlopen", return_value=response) as request:
        rows = client.load_channels([CHANNEL_REFERENCE])
    assert rows[0]["x"] == 1
    sent = request.call_args.args[0]
    assert sent.full_url == "https://example.test/api/jobs/channel-data"
    assert json.loads(sent.data)["channels"] == [CHANNEL_REFERENCE]
    assert "secret" not in repr(client)


def test_platform_client_requires_explicit_connection(monkeypatch):
    monkeypatch.delenv("ANOMX_URL", raising=False)
    monkeypatch.delenv("ANOMX_TOKEN", raising=False)
    with pytest.raises(PlatformError, match="ANOMX_URL"):
        PlatformClient.from_env()
    with pytest.raises(ValueError, match="HTTP"):
        PlatformClient("file:///tmp/whatever", "token")
