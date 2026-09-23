import pytest

from anomx.agent.helpers.http_pool import reset_pool


@pytest.fixture(autouse=True)
def _reset_http_pool():
    # A test that doesn't fully mock the network layer can leave a real, live
    # connection in the process-global keep-alive pool. Clearing it before and
    # after every test keeps that from leaking into unrelated tests.
    reset_pool()
    yield
    reset_pool()
