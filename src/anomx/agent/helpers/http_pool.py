"""A tiny keep-alive connection pool for the hot streaming request path.

`urllib.request.urlopen` opens a brand-new TCP connection and does a full TLS
handshake on every call. Backend instances are recreated per turn
(`backend_for_provider`), so a fresh connection was being paid for on every single
agent turn even though the same host is hit repeatedly for the lifetime of the CLI
process. This module keeps a small pool of `http.client.HTTPSConnection` objects
per host alive across turns and reuses one whenever its previous response was
fully consumed.

Connections are checked out of the pool for the duration of one request and
checked back in afterwards, rather than being looked up and left in the pool
while in use -- the main agent turn and up to several subagents run concurrently
on separate threads (see the "run up to five subagents concurrently" guidance in
the agent prompts) and commonly share the same provider, so a single connection
per host handed out to whichever thread asks would let two threads write to and
read from the same socket at once, corrupting or cross-delivering responses
between unrelated turns.

Only the streaming Messages-API loop (the request made on every turn) is routed
through this pool; the smaller one-off calls elsewhere keep using
`urllib.request.urlopen` since they are not on the hot per-turn path.
"""

from __future__ import annotations

import http.client
import ssl
import threading
from collections.abc import Iterator
from types import TracebackType
from typing import Literal
from urllib.parse import urlsplit

_POOL: dict[str, list[http.client.HTTPSConnection]] = {}
_LOCK = threading.Lock()
_CONTEXT = ssl.create_default_context()


class PooledStreamResponse:
    """Adapts a pooled `http.client.HTTPResponse` to the `urlopen` response shape."""

    def __init__(
        self,
        response: http.client.HTTPResponse,
        connection: http.client.HTTPSConnection,
        host: str,
    ) -> None:
        self._response = response
        self._connection = connection
        self._host = host

    def __enter__(self) -> PooledStreamResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        # Only a fully-drained response leaves the underlying socket in a state
        # where the next request can reuse it. An aborted turn or a mid-stream
        # error exits early without reading the rest of the body, so that
        # connection must be dropped rather than checked back in for reuse.
        if exc_type is not None or not self._response.isclosed():
            _discard(self._connection)
        else:
            _checkin(self._host, self._connection)
        return False

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._response)


def pooled_https_post(
    endpoint: str,
    *,
    data: bytes,
    headers: dict[str, str],
    timeout: float,
) -> PooledStreamResponse:
    """POST over a reused keep-alive connection, reconnecting once on a stale socket."""

    parts = urlsplit(endpoint)
    host = parts.netloc
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"

    last_error: Exception | None = None
    for attempt in range(2):
        connection = _checkout(host, timeout)
        try:
            connection.request("POST", path, body=data, headers=headers)
            response = connection.getresponse()
            return PooledStreamResponse(response, connection, host)
        except (http.client.HTTPException, OSError) as error:
            _discard(connection)
            last_error = error
            continue
    assert last_error is not None
    raise last_error


def _checkout(host: str, timeout: float) -> http.client.HTTPSConnection:
    with _LOCK:
        pool = _POOL.get(host)
        connection = pool.pop() if pool else None
    if connection is None:
        return http.client.HTTPSConnection(host, timeout=timeout, context=_CONTEXT)
    connection.timeout = timeout
    if connection.sock is not None:
        connection.sock.settimeout(timeout)
    return connection


def _checkin(host: str, connection: http.client.HTTPSConnection) -> None:
    with _LOCK:
        _POOL.setdefault(host, []).append(connection)


def reset_pool() -> None:
    """Close and forget every pooled connection.

    Used by tests to keep the process-global pool from leaking a live socket
    (opened by a test that didn't fully mock the network layer) into unrelated
    tests that run afterwards.
    """

    with _LOCK:
        connections = [connection for pool in _POOL.values() for connection in pool]
        _POOL.clear()
    for connection in connections:
        try:
            connection.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


def _discard(connection: http.client.HTTPSConnection) -> None:
    try:
        connection.close()
    except Exception:  # noqa: BLE001 - best-effort cleanup of a dead socket
        pass


__all__ = ["PooledStreamResponse", "pooled_https_post", "reset_pool"]
