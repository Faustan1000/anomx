"""Regression tests for the keep-alive connection pool's thread-safety."""

import threading
import time

from anomx.agent.helpers import http_pool


def test_checkout_never_hands_out_the_same_connection_twice():
    http_pool.reset_pool()
    try:
        first = http_pool._checkout("example.com", 5)
        second = http_pool._checkout("example.com", 5)

        # Two checkouts before either is checked back in must be distinct
        # connections -- handing out the same one would let two threads write to
        # and read from the same socket concurrently, corrupting or
        # cross-delivering responses between unrelated agent turns.
        assert first is not second

        http_pool._checkin("example.com", first)
        third = http_pool._checkout("example.com", 5)
        assert third is first
    finally:
        http_pool.reset_pool()


def test_concurrent_checkouts_are_never_outstanding_at_the_same_time():
    http_pool.reset_pool()
    outstanding: set[int] = set()
    violations: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(50):
            connection = http_pool._checkout("example.com", 5)
            key = id(connection)
            with lock:
                if key in outstanding:
                    violations.append("connection handed out while already in use")
                outstanding.add(key)
            time.sleep(0)  # yield, maximize the chance of a race if one exists
            with lock:
                outstanding.discard(key)
            http_pool._checkin("example.com", connection)

    try:
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not violations
    finally:
        http_pool.reset_pool()
