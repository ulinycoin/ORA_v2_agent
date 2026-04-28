"""Unit tests: RateLimiter."""
import pytest
from ora_v2.transport.rate_limiter import RateLimiter


def test_allows_within_limit():
    rl = RateLimiter(max_requests=3, window_seconds=60)
    assert rl.check(1) is True
    assert rl.check(1) is True
    assert rl.check(1) is True


def test_blocks_over_limit():
    rl = RateLimiter(max_requests=2, window_seconds=60)
    rl.check(1)
    rl.check(1)
    assert rl.check(1) is False


def test_different_users_independent():
    rl = RateLimiter(max_requests=1, window_seconds=60)
    assert rl.check(1) is True
    assert rl.check(2) is True
    assert rl.check(1) is False
    assert rl.check(2) is False


def test_window_expiry():
    import time
    rl = RateLimiter(max_requests=1, window_seconds=1)
    rl.check(42)
    assert rl.check(42) is False
    time.sleep(1.1)
    assert rl.check(42) is True
