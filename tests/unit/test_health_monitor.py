"""HealthMonitor circuit-breaker logic.

Tests the FAILED/HEALTHY transition logic directly via _on_success /
_on_failure to keep the suite fast and asyncio-free. The end-to-end async
HTTP path is covered by tests/integration/test_distributed.py.
"""
from __future__ import annotations

import pytest

from master.health_monitor import HealthMonitor
from workers import WorkerStatus
from workers.remote_proxy import RemoteWorkerProxy


def _proxy(worker_id: str = "w1", url: str = "http://stub:0") -> RemoteWorkerProxy:
    # The constructor builds an httpx.Client; that's fine — the tests below
    # never actually call probe_health(), and the client is closed via close().
    return RemoteWorkerProxy(worker_id=worker_id, url=url, max_concurrent_tasks=8)


def test_three_failures_flip_to_failed():
    p = _proxy()
    monitor = HealthMonitor([p], failure_threshold=3, recovery_threshold=3)
    try:
        for _ in range(2):
            monitor._on_failure(p)
            assert p.status == WorkerStatus.HEALTHY  # still under threshold
        monitor._on_failure(p)
        assert p.status == WorkerStatus.FAILED
    finally:
        p.close()


def test_one_success_resets_fail_streak():
    p = _proxy()
    monitor = HealthMonitor([p], failure_threshold=3)
    try:
        monitor._on_failure(p)
        monitor._on_failure(p)
        monitor._on_success(p)
        # streak reset → next failure starts at 1, not 3
        monitor._on_failure(p)
        assert p.status == WorkerStatus.HEALTHY
    finally:
        p.close()


def test_three_successes_recover_failed_worker():
    p = _proxy()
    monitor = HealthMonitor([p], failure_threshold=2, recovery_threshold=3)
    try:
        monitor._on_failure(p)
        monitor._on_failure(p)
        assert p.status == WorkerStatus.FAILED

        for _ in range(2):
            monitor._on_success(p)
            assert p.status == WorkerStatus.FAILED  # still under recovery threshold
        monitor._on_success(p)
        assert p.status == WorkerStatus.HEALTHY
    finally:
        p.close()


def test_one_failure_resets_recovery_streak():
    p = _proxy()
    monitor = HealthMonitor([p], failure_threshold=2, recovery_threshold=3)
    try:
        monitor._on_failure(p)
        monitor._on_failure(p)
        monitor._on_success(p)
        monitor._on_success(p)
        monitor._on_failure(p)  # recovery streak resets
        # Need 3 fresh successes to recover.
        monitor._on_success(p)
        monitor._on_success(p)
        assert p.status == WorkerStatus.FAILED
        monitor._on_success(p)
        assert p.status == WorkerStatus.HEALTHY
    finally:
        p.close()


def test_snapshot_reports_per_worker_state():
    a = _proxy("a")
    b = _proxy("b")
    monitor = HealthMonitor([a, b], failure_threshold=3)
    try:
        monitor._on_success(a)
        monitor._on_failure(b)
        snap = monitor.snapshot()
        by_id = {row["worker_id"]: row for row in snap}
        assert by_id["a"]["ok_streak"] == 1
        assert by_id["a"]["fail_streak"] == 0
        assert by_id["b"]["fail_streak"] == 1
        assert by_id["b"]["ok_streak"] == 0
    finally:
        a.close()
        b.close()


def test_invalid_intervals_rejected():
    p = _proxy()
    try:
        with pytest.raises(ValueError):
            HealthMonitor([p], poll_interval_seconds=0)
        with pytest.raises(ValueError):
            HealthMonitor([p], probe_timeout_seconds=0)
    finally:
        p.close()
