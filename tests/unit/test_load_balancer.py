"""LoadBalancer correctness + thundering-herd regression."""
from __future__ import annotations

import threading

import pytest

from common import Request
from lb import LoadBalancer, LoadBalancingStrategy
from workers import GPUWorkerNode


def _req(rid: str = "r") -> Request:
    return Request(request_id=rid, user_id="u", prompt="hi", metadata={})


def _make_workers(n: int = 3, max_concurrent: int = 8) -> list[GPUWorkerNode]:
    return [
        GPUWorkerNode(
            worker_id=f"w{i}", gpu_name="sim", max_concurrent_tasks=max_concurrent
        )
        for i in range(n)
    ]


# ── Strategy correctness ──────────────────────────────────────────────────────


def test_round_robin_cycles_through_workers():
    workers = _make_workers(3)
    lb = LoadBalancer(workers, strategy=LoadBalancingStrategy.ROUND_ROBIN)

    picked = [lb.select_worker(_req(f"r{i}")).worker_id for i in range(6)]
    assert picked == ["w0", "w1", "w2", "w0", "w1", "w2"]


def test_least_connections_picks_lowest_pending():
    workers = _make_workers(3)
    workers[0].pending_tasks = 5
    workers[1].pending_tasks = 1
    workers[2].pending_tasks = 3
    lb = LoadBalancer(workers, strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)

    chosen = lb.select_worker(_req())
    assert chosen.worker_id == "w1"


def test_load_aware_uses_capacity_ratio():
    workers = _make_workers(3, max_concurrent=10)
    workers[0].pending_tasks = 8  # 80%
    workers[1].pending_tasks = 5  # 50%
    workers[2].pending_tasks = 2  # 20%
    lb = LoadBalancer(workers, strategy=LoadBalancingStrategy.LOAD_AWARE)

    assert lb.select_worker(_req()).worker_id == "w2"


# ── FAILED filtering + LB resilience ──────────────────────────────────────────


def test_failed_workers_are_skipped():
    workers = _make_workers(3)
    workers[0].mark_failed()
    workers[2].mark_failed()
    lb = LoadBalancer(workers, strategy=LoadBalancingStrategy.ROUND_ROBIN)

    for _ in range(5):
        assert lb.select_worker(_req()).worker_id == "w1"


def test_all_failed_raises():
    workers = _make_workers(2)
    for w in workers:
        w.mark_failed()
    lb = LoadBalancer(workers, strategy=LoadBalancingStrategy.ROUND_ROBIN)

    with pytest.raises(RuntimeError, match="No healthy"):
        lb.select_worker(_req())


def test_constructor_rejects_empty():
    with pytest.raises(ValueError):
        LoadBalancer([], strategy=LoadBalancingStrategy.ROUND_ROBIN)


# ── Thundering-herd regression ────────────────────────────────────────────────


def test_concurrent_least_connections_distributes():
    """50 threads racing select_worker() should balance across 3 workers.

    Before the lock + reservation fix, every thread saw active_tasks=0 and
    Python's min() returned the first worker on a tie, so all 50 picks landed
    on w0. With pending_tasks reservations under a lock, each pick increments
    the chosen worker's counter, so subsequent threads see different values
    and pick differently.
    """
    workers = _make_workers(3, max_concurrent=64)
    lb = LoadBalancer(workers, strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
    barrier = threading.Barrier(50)
    picks: list[str] = []
    picks_lock = threading.Lock()

    def fire(i: int) -> None:
        barrier.wait()  # release all threads at once
        w = lb.select_worker(_req(f"r{i}"))
        with picks_lock:
            picks.append(w.worker_id)

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    counts = {w.worker_id: picks.count(w.worker_id) for w in workers}
    # With perfect distribution each worker would get 50/3 ≈ 16.6; allow ±5.
    for count in counts.values():
        assert 12 <= count <= 22, (
            f"thundering-herd regression: {counts} (expected ~16 per worker)"
        )

    # Each select_worker() reserved its pick. pending_tasks should equal the
    # picks per worker until callers release.
    for w in workers:
        assert w.pending_tasks == counts[w.worker_id]


def test_release_returns_pending_to_zero():
    workers = _make_workers(2)
    lb = LoadBalancer(workers, strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)

    picked = [lb.select_worker(_req(f"r{i}")) for i in range(10)]
    for w in picked:
        w.release()

    for w in workers:
        assert w.pending_tasks == 0


# LoadBalancer treats only FAILED as ineligible; HEALTHY workers are
# always candidates. (DEGRADED was removed when self-shedding made the
# overflow transition unreachable.)


# -- Power-of-Two-Choices ------------------------------------------------------


def test_power_of_two_picks_less_loaded_under_skew():
    """With one heavily loaded worker and two light, P2C should rarely
    pick the heavy one."""
    workers = _make_workers(3)
    workers[0].pending_tasks = 100  # the "hot" worker
    lb = LoadBalancer(workers, strategy=LoadBalancingStrategy.POWER_OF_TWO)

    picks = []
    n = 200
    for i in range(n):
        chosen = lb.select_worker(_req(f"r{i}"))
        picks.append(chosen.worker_id)
        # release so pending_tasks doesn't snowball during the test
        chosen.release()
    # Restore the seeded skew (release decremented w0)
    counts = {w.worker_id: picks.count(w.worker_id) for w in workers}
    # P2C never picks the hottest worker unless both samples ARE the hottest,
    # which can't happen because there's only one of it. Expected: w0 ~ 0,
    # w1 + w2 ~ n.
    assert counts["w0"] < n // 10, (
        f"P2C picked the hot worker {counts['w0']} / {n} times: {counts}"
    )


def test_power_of_two_handles_single_worker():
    workers = _make_workers(1)
    lb = LoadBalancer(workers, strategy=LoadBalancingStrategy.POWER_OF_TWO)
    assert lb.select_worker(_req()).worker_id == "w0"


def test_set_strategy_switches_at_runtime():
    workers = _make_workers(3)
    lb = LoadBalancer(workers, strategy=LoadBalancingStrategy.ROUND_ROBIN)
    assert lb.strategy == LoadBalancingStrategy.ROUND_ROBIN
    lb.set_strategy(LoadBalancingStrategy.POWER_OF_TWO)
    assert lb.strategy == LoadBalancingStrategy.POWER_OF_TWO
    # verify selection still works on the new strategy
    chosen = lb.select_worker(_req())
    assert chosen.worker_id in {"w0", "w1", "w2"}
