"""GPUWorkerNode capacity, health-state, reservation, and metrics."""
from __future__ import annotations

import pytest

from common import Request
from llm import LLMInferenceEngine, SimulatedLLMBackend
from workers import (
    GPUWorkerNode,
    WorkerAtCapacityError,
    WorkerStatus,
    WorkerTransientError,
    WorkerUnavailableError,
)


def _engine(failure_rate: float = 0.0) -> LLMInferenceEngine:
    return LLMInferenceEngine(
        backend=SimulatedLLMBackend(
            base_latency_s=0.001,
            per_token_latency_s=0.0,
            jitter_s=0.0,
            failure_rate=failure_rate,
            rng_seed=0,
        )
    )


def _req(rid: str = "r") -> Request:
    return Request(request_id=rid, user_id="u", prompt="hi", metadata={})


# ── Construction validation ───────────────────────────────────────────────────


def test_invalid_max_concurrent_raises():
    with pytest.raises(ValueError):
        GPUWorkerNode(worker_id="w", gpu_name="x", max_concurrent_tasks=0)


def test_invalid_failure_rate_raises():
    with pytest.raises(ValueError):
        GPUWorkerNode(worker_id="w", gpu_name="x", failure_rate=1.5)


# ── Reservation primitives ────────────────────────────────────────────────────


def test_reserve_and_release_balance():
    w = GPUWorkerNode(worker_id="w", gpu_name="x")
    for _ in range(10):
        w.reserve()
    assert w.pending_tasks == 10
    for _ in range(10):
        w.release()
    assert w.pending_tasks == 0


def test_release_does_not_go_negative():
    w = GPUWorkerNode(worker_id="w", gpu_name="x")
    w.release()
    w.release()
    assert w.pending_tasks == 0


# ── process() lifecycle ───────────────────────────────────────────────────────


def test_process_increments_completed():
    w = GPUWorkerNode(worker_id="w", gpu_name="x")
    w.process(_req(), context="ctx", inference_engine=_engine())
    assert w.completed_tasks == 1
    assert w.failed_tasks == 0
    assert w.active_tasks == 0


def test_failed_worker_refuses_immediately():
    w = GPUWorkerNode(worker_id="w", gpu_name="x")
    w.mark_failed()
    with pytest.raises(WorkerUnavailableError):
        w.process(_req(), context="", inference_engine=_engine())
    # Counters untouched on refusal.
    assert w.active_tasks == 0
    assert w.completed_tasks == 0


def test_at_capacity_worker_rejects_without_running():
    """Self-shed: a worker already at MAX_CONCURRENT_TASKS must refuse
    new work rather than queue it on the (single) GPU. The scheduler's
    retry loop treats this as a routing miss, not a failure."""
    w = GPUWorkerNode(worker_id="w", gpu_name="x", max_concurrent_tasks=2)
    w.active_tasks = 2  # simulate 2 in-flight requests
    with pytest.raises(WorkerAtCapacityError):
        w.process(_req(), context="", inference_engine=_engine())
    # active_tasks must NOT have been incremented past capacity, and
    # completed/failed counters stay clean (this isn't a failure).
    assert w.active_tasks == 2
    assert w.completed_tasks == 0
    assert w.failed_tasks == 0
    assert w.status == WorkerStatus.HEALTHY


def test_transient_failure_decrements_active_tasks():
    w = GPUWorkerNode(worker_id="w", gpu_name="x", failure_rate=1.0)
    with pytest.raises(WorkerTransientError):
        w.process(_req(), context="", inference_engine=_engine())
    assert w.active_tasks == 0
    assert w.failed_tasks == 1


def test_draining_worker_refuses_new_work():
    """Once `begin_drain()` is called, process() must refuse new work
    immediately so in-flight requests can finish without competition for
    the (single) GPU. Same WorkerUnavailableError path as a FAILED worker."""
    w = GPUWorkerNode(worker_id="w", gpu_name="x")
    w.begin_drain()
    assert w.is_draining
    with pytest.raises(WorkerUnavailableError, match="draining"):
        w.process(_req(), context="", inference_engine=_engine())
    # Counters untouched; this is not a failure.
    assert w.completed_tasks == 0
    assert w.failed_tasks == 0
    assert w.active_tasks == 0


def test_status_stays_healthy_through_normal_flow():
    """The capacity guard prevents active_tasks from ever exceeding max via
    the request path, so the worker never auto-flips to a degraded state.
    DEGRADED used to exist as an "overflow" status; with self-shedding it
    became unreachable, so it was removed from the model."""
    w = GPUWorkerNode(worker_id="w", gpu_name="x", max_concurrent_tasks=2)
    w.active_tasks = 1
    w.process(_req(), context="", inference_engine=_engine())
    assert w.status == WorkerStatus.HEALTHY
    # Trying to process while at capacity raises WorkerAtCapacityError
    # without changing the status.
    w.active_tasks = 2
    with pytest.raises(WorkerAtCapacityError):
        w.process(_req(), context="", inference_engine=_engine())
    assert w.status == WorkerStatus.HEALTHY


# ── snapshot_metrics() ────────────────────────────────────────────────────────


def test_snapshot_keys_present():
    w = GPUWorkerNode(worker_id="w", gpu_name="x")
    w.process(_req(), context="", inference_engine=_engine())
    snap = w.snapshot_metrics()
    expected = {
        "worker_id", "gpu_name", "status", "active_tasks", "pending_tasks",
        "max_concurrent_tasks", "completed_tasks", "failed_tasks",
        "total_tasks", "avg_latency_seconds", "last_latency_seconds",
    }
    assert expected.issubset(snap.keys())
    assert snap["completed_tasks"] == 1
    assert snap["avg_latency_seconds"] > 0
