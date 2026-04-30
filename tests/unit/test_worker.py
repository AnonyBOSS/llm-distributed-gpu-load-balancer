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


def test_recovery_to_healthy_after_degraded():
    # The capacity guard now prevents active from exceeding max, so the
    # only way to be DEGRADED is via external state change. Once a request
    # finishes and active <= max, recovery fires.
    w = GPUWorkerNode(worker_id="w", gpu_name="x", max_concurrent_tasks=2)
    w.active_tasks = 1
    w.status = WorkerStatus.DEGRADED  # set artificially
    w.process(_req(), context="", inference_engine=_engine())
    # active goes 1 -> 2 -> 1; 1 <= 2 satisfies recovery.
    assert w.status == WorkerStatus.HEALTHY


def test_capacity_guard_prevents_degraded_transition():
    """With the new self-shed guard, active can never exceed max via
    process(). The DEGRADED-on-overflow path is therefore unreachable
    through the request flow -- DEGRADED is reserved for explicit
    external state changes."""
    w = GPUWorkerNode(worker_id="w", gpu_name="x", max_concurrent_tasks=2)
    w.active_tasks = 1
    w.process(_req(), context="", inference_engine=_engine())
    # active touched 2 mid-call but never exceeded max -> stayed HEALTHY.
    assert w.status == WorkerStatus.HEALTHY
    # Trying to process while at capacity raises WorkerAtCapacityError.
    w.active_tasks = 2
    with pytest.raises(WorkerAtCapacityError):
        w.process(_req(), context="", inference_engine=_engine())


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
