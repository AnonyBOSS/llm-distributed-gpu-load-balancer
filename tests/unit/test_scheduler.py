"""MasterScheduler retry, fallover, RAG, and reservation handling."""
from __future__ import annotations

import pytest

from common import Request
from llm import LLMInferenceEngine, SimulatedLLMBackend
from master import MasterScheduler
from rag import RAGRetriever
from workers import GPUWorkerNode


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


def _retriever() -> RAGRetriever:
    return RAGRetriever(use_stub=True)


# ── Happy path ────────────────────────────────────────────────────────────────


def test_handle_request_completes():
    workers = [GPUWorkerNode(worker_id=f"w{i}", gpu_name="x") for i in range(2)]
    sched = MasterScheduler(_retriever(), _engine(), workers=workers)
    resp = sched.handle_request(_req())
    assert resp.status == "completed"
    assert resp.worker_id in {"w0", "w1"}
    assert sched.stats.successful_requests == 1
    assert sched.stats.failed_requests == 0


# ── Dead-worker fallover ──────────────────────────────────────────────────────


def test_failed_explicit_worker_falls_over_to_healthy():
    """If the LB pre-selected a worker and that worker is FAILED, the
    scheduler's retry loop must move on to a healthy candidate."""
    dead = GPUWorkerNode(worker_id="dead", gpu_name="x")
    alive = GPUWorkerNode(worker_id="alive", gpu_name="x")
    dead.mark_failed()
    sched = MasterScheduler(
        _retriever(), _engine(), workers=[dead, alive], max_retries=1,
    )
    resp = sched.handle_request(_req(), worker=dead)
    assert resp.status == "completed"
    assert resp.worker_id == "alive"


# ── Retry exhaustion ──────────────────────────────────────────────────────────


def test_all_workers_failing_returns_failed_status():
    workers = [
        GPUWorkerNode(
            worker_id=f"w{i}", gpu_name="x", failure_rate=1.0, rng_seed=i,
        )
        for i in range(2)
    ]
    sched = MasterScheduler(
        _retriever(),
        _engine(failure_rate=1.0),
        workers=workers,
        max_retries=1,
    )
    resp = sched.handle_request(_req())
    assert resp.status == "failed"
    assert sched.stats.failed_requests == 1


# ── Reservation handoff ───────────────────────────────────────────────────────


def test_pre_reserved_worker_is_released_at_end():
    """If the caller reserved before passing in, the scheduler must release
    that handoff regardless of success/failure."""
    workers = [GPUWorkerNode(worker_id=f"w{i}", gpu_name="x") for i in range(2)]
    sched = MasterScheduler(_retriever(), _engine(), workers=workers)

    workers[0].reserve()  # simulate LB.select_worker()'s pre-reservation
    assert workers[0].pending_tasks == 1

    sched.handle_request(_req(), worker=workers[0])
    assert workers[0].pending_tasks == 0
    assert workers[1].pending_tasks == 0


# ── Validation ────────────────────────────────────────────────────────────────


def test_no_workers_and_no_explicit_raises():
    sched = MasterScheduler(_retriever(), _engine(), workers=[])
    # ValueError from _resolve_candidate_workers; bubbles through finally
    # block in handle_request without being caught.
    with pytest.raises(ValueError, match="requires either"):
        sched.handle_request(_req())


# ── Batch ─────────────────────────────────────────────────────────────────────


def test_handle_batch_returns_per_request_responses():
    workers = [GPUWorkerNode(worker_id=f"w{i}", gpu_name="x") for i in range(2)]
    sched = MasterScheduler(_retriever(), _engine(), workers=workers)
    requests = [_req(f"r{i}") for i in range(5)]
    responses = sched.handle_batch(requests)
    assert len(responses) == 5
    assert all(r.status == "completed" for r in responses)


# -- At-capacity fallover (load shedding without consuming retries) -----------


def test_at_capacity_worker_falls_over_without_consuming_retries():
    """A worker at capacity is a routing miss, not a failure. The scheduler
    walks past it to the next candidate without spending a retry attempt,
    and does NOT record the rejection as a per-worker failure."""
    busy = GPUWorkerNode(worker_id="busy", gpu_name="x", max_concurrent_tasks=2)
    busy.active_tasks = 2  # pinned at capacity
    free = GPUWorkerNode(worker_id="free", gpu_name="x", max_concurrent_tasks=2)

    # Pre-reserved order: scheduler tries busy first (passed in), then free.
    # max_retries=0 means only one *failure* attempt is allowed -- if the
    # at-capacity rejection counted as a failure, we'd give up after busy.
    sched = MasterScheduler(
        _retriever(), _engine(), workers=[busy, free], max_retries=0,
    )
    resp = sched.handle_request(_req(), worker=busy)

    assert resp.status == "completed"
    assert resp.worker_id == "free"
    # The capacity miss did NOT increment busy's failure counter.
    assert sched.worker_failures.get("busy", 0) == 0
    assert sched.stats.successful_requests == 1


def test_all_workers_at_capacity_returns_failed():
    """If every candidate is at capacity, the request fails gracefully
    rather than looping forever."""
    workers = []
    for i in range(2):
        w = GPUWorkerNode(worker_id=f"w{i}", gpu_name="x", max_concurrent_tasks=2)
        w.active_tasks = 2
        workers.append(w)
    sched = MasterScheduler(_retriever(), _engine(), workers=workers)
    resp = sched.handle_request(_req())
    assert resp.status == "failed"
