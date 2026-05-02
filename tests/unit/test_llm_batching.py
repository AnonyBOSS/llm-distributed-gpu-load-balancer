"""Continuous-batching backend correctness + amortisation."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from llm.inference import BatchedSimulatedLLMBackend, LLMInferenceError


def _fast_backend(**kwargs) -> BatchedSimulatedLLMBackend:
    """A backend with deterministic latency and small windows for fast tests."""
    return BatchedSimulatedLLMBackend(
        base_latency_s=0.05,
        per_token_latency_s=0.0,
        jitter_s=0.0,
        batch_max_size=8,
        batch_window_s=0.005,
        rng_seed=0,
        **kwargs,
    )


def test_single_call_returns_answer():
    b = _fast_backend()
    ans = b.generate("ping", "ctx")
    assert "Answer to 'ping'" in ans
    assert "tokens" in ans


def test_concurrent_batch_amortises_latency():
    """16 concurrent calls should complete in ~2 batches of 8 - far less
    than 16x a single-call latency."""
    b = _fast_backend()

    # Warm-up call to amortise any thread-startup cost.
    b.generate("warmup", "")

    n = 16
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(lambda i: b.generate(f"q{i}", "ctx"), range(n)))
    elapsed = time.perf_counter() - start

    # 16 sequential calls would take ~16 * 0.05 + window = ~0.85 s.
    # Two batches of 8 should take ~2 * 0.05 + 2 * 0.005 = ~0.11 s.
    # Allow some slack for scheduler jitter.
    assert elapsed < 0.5, f"batch did not amortise: {elapsed:.3f}s for {n} calls"


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        BatchedSimulatedLLMBackend(batch_max_size=0)
    with pytest.raises(ValueError):
        BatchedSimulatedLLMBackend(batch_window_s=0)
    with pytest.raises(ValueError):
        BatchedSimulatedLLMBackend(failure_rate=2.0)
    with pytest.raises(ValueError):
        BatchedSimulatedLLMBackend(base_latency_s=-1)


def test_failure_rate_propagates():
    b = BatchedSimulatedLLMBackend(
        base_latency_s=0.0,
        per_token_latency_s=0.0,
        jitter_s=0.0,
        failure_rate=1.0,
        batch_max_size=4,
        batch_window_s=0.005,
        rng_seed=1,
    )
    with pytest.raises(LLMInferenceError):
        b.generate("never", "")


def test_calls_in_same_window_share_latency():
    """Two callers arriving inside one batch window finish at nearly the
    same wall-clock time -- the canonical batching invariant."""
    b = _fast_backend()

    # Warm-up so the first thread doesn't carry import-time costs.
    b.generate("warm", "")

    finished_at: list[float] = []
    finished_lock = threading.Lock()

    def fire(prompt: str) -> None:
        b.generate(prompt, "ctx")
        with finished_lock:
            finished_at.append(time.perf_counter())

    t1 = threading.Thread(target=fire, args=("a",))
    t2 = threading.Thread(target=fire, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    delta = abs(finished_at[0] - finished_at[1])
    # If they batched together they finish within ~jitter of each other.
    # If they serialised, delta >= one full call latency (~0.05 s).
    assert delta < 0.02, f"calls did not batch together: delta={delta:.3f}s"


def test_residual_batch_drains():
    """Submit 17 calls (>2 full batches). All must complete."""
    b = _fast_backend()

    n = 17
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(lambda i: b.generate(f"q{i}", "ctx"), range(n)))
    assert len(results) == n
    assert all("Answer to" in r for r in results)


def test_batched_beats_serialised_sim():
    """Apples-to-apples: with serialise=True, sim genuinely sequential and
    batched serves N requests in N/batch_size sleeps. Batching wins."""
    from llm.inference import SimulatedLLMBackend

    seq = SimulatedLLMBackend(
        base_latency_s=0.05, per_token_latency_s=0.0, jitter_s=0.0,
        rng_seed=1, serialise=True,
    )
    bat = BatchedSimulatedLLMBackend(
        base_latency_s=0.05, per_token_latency_s=0.0, jitter_s=0.0,
        batch_max_size=8, batch_window_s=0.005, rng_seed=2, serialise=True,
    )

    n = 16

    def time_n(backend) -> float:
        # Warm-up
        backend.generate("warm", "")
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(lambda i: backend.generate(f"q{i}", "ctx"), range(n)))
        return time.perf_counter() - start

    seq_time = time_n(seq)
    bat_time = time_n(bat)
    speedup = seq_time / max(bat_time, 1e-9)
    # With serialise=True the sim takes ~16 * 0.05 = 0.8 s; batched takes
    # 2 batches * 0.05 = ~0.1 s. Real speedup ~ 5-8x; require >= 3x.
    assert speedup >= 3.0, (
        f"expected batched >= 3x faster than serialised sim, "
        f"got seq={seq_time:.3f}s bat={bat_time:.3f}s speedup={speedup:.2f}x"
    )
