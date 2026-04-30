"""FastAPI worker service.

One process = one GPUWorkerNode + one LLMInferenceEngine.

Configure via env vars:
    WORKER_ID                 (default: gpu-worker-1)
    GPU_NAME                  (default: NVIDIA-A100-SIM)
    MAX_CONCURRENT_TASKS      (default: 8)
    FAILURE_RATE              (default: 0.0)
    THREADPOOL_TOKENS         (default: 1000)  -- anyio threadpool size
    LLM_BACKEND               (sim | hf, default: sim)
    LLM_MODEL                 (default: distilgpt2 when LLM_BACKEND=hf)

Endpoints:
    GET  /                — liveness ("ok")
    GET  /health          — full worker snapshot
    GET  /metrics         — same as /health (until Prometheus added in week 2)
    POST /process         — run one inference
    POST /admin/fail      — mark FAILED (for failure-simulation tests)
    POST /admin/recover   — mark HEALTHY
"""
from __future__ import annotations

import os
from time import perf_counter

import anyio
from fastapi import FastAPI, HTTPException

from common.metrics import MetricsBundle
from common.wire import (
    ProcessRequest,
    ProcessResponse,
    WorkerHealth,
)
from llm import LLMInferenceEngine
from llm.inference import (
    BatchedSimulatedLLMBackend,
    HuggingFaceLLMBackend,
    SimulatedLLMBackend,
)
from workers import GPUWorkerNode, WorkerAtCapacityError, WorkerUnavailableError


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"env {name}={raw!r} is not an int") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"env {name}={raw!r} is not a float") from exc


WORKER_ID = os.environ.get("WORKER_ID", "gpu-worker-1")
GPU_NAME = os.environ.get("GPU_NAME", "NVIDIA-A100-SIM")
MAX_CONCURRENT = _env_int("MAX_CONCURRENT_TASKS", 8)
FAILURE_RATE = _env_float("FAILURE_RATE", 0.0)
THREADPOOL_TOKENS = _env_int("THREADPOOL_TOKENS", 1000)


app = FastAPI(title=f"worker:{WORKER_ID}")
worker = GPUWorkerNode(
    worker_id=WORKER_ID,
    gpu_name=GPU_NAME,
    max_concurrent_tasks=MAX_CONCURRENT,
    failure_rate=FAILURE_RATE,
)
engine = LLMInferenceEngine()
metrics_bundle = MetricsBundle(service=f"worker:{WORKER_ID}")


@app.on_event("startup")
def _on_startup() -> None:
    # Default anyio threadpool is ~40 tokens; sync `def` endpoints + sync LLM
    # backend would block at that ceiling under 1000 concurrent users.
    anyio.to_thread.current_default_thread_limiter().total_tokens = THREADPOOL_TOKENS
    print(
        f"[worker-svc] {WORKER_ID} ready on {GPU_NAME} "
        f"(max_concurrent={MAX_CONCURRENT}, failure_rate={FAILURE_RATE}, "
        f"threadpool={THREADPOOL_TOKENS})"
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "worker_id": WORKER_ID}


@app.get("/health", response_model=WorkerHealth)
def health() -> dict[str, object]:
    snap = worker.snapshot_metrics()
    metrics_bundle.update_target_state(
        target=WORKER_ID,
        status=str(snap["status"]),
        active_tasks=int(snap["active_tasks"]),
        pending_tasks=int(snap["pending_tasks"]),
    )
    return snap


@app.get("/metrics")
def metrics():
    # Refresh gauges from the live worker snapshot before scraping so the
    # pulled values are always current, not just the last /process update.
    snap = worker.snapshot_metrics()
    metrics_bundle.update_target_state(
        target=WORKER_ID,
        status=str(snap["status"]),
        active_tasks=int(snap["active_tasks"]),
        pending_tasks=int(snap["pending_tasks"]),
    )
    return metrics_bundle.handler()


@app.post("/process", response_model=ProcessResponse)
def process(payload: ProcessRequest) -> ProcessResponse:
    request = payload.request.to_dataclass()
    start = perf_counter()
    try:
        with metrics_bundle.time_request(target=WORKER_ID):
            answer = worker.process(request, payload.context, engine)
    except WorkerAtCapacityError as exc:
        # Load shedding, not failure. 503 + Retry-After tells callers to
        # try a different worker; the proxy treats this as a routing miss
        # and does NOT trip its circuit breaker on this response.
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "0", "X-Reject-Reason": "at-capacity"},
        ) from exc
    except WorkerUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Transient or LLM error — surface as 500 so the caller's retry loop kicks in.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    latency = perf_counter() - start
    return ProcessResponse(
        worker_id=WORKER_ID,
        answer=answer,
        latency_seconds=latency,
    )


@app.post("/admin/fail")
def admin_fail() -> dict[str, str]:
    worker.mark_failed()
    return {"worker_id": WORKER_ID, "status": worker.status.value}


@app.post("/admin/recover")
def admin_recover() -> dict[str, str]:
    worker.mark_healthy()
    return {"worker_id": WORKER_ID, "status": worker.status.value}


@app.post("/admin/backend")
def admin_backend(payload: dict) -> dict[str, str]:
    """Hot-swap the LLM backend at runtime.

    Used by scripts/benchmark.py to A/B sim vs batched_sim without
    rebuilding containers. Note: HF backend swap performs a real model
    load and may take several seconds for the first request.
    """
    global engine
    name = str(payload.get("backend", "")).strip().lower()
    if name in ("sim", "simulated"):
        engine = LLMInferenceEngine(backend=SimulatedLLMBackend())
    elif name in ("batched", "batched_sim", "batched-sim"):
        engine = LLMInferenceEngine(backend=BatchedSimulatedLLMBackend())
    elif name in ("hf", "huggingface", "transformers"):
        model_name = str(payload.get("model", "distilgpt2"))
        device = str(payload.get("device", "auto"))
        engine = LLMInferenceEngine(
            backend=HuggingFaceLLMBackend(model_name=model_name, device=device)
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"backend={name!r} invalid. Use 'sim', 'batched_sim', or 'hf'.",
        )
    print(f"[worker-svc] Hot-swapped backend to {name}")
    return {"backend": name}
