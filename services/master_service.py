"""FastAPI master service.

Owns:
  - RAGRetriever (one per process)
  - LoadBalancer over RemoteWorkerProxy (one per configured worker URL)
  - MasterScheduler that orchestrates RAG + retry across remote workers

Configure via env vars:
    WORKER_URLS               (comma-separated, e.g. "http://worker-1:8000,http://worker-2:8000")
    WORKER_IDS                (optional; matches WORKER_URLS order; defaults gpu-worker-N)
    LB_STRATEGY               (round_robin | least_connections | load_aware, default: load_aware)
    MAX_CONCURRENT_TASKS      (default: 8) -- per-worker capacity hint for load_aware
    MAX_RETRIES               (default: 1) -- scheduler retries across workers
    RAG_USE_STUB              ("true" / "false", default: "true")
    RAG_TOP_K                 (default: 3)
    THREADPOOL_TOKENS         (default: 1000)
    WORKER_HTTP_TIMEOUT_SEC   (default: 30.0)
    MONITOR_INTERVAL_SEC      (default: 1.0)
    MONITOR_TIMEOUT_SEC       (default: 0.5)
    MONITOR_FAIL_THRESHOLD    (default: 3)
    MONITOR_RECOVER_THRESHOLD (default: 3)

Endpoints:
    GET  /                — liveness
    GET  /health          — master + per-worker snapshot
    GET  /metrics         — same as /health for now
    GET  /workers         — list of known workers + their cached state
    POST /request         — handle one client request end-to-end
"""
from __future__ import annotations

import os

import anyio
from fastapi import FastAPI, HTTPException

from common.metrics import MetricsBundle
from common.wire import (
    RequestPayload,
    ResponsePayload,
)
from lb import LoadBalancer, LoadBalancingStrategy
from llm import LLMInferenceEngine
from master import MasterScheduler
from master.health_monitor import HealthMonitor
from rag import RAGRetriever
from workers.remote_proxy import RemoteWorkerProxy


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


WORKER_URLS_RAW = os.environ.get(
    "WORKER_URLS",
    "http://localhost:8001,http://localhost:8002,http://localhost:8003",
)
WORKER_IDS_RAW = os.environ.get("WORKER_IDS", "")
LB_STRATEGY_RAW = os.environ.get("LB_STRATEGY", "load_aware").strip().lower()
MAX_CONCURRENT = _env_int("MAX_CONCURRENT_TASKS", 8)
MAX_RETRIES = _env_int("MAX_RETRIES", 1)
RAG_USE_STUB = _env_bool("RAG_USE_STUB", True)
RAG_TOP_K = _env_int("RAG_TOP_K", 3)
THREADPOOL_TOKENS = _env_int("THREADPOOL_TOKENS", 1000)
WORKER_HTTP_TIMEOUT = _env_float("WORKER_HTTP_TIMEOUT_SEC", 30.0)
MONITOR_INTERVAL = _env_float("MONITOR_INTERVAL_SEC", 1.0)
MONITOR_TIMEOUT = _env_float("MONITOR_TIMEOUT_SEC", 0.5)
MONITOR_FAIL_THRESHOLD = _env_int("MONITOR_FAIL_THRESHOLD", 3)
MONITOR_RECOVER_THRESHOLD = _env_int("MONITOR_RECOVER_THRESHOLD", 3)


def _parse_worker_urls() -> list[tuple[str, str]]:
    urls = [u.strip() for u in WORKER_URLS_RAW.split(",") if u.strip()]
    if not urls:
        raise RuntimeError("WORKER_URLS must contain at least one URL")

    if WORKER_IDS_RAW:
        ids = [i.strip() for i in WORKER_IDS_RAW.split(",") if i.strip()]
        if len(ids) != len(urls):
            raise RuntimeError(
                f"WORKER_IDS has {len(ids)} entries but WORKER_URLS has {len(urls)}"
            )
    else:
        ids = [f"gpu-worker-{i + 1}" for i in range(len(urls))]

    return list(zip(ids, urls, strict=False))


def _resolve_strategy(name: str) -> LoadBalancingStrategy:
    try:
        return LoadBalancingStrategy(name)
    except ValueError as exc:
        valid = [s.value for s in LoadBalancingStrategy]
        raise RuntimeError(
            f"LB_STRATEGY={name!r} invalid. Valid: {valid}"
        ) from exc


app = FastAPI(title="master")
metrics_bundle = MetricsBundle(service="master")

# Globals populated by startup.
proxies: list[RemoteWorkerProxy] = []
load_balancer: LoadBalancer | None = None
scheduler: MasterScheduler | None = None
retriever: RAGRetriever | None = None
inference_engine: LLMInferenceEngine | None = None
monitor: HealthMonitor | None = None


@app.on_event("startup")
async def _on_startup() -> None:
    global proxies, load_balancer, scheduler, retriever, inference_engine, monitor

    anyio.to_thread.current_default_thread_limiter().total_tokens = THREADPOOL_TOKENS

    pairs = _parse_worker_urls()
    proxies = [
        RemoteWorkerProxy(
            worker_id=worker_id,
            url=url,
            max_concurrent_tasks=MAX_CONCURRENT,
            timeout_seconds=WORKER_HTTP_TIMEOUT,
        )
        for worker_id, url in pairs
    ]

    # Probe each worker's /health to discover its actual MAX_CONCURRENT_TASKS,
    # so heterogeneous worker pools (different sizes) feed the right capacity
    # into load_aware / power_of_two strategies. Best-effort: workers that
    # don't answer keep the env-default capacity.
    for proxy in proxies:
        info = proxy.probe_health()
        if info is not None and info.max_concurrent_tasks > 0:
            proxy.max_concurrent_tasks = info.max_concurrent_tasks
            print(
                f"[master-svc] {proxy.worker_id}: adopting "
                f"max_concurrent_tasks={info.max_concurrent_tasks} from worker"
            )

    strategy = _resolve_strategy(LB_STRATEGY_RAW)
    # LoadBalancer's static type hints want GPUWorkerNode but it only touches
    # the duck-typed surface (status, pending_tasks, max_concurrent_tasks,
    # reserve, worker_id) — RemoteWorkerProxy implements all of these.
    load_balancer = LoadBalancer(proxies, strategy=strategy)  # type: ignore[arg-type]

    retriever = RAGRetriever(use_stub=RAG_USE_STUB, top_k=RAG_TOP_K)
    # Inference engine here is unused by RemoteWorkerProxy.process() but is
    # required by the MasterScheduler signature (it's what gets passed through
    # to local workers in the in-process variant).
    inference_engine = LLMInferenceEngine()
    scheduler = MasterScheduler(
        retriever=retriever,
        inference_engine=inference_engine,
        workers=proxies,  # type: ignore[arg-type]
        max_retries=MAX_RETRIES,
    )

    monitor = HealthMonitor(
        proxies,
        poll_interval_seconds=MONITOR_INTERVAL,
        probe_timeout_seconds=MONITOR_TIMEOUT,
        failure_threshold=MONITOR_FAIL_THRESHOLD,
        recovery_threshold=MONITOR_RECOVER_THRESHOLD,
    )
    await monitor.start()

    print(
        f"[master-svc] Ready. workers={[p.worker_id for p in proxies]}, "
        f"strategy={strategy.value}, max_retries={MAX_RETRIES}, "
        f"rag_stub={RAG_USE_STUB}, threadpool={THREADPOOL_TOKENS}"
    )


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    if monitor is not None:
        await monitor.stop()
    for proxy in proxies:
        try:
            proxy.close()
        except Exception as exc:  # noqa: BLE001 -- best effort cleanup
            print(f"[master-svc] Failed to close {proxy.worker_id}: {exc}")
    print("[master-svc] Shutdown complete.")


@app.get("/")
def root() -> dict[str, object]:
    return {
        "status": "ok",
        "workers": [p.worker_id for p in proxies],
        "strategy": load_balancer.strategy.value if load_balancer else None,
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "scheduler": {
            "total_requests": scheduler.stats.total_requests if scheduler else 0,
            "successful_requests": scheduler.stats.successful_requests if scheduler else 0,
            "failed_requests": scheduler.stats.failed_requests if scheduler else 0,
        },
        "workers": [p.snapshot_metrics() for p in proxies],
        "monitor": monitor.snapshot() if monitor else [],
    }


@app.get("/metrics")
def metrics():
    # Refresh per-worker gauges before scraping so Prometheus sees current state.
    for p in proxies:
        snap = p.snapshot_metrics()
        metrics_bundle.update_target_state(
            target=p.worker_id,
            status=str(snap["status"]),
            active_tasks=int(snap["active_tasks"]),
            pending_tasks=int(snap["pending_tasks"]),
        )
    return metrics_bundle.handler()


@app.get("/workers")
def workers_endpoint() -> list[dict[str, object]]:
    return [p.snapshot_metrics() for p in proxies]


@app.post("/admin/strategy")
def admin_strategy(payload: dict) -> dict[str, str]:
    """Switch the LB strategy at runtime. Used by scripts/benchmark.py to
    A/B compare strategies without restarting the master process."""
    if load_balancer is None:
        raise HTTPException(status_code=503, detail="master not initialised")
    name = str(payload.get("strategy", "")).strip().lower()
    try:
        strategy = LoadBalancingStrategy(name)
    except ValueError as exc:
        valid = [s.value for s in LoadBalancingStrategy]
        raise HTTPException(
            status_code=400,
            detail=f"strategy={name!r} invalid. Valid: {valid}",
        ) from exc
    load_balancer.set_strategy(strategy)
    return {"strategy": strategy.value}


@app.post("/admin/backend")
def admin_backend_fanout(payload: dict) -> dict[str, object]:
    """Fan out a backend swap to every registered worker.

    This is a convenience for the benchmark harness so it doesn't need
    each worker port published to the host. Failures on individual
    workers are reported per-worker; the master endpoint itself returns 200
    as long as it could reach at least one worker.
    """
    results: dict[str, object] = {}
    any_ok = False
    for proxy in proxies:
        try:
            body = proxy.post_json("/admin/backend", payload)
            results[proxy.worker_id] = body
            any_ok = True
        except Exception as exc:  # noqa: BLE001
            results[proxy.worker_id] = {"error": str(exc)}
    if not any_ok:
        raise HTTPException(status_code=503, detail="no workers reachable")
    return {"workers": results}


@app.post("/request", response_model=ResponsePayload)
def handle_request(payload: RequestPayload) -> ResponsePayload:
    if scheduler is None or load_balancer is None:
        raise HTTPException(status_code=503, detail="master not initialised")

    request = payload.to_dataclass()
    try:
        worker = load_balancer.select_worker(request)
    except RuntimeError as exc:
        # All workers are FAILED — surface as 503 so the caller (LB / NGINX)
        # can return a graceful error to the client.
        metrics_bundle.requests_total.labels("master", "no-worker", "all").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    target = worker.worker_id
    try:
        with metrics_bundle.time_request(target=target):
            response = scheduler.handle_request(request, worker)  # type: ignore[arg-type]
            if response.status != "completed":
                raise HTTPException(status_code=502, detail="all retries exhausted")
    except HTTPException:
        raise
    return ResponsePayload.from_dataclass(response)
