"""FastAPI load-balancer service.

Sits between NGINX (or the client) and one or more master services. Routes
each incoming /request to a master using the same `LoadBalancer` class that
the master uses for worker selection — so all three strategies (round_robin,
least_connections, load_aware) work at the LB tier too.

Configure via env vars:
    MASTER_URLS               (comma-separated, e.g. "http://master:9000")
    MASTER_IDS                (optional; defaults master-N in order)
    LB_STRATEGY               (round_robin | least_connections | load_aware, default: round_robin)
    MASTER_MAX_INFLIGHT       (default: 100) -- capacity hint per master for load_aware
    MASTER_HTTP_TIMEOUT_SEC   (default: 60.0)
    THREADPOOL_TOKENS         (default: 1000)

Endpoints:
    GET  /                — liveness
    GET  /health          — LB + per-master snapshot
    GET  /metrics         — same as /health for now
    POST /request         — proxy to a master using configured strategy
"""
from __future__ import annotations

import os

import anyio
from fastapi import FastAPI, HTTPException

from common.wire import RequestPayload, ResponsePayload
from lb import LoadBalancer, LoadBalancingStrategy
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


MASTER_URLS_RAW = os.environ.get("MASTER_URLS", "http://localhost:9000")
MASTER_IDS_RAW = os.environ.get("MASTER_IDS", "")
LB_STRATEGY_RAW = os.environ.get("LB_STRATEGY", "round_robin").strip().lower()
MASTER_MAX_INFLIGHT = _env_int("MASTER_MAX_INFLIGHT", 100)
MASTER_HTTP_TIMEOUT = _env_float("MASTER_HTTP_TIMEOUT_SEC", 60.0)
THREADPOOL_TOKENS = _env_int("THREADPOOL_TOKENS", 1000)


def _parse_master_urls() -> list[tuple[str, str]]:
    urls = [u.strip() for u in MASTER_URLS_RAW.split(",") if u.strip()]
    if not urls:
        raise RuntimeError("MASTER_URLS must contain at least one URL")

    if MASTER_IDS_RAW:
        ids = [i.strip() for i in MASTER_IDS_RAW.split(",") if i.strip()]
        if len(ids) != len(urls):
            raise RuntimeError(
                f"MASTER_IDS has {len(ids)} entries but MASTER_URLS has {len(urls)}"
            )
    else:
        ids = [f"master-{i + 1}" for i in range(len(urls))]

    return list(zip(ids, urls))


def _resolve_strategy(name: str) -> LoadBalancingStrategy:
    try:
        return LoadBalancingStrategy(name)
    except ValueError as exc:
        valid = [s.value for s in LoadBalancingStrategy]
        raise RuntimeError(
            f"LB_STRATEGY={name!r} invalid. Valid: {valid}"
        ) from exc


app = FastAPI(title="lb")

# Globals populated by startup. Each master is wrapped in a RemoteWorkerProxy
# so the existing LoadBalancer can route to it without a separate abstraction.
master_proxies: list[RemoteWorkerProxy] = []
load_balancer: LoadBalancer | None = None


@app.on_event("startup")
def _on_startup() -> None:
    global master_proxies, load_balancer

    anyio.to_thread.current_default_thread_limiter().total_tokens = THREADPOOL_TOKENS

    pairs = _parse_master_urls()
    master_proxies = [
        RemoteWorkerProxy(
            worker_id=master_id,
            url=url,
            max_concurrent_tasks=MASTER_MAX_INFLIGHT,
            timeout_seconds=MASTER_HTTP_TIMEOUT,
        )
        for master_id, url in pairs
    ]

    strategy = _resolve_strategy(LB_STRATEGY_RAW)
    load_balancer = LoadBalancer(master_proxies, strategy=strategy)  # type: ignore[arg-type]

    print(
        f"[lb-svc] Ready. masters={[p.worker_id for p in master_proxies]}, "
        f"strategy={strategy.value}, threadpool={THREADPOOL_TOKENS}"
    )


@app.on_event("shutdown")
def _on_shutdown() -> None:
    for proxy in master_proxies:
        try:
            proxy.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[lb-svc] Failed to close {proxy.worker_id}: {exc}")
    print("[lb-svc] Shutdown complete.")


@app.get("/")
def root() -> dict[str, object]:
    return {
        "status": "ok",
        "masters": [p.worker_id for p in master_proxies],
        "strategy": load_balancer.strategy.value if load_balancer else None,
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "masters": [p.snapshot_metrics() for p in master_proxies],
    }


@app.get("/metrics")
def metrics() -> dict[str, object]:
    return health()


@app.post("/request", response_model=ResponsePayload)
def handle_request(payload: RequestPayload) -> ResponsePayload:
    if load_balancer is None:
        raise HTTPException(status_code=503, detail="lb not initialised")

    request = payload.to_dataclass()
    try:
        master = load_balancer.select_worker(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        body = master.post_json("/request", payload.model_dump())
        return ResponsePayload.model_validate(body)
    except Exception as exc:  # noqa: BLE001 -- post_json raises WorkerTransientError, body validation can also fail
        raise HTTPException(status_code=502, detail=f"master error: {exc}") from exc
    finally:
        master.release()
