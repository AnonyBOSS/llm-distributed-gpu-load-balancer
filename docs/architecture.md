# System Architecture

## Topology

```
client  →  nginx (8080)  →  lb (7000)  →  master (9000)  →  worker-{1,2,3} (8000)
                                              │                       ▲
                                              │  active health probes │
                                              └───────────────────────┘
                            ↓
                   prometheus (9090) ──→ grafana (3000)
```

Eight processes, each its own container. Communication is JSON over HTTP.

## Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **nginx** | [deploy/nginx.conf](../deploy/nginx.conf) | Front-of-house reverse proxy. `least_conn` upstream. |
| **lb service** | [services/lb_service.py](../services/lb_service.py) | Receives `POST /request`, routes to one of N masters using a `LoadBalancer` over master URLs. |
| **master service** | [services/master_service.py](../services/master_service.py) | Owns the request lifecycle: RAG retrieval, worker selection via inner `LoadBalancer`, retry across remote workers. Runs the [HealthMonitor](../master/health_monitor.py) as a background asyncio task. |
| **worker service** | [services/worker_service.py](../services/worker_service.py) | One `GPUWorkerNode` + one `LLMInferenceEngine` per process. `POST /process` runs inference; `POST /admin/{fail,recover}` for failure simulation. |
| **prometheus** | [deploy/prometheus.yml](../deploy/prometheus.yml) | Scrapes every service's `/metrics` every 5 s. |
| **grafana** | [deploy/grafana/](../deploy/grafana/) | Auto-provisioned dashboard at `/d/cse354-overview`. |

## Request flow

1. Client POSTs `/request` to nginx (port 8080).
2. nginx forwards to the LB service (port 7000) via `least_conn`.
3. The LB service applies its configured strategy (default `round_robin`) to choose a master. With one master in the demo this is degenerate; with multiple masters it would distribute load.
4. The chosen master receives the request, runs RAG retrieval against its in-memory FAISS index (or stub), then asks its inner `LoadBalancer` (default `load_aware`) which **worker** to send it to.
5. `LoadBalancer.select_worker()` is lock-protected and atomically reserves the chosen worker's `pending_tasks` counter — the fix for the original thundering-herd bug.
6. The master forwards the request to the worker over HTTP via a `RemoteWorkerProxy`. The proxy duck-types as a `GPUWorkerNode`, so the existing `MasterScheduler` retry loop works without modification.
7. The worker runs `LLMInferenceEngine.generate(...)` (sim or HuggingFace backend) and returns the answer.
8. The master assembles the `Response` and returns it back up the chain.

## Failure recovery

Two independent layers of fault tolerance:

1. **Per-request retry** ([master/scheduler.py](../master/scheduler.py)) — if a worker raises a transient error, the scheduler retries on the next-best worker. Configurable via `MAX_RETRIES`.
2. **Active health monitor** ([master/health_monitor.py](../master/health_monitor.py)) — the master polls every worker's `GET /health` every 1 s with a 500 ms timeout. Three consecutive misses flip a worker `FAILED`; three consecutive successes after that revive it to `HEALTHY`. The `LoadBalancer` filters `FAILED` workers from selection.

Combined: a single transient blip costs at most one retry. A node that stays down is detected within ~3 s and removed from rotation. When it comes back, it is auto-recovered within ~3 s of the first successful probe.

## Concurrency model

- Each FastAPI service uses sync `def` endpoints with the anyio threadpool bumped to 1000 tokens — sufficient for the 1000-concurrent-user target without going async.
- Within the master, `httpx.Client` is sync but each `RemoteWorkerProxy` keeps a connection-pooled keepalive client per worker (200 max conns each).
- The health monitor is the one async piece — it runs as an `asyncio.Task` started in the FastAPI startup hook, sharing the loop with the request handlers.
- The `GPUWorkerNode` and `RemoteWorkerProxy` mutate counters under `threading.Lock`; `LoadBalancer.select_worker()` holds the lock for both selection and reservation, eliminating the herd race.

## Scaling story

Horizontal scale today is one-axis: add more `worker-N` containers and add their URLs to the master's `WORKER_URLS` env. The architecture also accommodates multiple masters (for higher RAG throughput) and multiple LBs (for HA), since the LB tier already routes across master URLs by the same `LoadBalancer` strategies. A future change would replace static env-based registration with a dynamic `POST /workers/register` endpoint so workers can join at runtime.

## What about real GPUs?

The system uses a `SimulatedLLMBackend` by default for fast iteration. Setting `LLM_BACKEND=hf` per worker container loads `distilgpt2` via the HuggingFace `transformers` pipeline. CUDA is not required; both backends run on CPU. The architecture above is identical with real GPUs — only the worker's `LLMInferenceEngine` swaps backend.
