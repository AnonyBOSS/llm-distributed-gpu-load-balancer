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

The system uses a `SimulatedLLMBackend` by default for fast iteration. Setting `LLM_BACKEND=hf` per worker container loads `distilgpt2` via the HuggingFace `transformers` pipeline.

**CPU mode (default):** the base [deploy/Dockerfile](../deploy/Dockerfile) installs CPU-only torch from PyTorch's CPU index (`https://download.pytorch.org/whl/cpu`). `make up` brings up the CPU stack. Both `sim` and `hf` backends run on CPU.

**GPU mode:** [deploy/Dockerfile.gpu](../deploy/Dockerfile.gpu) starts from `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`. [deploy/docker-compose.gpu.yml](../deploy/docker-compose.gpu.yml) attaches an NVIDIA GPU device per worker via `deploy.resources.reservations.devices`. The HF backend's `device="auto"` kwarg probes `torch.cuda.is_available()` and uses the GPU when present. `make gpu-up && make gpu-smoke` exercises the full path end-to-end. NVIDIA Container Toolkit (Linux) or Docker Desktop with WSL2-GPU (Windows) is required.

## Trade-offs and theory

This section makes the design decisions explicit so the reader can see *why* each was chosen, not just what it does.

### CAP positioning

Under the partition tolerance dimension of CAP (Brewer, PODC '00; Gilbert & Lynch, SIGACT '02), this system chooses **AP — availability with eventual consistency**. When the master temporarily can't reach a worker, it does NOT block requests waiting for ground truth; it serves from the cached `RemoteWorkerProxy` state and rolls forward. The active health monitor (1 s probes) is the eventual-consistency repair: within ~3 s of a real outage, the cached state catches up and the worker is removed from rotation.

The opposite choice (CP) would freeze incoming requests while ground truth is unknown — wrong for a serving system whose SLO is latency.

### Why a 3-strike circuit breaker?

The 3-failure / 3-recovery threshold in [master/health_monitor.py](../master/health_monitor.py) is not arbitrary: it's the standard pattern from Nygard's *Release It!* (2nd ed., Pragmatic Bookshelf 2018) and Netflix's Hystrix. Reasoning:

- **1-strike** would make any transient blip (a single dropped TCP packet) take a worker out of rotation. False positives dominate.
- **5+ strikes** delays detection of a real outage to 5+ × 1 s = 5+ s, during which the LB keeps sending traffic to a black hole.
- **3 strikes** is the Goldilocks point: 3 s detection latency, very low false-positive rate at our 1 s probe cadence, and trivially tunable per env (`MONITOR_FAIL_THRESHOLD`).

### Why `pending_tasks`, not `active_tasks`, for selection?

`active_tasks` increments inside `worker.process()`, which is *after* the LB has handed out the worker. Two concurrent selectors that both see `active_tasks == 0` for the same worker pick the same worker — the thundering herd. `pending_tasks` increments during selection (atomically with the lock-protected pick), so the second selector sees the first's reservation and chooses elsewhere. Equivalent to a token-bucket reservation in queueing-theory terms.

This is the same race that Linux's `epoll` exclusive mode (Linux 4.5+) and Cloudflare's [lock-then-decrement](https://blog.cloudflare.com/the-sad-state-of-linux-socket-balancing/) pattern address. A 50-thread regression test in [tests/unit/test_load_balancer.py::test_concurrent_least_connections_distributes](../tests/unit/test_load_balancer.py) would have caught the original bug; it now guards against regressions.

### Why a worker self-sheds when at capacity

Each `GPUWorkerNode.process()` raises `WorkerAtCapacityError` when `active_tasks >= max_concurrent_tasks`. The worker doesn't accept the request and queue it on the GPU; instead the HTTP boundary returns `503` with `X-Reject-Reason: at-capacity`, the proxy raises `WorkerAtCapacityError` to the scheduler, and the scheduler walks to the next candidate without spending a retry attempt.

Why is this an architectural improvement, not just defensive coding?

- **Without self-shedding** the worker accepts every request and queues them. On a single-GPU host that means N concurrent `transformers.pipeline(...)` calls fighting for one PCIe bus and one VRAM allocator. CUDA starts evicting and reloading KV-cache between calls. Every request slows down by 10–100×, and because each Python thread is busy-spinning in CUDA driver code waiting for the GPU, the host CPU gets pegged at 100 %. This *is* what bricked the host on the first GPU run; see commit `bc37a36`.
- **With self-shedding** load above capacity becomes a *routing problem*, not a *resource-contention problem*. The master sees the 503, the LB picks a different worker (or returns a graceful failure if every worker is busy), and the GPU keeps running its in-flight batch at full speed.

This is the *load shedding* pattern from production serving (Brooker, [Amazon Builders' Library: Caching challenges and strategies](https://aws.amazon.com/builders-library/caching-challenges-and-strategies/), 2020): refusing work you can't do well is strictly better than accepting work you can't finish. Two production techniques in this family — graceful degradation and admission control — collapse to the same insight.

The 503 carries a custom `X-Reject-Reason: at-capacity` header so the proxy can distinguish "worker overloaded" from "worker is dying." A real failure (5xx without that header, or a transport error) still trips the consecutive-failure counter and eventually marks the worker FAILED. Capacity rejections do not.

### Why two independent fault-tolerance layers?

Per-request retry (in [master/scheduler.py](../master/scheduler.py)) and active health probing (in [master/health_monitor.py](../master/health_monitor.py)) have *different* MTTR profiles and fail open in *different* scenarios. The pattern is documented in the Google SRE Workbook ("Managing Load"):

- **Single transient error** (e.g. a momentary GC pause): retry catches it, monitor never sees it.
- **Sustained outage** (e.g. process crashed): retry exhausts attempts on the first request, but the monitor moves the worker out of the pool within ~3 s so subsequent requests don't waste time on it.
- **Silent recovery** (e.g. operator restarted the container): monitor's recovery threshold notices and re-admits the worker; per-request retry has no way to discover this.

Combining them is robust against any *one* of these failing, which is the property the SRE Workbook calls "defence in depth."

### Strategy choice for heterogeneous workers

In the homogeneous default compose (3 workers × 8 slots each), all four strategies converge — every worker is identical, so any selection rule produces near-uniform load. The strategies *only* differentiate themselves when workers differ. The heterogeneous benchmark ([scripts/heterogeneous_bench.py](../scripts/heterogeneous_bench.py), with [deploy/docker-compose.heterogeneous.yml](../deploy/docker-compose.heterogeneous.yml) at 1 : 2 : 8 capacity ratio) is what surfaces:

- `round_robin` ignores capacity entirely → systematic overload of the small worker.
- `least_connections` ignores capacity → same problem.
- `load_aware` (Reiss et al., SoCC '12) divides queue depth by capacity → tracks the actual loadable amount.
- `power_of_two` (Mitzenmacher 2001) gets close to global least-loaded with O(1) state per request — important when the worker pool is large enough that scanning all of them is itself a bottleneck.

The standard result from Mitzenmacher's analysis is: under uniform random arrival, `power_of_two` reduces maximum queue length from $\Theta(\log n / \log \log n)$ to $\Theta(\log \log n)$ — exponentially better tails for the same per-request work.

## References

- Mitzenmacher, M. (2001). *The Power of Two Choices in Randomized Load Balancing.* IEEE TPDS.
- Reiss, C., Tumanov, A., Ganger, G. R., Katz, R. H., Kozuch, M. A. (2012). *Heterogeneity and Dynamicity of Clouds at Scale: Google Trace Analysis.* SoCC '12.
- Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., Chun, B.-G. (2022). *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI '22.
- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., Stoica, I. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). SOSP '23.
- Nygard, M. (2018). *Release It! Design and Deploy Production-Ready Software*, 2nd ed. Pragmatic Bookshelf. (Circuit breaker pattern.)
- Brewer, E. (2000). *Towards Robust Distributed Systems.* PODC '00 keynote. (CAP theorem.)
- Gilbert, S., Lynch, N. (2002). *Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-Tolerant Web Services.* SIGACT.
- Beyer, B., Murphy, N. R., Rensin, D. K., Kawahara, K., Thorne, S. (eds.) (2018). *The Site Reliability Workbook.* O'Reilly. (Defence-in-depth load management.)
