# Distributed GPU Load Balancer for LLM Requests

This repository implements the CSE354 Distributed Computing project:
*"Efficient Load Balancing and GPU Cluster Task Distribution for Handling 1000+ Concurrent LLM Requests."*

The project models a distributed AI serving platform that accepts many simultaneous user requests, routes them across multiple GPU-backed workers, enriches prompts with Retrieval-Augmented Generation (RAG), and stays responsive when the system is under heavy load or partial failure.

## Project Goals

- handle 1000+ concurrent LLM requests
- distribute work across multiple compute nodes using several scheduling strategies
- simulate GPU task scheduling and inference execution
- integrate a real RAG pipeline for context-aware responses
- support scalability, observability, and fault tolerance

## Current Status

The repository runs as a real distributed system: each component is its own
FastAPI process in its own Docker container, with active health monitoring,
Prometheus + Grafana observability, a pytest CI suite, and a benchmark
harness that drives the full 100→1000 user ramp across all three LB
strategies plus a fault-injection scenario.

The single-process simulation (`main.py`, `scripts/smoke_concurrent.py`)
is preserved for fast local iteration and is what the unit tests target.

### Quickstart (distributed mode)

```bash
make up                                       # CPU stack (8 containers)
curl -X POST http://localhost:8080/request \
     -H 'Content-Type: application/json' \
     -d '{"request_id":"r1","user_id":"u1","prompt":"hello","metadata":{}}'
make bench-quick                              # 50- + 200-user smoke benchmark
make down                                     # tear down
```

`make help` lists every available target. Live dashboards once the stack is up:

| URL | What |
|---|---|
| <http://localhost:3000/d/cse354-overview> | Grafana dashboard (no login) |
| <http://localhost:9090/targets> | Prometheus scrape targets |
| <http://localhost:9000/docs> | Master service auto-generated OpenAPI |
| <http://localhost:7000/docs> | Load-balancer service OpenAPI |

Benchmark variants:

```bash
make bench           # full 4-strategy x 4-user-count run + fault injection (~10 min)
make bench-batching  # sim vs continuous-batching head-to-head
make bench-hetero    # heterogeneous workers (capacity 1:2:8) — strategies actually differ
```

GPU mode (requires NVIDIA GPU + `nvidia-container-toolkit`):

```bash
make gpu-up          # 2 workers on CUDA + 1 sim worker behind one master
make gpu-smoke       # one real inference end-to-end
make bench-gpu       # GPU benchmark (defaults to 50 users — see hardware notes)
make gpu-down
```

### Hardware notes

The default GPU compose ([deploy/docker-compose.gpu.yml](deploy/docker-compose.gpu.yml)) is tuned for ~6 GB consumer GPUs (RTX 3060 / 4060) and runs **2 GPU workers + 1 CPU/sim worker**. The math:

| Setup | Per-worker VRAM (distilgpt2 fp32) | Total | Free on 6 GB | Outcome |
|---|---:|---:|---:|---|
| 1 GPU worker | ~2 GB | 2 GB | ~4 GB | Plenty of headroom; loses redundancy |
| **2 GPU workers (default)** | ~2 GB | ~4 GB | ~2 GB | Comfortable for 50–100 user benchmarks |
| 3 GPU workers | ~2 GB | ~6 GB | ~0 GB | OOMs under load, CPU pegs at 100 % |

Three guardrails prevent the OOM-then-CPU-thrash failure mode:

1. **Worker self-shedding** — each worker's `process()` raises `WorkerAtCapacityError` when `active_tasks >= MAX_CONCURRENT_TASKS`. The HTTP boundary returns 503 with `X-Reject-Reason: at-capacity`; the master's scheduler falls over to the next candidate without spending a retry attempt and without tripping the proxy's circuit breaker. Tested in [tests/unit/test_worker.py::test_at_capacity_worker_rejects_without_running](tests/unit/test_worker.py) and [tests/unit/test_scheduler.py::test_at_capacity_worker_falls_over_without_consuming_retries](tests/unit/test_scheduler.py).
2. **Conservative `MAX_CONCURRENT_TASKS=2` per GPU worker** — caps in-flight inference per GPU process to leave KV-cache room.
3. **`make bench-gpu` pre-flight VRAM check** — queries `nvidia-smi` before the run, aborts if free VRAM is too low for the requested peak user count.

For larger GPUs (24 GB+), edit `deploy/docker-compose.gpu.yml` to put `worker-3` on GPU too and bump `MAX_CONCURRENT_TASKS`.

### Topology

```
client → nginx (8080) → lb (7000) → master (9000) → worker-{1,2,3} (8000)
                                        │                   ▲
                                        │  health probes    │
                                        └───────────────────┘
                            ↓
                   prometheus (9090) ──→ grafana (3000)
```

- Three GPU workers, each its own process with its own `LLMInferenceEngine`.
- One master orchestrating RAG + worker selection + per-request retry.
- One LB tier (extensible to N masters).
- One nginx upstream (matching the brief's "Load Balancing Tools: NGINX" requirement).
- Prometheus scrapes every service every 5 s; Grafana auto-loads the dashboard.

See [docs/architecture.md](docs/architecture.md) for the full request flow,
failure-recovery model, and concurrency story.

### Implemented

- shared `Request`/`Response` data models in `common`
- Pydantic wire models in [common/wire.py](common/wire.py) for HTTP serialisation
- a configurable `LoadBalancer` in `lb` with three strategies: round-robin, least-connections, and load-aware, with a lock + reservation counter that fixes a real thundering-herd bug discovered during smoke testing
- a `MasterScheduler` in `master` that drives the RAG step, dispatches work to workers, and retries on failure
- an active `HealthMonitor` in [master/health_monitor.py](master/health_monitor.py) that polls every worker every 1 s and auto-flips them FAILED/HEALTHY with a 3-strike circuit breaker
- thread-safe `GPUWorkerNode` instances in `workers` with capacity, health states, latency tracking, completion counters, and injectable failures
- a `RemoteWorkerProxy` in [workers/remote_proxy.py](workers/remote_proxy.py) that duck-types as `GPUWorkerNode` and speaks HTTP, so the LB and scheduler work over the network unchanged
- an `LLMInferenceEngine` in `llm` with a default simulated backend and an optional HuggingFace backend selectable via an environment variable
- a real `RAGRetriever` in `rag` backed by `sentence-transformers` and FAISS, with a fast deterministic stub
- FastAPI services in [services/](services/) for worker, master, and LB tiers, each exposing `/health` and Prometheus `/metrics`
- Prometheus + Grafana stack pre-provisioned in [deploy/](deploy/) with a dashboard for throughput, p50/p95/p99 latency, per-worker utilisation, and live worker status
- 36 unit tests + 4 integration tests in [tests/](tests/) covering the herd regression, scheduler retry/fallover, monitor circuit breaker, RAG, and the live compose stack
- GitHub Actions CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) running unit tests on push and integration tests on PRs
- benchmark harness in [scripts/benchmark.py](scripts/benchmark.py) running 100→1000 ramps × 3 strategies × clean/fault, saving CSV + chart PNGs

### Documentation

- [docs/architecture.md](docs/architecture.md) — components, request flow, failure recovery, concurrency model
- [docs/benchmarks.md](docs/benchmarks.md) — methodology, headline numbers, charts
- [benchmarks/results.csv](benchmarks/results.csv) — raw aggregated benchmark table

## System Architecture

The architecture follows the CSE354 brief.

### 1. Client Layer (`client/`)

`ClientLoadGenerator` produces synthetic `Request` objects with stable IDs, sample prompts, and metadata. It is the source of traffic for the dry run and the smoke test.

### 2. Load Balancer (`lb/`)

`LoadBalancer` accepts a list of workers and a `LoadBalancingStrategy`. Three strategies are supported:

| Strategy            | Selection rule                                                                                                 | Cite |
|---------------------|----------------------------------------------------------------------------------------------------------------|------|
| `ROUND_ROBIN`       | next worker in a fixed rotation, regardless of load                                                            | — |
| `LEAST_CONNECTIONS` | worker with the lowest current `pending_tasks`                                                                 | classical |
| `LOAD_AWARE`        | worker with the lowest `pending_tasks / max_concurrent_tasks` ratio (capacity-normalised utilisation)          | Reiss et al., SoCC '12 |
| `POWER_OF_TWO`      | sample two workers uniformly at random; pick the less-loaded one. Approaches global least-loaded with O(1) state | Mitzenmacher, IEEE TPDS 2001 |

Failed workers are filtered out of the candidate pool before selection, so a node that has been marked `WorkerStatus.FAILED` stops receiving traffic without any extra coordination from the scheduler.

### 3. Master Scheduler (`master/`)

`MasterScheduler.handle_request(request, worker)` owns the request lifecycle:

1. Calls `RAGRetriever.retrieve_context(request)` to gather supporting context.
2. Picks a candidate-worker list (the explicit worker first, then the rest sorted by `active_tasks`).
3. Calls `worker.process(request, context, inference_engine)` and retries on a different worker if it raises.
4. Returns a `Response` with the request ID, the worker that succeeded, the answer, and the context.
5. Tracks per-worker success/failure counts and overall scheduler stats.

If every retry fails, the scheduler records the failure and returns a `Response` with `status="failed"` rather than propagating the exception.

### 4. GPU Worker Nodes (`workers/`)

`GPUWorkerNode` simulates a GPU-backed inference server with explicit capacity and health.

Public surface:

- `worker_id`, `gpu_name`, `active_tasks`, `max_concurrent_tasks`
- `status` (`HEALTHY`, `DEGRADED`, `FAILED`)
- counters: `completed_tasks`, `failed_tasks`, `total_latency_seconds`, `last_latency`
- `process(request, context, inference_engine)` — the entry point used by the scheduler
- `mark_failed()` / `mark_healthy()` — operator hooks used by fault-tolerance demos
- `snapshot_metrics()` — returns a plain `dict` that the load balancer or a reporting layer can consume safely

Internals:

- a single `threading.Lock` guards every counter mutation, so 1000+ concurrent threads cannot tear the state
- a per-worker `random.Random` (seeded for reproducibility) drives a configurable `failure_rate` so transient failures can be exercised under load
- the `process()` flow is structured so `active_tasks` is decremented on every exit path, including injected failures

### 5. RAG Module (`rag/`)

`RAGRetriever` builds an in-process vector index over `rag/corpus.py` (16 short documents about distributed LLM serving, load balancing, fault tolerance, and concurrency).

- Embeddings come from `sentence-transformers/all-MiniLM-L6-v2` (configurable).
- The index is a `faiss.IndexFlatIP` with normalised vectors, equivalent to cosine similarity.
- The model and index are built lazily on the first `retrieve_context` call to keep import time low.
- A `use_stub=True` constructor flag returns a deterministic snippet from the corpus without loading the model — used by `main.py` so the dry run stays sub-second.

### 6. LLM Inference Module (`llm/`)

`LLMInferenceEngine` delegates to a backend that conforms to the `LLMBackend` protocol.

- `SimulatedLLMBackend` (default) sleeps for a duration that depends on the prompt and context length, plus uniform jitter, and returns a templated answer that mentions a few keywords from the retrieved context. It accepts an optional `failure_rate` so the LLM step can also fail under stress.
- `HuggingFaceLLMBackend` runs `transformers.pipeline("text-generation", model=...)` and is selected by setting `LLM_BACKEND=hf`. Model name is configurable via `LLM_MODEL` (default `distilgpt2`).
- The backend is resolved from the `LLM_BACKEND` environment variable inside `LLMInferenceEngine.__init__`, so existing call sites that do `LLMInferenceEngine()` keep working without code changes.

## Request Lifecycle

```
ClientLoadGenerator
        |
        v
   LoadBalancer.select_worker(request)        <-- round_robin | least_connections | load_aware
        |
        v
MasterScheduler.handle_request(request, worker)
        |
        +---> RAGRetriever.retrieve_context(request)        --> context
        |
        +---> worker.process(request, context, llm)         --> answer
        |              |
        |              +---> SimulatedLLMBackend.generate(prompt, context)
        |                     (or HuggingFaceLLMBackend if LLM_BACKEND=hf)
        |
        v
     Response(request_id, worker_id, answer, context, status)
```

## Repository Structure

```text
.
|-- client/
|   `-- generator.py            # synthetic incoming requests
|-- common/
|   `-- models.py               # Request and Response dataclasses
|-- lb/
|   `-- round_robin.py          # LoadBalancer + LoadBalancingStrategy enum
|-- master/
|   `-- scheduler.py            # MasterScheduler with retry and per-worker stats
|-- workers/
|   `-- gpu_worker.py           # GPUWorkerNode, WorkerStatus, exception types
|-- llm/
|   `-- inference.py            # LLMInferenceEngine + simulated and HF backends
|-- rag/
|   |-- corpus.py               # in-memory document collection (16 docs)
|   `-- retriever.py            # FAISS-backed retriever with stub fast path
|-- scripts/
|   `-- smoke_concurrent.py     # 50-thread concurrent smoke test
|-- main.py                     # deterministic dry-run entrypoint
|-- requirements.txt
`-- README.md
```

## Setup

Use Python 3.9 or newer.

### Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

The required packages are `sentence-transformers`, `faiss-cpu`, and `numpy`. The optional HuggingFace backend additionally needs `transformers` and `torch`; these are documented in `requirements.txt` and are not installed by default.

## Running the Project

### Dry Run

```bash
python main.py
```

This builds three workers, configures the load balancer, sends one synthetic request, and prints a summary. The retriever is constructed with `use_stub=True`, so the dry run does not download the embedding model and finishes in well under a second:

```text
[client] Generating 1 synthetic request(s)
[lb] Routed req-0001 -> gpu-worker-1 using load_aware
[rag] Retrieved stub context for req-0001 (doc_id=doc-007)
[master] Scheduling req-0001 on worker gpu-worker-1 (attempt 1/1)
[worker:gpu-worker-1] Starting req-0001 on NVIDIA-A100-SIM
[llm] Generating response for req-0001
[worker:gpu-worker-1] Finished req-0001 in 0.475s
[master] Completed req-0001

=== Dry Run Summary ===
Request ID : req-0001
...
```

### Concurrent Smoke Test

```bash
python scripts/smoke_concurrent.py --users 50 --strategy least_connections
```

The smoke test fans out N threads, each one routing through the load balancer, the scheduler, the RAG retriever, and a worker. It then prints per-worker metrics and scheduler stats so balance, latency, and recovery behaviour are easy to inspect.

Useful flags:

- `--users N` — number of concurrent client threads (default 50)
- `--strategy {round_robin,least_connections,load_aware}` — load-balancing policy (default `least_connections`)
- `--failure-rate F` — per-worker probability of a transient failure on each request, in `[0.0, 1.0]`
- `--fault-after K` — after K requests have been dispatched, mark `gpu-worker-2` as `FAILED`; this exercises hard-failure detection and reassignment
- `--real-rag` — use the FAISS-backed retriever instead of the stub (downloads the embedding model on first run)

Example output (50 users, fault injection after 15 requests, 5% transient failure rate):

```text
=== Per-worker metrics ===
  gpu-worker-1     status=healthy   completed=  22 failed=   5 avg_latency=0.133s last_latency=0.153s
  gpu-worker-2     status=failed    completed=   5 failed=   0 avg_latency=0.126s last_latency=0.143s
  gpu-worker-3     status=healthy   completed=  23 failed=   1 avg_latency=0.135s last_latency=0.160s

=== Scheduler metrics ===
  total=50 successful=50 failed=0 total_time=6.666s
  worker_successes={'gpu-worker-1': 22, 'gpu-worker-2': 5, 'gpu-worker-3': 23}
  worker_failures={'gpu-worker-3': 1, 'gpu-worker-2': 6, 'gpu-worker-1': 5}

=== Summary ===
  requests=50 completed=50 failed=0
  wall_time=0.164s throughput=304.78 req/s
```

All 50 requests still complete even though one worker was hard-killed mid-run and 5 % of all attempts were transient failures.

### Optional: Real LLM Backend

```bash
pip install transformers torch
LLM_BACKEND=hf python main.py
```

This routes the LLM step through `transformers.pipeline("text-generation", model="distilgpt2")` instead of the simulated backend. Note that this path is for demos with a handful of requests, not the 1000+ load test; the simulated backend is the realistic choice on a laptop.

### Optional: Real RAG Retriever in `main.py`

`main.py` uses `use_stub=True` for speed. To exercise the real FAISS path with a single request, change the line in `main.py` to `RAGRetriever()` or run the smoke test with `--real-rag`. The first call downloads the `all-MiniLM-L6-v2` model (~90 MB).

## Fault-Tolerance Story

The fault-tolerance flow has three independent levers and one final guardrail:

1. **Hard failure (`mark_failed()`)** — operator-style failure. The next `process()` call raises immediately, so no work is wasted on a dead node, and the load balancer drops the worker from the candidate pool.
2. **Transient failure (`failure_rate`)** — probabilistic. Lets the test surface flaky behaviour where a worker sometimes errors but is otherwise healthy.
3. **LLM failure** — the simulated backend can also be configured with a non-zero `failure_rate`, so an isolated inference error can be tested without making the whole worker bad.
4. **Scheduler retry** — `MasterScheduler` catches every failure and replays the request on the next-best worker up to `max_retries + 1` total attempts. Per-worker successes and failures are recorded.

## Performance Notes

- All worker counters are guarded by a single `threading.Lock`. CPython integer reads are atomic, so the load balancer can read `active_tasks` without taking the lock and still get a usable value for least-connections / load-aware routing.
- Latency in the simulated backend is dominated by `time.sleep`, which releases the GIL, so the simulation scales nearly linearly with thread count up to the worker capacity limit.
- The FAISS index is built once per process, behind a lock. Subsequent queries are lock-free; both `SentenceTransformer.encode` and `IndexFlatIP.search` are safe to call concurrently after the index is populated.

## Testing and Evaluation Plan

Useful experiments to run against the current implementation:

| Experiment                              | Command                                                                                                     |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Strategy comparison under uniform load  | `python scripts/smoke_concurrent.py --users 200 --strategy {round_robin,least_connections,load_aware}`      |
| Fault tolerance                         | `python scripts/smoke_concurrent.py --users 200 --fault-after 50`                                           |
| Resilience to flakiness                 | `python scripts/smoke_concurrent.py --users 200 --failure-rate 0.1`                                         |
| Real retrieval correctness              | `python scripts/smoke_concurrent.py --users 20 --real-rag`                                                  |

Suggested metrics to record from the smoke output:

- average and tail latency per worker (`avg_latency_seconds`, `last_latency_seconds`)
- throughput (`requests / wall_time`)
- distribution fairness across workers (`completed_tasks` per worker)
- success rate after fault injection (`scheduler.successful_requests / total_requests`)

## Notes on the Code Skeleton in the Brief

The PDF skeleton suggests calling RAG inside the worker. This implementation places RAG inside the scheduler, which is a small but important divergence: it lets the worker remain a pure compute unit and keeps the request lifecycle visible at one layer. The public signatures of the worker, the LLM engine, and the RAG retriever are otherwise compatible with the brief.
