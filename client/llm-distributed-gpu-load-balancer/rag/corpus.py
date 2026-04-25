from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    text: str


DEFAULT_CORPUS: tuple[Document, ...] = (
    Document(
        doc_id="doc-001",
        title="Round-robin load balancing",
        text=(
            "Round-robin load balancing distributes each incoming request to the next "
            "worker in a fixed rotation. It is simple, stateless, and fair when workers "
            "have similar capacity. It can be unfair when workers differ in speed or "
            "when request sizes vary, because slow workers accumulate queue depth."
        ),
    ),
    Document(
        doc_id="doc-002",
        title="Least-connections scheduling",
        text=(
            "Least-connections routing picks the worker with the fewest in-flight "
            "requests. It is well suited to heterogeneous workloads where request "
            "latency varies, because it naturally steers traffic away from workers "
            "that are already saturated."
        ),
    ),
    Document(
        doc_id="doc-003",
        title="Load-aware routing",
        text=(
            "Load-aware routing generalises least-connections by combining active "
            "task count, recent latency, GPU utilisation, and queue depth into a "
            "composite score. It improves tail latency under mixed traffic but "
            "requires the load balancer to observe worker-side metrics."
        ),
    ),
    Document(
        doc_id="doc-004",
        title="GPU cluster task distribution",
        text=(
            "A GPU cluster assigns inference tasks to nodes with free accelerator "
            "capacity. Good distribution minimises idle time on expensive GPUs, "
            "keeps memory fragmentation low, and supports parallel execution by "
            "pinning batches to specific devices."
        ),
    ),
    Document(
        doc_id="doc-005",
        title="LLM inference concurrency",
        text=(
            "High-concurrency LLM serving relies on continuous batching, paged "
            "attention, and asynchronous dispatch. Throughput scales with how many "
            "decode steps a single GPU can interleave across independent requests "
            "without stalling on memory transfers."
        ),
    ),
    Document(
        doc_id="doc-006",
        title="Retrieval-augmented generation",
        text=(
            "Retrieval-augmented generation (RAG) fetches relevant documents from "
            "a knowledge base and concatenates them with the user prompt before "
            "calling the language model. This grounds answers in external data, "
            "reduces hallucinations, and lets the system update without retraining."
        ),
    ),
    Document(
        doc_id="doc-007",
        title="Vector databases and embeddings",
        text=(
            "Vector databases index dense embeddings produced by a sentence encoder. "
            "Similarity search uses cosine distance or inner product to find the "
            "documents that are semantically closest to the query. FAISS is a common "
            "in-process backend for small to medium corpora."
        ),
    ),
    Document(
        doc_id="doc-008",
        title="Fault tolerance in distributed services",
        text=(
            "Fault tolerance keeps a distributed service responsive when nodes fail. "
            "Common techniques include health checks, heartbeats, automatic retry, "
            "task reassignment, and partial-failure degradation modes that keep "
            "healthy nodes serving traffic while failed ones recover."
        ),
    ),
    Document(
        doc_id="doc-009",
        title="Worker node failure detection",
        text=(
            "Worker failure detection uses periodic heartbeats, synthetic probes, "
            "or in-band error signalling to mark a node unavailable. Once a node is "
            "marked failed, the scheduler stops sending it work and reassigns any "
            "outstanding tasks to healthy peers."
        ),
    ),
    Document(
        doc_id="doc-010",
        title="Task reassignment and retries",
        text=(
            "Reassignment replays a failed task on a different worker so that no "
            "request is lost. Idempotent handlers, bounded retry counts, and "
            "exponential backoff prevent retry storms from overloading the cluster "
            "when many workers fail at once."
        ),
    ),
    Document(
        doc_id="doc-011",
        title="Horizontal scaling of inference",
        text=(
            "Horizontal scaling adds more GPU workers to serve additional traffic. "
            "It depends on a stateless request path, a scalable router, and a "
            "knowledge base that can be read concurrently. Each worker should be "
            "interchangeable so the load balancer can freely reshape traffic."
        ),
    ),
    Document(
        doc_id="doc-012",
        title="Latency and throughput metrics",
        text=(
            "Latency measures how long a single request takes end to end. Throughput "
            "measures how many requests complete per second. Reporting p50, p95, and "
            "p99 latency alongside throughput gives a fairer picture than averages "
            "because LLM traffic is often heavy tailed."
        ),
    ),
    Document(
        doc_id="doc-013",
        title="GPU utilisation monitoring",
        text=(
            "GPU utilisation tracks how much of each accelerator's time is spent "
            "doing useful compute. Low utilisation with high queue depth usually "
            "signals a bottleneck elsewhere in the pipeline, such as tokenisation, "
            "retrieval, or network transfer."
        ),
    ),
    Document(
        doc_id="doc-014",
        title="Concurrency with threads",
        text=(
            "Python threads share memory and are a good fit for I/O-bound or "
            "blocking workloads such as simulated GPU calls. Shared counters must "
            "be protected by a lock to avoid torn reads, even though individual "
            "integer reads are atomic in CPython."
        ),
    ),
    Document(
        doc_id="doc-015",
        title="Simulating GPU inference workloads",
        text=(
            "A simulation models GPU inference without an accelerator by sleeping "
            "for a duration that depends on prompt length, context size, and a "
            "little random jitter. This lets a distributed system be exercised end "
            "to end on a laptop before being connected to real hardware."
        ),
    ),
    Document(
        doc_id="doc-016",
        title="Health states of a worker",
        text=(
            "A worker can report healthy, degraded, or failed. Healthy workers "
            "accept new work. Degraded workers still accept work but are over "
            "capacity, so the scheduler should prefer other nodes when possible. "
            "Failed workers reject new work until an operator restores them."
        ),
    ),
)
