# client/config.py
"""
All client-side configuration lives here.
Teammates or the grader can change load parameters without touching logic files.
"""

from __future__ import annotations

# ── Load test volumes ─────────────────────────────────────────────────────────
DEFAULT_NUM_USERS: int = 1000
RAMP_UP_STEPS: list[int] = [100, 250, 500, 750, 1000]  # gradual ramp levels
RAMP_STEP_PAUSE_SEC: float = 2.0  # wait between ramp steps

# ── Concurrency ───────────────────────────────────────────────────────────────
MAX_THREAD_WORKERS: int = 500  # ThreadPoolExecutor cap (OS thread limit)

# ── Per-request behaviour ─────────────────────────────────────────────────────
REQUEST_TIMEOUT_SEC: float = 30.0  # seconds before a request is marked timeout
MAX_RETRIES: int = 2  # retries on transient failures
RETRY_BACKOFF_SEC: float = 0.3  # base backoff (doubles each attempt)

# ── Live reporting ────────────────────────────────────────────────────────────
REPORT_INTERVAL_SEC: float = 5.0  # how often the progress bar updates

# ── Results output ────────────────────────────────────────────────────────────
SAVE_RESULTS: bool = True
RESULTS_FILE: str = "client/results/load_test_results.json"

# ── Sample prompts (realistic LLM queries) ────────────────────────────────────
SAMPLE_PROMPTS: list[str] = [
    "What is distributed computing and why does it matter?",
    "Explain the CAP theorem with a practical example.",
    "How does load balancing improve system scalability?",
    "What is a GPU cluster and how is it used for ML workloads?",
    "Describe the difference between latency and throughput.",
    "What is Retrieval-Augmented Generation (RAG)?",
    "How do you handle fault tolerance in distributed systems?",
    "Explain consistent hashing and its use in distributed caches.",
    "What is the role of a master node in a GPU cluster?",
    "How does round-robin scheduling work for load balancing?",
    "What are the advantages of parallel processing over sequential?",
    "Describe how a vector database enables semantic search.",
    "What is the difference between synchronous and async request handling?",
    "Compare gRPC and REST for high-throughput microservices.",
    "How does Kubernetes manage container orchestration at scale?",
    "What is Apache Kafka and when should you use it?",
    "Explain the MapReduce programming model with an example.",
    "What causes GPU memory bottlenecks during LLM inference?",
    "How do you monitor SLAs in a distributed AI system?",
    "What is the Byzantine Generals problem in distributed systems?",
    "Explain tensor parallelism for large language models.",
    "What is KV-cache and how does it speed up autoregressive decoding?",
    "How does vLLM achieve high throughput for LLM serving?",
    "What trade-offs exist between batch size and latency in ML serving?",
    "Describe the architecture of a production RAG pipeline.",
]
