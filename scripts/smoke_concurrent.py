from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

# Ensure project root is on PYTHONPATH when this script is invoked directly
# (e.g. `python scripts/smoke_concurrent.py`) instead of as a module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client import ClientLoadGenerator  # noqa: E402
from lb import LoadBalancer, LoadBalancingStrategy  # noqa: E402
from llm import LLMInferenceEngine, SimulatedLLMBackend  # noqa: E402
from master import MasterScheduler  # noqa: E402
from rag import RAGRetriever  # noqa: E402
from workers import GPUWorkerNode  # noqa: E402


def build_workers(failure_rate: float) -> list[GPUWorkerNode]:
    return [
        GPUWorkerNode(
            worker_id="gpu-worker-1",
            gpu_name="NVIDIA-A100-SIM",
            max_concurrent_tasks=400,
            failure_rate=failure_rate,
            rng_seed=1,
        ),
        GPUWorkerNode(
            worker_id="gpu-worker-2",
            gpu_name="NVIDIA-A100-SIM",
            max_concurrent_tasks=400,
            failure_rate=failure_rate,
            rng_seed=2,
        ),
        GPUWorkerNode(
            worker_id="gpu-worker-3",
            gpu_name="NVIDIA-A100-SIM",
            max_concurrent_tasks=400,
            failure_rate=failure_rate,
            rng_seed=3,
        ),
    ]


def run(
    num_users: int,
    failure_rate: float,
    fault_after: int | None,
    use_stub_rag: bool,
    strategy: LoadBalancingStrategy,
    real_llm: bool,
) -> int:
    workers = build_workers(failure_rate=failure_rate)
    generator = ClientLoadGenerator()
    load_balancer = LoadBalancer(workers, strategy=strategy)
    retriever = RAGRetriever(use_stub=use_stub_rag)
    if real_llm:
        # Defers to LLM_BACKEND env var (set to "hf" by --real-llm).
        # Loads the HuggingFace model once on first call.
        inference_engine = LLMInferenceEngine()
    else:
        inference_engine = LLMInferenceEngine(
            backend=SimulatedLLMBackend(
                base_latency_s=0.05,
                per_token_latency_s=0.001,
                jitter_s=0.02,
                rng_seed=42,
            )
        )
    scheduler = MasterScheduler(retriever, inference_engine, workers=workers, max_retries=2)

    requests = generator.generate_requests(count=num_users)

    results: list[str] = []
    results_lock = threading.Lock()

    def send(request_index: int) -> None:
        request = requests[request_index]
        selected = load_balancer.select_worker(request)
        response = scheduler.handle_request(request, selected)
        with results_lock:
            results.append(response.status)

    fault_triggered = threading.Event()

    def maybe_trigger_fault(index: int) -> None:
        if fault_after is None or fault_triggered.is_set():
            return
        if index >= fault_after:
            fault_triggered.set()
            print("\n*** Injecting fault: marking gpu-worker-2 as FAILED ***\n")
            workers[1].mark_failed()

    start = time.perf_counter()
    threads: list[threading.Thread] = []
    for i in range(num_users):
        maybe_trigger_fault(i)
        t = threading.Thread(target=send, args=(i,), name=f"client-{i:04d}")
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    print("\n=== Per-worker metrics ===")
    for worker in workers:
        snapshot = worker.snapshot_metrics()
        print(
            f"  {snapshot['worker_id']:16s} "
            f"status={snapshot['status']:<9s} "
            f"completed={snapshot['completed_tasks']:4d} "
            f"failed={snapshot['failed_tasks']:4d} "
            f"avg_latency={snapshot['avg_latency_seconds']:.3f}s "
            f"last_latency={snapshot['last_latency_seconds']:.3f}s"
        )

    print("\n=== Scheduler metrics ===")
    stats = scheduler.stats
    print(
        f"  total={stats.total_requests} "
        f"successful={stats.successful_requests} "
        f"failed={stats.failed_requests} "
        f"total_time={stats.total_processing_time_seconds:.3f}s"
    )
    print(f"  worker_successes={scheduler.worker_successes}")
    print(f"  worker_failures={scheduler.worker_failures}")

    print("\n=== Summary ===")
    completed = sum(1 for r in results if r == "completed")
    failed = sum(1 for r in results if r != "completed")
    throughput = num_users / elapsed if elapsed > 0 else float("inf")
    print(f"  requests={num_users} completed={completed} failed={failed}")
    print(f"  wall_time={elapsed:.3f}s throughput={throughput:.2f} req/s")

    return 0 if completed == num_users else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent smoke test for the AI pipeline.")
    parser.add_argument("--users", type=int, default=50, help="Number of concurrent requests.")
    parser.add_argument(
        "--failure-rate",
        type=float,
        default=0.0,
        help="Per-worker probability of a transient failure in [0.0, 1.0].",
    )
    parser.add_argument(
        "--fault-after",
        type=int,
        default=None,
        help="Mark gpu-worker-2 as FAILED after dispatching this many requests.",
    )
    parser.add_argument(
        "--real-rag",
        action="store_true",
        help=(
            "Use the FAISS-backed RAG retriever. "
            "Triggers a one-time sentence-transformers model download."
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in LoadBalancingStrategy],
        default=LoadBalancingStrategy.LEAST_CONNECTIONS.value,
        help="Load-balancing strategy used by the LoadBalancer.",
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help=(
            "Use the HuggingFace LLM backend (sets LLM_BACKEND=hf). "
            "Requires `pip install transformers torch`. Much slower than the simulated backend."
        ),
    )
    args = parser.parse_args()

    if args.real_llm:
        os.environ["LLM_BACKEND"] = "hf"
    else:
        os.environ.setdefault("LLM_BACKEND", "sim")

    return run(
        num_users=args.users,
        failure_rate=args.failure_rate,
        fault_after=args.fault_after,
        use_stub_rag=not args.real_rag,
        strategy=LoadBalancingStrategy(args.strategy),
        real_llm=args.real_llm,
    )


if __name__ == "__main__":
    raise SystemExit(main())
