from __future__ import annotations

import random
import threading
from enum import Enum
from time import perf_counter

from common import Request
from llm import LLMInferenceEngine


class WorkerStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


class WorkerUnavailableError(RuntimeError):
    pass


class WorkerTransientError(RuntimeError):
    pass


class GPUWorkerNode:
    def __init__(
        self,
        worker_id: str,
        gpu_name: str,
        max_concurrent_tasks: int = 8,
        failure_rate: float = 0.0,
        rng_seed: int | None = None,
    ) -> None:
        if max_concurrent_tasks < 1:
            raise ValueError("max_concurrent_tasks must be at least 1")
        if not 0.0 <= failure_rate <= 1.0:
            raise ValueError("failure_rate must be in [0.0, 1.0]")

        self.worker_id = worker_id
        self.gpu_name = gpu_name
        self.max_concurrent_tasks = max_concurrent_tasks
        self.failure_rate = failure_rate

        # Counters (read unsynchronised by the scheduler for least-busy routing).
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_latency_seconds = 0.0
        self.last_latency = 0.0
        self.status: WorkerStatus = WorkerStatus.HEALTHY

        self._lock = threading.Lock()
        self._rng = random.Random(rng_seed)

    def mark_failed(self) -> None:
        with self._lock:
            self.status = WorkerStatus.FAILED
        print(f"[worker:{self.worker_id}] Marked FAILED")

    def mark_healthy(self) -> None:
        with self._lock:
            self.status = WorkerStatus.HEALTHY
        print(f"[worker:{self.worker_id}] Marked HEALTHY")

    def snapshot_metrics(self) -> dict[str, object]:
        with self._lock:
            total = self.completed_tasks + self.failed_tasks
            avg_latency = (
                self.total_latency_seconds / self.completed_tasks
                if self.completed_tasks
                else 0.0
            )
            return {
                "worker_id": self.worker_id,
                "gpu_name": self.gpu_name,
                "status": self.status.value,
                "active_tasks": self.active_tasks,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                "total_tasks": total,
                "avg_latency_seconds": avg_latency,
                "last_latency_seconds": self.last_latency,
            }

    def process(
        self,
        request: Request,
        context: str,
        inference_engine: LLMInferenceEngine,
    ) -> str:
        # Reject immediately if the operator has marked this node down.
        # The scheduler's retry loop will reassign to another worker.
        if self.status == WorkerStatus.FAILED:
            raise WorkerUnavailableError(
                f"worker {self.worker_id} is marked FAILED"
            )

        with self._lock:
            self.active_tasks += 1
            if (
                self.status == WorkerStatus.HEALTHY
                and self.active_tasks > self.max_concurrent_tasks
            ):
                self.status = WorkerStatus.DEGRADED

        print(
            f"[worker:{self.worker_id}] Starting {request.request_id} on {self.gpu_name}"
        )
        start = perf_counter()

        try:
            # Failure-rate roll lives INSIDE the try so the finally block
            # still decrements active_tasks on an injected failure.
            if self._rng.random() < self.failure_rate:
                raise WorkerTransientError(
                    f"injected transient failure on {self.worker_id}"
                )

            answer = inference_engine.generate(request, context)
            latency = perf_counter() - start

            with self._lock:
                self.completed_tasks += 1
                self.last_latency = latency
                self.total_latency_seconds += latency

            print(
                f"[worker:{self.worker_id}] Finished {request.request_id} "
                f"in {latency:.3f}s"
            )
            return answer
        except Exception:
            with self._lock:
                self.failed_tasks += 1
            print(f"[worker:{self.worker_id}] Failed {request.request_id}")
            raise
        finally:
            with self._lock:
                self.active_tasks -= 1
                if (
                    self.status == WorkerStatus.DEGRADED
                    and self.active_tasks <= self.max_concurrent_tasks
                ):
                    self.status = WorkerStatus.HEALTHY
