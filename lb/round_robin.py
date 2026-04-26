from __future__ import annotations

import threading
from enum import Enum
from typing import Sequence

from common import Request
from workers import GPUWorkerNode, WorkerStatus


class LoadBalancingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LOAD_AWARE = "load_aware"


class LoadBalancer:
    """Selects a worker per request using one of three strategies.

    `select_worker()` is atomic: selection AND reservation happen under a
    single lock. The caller MUST call `worker.release()` after the request
    finishes (success or failure) — usually in a `finally` block. Without
    that, `pending_tasks` leaks and load_aware/least_connections degrade.
    """

    def __init__(
        self,
        workers: Sequence[GPUWorkerNode],
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
    ) -> None:
        if not workers:
            raise ValueError("LoadBalancer requires at least one worker.")

        self._workers = list(workers)
        self._strategy = strategy
        self._next_index = 0
        self._lock = threading.Lock()

    @property
    def workers(self) -> tuple[GPUWorkerNode, ...]:
        return tuple(self._workers)

    @property
    def strategy(self) -> LoadBalancingStrategy:
        return self._strategy

    def set_strategy(self, strategy: LoadBalancingStrategy) -> None:
        """Atomically switch the routing strategy at runtime.

        Used by the benchmark harness to compare strategies back-to-back
        without restarting the master process. Resets the round-robin
        cursor so a switch into ROUND_ROBIN starts at worker 0.
        """
        with self._lock:
            self._strategy = strategy
            self._next_index = 0

    def select_worker(self, request: Request) -> GPUWorkerNode:
        with self._lock:
            available = [w for w in self._workers if w.status != WorkerStatus.FAILED]
            if not available:
                raise RuntimeError("No healthy GPU workers available.")

            if self._strategy == LoadBalancingStrategy.ROUND_ROBIN:
                worker = self._round_robin(available)
            elif self._strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                worker = self._least_connections(available)
            elif self._strategy == LoadBalancingStrategy.LOAD_AWARE:
                worker = self._load_aware(available)
            else:
                raise ValueError(f"Unsupported strategy: {self._strategy}")

            worker.reserve()

        print(
            f"[lb] Routed {request.request_id} -> {worker.worker_id} "
            f"using {self._strategy.value} (pending={worker.pending_tasks})"
        )
        return worker

    def _round_robin(self, workers: Sequence[GPUWorkerNode]) -> GPUWorkerNode:
        worker = workers[self._next_index % len(workers)]
        self._next_index += 1
        return worker

    def _least_connections(self, workers: Sequence[GPUWorkerNode]) -> GPUWorkerNode:
        return min(workers, key=lambda w: w.pending_tasks)

    def _load_aware(self, workers: Sequence[GPUWorkerNode]) -> GPUWorkerNode:
        return min(
            workers,
            key=lambda w: (
                w.pending_tasks / w.max_concurrent_tasks
                if w.max_concurrent_tasks > 0 else float("inf")
            ),
        )
