from _future_ import annotations

from enum import Enum
from typing import Sequence

from common import Request
from workers import GPUWorkerNode, WorkerStatus


class LoadBalancingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LOAD_AWARE = "load_aware"


class LoadBalancer:
    def _init_(
        self,
        workers: Sequence[GPUWorkerNode],
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
    ) -> None:
        if not workers:
            raise ValueError("LoadBalancer requires at least one worker.")

        self._workers = list(workers)
        self._strategy = strategy
        self._next_index = 0

    def select_worker(self, request: Request) -> GPUWorkerNode:
        available_workers = self._get_available_workers()

        if not available_workers:
            raise RuntimeError("No healthy GPU workers available.")

        if self._strategy == LoadBalancingStrategy.ROUND_ROBIN:
            worker = self._round_robin(available_workers)

        elif self._strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            worker = self._least_connections(available_workers)

        elif self._strategy == LoadBalancingStrategy.LOAD_AWARE:
            worker = self._load_aware(available_workers)

        else:
            raise ValueError(f"Unsupported strategy: {self._strategy}")

        print(
            f"[lb] Routed {request.request_id} to {worker.worker_id} "
            f"using {self._strategy.value} scheduling"
        )

        return worker

    def _get_available_workers(self) -> list[GPUWorkerNode]:
        return [
            worker
            for worker in self._workers
            if worker.status != WorkerStatus.FAILED
        ]

    def _round_robin(self, workers: Sequence[GPUWorkerNode]) -> GPUWorkerNode:
        worker = workers[self._next_index % len(workers)]
        self._next_index = (self._next_index + 1) % len(workers)
        return worker

    def _least_connections(self, workers: Sequence[GPUWorkerNode]) -> GPUWorkerNode:
        return min(workers, key=lambda worker: worker.active_tasks)

    def _load_aware(self, workers: Sequence[GPUWorkerNode]) -> GPUWorkerNode:
        return min(
            workers,
            key=lambda worker: worker.active_tasks / worker.max_concurrent_tasks,
        )
