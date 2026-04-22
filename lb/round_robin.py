from __future__ import annotations

from typing import Sequence

from common import Request
from workers import GPUWorkerNode


class RoundRobinLoadBalancer:
    def __init__(self, workers: Sequence[GPUWorkerNode]) -> None:
        if not workers:
            raise ValueError("RoundRobinLoadBalancer requires at least one worker.")

        self._workers = list(workers)
        self._next_index = 0

    def select_worker(self, request: Request) -> GPUWorkerNode:
        worker = self._workers[self._next_index]
        print(
            f"[lb] Routed {request.request_id} to {worker.worker_id} "
            "using round-robin scheduling"
        )
        self._next_index = (self._next_index + 1) % len(self._workers)
        return worker
