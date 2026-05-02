from __future__ import annotations

import random
import threading
from enum import Enum
from typing import Sequence

from common import Request
from workers import GPUWorkerNode, WorkerStatus


class LoadBalancingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LOAD_AWARE = "load_aware"
    POWER_OF_TWO = "power_of_two"


class LoadBalancer:
    """Selects a worker per request using one of four strategies.

    `select_worker()` is atomic: selection AND reservation happen under a
    single lock. The caller MUST call `worker.release()` after the request
    finishes (success or failure) — usually in a `finally` block. Without
    that, `pending_tasks` leaks and load_aware/least_connections degrade.

    Why atomic select-and-reserve? Without the lock, N concurrent threads
    that all see `pending_tasks==0` for the same worker pick that worker,
    creating a thundering herd that defeats least_connections / load_aware
    entirely. The fix mirrors how Linux's `epoll` exclusive mode and
    Cloudflare's lock-then-decrement counter pattern handle the same race.

    Why pending_tasks instead of active_tasks? `active_tasks` only
    increments inside `worker.process()`, which is *after* selection
    finishes — so concurrent selectors all see the same stale value.
    `pending_tasks` is incremented during selection, so the next selector
    sees the just-reserved slot. Equivalent to a token-bucket reservation
    in queueing-theory terms.
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
        self._rng = random.Random()

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
            elif self._strategy == LoadBalancingStrategy.POWER_OF_TWO:
                worker = self._power_of_two(available)
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
        # Classical join-the-shortest-queue policy, optimal under M/M/c
        # assumptions but sensitive to heterogeneous worker capacity (a 1-slot
        # worker and a 16-slot worker look the same when both are at zero).
        return min(workers, key=lambda w: w.pending_tasks)

    def _load_aware(self, workers: Sequence[GPUWorkerNode]) -> GPUWorkerNode:
        # Capacity-normalised utilisation. Reiss et al., "Heterogeneity and
        # Dynamicity of Clouds at Scale: Google Trace Analysis" (SoCC '12)
        # documents how production clusters always have heterogeneous nodes;
        # routing on raw queue length systematically underloads big workers.
        return min(
            workers,
            key=lambda w: (
                w.pending_tasks / w.max_concurrent_tasks
                if w.max_concurrent_tasks > 0
                else float("inf")
            ),
        )

    def _power_of_two(self, workers: Sequence[GPUWorkerNode]) -> GPUWorkerNode:
        """Pick two random workers, send to the less-loaded one.

        Mitzenmacher (2001), "The Power of Two Choices in Randomized Load
        Balancing" (IEEE TPDS): selecting the lesser-loaded of two random
        candidates yields exponentially better worst-case load than random,
        and approaches the global least-loaded with O(1) state per request --
        no shared sorted structure to contend on. Falls back to the single
        candidate when only one healthy worker remains.
        """
        if len(workers) == 1:
            return workers[0]
        a, b = self._rng.sample(list(workers), 2)
        if a.max_concurrent_tasks == 0 or b.max_concurrent_tasks == 0:
            return a if a.pending_tasks <= b.pending_tasks else b
        a_ratio = a.pending_tasks / a.max_concurrent_tasks
        b_ratio = b.pending_tasks / b.max_concurrent_tasks
        return a if a_ratio <= b_ratio else b
