from __future__ import annotations

import inspect
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from common import Request
from workers import GPUWorkerNode, WorkerStatus


@dataclass(slots=True)
class LoadBalancerMetrics:
    """Metrics tracked by the load balancer."""

    total_requests_routed: int = 0
    requests_per_worker: dict[str, int] = field(default_factory=dict)
    successful_routes: int = 0
    failed_routes: int = 0
    skipped_failed_workers: int = 0
    avg_selection_time_ms: float = 0.0
    total_selection_time_ms: float = 0.0


class RoundRobinLoadBalancer:
    """
    Health-aware round-robin load balancer with resilience and logging.

    Features:
    - Distributes requests across healthy workers in round-robin order
    - Skips workers marked as FAILED to ensure resilience
    - Tracks detailed metrics for observability
    - Supports both the project API and the simpler design-snippet API
    """

    def __init__(self, workers: Sequence[GPUWorkerNode]) -> None:
        if not workers:
            raise ValueError("RoundRobinLoadBalancer requires at least one worker.")

        self._workers = list(workers)
        self._next_index = 0
        self._lock = threading.Lock()
        self._metrics = LoadBalancerMetrics()
        self._start_time = time.time()

        for worker in self._workers:
            self._metrics.requests_per_worker[worker.worker_id] = 0

        self._log(
            f"Initialized Round Robin Load Balancer with {len(self._workers)} workers"
        )
        self._log_worker_info()

    def select_worker(self, request: Request) -> GPUWorkerNode:
        """Select the next healthy worker in round-robin order."""
        start_time = time.perf_counter_ns()

        with self._lock:
            selected_worker = self._select_next_healthy_worker()
            self._record_route(selected_worker)

            selection_time_ms = (time.perf_counter_ns() - start_time) / 1_000_000
            self._update_avg_selection_time(selection_time_ms)

        worker_status = self._get_worker_status_string(selected_worker)
        self._log(
            f"[{request.request_id}] Routed to {selected_worker.worker_id} "
            f"({selected_worker.gpu_name}) | Status: {worker_status} | "
            f"Active: {selected_worker.active_tasks}/"
            f"{selected_worker.max_concurrent_tasks}"
        )
        return selected_worker

    def get_next_worker(self, request: Request | None = None) -> GPUWorkerNode:
        """
        Compatibility helper for the simpler load balancer API from the slide.

        When a request is supplied, metrics are recorded exactly as they are for
        select_worker(). When it is omitted, selection still stays health-aware
        and round-robin, but request-scoped metrics/logging are skipped.
        """
        with self._lock:
            worker = self._select_next_healthy_worker()
            if request is not None:
                self._record_route(worker)
            return worker

    def dispatch(
        self,
        request: Request,
        *process_args: Any,
        **process_kwargs: Any,
    ) -> Any:
        """
        Dispatch a request using round-robin selection.

        Supports both:
        - worker.process(request)
        - worker.process(request, context, inference_engine)
        """
        last_error: Exception | None = None

        for _ in range(len(self._workers)):
            worker = self.get_next_worker(request=request)

            try:
                return self._invoke_worker(
                    worker, request, *process_args, **process_kwargs
                )
            except Exception as exc:
                last_error = exc
                if worker.status == WorkerStatus.FAILED:
                    self._metrics.skipped_failed_workers += 1
                    self._log(
                        f"Worker {worker.worker_id} became unavailable during "
                        "dispatch; trying the next worker"
                    )
                    continue
                raise

        self._metrics.failed_routes += 1
        if last_error is not None:
            raise RuntimeError(
                "Load balancer resilience failure: no available worker could "
                "complete the dispatched request."
            ) from last_error
        raise RuntimeError("Load balancer has no workers available for dispatch.")

    def _record_route(self, worker: GPUWorkerNode) -> None:
        self._metrics.total_requests_routed += 1
        self._metrics.requests_per_worker[worker.worker_id] += 1
        self._metrics.successful_routes += 1

    def _select_next_healthy_worker(self) -> GPUWorkerNode:
        """Select the next worker in round-robin order, skipping failed ones."""
        num_workers = len(self._workers)
        attempts = 0
        max_attempts = num_workers * 2

        while attempts < max_attempts:
            worker = self._workers[self._next_index]
            self._next_index = (self._next_index + 1) % num_workers

            if worker.status == WorkerStatus.FAILED:
                self._metrics.skipped_failed_workers += 1
                self._log(
                    f"! Skipping failed worker {worker.worker_id} - "
                    "searching for healthy alternative"
                )
                attempts += 1
                continue

            return worker

        self._metrics.failed_routes += 1
        raise RuntimeError(
            "Load balancer resilience failure: all workers are marked FAILED. "
            "No healthy workers available for request distribution."
        )

    @staticmethod
    def _invoke_worker(
        worker: GPUWorkerNode,
        request: Request,
        *process_args: Any,
        **process_kwargs: Any,
    ) -> Any:
        """
        Call the worker using either the project's full processing signature or
        the simpler one-argument variant used in the design snippet.
        """
        process_signature = inspect.signature(worker.process)
        accepts_varargs = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in process_signature.parameters.values()
        )

        if process_args or process_kwargs or accepts_varargs:
            return worker.process(request, *process_args, **process_kwargs)

        positional_parameters = [
            parameter
            for parameter in process_signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        if len(positional_parameters) <= 1:
            return worker.process(request)

        raise TypeError(
            "worker.process requires additional arguments. "
            "Provide them to dispatch(request, ...)."
        )

    def get_metrics(self) -> LoadBalancerMetrics:
        """Return a snapshot of current load balancer metrics."""
        with self._lock:
            return LoadBalancerMetrics(
                total_requests_routed=self._metrics.total_requests_routed,
                requests_per_worker=dict(self._metrics.requests_per_worker),
                successful_routes=self._metrics.successful_routes,
                failed_routes=self._metrics.failed_routes,
                skipped_failed_workers=self._metrics.skipped_failed_workers,
                avg_selection_time_ms=self._metrics.avg_selection_time_ms,
                total_selection_time_ms=self._metrics.total_selection_time_ms,
            )

    def report_metrics(self) -> None:
        """Print a detailed metrics report to the console."""
        metrics = self.get_metrics()
        uptime = time.time() - self._start_time

        print("\n" + "=" * 80)
        print("LOAD BALANCER METRICS REPORT")
        print("=" * 80)
        print(f"Uptime (seconds):              {uptime:.2f}")
        print(f"Total Requests Routed:         {metrics.total_requests_routed}")
        print(f"Successful Routes:             {metrics.successful_routes}")
        print(f"Failed Routes:                 {metrics.failed_routes}")
        print(f"Skipped Failed Workers:        {metrics.skipped_failed_workers}")
        print(f"Avg Selection Time (ms):       {metrics.avg_selection_time_ms:.3f}")
        print()
        print("Requests per Worker:")
        for worker_id in sorted(self._metrics.requests_per_worker.keys()):
            count = metrics.requests_per_worker[worker_id]
            percentage = (
                count / metrics.total_requests_routed * 100
                if metrics.total_requests_routed > 0
                else 0
            )
            print(f"  {worker_id:20} {count:6} requests ({percentage:5.1f}%)")
        print("=" * 80 + "\n")

    def health_summary(self) -> None:
        """Print current health status of all workers."""
        print("\n" + "-" * 80)
        print("WORKER HEALTH STATUS")
        print("-" * 80)
        for worker in self._workers:
            status = worker.snapshot_metrics()
            health_icon = self._get_health_icon(worker.status)
            print(
                f"{health_icon} {status['worker_id']:20} | "
                f"Status: {status['status']:10} | "
                f"Active: {status['active_tasks']:2}/{status['max_concurrent_tasks']:2} | "
                f"Completed: {status['completed_tasks']:4} | "
                f"Failed: {status['failed_tasks']:4} | "
                f"Avg Latency: {status['avg_latency_seconds']:.3f}s"
            )
        print("-" * 80 + "\n")

    def _get_worker_status_string(self, worker: GPUWorkerNode) -> str:
        """Return a string representation of worker status."""
        icon = self._get_health_icon(worker.status)
        return f"{icon} {worker.status.value}"

    @staticmethod
    def _get_health_icon(status: WorkerStatus) -> str:
        """Return a visual indicator for worker health status."""
        if status == WorkerStatus.HEALTHY:
            return "OK"
        if status == WorkerStatus.DEGRADED:
            return "WARN"
        return "DOWN"

    def _update_avg_selection_time(self, new_time_ms: float) -> None:
        """Update the rolling average selection time."""
        total_time = self._metrics.total_selection_time_ms + new_time_ms
        total_count = self._metrics.total_requests_routed
        self._metrics.avg_selection_time_ms = (
            total_time / total_count if total_count > 0 else 0.0
        )
        self._metrics.total_selection_time_ms = total_time

    def _log_worker_info(self) -> None:
        """Log initial information about all workers."""
        print(f"[lb] Available workers ({len(self._workers)} total):")
        for index, worker in enumerate(self._workers, start=1):
            print(
                f"[lb]   {index}. {worker.worker_id:20} -> {worker.gpu_name:20} "
                f"(max concurrent: {worker.max_concurrent_tasks})"
            )

    @staticmethod
    def _log(message: str) -> None:
        """Log a message with timestamp from the load balancer."""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{timestamp}] [lb] {message}")


class LoadBalancer(RoundRobinLoadBalancer):
    """Compatibility alias for the simpler round-robin load balancer name."""
