"""RemoteWorkerProxy — a duck-typed GPUWorkerNode backed by HTTP.

Lives in the master process. Exposes the same surface as `GPUWorkerNode`
(worker_id, status, pending_tasks, reserve/release, process, snapshot_metrics)
so `LoadBalancer` and `MasterScheduler` can use it without changes. The proxy
tracks `pending_tasks` locally so LB selection stays a single in-process
decision (no extra round-trip per request).
"""
from __future__ import annotations

import threading
from time import perf_counter

import httpx

from common import Request
from common.wire import ProcessRequest, ProcessResponse, RequestPayload, WorkerHealth

from .gpu_worker import (
    WorkerAtCapacityError,
    WorkerStatus,
    WorkerTransientError,
    WorkerUnavailableError,
)


class RemoteWorkerProxy:
    DEFAULT_FAILURE_THRESHOLD = 3

    def __init__(
        self,
        worker_id: str,
        url: str,
        *,
        max_concurrent_tasks: int = 8,
        gpu_name: str = "remote",
        timeout_seconds: float = 30.0,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
    ) -> None:
        self.worker_id = worker_id
        self.gpu_name = gpu_name
        self.url = url.rstrip("/")
        self.max_concurrent_tasks = max_concurrent_tasks

        self.active_tasks = 0
        self.pending_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_latency_seconds = 0.0
        self.last_latency = 0.0
        self.status: WorkerStatus = WorkerStatus.HEALTHY

        # Circuit-breaker state. Flip to FAILED after N consecutive HTTP errors
        # so a single network blip doesn't permanently sink a worker. Reset on
        # any successful call. Week 2's active monitor will revive FAILED workers.
        self._failure_threshold = max(1, failure_threshold)
        self._consecutive_failures = 0

        self._lock = threading.Lock()
        # Connection pool keep-alive amortises TLS / TCP setup across requests.
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=5.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )

    def reserve(self) -> None:
        with self._lock:
            self.pending_tasks += 1

    def release(self) -> None:
        with self._lock:
            self.pending_tasks = max(0, self.pending_tasks - 1)

    def mark_failed(self) -> None:
        with self._lock:
            self.status = WorkerStatus.FAILED
        print(f"[remote:{self.worker_id}] Marked FAILED")

    def mark_healthy(self) -> None:
        with self._lock:
            self.status = WorkerStatus.HEALTHY
        print(f"[remote:{self.worker_id}] Marked HEALTHY")

    def probe_health(self) -> WorkerHealth | None:
        """Best-effort GET /health. Returns None on failure."""
        try:
            r = self._client.get(f"{self.url}/health", timeout=2.0)
            r.raise_for_status()
            return WorkerHealth.model_validate(r.json())
        except httpx.HTTPError:
            return None

    def post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        """Generic JSON POST that updates failure-counter state.

        Used by the LB tier when forwarding to a master (which returns a full
        ResponsePayload, not the worker /process body). Exposes circuit-breaker
        bookkeeping without callers reaching into private fields.
        """
        path = path if path.startswith("/") else f"/{path}"
        try:
            r = self._client.post(f"{self.url}{path}", json=payload)
            r.raise_for_status()
            with self._lock:
                self.completed_tasks += 1
                self._consecutive_failures = 0
            return r.json()
        except httpx.HTTPError as exc:
            with self._lock:
                self.failed_tasks += 1
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._failure_threshold:
                    self.status = WorkerStatus.FAILED
            raise WorkerTransientError(
                f"remote {self.worker_id} HTTP failure on {path}: {exc}"
            ) from exc

    def process(self, request: Request, context: str, inference_engine=None) -> str:
        # `inference_engine` arg kept for signature parity with GPUWorkerNode;
        # remote workers own their own engine on the server side.
        del inference_engine

        if self.status == WorkerStatus.FAILED:
            raise WorkerUnavailableError(
                f"remote worker {self.worker_id} is marked FAILED"
            )

        with self._lock:
            self.active_tasks += 1

        start = perf_counter()
        try:
            payload = ProcessRequest(
                request=RequestPayload.from_dataclass(request),
                context=context,
            )
            response = self._client.post(
                f"{self.url}/process",
                json=payload.model_dump(),
            )
            # 503 with X-Reject-Reason=at-capacity is load shedding, not a
            # failure. Surface as WorkerAtCapacityError without incrementing
            # the consecutive-failure counter -- a worker that is healthy but
            # busy must not be marked FAILED by the proxy.
            if (
                response.status_code == 503
                and response.headers.get("X-Reject-Reason") == "at-capacity"
            ):
                raise WorkerAtCapacityError(
                    f"remote {self.worker_id} reports at-capacity"
                )
            response.raise_for_status()
            body = ProcessResponse.model_validate(response.json())
            latency = perf_counter() - start

            with self._lock:
                self.completed_tasks += 1
                self.last_latency = latency
                self.total_latency_seconds += latency
                self._consecutive_failures = 0

            return body.answer
        except WorkerAtCapacityError:
            # No accounting beyond active_tasks decrement in finally; not a
            # failure for the proxy's circuit breaker.
            raise
        except httpx.HTTPError as exc:
            with self._lock:
                self.failed_tasks += 1
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._failure_threshold:
                    self.status = WorkerStatus.FAILED
            raise WorkerTransientError(
                f"remote {self.worker_id} HTTP failure: {exc}"
            ) from exc
        finally:
            with self._lock:
                self.active_tasks -= 1

    def snapshot_metrics(self) -> dict[str, object]:
        with self._lock:
            total = self.completed_tasks + self.failed_tasks
            avg_latency = (
                self.total_latency_seconds / self.completed_tasks
                if self.completed_tasks else 0.0
            )
            return {
                "worker_id": self.worker_id,
                "gpu_name": self.gpu_name,
                "status": self.status.value,
                "active_tasks": self.active_tasks,
                "pending_tasks": self.pending_tasks,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                "total_tasks": total,
                "avg_latency_seconds": avg_latency,
                "last_latency_seconds": self.last_latency,
                "remote_url": self.url,
            }

    def close(self) -> None:
        self._client.close()
