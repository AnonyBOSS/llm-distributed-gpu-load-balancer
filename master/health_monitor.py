"""Active health monitor for remote workers.

Runs as an asyncio background task in the master service. Polls each worker's
GET /health on a fixed interval; after `failure_threshold` consecutive
failures the worker is marked FAILED so the LoadBalancer stops routing to
it. After `recovery_threshold` consecutive successes on a previously-FAILED
worker, it is brought back to HEALTHY automatically.

The monitor is the only piece that knows how to revive a FAILED worker
without operator intervention. Without it, RemoteWorkerProxy can flip a
worker FAILED on three transient HTTP errors but has no path back.

Pattern: this is the classic *circuit breaker* (Nygard, "Release It!" 2nd ed.,
Pragmatic Bookshelf 2018) plus an *active health check* (Google SRE Workbook,
Ch. "Managing Load"). The two layers are deliberately independent:
  - per-request retry (in MasterScheduler) handles single transient errors;
  - active probing handles sustained outages and silently-recovered nodes.
Independence matters because each layer has different MTTR characteristics
and they fail open in different scenarios.
"""
from __future__ import annotations

import asyncio

import httpx

from workers import WorkerStatus
from workers.remote_proxy import RemoteWorkerProxy


class HealthMonitor:
    DEFAULT_POLL_INTERVAL = 1.0
    DEFAULT_PROBE_TIMEOUT = 0.5
    DEFAULT_FAILURE_THRESHOLD = 3
    DEFAULT_RECOVERY_THRESHOLD = 3

    def __init__(
        self,
        proxies: list[RemoteWorkerProxy],
        *,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL,
        probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_threshold: int = DEFAULT_RECOVERY_THRESHOLD,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        if probe_timeout_seconds <= 0:
            raise ValueError("probe_timeout_seconds must be > 0")

        self._proxies = list(proxies)
        self._poll_interval = poll_interval_seconds
        self._probe_timeout = probe_timeout_seconds
        self._failure_threshold = max(1, failure_threshold)
        self._recovery_threshold = max(1, recovery_threshold)

        self._fail_streaks: dict[str, int] = {p.worker_id: 0 for p in self._proxies}
        self._ok_streaks: dict[str, int] = {p.worker_id: 0 for p in self._proxies}

        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def snapshot(self) -> list[dict[str, object]]:
        """Per-worker monitor state for /health and /metrics endpoints."""
        return [
            {
                "worker_id": p.worker_id,
                "url": p.url,
                "status": p.status.value,
                "fail_streak": self._fail_streaks.get(p.worker_id, 0),
                "ok_streak": self._ok_streaks.get(p.worker_id, 0),
            }
            for p in self._proxies
        ]

    async def start(self) -> None:
        if self.is_running:
            return
        self._stop_event = asyncio.Event()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._probe_timeout, connect=self._probe_timeout)
        )
        self._task = asyncio.create_task(self._run(), name="health-monitor")
        print(
            f"[monitor] Started: {len(self._proxies)} workers, "
            f"interval={self._poll_interval}s, timeout={self._probe_timeout}s, "
            f"fail_threshold={self._failure_threshold}, "
            f"recovery_threshold={self._recovery_threshold}"
        )

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        print("[monitor] Stopped.")

    async def _run(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            await self._probe_all()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval,
                )
            except asyncio.TimeoutError:
                continue

    async def _probe_all(self) -> None:
        # gather() with return_exceptions=True so one slow/dead worker can't
        # stall probes for the others.
        await asyncio.gather(
            *(self._probe_one(p) for p in self._proxies),
            return_exceptions=True,
        )

    async def _probe_one(self, proxy: RemoteWorkerProxy) -> None:
        assert self._client is not None
        try:
            r = await self._client.get(f"{proxy.url}/health")
            r.raise_for_status()
            self._on_success(proxy)
        except (httpx.HTTPError, ValueError):
            self._on_failure(proxy)

    def _on_success(self, proxy: RemoteWorkerProxy) -> None:
        self._fail_streaks[proxy.worker_id] = 0
        self._ok_streaks[proxy.worker_id] += 1
        if (
            proxy.status == WorkerStatus.FAILED
            and self._ok_streaks[proxy.worker_id] >= self._recovery_threshold
        ):
            print(
                f"[monitor] {proxy.worker_id} recovered "
                f"({self._ok_streaks[proxy.worker_id]} consecutive ok probes)"
            )
            proxy.mark_healthy()
            self._ok_streaks[proxy.worker_id] = 0

    def _on_failure(self, proxy: RemoteWorkerProxy) -> None:
        self._ok_streaks[proxy.worker_id] = 0
        self._fail_streaks[proxy.worker_id] += 1
        if (
            proxy.status != WorkerStatus.FAILED
            and self._fail_streaks[proxy.worker_id] >= self._failure_threshold
        ):
            print(
                f"[monitor] {proxy.worker_id} marked FAILED "
                f"({self._fail_streaks[proxy.worker_id]} consecutive bad probes)"
            )
            proxy.mark_failed()
            self._fail_streaks[proxy.worker_id] = 0
