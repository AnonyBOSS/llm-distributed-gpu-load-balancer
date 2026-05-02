# client/runner.py
"""
LoadTestRunner
==============
Runs the actual concurrent load test by wiring together:

    ClientLoadGenerator  →  Scheduler.handle_request(request)
                                  ↓  (internally calls lb.dispatch → worker.process)
                         →  MetricsCollector

This matches the PDF skeleton's call pattern exactly:

    # master/scheduler.py  (skeleton)
    def handle_request(self, request):
        response = self.lb.dispatch(request)
        return response

    # lb/load_balancer.py  (skeleton)
    def dispatch(self, request):
        worker = self.get_next_worker()
        return worker.process(request)

Supports two modes
------------------
flat     — all N users fire simultaneously (default)
ramp_up  — gradually steps through RAMP_UP_STEPS, pausing between each
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from client.config import (
    DEFAULT_NUM_USERS,
    MAX_RETRIES,
    MAX_THREAD_WORKERS,
    RAMP_STEP_PAUSE_SEC,
    RAMP_UP_STEPS,
    REPORT_INTERVAL_SEC,
    REQUEST_TIMEOUT_SEC,
    RESULTS_FILE,
    RETRY_BACKOFF_SEC,
    SAVE_RESULTS,
)
from client.generator import ClientLoadGenerator
from client.metrics_collector import MetricsCollector, RequestRecord

# ── Live progress reporter ────────────────────────────────────────────────────


class _LiveReporter:
    """Prints a one-line animated progress bar in a background daemon thread."""

    def __init__(self, collector: MetricsCollector, total: int) -> None:
        self._c = collector
        self._total = total
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=2)
        print()  # newline after the progress bar

    def _run(self) -> None:
        t0 = time.perf_counter()
        while not self._stop.is_set():
            done = self._c.total_completed
            ok = self._c.total_successful
            elapsed = time.perf_counter() - t0
            rps = done / max(elapsed, 1e-6)
            pct = done / max(self._total, 1) * 100
            filled = int(30 * done / max(self._total, 1))
            bar = "█" * filled + "░" * (30 - filled)
            print(
                f"\r[live] [{bar}] {pct:5.1f}%  "
                f"{done}/{self._total} req  ok={ok}  "
                f"rps={rps:.1f}  t={elapsed:.1f}s",
                end="",
                flush=True,
            )
            self._stop.wait(REPORT_INTERVAL_SEC)


# ── Per-user task (runs inside thread pool) ───────────────────────────────────


def _run_single_user(
    index: int,
    generator: ClientLoadGenerator,
    scheduler,  # master.Scheduler — exposes handle_request(request)
    collector: MetricsCollector,
) -> None:
    """
    Simulate one user, aligned with the PDF skeleton's Scheduler interface:

        response = scheduler.handle_request(request)

    The scheduler internally calls lb.dispatch(request) → worker.process(request),
    so the client does NOT call the load balancer directly — exactly matching
    the skeleton's architecture.

    Retried up to MAX_RETRIES times on transient failures, with exponential
    back-off. Abandoned as "timeout" if total elapsed exceeds REQUEST_TIMEOUT_SEC.
    """
    request = generator.generate_requests(count=1)[0]
    status = "error"
    worker_id = "unassigned"
    start = time.perf_counter()

    for attempt in range(MAX_RETRIES + 1):
        # ── Timeout guard ────────────────────────────────────────────────────
        elapsed = time.perf_counter() - start
        if elapsed >= REQUEST_TIMEOUT_SEC:
            status = "timeout"
            print(
                f"[runner] user-{index:05d} timed out after " f"{elapsed:.1f}s (attempt {attempt})"
            )
            break

        try:
            # ── Skeleton-aligned call ────────────────────────────────────────
            # Matches: response = scheduler.handle_request(request)
            # The scheduler owns the load balancer and dispatches internally.
            response = scheduler.handle_request(request)

            # Handle both the extended Response dataclass and the dict the
            # skeleton's GPUWorker.process() returns {"id", "result", "latency"}
            if hasattr(response, "status"):
                status = response.status  # "completed" | "failed"
                worker_id = getattr(response, "worker_id", "unknown")
            elif isinstance(response, dict):
                status = response.get("status", "completed")
                worker_id = str(response.get("id", "unknown"))
            else:
                status = "completed"

            if status == "completed":
                break  # success — no retry needed

            # Worker returned "failed" → retry if attempts remain
            if attempt < MAX_RETRIES:
                backoff = RETRY_BACKOFF_SEC * (2**attempt)
                print(
                    f"[runner] user-{index:05d} got '{status}', "
                    f"retrying in {backoff:.2f}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(backoff)

        except RuntimeError as exc:
            # All workers are down — no point retrying
            status = "failed"
            print(f"[runner] user-{index:05d} fatal RuntimeError: {exc}")
            break

        except Exception as exc:
            status = "error"
            if attempt < MAX_RETRIES:
                backoff = RETRY_BACKOFF_SEC * (2**attempt)
                print(f"[runner] user-{index:05d} error ({exc}), " f"retrying in {backoff:.2f}s")
                time.sleep(backoff)
            else:
                print(f"[runner] user-{index:05d} gave up after " f"{MAX_RETRIES} retries: {exc}")

    latency = time.perf_counter() - start
    collector.record(
        RequestRecord(
            request_id=request.request_id,
            user_id=request.user_id,
            status=status,
            latency_sec=latency,
            worker_id=worker_id,
            timestamp=time.time(),
        )
    )


# ── Public runner ─────────────────────────────────────────────────────────────


class LoadTestRunner:
    """
    Drives the full load test.

    Mirrors the PDF skeleton's run_load_test() function but wraps it in a
    class with ThreadPoolExecutor (instead of raw threads), live reporting,
    retry/timeout logic, and metrics collection.

    Parameters
    ----------
    scheduler : master.Scheduler
        Must expose ``handle_request(request) -> response`` and optionally
        ``handle_batch(requests) -> list[response]``.
        The scheduler owns the LoadBalancer internally — matching the PDF
        skeleton architecture exactly.
    generator : ClientLoadGenerator (optional, created if not provided)
    """

    def __init__(
        self,
        scheduler,
        generator: ClientLoadGenerator | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._generator = generator or ClientLoadGenerator()
        self._collector = MetricsCollector()

    # ── Entry points ──────────────────────────────────────────────────────────

    def run(
        self,
        num_users: int = DEFAULT_NUM_USERS,
        ramp_up: bool = False,
    ) -> MetricsCollector:
        """
        Run the load test and return the populated MetricsCollector.

        Parameters
        ----------
        num_users : total concurrent users to simulate (default 1000)
        ramp_up   : True  → gradual ramp through RAMP_UP_STEPS (100→1000)
                    False → instant flat load (default)
        """
        self._collector.start()

        if ramp_up:
            self._run_ramp_up(num_users)
        else:
            self._run_flat(num_users)

        self._collector.stop()
        stats = self._collector.compute_stats()
        self._collector.print_summary(stats)

        if SAVE_RESULTS:
            self._collector.save_to_file(RESULTS_FILE)

        return self._collector

    def run_batch(self, num_requests: int = 100) -> MetricsCollector:
        """
        Uses Scheduler.handle_batch() — sends all requests through the
        scheduler in one call, recording accurate per-request latency.
        """
        print(f"\n[runner] Batch mode — {num_requests} requests via handle_batch()")
        requests = self._generator.generate_requests(count=num_requests)

        self._collector.start()
        batch_start = time.perf_counter()

        responses = self._scheduler.handle_batch(requests)

        batch_end = time.perf_counter()
        per_req_latency = (batch_end - batch_start) / max(num_requests, 1)

        for i, (req, resp) in enumerate(zip(requests, responses, strict=False)):
            if hasattr(resp, "status"):
                status = resp.status
                worker_id = getattr(resp, "worker_id", "unknown")
            elif isinstance(resp, dict):
                status = resp.get("status", "completed")
                worker_id = str(resp.get("id", "unknown"))
            else:
                status = "completed"
                worker_id = "unknown"

            self._collector.record(
                RequestRecord(
                    request_id=req.request_id,
                    user_id=req.user_id,
                    status=status,
                    latency_sec=per_req_latency * (i + 1),
                    worker_id=worker_id,
                    timestamp=time.time(),
                )
            )

        self._collector.stop()
        self._collector.print_summary()

        if SAVE_RESULTS:
            self._collector.save_to_file(RESULTS_FILE)

        return self._collector

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_flat(self, num_users: int) -> None:
        """
        Fire all num_users simultaneously using a thread pool.
        Equivalent to the skeleton's run_load_test() but with a
        ThreadPoolExecutor cap (MAX_THREAD_WORKERS) instead of unbounded threads.
        """
        print(f"\n[runner] Flat load — {num_users} concurrent users")
        reporter = _LiveReporter(self._collector, num_users)
        reporter.start()

        with ThreadPoolExecutor(max_workers=min(num_users, MAX_THREAD_WORKERS)) as pool:
            futures = [
                pool.submit(
                    _run_single_user,
                    i,
                    self._generator,
                    self._scheduler,
                    self._collector,
                )
                for i in range(num_users)
            ]
            for f in as_completed(futures):
                f.result()  # surface unhandled exceptions immediately

        reporter.stop()

    def _run_ramp_up(self, num_users: int) -> None:
        """
        Gradually ramp load — simulates 100 → 1000 users as required by
        the project's Testing and Evaluation section (Simulate 100→1000
        concurrent users).
        """
        steps = [s for s in RAMP_UP_STEPS if s < num_users] + [num_users]
        print(f"\n[runner] Ramp-up load — steps: {steps}")

        reporter = _LiveReporter(self._collector, num_users)
        reporter.start()

        submitted = 0
        with ThreadPoolExecutor(max_workers=min(num_users, MAX_THREAD_WORKERS)) as pool:
            for step in steps:
                batch = step - submitted
                if batch <= 0:
                    continue

                print(f"\n[ramp]   +{batch} users  (cumulative: {step})")
                futures = [
                    pool.submit(
                        _run_single_user,
                        submitted + i,
                        self._generator,
                        self._scheduler,
                        self._collector,
                    )
                    for i in range(batch)
                ]
                submitted += batch

                for f in as_completed(futures):
                    f.result()

                if step < num_users:
                    print(f"[ramp]   Pausing {RAMP_STEP_PAUSE_SEC}s before next step…")
                    time.sleep(RAMP_STEP_PAUSE_SEC)

        reporter.stop()
