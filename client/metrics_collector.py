# client/metrics_collector.py
"""
Thread-safe metrics collector.
Records one RequestRecord per completed request, then computes
aggregate statistics (latency percentiles, throughput, error rate,
per-worker distribution) for the final report.
"""
from __future__ import annotations

import json
import os
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional

# ── Per-request record ────────────────────────────────────────────────────────

@dataclass
class RequestRecord:
    request_id : str
    user_id    : str
    status     : str    # "completed" | "failed" | "timeout" | "error"
    latency_sec: float
    worker_id  : str
    timestamp  : float  # epoch time when the response was received


# ── Aggregate statistics ──────────────────────────────────────────────────────

@dataclass
class SummaryStats:
    # volumes
    total_requests   : int
    completed        : int
    failed           : int
    timeouts         : int
    error_rate_pct   : float

    # throughput
    throughput_rps   : float
    test_duration_sec: float

    # latency (milliseconds, successful requests only)
    min_latency_ms   : float
    avg_latency_ms   : float
    median_latency_ms: float
    p95_latency_ms   : float
    p99_latency_ms   : float
    max_latency_ms   : float

    # per-worker breakdown
    worker_distribution: dict[str, int] = field(default_factory=dict)


# ── Collector ─────────────────────────────────────────────────────────────────

class MetricsCollector:
    """
    All public methods are safe to call from multiple threads simultaneously.
    """

    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._records   : list[RequestRecord] = []
        self._start_time: Optional[float]     = None
        self._end_time  : Optional[float]     = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        with self._lock:
            self._records.clear()
            self._start_time = time.perf_counter()
            self._end_time   = None

    def stop(self) -> None:
        with self._lock:
            self._end_time = time.perf_counter()

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, rec: RequestRecord) -> None:
        with self._lock:
            self._records.append(rec)

    # ── Live counters (approximate — no lock for speed) ───────────────────────

    @property
    def total_completed(self) -> int:
        return len(self._records)

    @property
    def total_successful(self) -> int:
        return sum(1 for r in self._records if r.status == "completed")

    # ── Aggregate computation ─────────────────────────────────────────────────

    def compute_stats(self) -> SummaryStats:
        with self._lock:
            records  = list(self._records)
            start    = self._start_time or time.perf_counter()
            end      = self._end_time   or time.perf_counter()

        duration = max(end - start, 1e-6)
        total    = len(records)

        if total == 0:
            return SummaryStats(
                total_requests=0, completed=0, failed=0, timeouts=0,
                error_rate_pct=0.0, throughput_rps=0.0,
                test_duration_sec=round(duration, 3),
                min_latency_ms=0.0, avg_latency_ms=0.0, median_latency_ms=0.0,
                p95_latency_ms=0.0, p99_latency_ms=0.0, max_latency_ms=0.0,
            )

        completed = [r for r in records if r.status == "completed"]
        failed    = [r for r in records if r.status == "failed"]
        timeouts  = [r for r in records if r.status == "timeout"]
        bad       = len(failed) + len(timeouts)

        # latencies in ms (successful only)
        lats_ms = sorted(r.latency_sec * 1_000 for r in completed) or [0.0]

        def _pct(data: list[float], p: float) -> float:
            idx = min(int(len(data) * p / 100), len(data) - 1)
            return round(data[idx], 2)

        # worker distribution across ALL responses
        wdist: dict[str, int] = defaultdict(int)
        for r in records:
            wdist[r.worker_id] += 1

        return SummaryStats(
            total_requests    = total,
            completed         = len(completed),
            failed            = len(failed),
            timeouts          = len(timeouts),
            error_rate_pct    = round(bad / total * 100, 2),
            throughput_rps    = round(total / duration, 2),
            test_duration_sec = round(duration, 3),
            min_latency_ms    = round(min(lats_ms), 2),
            avg_latency_ms    = round(statistics.mean(lats_ms), 2),
            median_latency_ms = round(statistics.median(lats_ms), 2),
            p95_latency_ms    = _pct(lats_ms, 95),
            p99_latency_ms    = _pct(lats_ms, 99),
            max_latency_ms    = round(max(lats_ms), 2),
            worker_distribution = dict(wdist),
        )

    # ── Output ────────────────────────────────────────────────────────────────

    def print_summary(self, stats: Optional[SummaryStats] = None) -> None:
        s   = stats or self.compute_stats()
        sep = "─" * 54

        print(f"\n{'═' * 54}")
        print("  LOAD TEST SUMMARY")
        print(f"{'═' * 54}")
        print(f"  Duration          : {s.test_duration_sec:.3f}s")
        print(f"  Total Requests    : {s.total_requests}")
        print(f"  Completed         : {s.completed}")
        print(f"  Failed            : {s.failed}")
        print(f"  Timeouts          : {s.timeouts}")
        print(f"  Error Rate        : {s.error_rate_pct:.2f}%")
        print(f"  Throughput        : {s.throughput_rps} req/s")
        print(f"  {sep}")
        print("  Latency (ms)  — successful requests only:")
        print(f"    Min             : {s.min_latency_ms}")
        print(f"    Avg             : {s.avg_latency_ms}")
        print(f"    Median (p50)    : {s.median_latency_ms}")
        print(f"    p95             : {s.p95_latency_ms}")
        print(f"    p99             : {s.p99_latency_ms}")
        print(f"    Max             : {s.max_latency_ms}")
        print(f"  {sep}")
        print("  Worker Distribution:")
        for worker, count in sorted(s.worker_distribution.items()):
            pct = count / max(s.total_requests, 1)
            bar = "█" * min(36, int(pct * 36))
            print(f"    {worker:<22}: {count:>5}  {bar}")
        print(f"{'═' * 54}\n")

    def save_to_file(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        stats   = self.compute_stats()
        payload = {
            "summary": asdict(stats),
            "records": [
                {
                    "request_id" : r.request_id,
                    "user_id"    : r.user_id,
                    "status"     : r.status,
                    "latency_ms" : round(r.latency_sec * 1_000, 2),
                    "worker_id"  : r.worker_id,
                    "timestamp"  : r.timestamp,
                }
                for r in self._records
            ],
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[metrics] Full results saved → {path}")
