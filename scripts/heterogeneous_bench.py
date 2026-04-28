"""Benchmark all four LB strategies against heterogeneous workers.

The default compose stack runs three identical workers, so the
strategy choice barely matters -- round_robin already balances perfectly.
This script targets the 1:2:8 capacity ratio defined in
deploy/docker-compose.heterogeneous.yml and shows where capacity-aware
strategies (load_aware, power_of_two) actually win.

Bring up the heterogeneous stack first:

    docker compose -f deploy/docker-compose.yml \
                   -f deploy/docker-compose.heterogeneous.yml \
                   up -d --force-recreate

Then:

    python scripts/heterogeneous_bench.py

Saves benchmarks/heterogeneous_results.csv and one chart at
benchmarks/charts/heterogeneous_strategy_comparison.png.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark import (  # noqa: E402
    CHARTS_DIR,
    OUT_DIR,
    RunSummary,
    _check_stack,
    _save_raw,
    _set_backend,
    run_one,
)


STRATEGIES = ["round_robin", "least_connections", "load_aware", "power_of_two"]
USER_COUNTS = [200, 500, 1000]


def _save_csv(rows: list[RunSummary]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "heterogeneous_results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy", "users", "throughput_rps", "p50_ms", "p95_ms", "p99_ms",
            "errors", "worker_dist",
        ])
        for r in rows:
            writer.writerow([
                r.strategy, r.users, round(r.throughput_rps, 2),
                round(r.p50_seconds * 1000, 1),
                round(r.p95_seconds * 1000, 1),
                round(r.p99_seconds * 1000, 1),
                r.errors,
                r.worker_distribution,
            ])
    return path


def _save_chart(rows: list[RunSummary]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    user_counts = sorted({r.users for r in rows})
    strategies = sorted({r.strategy for r in rows})

    # Two panels: throughput, p99 latency.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for s in strategies:
        xs = [r.users for r in rows if r.strategy == s]
        ys_thr = [r.throughput_rps for r in rows if r.strategy == s]
        ys_p99 = [r.p99_seconds * 1000 for r in rows if r.strategy == s]
        ax1.plot(xs, ys_thr, marker="o", label=s)
        ax2.plot(xs, ys_p99, marker="o", label=s)

    ax1.set_xlabel("Concurrent users")
    ax1.set_ylabel("Throughput (req/s)")
    ax1.set_title("Throughput across LB strategies\n(heterogeneous workers, 1:2:8 capacity)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Concurrent users")
    ax2.set_ylabel("p99 latency (ms)")
    ax2.set_title("Tail latency across LB strategies\n(heterogeneous workers)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(CHARTS_DIR / "heterogeneous_strategy_comparison.png", dpi=120)
    plt.close()


def main() -> None:
    _check_stack()
    print("[hetero] Resetting backend to sim across workers")
    _set_backend("sim")
    time.sleep(1)

    rows: list[RunSummary] = []
    for strategy in STRATEGIES:
        for users in USER_COUNTS:
            summary, results = run_one(strategy, users)
            rows.append(summary)
            _save_raw(f"hetero_{strategy}_u{users}", results)
            time.sleep(2)

    _save_csv(rows)
    _save_chart(rows)
    print("[hetero] === HEADLINE ===")
    for r in rows:
        print(
            f"  {r.strategy:18} users={r.users:5d} "
            f"throughput={r.throughput_rps:6.1f} rps "
            f"p99={r.p99_seconds * 1000:6.0f}ms "
            f"errors={r.errors} dist={r.worker_distribution}"
        )


if __name__ == "__main__":
    main()
