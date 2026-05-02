"""Benchmark harness for the distributed LLM cluster.

Runs the request fan-in at multiple concurrency levels (default 100, 250,
500, 1000) for each LB strategy (round_robin, least_connections,
load_aware), with an optional fault-injection run that kills a worker
container mid-flight.

Outputs into ./benchmarks/:
    raw/{strategy}_{users}_{tag}.json   per-request data per run
    results.csv                         summary table (one row per run)
    charts/throughput_vs_users.png
    charts/latency_p99_vs_users.png
    charts/worker_distribution.png
    charts/recovery_after_fault.png

Prerequisites:
    docker compose -f deploy/docker-compose.yml up -d --build

Then:
    python scripts/benchmark.py
    python scripts/benchmark.py --quick           # 50 + 200 only, single strategy
    python scripts/benchmark.py --no-fault        # skip fault-injection run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LB_URL = "http://localhost:8080"
MASTER_URL = "http://localhost:9000"
PROMETHEUS_URL = "http://localhost:9090"
FAULT_CONTAINER = "deploy-worker-2-1"

OUT_DIR = PROJECT_ROOT / "benchmarks"
RAW_DIR = OUT_DIR / "raw"
CHARTS_DIR = OUT_DIR / "charts"


@dataclass
class RequestResult:
    request_id: str
    status_code: int
    worker_id: str
    latency_seconds: float
    timestamp: float


@dataclass
class RunSummary:
    strategy: str
    users: int
    fault: bool
    elapsed_seconds: float
    throughput_rps: float
    successful: int
    errors: int
    error_rate: float
    p50_seconds: float
    p95_seconds: float
    p99_seconds: float
    worker_distribution: dict[str, int]


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _set_strategy(strategy: str) -> None:
    with httpx.Client(timeout=5.0) as c:
        r = c.post(f"{MASTER_URL}/admin/strategy", json={"strategy": strategy})
        r.raise_for_status()
        print(f"[bench] master strategy -> {r.json()['strategy']}")


def _set_backend(backend: str) -> None:
    """Fan out a backend swap to every worker via the master proxy."""
    with httpx.Client(timeout=120.0) as c:  # HF first-load can be slow
        r = c.post(f"{MASTER_URL}/admin/backend", json={"backend": backend})
        r.raise_for_status()
        result = r.json()
        bad = {k: v for k, v in result["workers"].items() if isinstance(v, dict) and "error" in v}
        ok_count = len(result["workers"]) - len(bad)
        print(
            f"[bench] backend -> {backend} on {ok_count} workers"
            + (f" (failures: {bad})" if bad else "")
        )


def _fire_one(client: httpx.Client, i: int) -> RequestResult:
    payload = {
        "request_id": f"bench-{i:06d}",
        "user_id": f"u{i}",
        "prompt": f"benchmark query #{i}",
        "metadata": {"client_index": i, "source": "benchmark"},
    }
    start = time.perf_counter()
    # Client-side retry with backoff. With Week 4's worker self-shedding,
    # 502/503 means "system at capacity, try again" -- the production-correct
    # response is to back off briefly and retry, not give up. Without this,
    # any benchmark whose offered load exceeds total slot capacity reports
    # massive errors even though the server is healthy.
    #
    # Budget: 10 attempts with exponential backoff starting at 0.1 s
    # gives ~ 0.1+0.2+0.4+0.8+1.6+3.2+6.4+12.8+25.6 = 51 s total before
    # giving up -- enough to drain a GPU-bound queue at ~ 5 rps.
    max_attempts = 10
    backoff = 0.1
    last_status = 0
    last_worker = "?"
    for attempt in range(max_attempts):
        try:
            r = client.post(f"{LB_URL}/request", json=payload, timeout=120.0)
            last_status = r.status_code
            if r.status_code == 200:
                try:
                    last_worker = r.json().get("worker_id", "?")
                except Exception:  # noqa: BLE001
                    pass
                latency = time.perf_counter() - start
                return RequestResult(
                    request_id=payload["request_id"],
                    status_code=200,
                    worker_id=last_worker,
                    latency_seconds=latency,
                    timestamp=time.time(),
                )
            if r.status_code in (502, 503) and attempt + 1 < max_attempts:
                # capacity-related; retry with exponential backoff + jitter
                time.sleep(backoff * (2**attempt) + 0.001 * (i % 10))
                continue
            break
        except httpx.HTTPError:
            if attempt + 1 < max_attempts:
                time.sleep(backoff * (2**attempt))
                continue
            break
    return RequestResult(
        request_id=payload["request_id"],
        status_code=last_status,
        worker_id=last_worker,
        latency_seconds=time.perf_counter() - start,
        timestamp=time.time(),
    )


def _inject_fault() -> None:
    try:
        subprocess.run(
            ["docker", "stop", FAULT_CONTAINER],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"[bench] FAULT injected: stopped {FAULT_CONTAINER}")
    except subprocess.CalledProcessError as exc:
        print(f"[bench] fault injection failed: {exc.stderr}")


def _recover_fault() -> None:
    try:
        subprocess.run(
            ["docker", "start", FAULT_CONTAINER],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"[bench] recovered: started {FAULT_CONTAINER}")
        # Give the monitor up to ~10 s to recover the worker.
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with httpx.Client(timeout=2.0) as c:
                    body = c.get(f"{MASTER_URL}/health").json()
                states = {row["worker_id"]: row["status"] for row in body.get("monitor", [])}
                if all(s == "healthy" for s in states.values()):
                    print(f"[bench] all workers HEALTHY: {states}")
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
        print("[bench] WARNING: workers did not all return to HEALTHY in 30s")
    except subprocess.CalledProcessError as exc:
        print(f"[bench] recovery failed: {exc.stderr}")


def run_one(
    strategy: str,
    num_users: int,
    *,
    fault_after: int | None = None,
) -> tuple[RunSummary, list[RequestResult]]:
    print(
        f"[bench] === RUN strategy={strategy} users={num_users} " f"fault_after={fault_after} ==="
    )
    _set_strategy(strategy)

    results: list[RequestResult] = []
    fault_triggered = False

    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=200)
    with httpx.Client(timeout=120.0, limits=limits) as client:
        with ThreadPoolExecutor(max_workers=min(num_users, 1000)) as pool:
            futures = [pool.submit(_fire_one, client, i) for i in range(num_users)]
            t0 = time.perf_counter()
            for fut in as_completed(futures):
                results.append(fut.result())
                if fault_after is not None and not fault_triggered and len(results) >= fault_after:
                    _inject_fault()
                    fault_triggered = True
            elapsed = time.perf_counter() - t0

    if fault_triggered:
        _recover_fault()

    ok_results = [r for r in results if r.status_code == 200]
    latencies = sorted(r.latency_seconds for r in ok_results)
    worker_dist: dict[str, int] = {}
    for r in ok_results:
        worker_dist[r.worker_id] = worker_dist.get(r.worker_id, 0) + 1

    summary = RunSummary(
        strategy=strategy,
        users=num_users,
        fault=fault_triggered,
        elapsed_seconds=elapsed,
        throughput_rps=len(ok_results) / elapsed if elapsed > 0 else 0.0,
        successful=len(ok_results),
        errors=len(results) - len(ok_results),
        error_rate=(len(results) - len(ok_results)) / max(len(results), 1),
        p50_seconds=_percentile(latencies, 0.50),
        p95_seconds=_percentile(latencies, 0.95),
        p99_seconds=_percentile(latencies, 0.99),
        worker_distribution=worker_dist,
    )

    print(
        f"[bench] DONE elapsed={summary.elapsed_seconds:.2f}s "
        f"throughput={summary.throughput_rps:.1f} rps "
        f"ok={summary.successful}/{num_users} "
        f"errors={summary.errors} "
        f"p50={summary.p50_seconds * 1000:.0f}ms "
        f"p95={summary.p95_seconds * 1000:.0f}ms "
        f"p99={summary.p99_seconds * 1000:.0f}ms "
        f"dist={worker_dist}"
    )
    return summary, results


def _save_raw(tag: str, results: list[RequestResult]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{tag}.json"
    path.write_text(json.dumps([asdict(r) for r in results]))
    return path


def _save_csv(rows: list[RunSummary], filename: str = "results.csv") -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "strategy",
                "users",
                "fault",
                "elapsed_sec",
                "throughput_rps",
                "successful",
                "errors",
                "error_rate",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "worker_dist",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.strategy,
                    r.users,
                    r.fault,
                    round(r.elapsed_seconds, 3),
                    round(r.throughput_rps, 2),
                    r.successful,
                    r.errors,
                    round(r.error_rate, 4),
                    round(r.p50_seconds * 1000, 1),
                    round(r.p95_seconds * 1000, 1),
                    round(r.p99_seconds * 1000, 1),
                    json.dumps(r.worker_distribution, sort_keys=True),
                ]
            )
    return path


def _save_compare_chart(rows: list[RunSummary]) -> None:
    """Side-by-side bar chart of throughput + p99 for sim vs batched_sim."""
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [r.strategy.split("/", 1)[-1] for r in rows]
    throughputs = [r.throughput_rps for r in rows]
    p99s = [r.p99_seconds * 1000 for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = list(range(len(labels)))
    width = 0.35
    bars1 = ax1.bar(
        [i - width / 2 for i in x],
        throughputs,
        width=width,
        label="throughput (rps)",
        color="#2b8cbe",
    )
    ax1.set_ylabel("Throughput (req/s)", color="#2b8cbe")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis="y", labelcolor="#2b8cbe")
    for b, v in zip(bars1, throughputs, strict=False):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        [i + width / 2 for i in x], p99s, width=width, label="p99 latency (ms)", color="#e34a33"
    )
    ax2.set_ylabel("p99 latency (ms)", color="#e34a33")
    ax2.tick_params(axis="y", labelcolor="#e34a33")
    for b, v in zip(bars2, p99s, strict=False):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom")

    users = rows[0].users
    plt.title(f"Sim vs continuous batching at {users} users (load_aware)")
    fig.tight_layout()
    plt.savefig(CHARTS_DIR / "sim_vs_batched.png", dpi=120)
    plt.close()


def _draw_charts(rows: list[RunSummary], fault_results: list[RequestResult] | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_rows = [r for r in rows if not r.fault]

    strategies = sorted({r.strategy for r in clean_rows})
    user_counts = sorted({r.users for r in clean_rows})

    # 1. Throughput vs users.
    plt.figure(figsize=(8, 5))
    for s in strategies:
        xs = [r.users for r in clean_rows if r.strategy == s]
        ys = [r.throughput_rps for r in clean_rows if r.strategy == s]
        plt.plot(xs, ys, marker="o", label=s)
    plt.xlabel("Concurrent users")
    plt.ylabel("Throughput (req/s)")
    plt.title("Throughput vs concurrency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "throughput_vs_users.png", dpi=120)
    plt.close()

    # 2. p99 latency vs users.
    plt.figure(figsize=(8, 5))
    for s in strategies:
        xs = [r.users for r in clean_rows if r.strategy == s]
        ys = [r.p99_seconds * 1000 for r in clean_rows if r.strategy == s]
        plt.plot(xs, ys, marker="o", label=s)
    plt.xlabel("Concurrent users")
    plt.ylabel("p99 latency (ms)")
    plt.title("Tail latency vs concurrency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "latency_p99_vs_users.png", dpi=120)
    plt.close()

    # 3. Worker distribution at the largest user count.
    largest = max(user_counts) if user_counts else 0
    plt.figure(figsize=(9, 5))
    width = 0.25
    workers = sorted({w for r in clean_rows if r.users == largest for w in r.worker_distribution})
    for idx, s in enumerate(strategies):
        for r in clean_rows:
            if r.strategy == s and r.users == largest:
                values = [r.worker_distribution.get(w, 0) for w in workers]
                xs = [i + idx * width for i in range(len(workers))]
                plt.bar(xs, values, width=width, label=s)
                break
    plt.xticks(
        [i + width * (len(strategies) - 1) / 2 for i in range(len(workers))],
        workers,
        rotation=15,
    )
    plt.ylabel(f"Requests served (of {largest})")
    plt.title(f"Per-worker distribution at {largest} users")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "worker_distribution.png", dpi=120)
    plt.close()

    # 4. Recovery curve from the fault run, if available.
    if fault_results:
        sorted_results = sorted(fault_results, key=lambda r: r.timestamp)
        t0 = sorted_results[0].timestamp
        # Bin into 0.5s windows; per-bin error rate.
        bin_width = 0.5
        if not sorted_results:
            return
        max_offset = sorted_results[-1].timestamp - t0
        bins = [
            (i * bin_width, (i + 1) * bin_width) for i in range(int(max_offset / bin_width) + 1)
        ]
        bin_rates = []
        bin_centers = []
        for lo, hi in bins:
            window = [r for r in sorted_results if lo <= (r.timestamp - t0) < hi]
            if not window:
                continue
            errors = sum(1 for r in window if r.status_code != 200)
            bin_rates.append(errors / len(window))
            bin_centers.append((lo + hi) / 2)

        plt.figure(figsize=(9, 4))
        plt.plot(bin_centers, [r * 100 for r in bin_rates], marker="o")
        plt.xlabel("Wall-clock seconds since first request")
        plt.ylabel("Error rate per 0.5s bin (%)")
        plt.title("Error rate during fault-injection run (worker-2 stopped mid-flight)")
        plt.ylim(0, max(5, max((r * 100 for r in bin_rates), default=5) * 1.2))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / "recovery_after_fault.png", dpi=120)
        plt.close()


def _check_stack() -> None:
    try:
        with httpx.Client(timeout=2.0) as c:
            c.get(f"{LB_URL}/healthz").raise_for_status()
            c.get(f"{MASTER_URL}/").raise_for_status()
    except httpx.HTTPError as exc:
        print(
            f"[bench] cannot reach the compose stack: {exc}\n"
            f"        run: docker compose -f deploy/docker-compose.yml up -d"
        )
        sys.exit(1)


def _gpu_preflight(user_counts: list[int]) -> None:
    """In --mode gpu, warn / abort if VRAM headroom is dangerously low.

    A model copy on GPU costs ~2 GB for distilgpt2 (fp32 weights + activations
    + KV-cache room). The compose default is 2 GPU workers ~ 4 GB used, ~2 GB
    free on a 6 GB card. If less than ~1 GB is free *now*, the next
    benchmark run is likely to OOM the GPU mid-flight, which on the first
    run cooked the host CPU to 95 C while CUDA thrashed.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("[bench] --mode gpu: nvidia-smi not available; skipping VRAM pre-flight")
        return

    first_line = out.splitlines()[0]
    free_mib_s, total_mib_s = (p.strip() for p in first_line.split(","))
    try:
        free_mib = int(free_mib_s)
        total_mib = int(total_mib_s)
    except ValueError:
        print(f"[bench] could not parse nvidia-smi output: {first_line!r}; skipping pre-flight")
        return

    print(f"[bench] GPU memory: free={free_mib} MiB / total={total_mib} MiB")

    # Hard abort: less than 1 GB free means models themselves may OOM.
    if free_mib < 1024:
        print(
            f"[bench] ABORT: only {free_mib} MiB VRAM free — not enough headroom "
            f"for model weights. Free at least 1 GB before running."
        )
        sys.exit(2)

    # KV-cache budget: bounded by MAX_CONCURRENT_TASKS (GPU compose default = 2
    # per worker × 2 GPU workers = 4), NOT by the number of queued users.
    # Users beyond the concurrency cap queue in Python and consume zero VRAM.
    # Cap at 20 to stay accurate even for larger GPU pools.
    peak_users = max(user_counts) if user_counts else 0
    gpu_concurrent = min(peak_users, 20)
    needed_mib = 200 + 5 * gpu_concurrent

    if free_mib < needed_mib:
        print(
            f"[bench] WARNING: only {free_mib} MiB VRAM free for an estimated "
            f"{needed_mib} MiB KV-cache peak; running anyway"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user-counts",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[100, 250, 500, 1000],
        help="comma-separated concurrency levels",
    )
    parser.add_argument(
        "--strategies",
        type=lambda s: s.split(","),
        default=["round_robin", "least_connections", "load_aware"],
        help="comma-separated LB strategies",
    )
    parser.add_argument(
        "--fault-users",
        type=int,
        default=250,
        help="user count for the single fault-injection run",
    )
    parser.add_argument(
        "--fault-after",
        type=int,
        default=80,
        help="trigger the fault after this many completed requests",
    )
    parser.add_argument("--no-fault", action="store_true", help="skip fault run")
    parser.add_argument(
        "--backend",
        choices=["sim", "batched_sim"],
        default="sim",
        help="LLM backend on each worker; 'batched_sim' models continuous batching",
    )
    parser.add_argument(
        "--compare-backends",
        action="store_true",
        help="run twice (sim then batched_sim) at one user count and chart the difference",
    )
    parser.add_argument(
        "--compare-users",
        type=int,
        default=500,
        help="user count for --compare-backends (default 500)",
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "gpu"],
        default="sim",
        help="'gpu' assumes deploy/docker-compose.gpu.yml is up: smaller user "
        "counts, results saved to results_gpu.csv",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="shortcut: --user-counts=50,200 --strategies=load_aware --no-fault",
    )
    args = parser.parse_args()

    if args.quick:
        args.user_counts = [50, 200]
        args.strategies = ["load_aware"]
        args.no_fault = True

    if args.mode == "gpu":
        # Real-model HF inference is ~10-100x slower than sim. Don't cap by
        # default (user can choose --user-counts), but skip the fault run and
        # default to a single strategy unless overridden, so each invocation
        # finishes in a reasonable time.
        args.no_fault = True
        if args.strategies == ["round_robin", "least_connections", "load_aware"]:
            # User accepted the default; collapse to just round_robin in GPU mode
            # so we don't run 4x more than they expected.
            args.strategies = ["round_robin"]

    _check_stack()
    if args.mode == "gpu":
        _gpu_preflight(args.user_counts)
    OUT_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    # Apply requested LLM backend before any runs. In `--mode gpu` the
    # workers were started with LLM_BACKEND=hf via compose env; don't
    # override that or we'd silently fall back to the simulated backend
    # and the "GPU" benchmark would in fact measure sleeps.
    if args.mode != "gpu":
        try:
            _set_backend(args.backend)
        except httpx.HTTPError as exc:
            print(
                f"[bench] WARNING: could not set backend ({exc}); continuing with whatever the workers already have"
            )
    else:
        print("[bench] --mode gpu: leaving workers on their compose-configured HF backend")

    rows: list[RunSummary] = []
    fault_results: list[RequestResult] | None = None
    compare_rows: list[RunSummary] = []

    for strategy in args.strategies:
        for users in args.user_counts:
            summary, results = run_one(strategy, users)
            rows.append(summary)
            tag = f"{args.backend}_{strategy}_u{users}_clean"
            _save_raw(tag, results)
            time.sleep(2)  # let pending_tasks fully drain between runs

    if not args.no_fault:
        summary, results = run_one(
            "load_aware",
            args.fault_users,
            fault_after=args.fault_after,
        )
        rows.append(summary)
        fault_results = results
        _save_raw(f"{args.backend}_load_aware_u{args.fault_users}_fault", results)

    if args.compare_backends:
        # Run a sim vs batched_sim head-to-head at one user count, single strategy.
        for backend in ("sim", "batched_sim"):
            _set_backend(backend)
            time.sleep(1)
            summary, results = run_one("load_aware", args.compare_users)
            summary.strategy = f"{summary.strategy}/{backend}"
            compare_rows.append(summary)
            _save_raw(f"compare_{backend}_u{args.compare_users}", results)
        _save_compare_chart(compare_rows)
        rows.extend(compare_rows)

    csv_name = "results_gpu.csv" if args.mode == "gpu" else "results.csv"
    csv_path = _save_csv(rows, filename=csv_name)
    print(f"[bench] wrote {csv_path}")

    try:
        _draw_charts(rows, fault_results)
        print(f"[bench] wrote charts to {CHARTS_DIR}")
    except ImportError as exc:
        print(f"[bench] skipping charts: {exc}")

    print("[bench] === HEADLINE ===")
    for r in rows:
        suffix = " [FAULT]" if r.fault else ""
        print(
            f"  {r.strategy:18} users={r.users:5d} "
            f"throughput={r.throughput_rps:6.1f} rps "
            f"p99={r.p99_seconds * 1000:6.0f}ms "
            f"errors={r.errors}{suffix}"
        )


if __name__ == "__main__":
    main()
