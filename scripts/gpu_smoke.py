"""GPU mode smoke test.

After bringing up the GPU compose stack:

    docker compose -f deploy/docker-compose.yml \
                   -f deploy/docker-compose.gpu.yml \
                   up -d --build

run:

    python scripts/gpu_smoke.py

It hits each worker's /health (so you can see the device the model loaded
on), then POSTs one real inference through the LB and prints the answer +
latency. If you've never run distilgpt2 on this host, the very first
request includes a one-time model download — expect 10-30 s.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LB_URL = "http://localhost:8080"
MASTER_URL = "http://localhost:9000"


def main() -> None:
    print("=== Master /health (per-worker view) ===")
    with httpx.Client(timeout=10.0) as client:
        try:
            r = client.get(f"{MASTER_URL}/health")
            r.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"FAIL: master not reachable at {MASTER_URL}: {exc}")
            sys.exit(1)
        body = r.json()
        for w in body.get("workers", []):
            print(
                f"  {w['worker_id']:15s} status={w['status']:10s} "
                f"max_concurrent={w['max_concurrent_tasks']:2d} "
                f"completed={w['completed_tasks']:4d}"
            )
        for m in body.get("monitor", []):
            print(
                f"  monitor[{m['worker_id']}]: status={m['status']} "
                f"ok_streak={m['ok_streak']} fail_streak={m['fail_streak']}"
            )

    print("\n=== Real inference through LB (nginx -> lb -> master -> worker) ===")
    payload = {
        "request_id": "gpu-smoke-1",
        "user_id": "smoke",
        "prompt": "Explain in two sentences how a GPU cluster handles many LLM requests.",
        "metadata": {"source": "gpu_smoke"},
    }
    start = time.perf_counter()
    with httpx.Client(timeout=120.0) as client:
        try:
            r = client.post(f"{LB_URL}/request", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"FAIL: LB request failed: {exc}")
            sys.exit(2)
    latency = time.perf_counter() - start

    body = r.json()
    print(f"served by   : {body.get('worker_id')}")
    print(f"latency     : {latency:.2f} s (includes any first-call model download)")
    print(f"context (kB): {len(body.get('context', '')) // 1024 + 1}")
    print(f"answer      : {body.get('answer', '')[:400]}")


if __name__ == "__main__":
    main()
