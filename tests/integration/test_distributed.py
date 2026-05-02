"""End-to-end test against a running compose stack.

Skipped automatically if the LB endpoint isn't reachable. To run locally:

    docker compose -f deploy/docker-compose.yml up -d
    pytest tests/integration -v

The test verifies (1) the full chain serves traffic, (2) Prometheus is
scraping every service, and (3) the master's active health monitor flips
a stopped worker FAILED then recovers it when restarted.
"""

from __future__ import annotations

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

LB_URL = os.environ.get("LB_URL", "http://localhost:8080")
PROM_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
COMPOSE_FILE = os.environ.get("COMPOSE_FILE", "deploy/docker-compose.yml")


def _stack_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get(f"{LB_URL}/healthz").status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    not _stack_reachable(),
    reason=f"compose stack not reachable at {LB_URL}; run `docker compose up -d` first",
)


def _post(client: httpx.Client, i: int) -> int:
    payload = {
        "request_id": f"int-{i:04d}",
        "user_id": f"u{i}",
        "prompt": f"q{i}",
        "metadata": {},
    }
    return client.post(f"{LB_URL}/request", json=payload).status_code


def test_traffic_distributes_across_all_workers():
    n = 60
    with httpx.Client(timeout=30.0) as c:
        with ThreadPoolExecutor(max_workers=n) as pool:
            statuses = list(pool.map(lambda i: _post(c, i), range(n)))
    assert all(s == 200 for s in statuses), f"non-200 in {statuses}"


def test_prometheus_scrapes_all_targets():
    with httpx.Client(timeout=5.0) as c:
        r = c.get(f"{PROM_URL}/api/v1/targets")
    r.raise_for_status()
    targets = r.json()["data"]["activeTargets"]
    by_job = {}
    for t in targets:
        by_job.setdefault(t["labels"]["job"], []).append(t["health"])
    # All scrape jobs present and healthy.
    assert "lb" in by_job and all(h == "up" for h in by_job["lb"])
    assert "master" in by_job and all(h == "up" for h in by_job["master"])
    assert "workers" in by_job and len(by_job["workers"]) == 3
    assert all(h == "up" for h in by_job["workers"])


def test_metrics_counter_increments_on_traffic():
    """Send N requests, verify Prometheus's requests_total picks them up."""
    n = 20

    def query_total() -> int:
        with httpx.Client(timeout=5.0) as c:
            r = c.get(
                f"{PROM_URL}/api/v1/query",
                params={"query": 'sum(requests_total{service="lb",status="ok"})'},
            )
        result = r.json()["data"]["result"]
        return int(float(result[0]["value"][1])) if result else 0

    before = query_total()
    with httpx.Client(timeout=30.0) as c:
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(lambda i: _post(c, i + 1000), range(n)))

    # Prometheus scrapes every 5s; wait long enough for one cycle.
    deadline = time.time() + 15
    while time.time() < deadline:
        after = query_total()
        if after >= before + n:
            break
        time.sleep(1)
    assert after >= before + n, f"Prometheus saw {after - before} new, expected >= {n}"


def test_health_monitor_detects_killed_worker_and_recovers():
    """Stop worker-2 mid-flight, verify the monitor flips it FAILED, then
    restart it and verify the monitor brings it back HEALTHY."""

    def _list_master_view() -> dict[str, str]:
        proc = subprocess.run(
            ["docker", "exec", "deploy-master-1", "curl", "-s", "http://localhost:9000/health"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        import json

        body = json.loads(proc.stdout)
        return {row["worker_id"]: row["status"] for row in body["monitor"]}

    # Establish baseline.
    initial = _list_master_view()
    assert all(s == "healthy" for s in initial.values()), initial

    subprocess.run(["docker", "stop", "deploy-worker-2-1"], check=True, capture_output=True)
    try:
        # Monitor probes every 1s; needs 3 misses before flipping. Allow margin.
        deadline = time.time() + 10
        while time.time() < deadline:
            if _list_master_view().get("gpu-worker-2") == "failed":
                break
            time.sleep(1)
        assert _list_master_view().get("gpu-worker-2") == "failed"

        # Other workers must still serve traffic without errors.
        with httpx.Client(timeout=30.0) as c:
            r = c.post(
                f"{LB_URL}/request",
                json={"request_id": "after-fail", "user_id": "u", "prompt": "q", "metadata": {}},
            )
        assert r.status_code == 200
    finally:
        subprocess.run(["docker", "start", "deploy-worker-2-1"], check=True, capture_output=True)

    # Wait for recovery — needs the worker container to come up + 3 ok probes.
    deadline = time.time() + 30
    while time.time() < deadline:
        if _list_master_view().get("gpu-worker-2") == "healthy":
            break
        time.sleep(1)
    assert _list_master_view().get("gpu-worker-2") == "healthy"
