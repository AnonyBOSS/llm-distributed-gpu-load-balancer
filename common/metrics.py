"""Prometheus metric definitions shared across services.

Each FastAPI service constructs its own `MetricsBundle` at startup and
calls `bundle.handler()` from a `/metrics` endpoint. The metric *names*
and *label keys* are standardised here so a single Grafana dashboard can
graph every service consistently.
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

# Histogram buckets covering simulated (~0.2 s) and HF (~30 s) latencies.
LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
)


# Worker status -> numeric for Prometheus gauge. Lower = healthier so
# `max_over_time(worker_status[5m])` picks the worst recent state.
STATUS_VALUE = {
    "healthy": 0,
    "degraded": 1,
    "failed": 2,
}


class MetricsBundle:
    """One bundle per service. Wraps a private registry so test runs that
    instantiate multiple bundles don't trip on Prometheus's global default."""

    def __init__(self, service: str) -> None:
        self.service = service
        self.registry = CollectorRegistry(auto_describe=True)

        # Per-service counters / histograms.
        self.requests_total = Counter(
            "requests_total",
            "Number of requests handled, labelled by outcome and downstream target.",
            labelnames=("service", "status", "target"),
            registry=self.registry,
        )
        self.request_latency_seconds = Histogram(
            "request_latency_seconds",
            "End-to-end request latency in seconds.",
            labelnames=("service", "target"),
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )

        # Per-target (worker / master) gauges.
        self.target_status = Gauge(
            "target_status",
            "0=healthy, 1=degraded, 2=failed.",
            labelnames=("service", "target"),
            registry=self.registry,
        )
        self.target_active_tasks = Gauge(
            "target_active_tasks",
            "In-flight tasks currently inside process().",
            labelnames=("service", "target"),
            registry=self.registry,
        )
        self.target_pending_tasks = Gauge(
            "target_pending_tasks",
            "Reserved tasks tracked at the LoadBalancer.",
            labelnames=("service", "target"),
            registry=self.registry,
        )

    @contextmanager
    def time_request(self, target: str) -> Iterator[None]:
        """Context manager that records latency and a +1 increment on the
        appropriate counter (status='ok' on clean exit, 'error' on raise)."""
        start = perf_counter()
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self.request_latency_seconds.labels(self.service, target).observe(
                perf_counter() - start
            )
            self.requests_total.labels(self.service, status, target).inc()

    def update_target_state(
        self,
        target: str,
        status: str,
        active_tasks: int,
        pending_tasks: int,
    ) -> None:
        self.target_status.labels(self.service, target).set(
            STATUS_VALUE.get(status, 2)
        )
        self.target_active_tasks.labels(self.service, target).set(active_tasks)
        self.target_pending_tasks.labels(self.service, target).set(pending_tasks)

    def handler(self) -> Response:
        """FastAPI/Starlette response body for GET /metrics."""
        return Response(
            content=generate_latest(self.registry),
            media_type=CONTENT_TYPE_LATEST,
        )
