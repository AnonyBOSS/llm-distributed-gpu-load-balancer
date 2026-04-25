# client/__init__.py
"""
Public API of the client package.

main.py does:
    from client import ClientLoadGenerator, LoadTestRunner

MetricsCollector and SummaryStats are also exported so the rest of the
system (e.g. a monitoring dashboard or admin layer) can import stats
without reaching into internal modules.
"""
from client.generator         import ClientLoadGenerator
from client.runner            import LoadTestRunner
from client.metrics_collector import MetricsCollector, SummaryStats, RequestRecord

__all__ = [
    "ClientLoadGenerator",
    "LoadTestRunner",
    "MetricsCollector",
    "SummaryStats",
    "RequestRecord",
]
