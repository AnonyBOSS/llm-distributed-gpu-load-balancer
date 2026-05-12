# tests/unit/test_lb_service.py
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import services.lb_service as lb_svc
from common.wire import RequestPayload
from workers import WorkerTransientError

_PAYLOAD = RequestPayload(request_id="r1", user_id="u1", prompt="hi", metadata={})
_SUCCESS = {
    "request_id": "r1",
    "worker_id": "master-2",
    "answer": "ok",
    "context": "",
    "status": "completed",
}


def _master(wid: str, *, fail: bool = False) -> MagicMock:
    m = MagicMock()
    m.worker_id = wid
    if fail:
        m.post_json.side_effect = WorkerTransientError("down")
    else:
        m.post_json.return_value = _SUCCESS
    return m


def test_retries_on_first_master_failure(monkeypatch):
    m1 = _master("master-1", fail=True)
    m2 = _master("master-2")
    lb = MagicMock()
    lb.select_worker.side_effect = [m1, m2]

    monkeypatch.setattr(lb_svc, "load_balancer", lb)
    monkeypatch.setattr(lb_svc, "LB_MASTER_RETRIES", 1)
    monkeypatch.setattr(lb_svc, "metrics_bundle", MagicMock())

    result = lb_svc.handle_request(_PAYLOAD)

    assert result.worker_id == "master-2"
    m1.release.assert_called_once()
    m2.release.assert_called_once()


def test_503_when_all_masters_fail(monkeypatch):
    m1 = _master("master-1", fail=True)
    m2 = _master("master-2", fail=True)
    lb = MagicMock()
    lb.select_worker.side_effect = [m1, m2]

    monkeypatch.setattr(lb_svc, "load_balancer", lb)
    monkeypatch.setattr(lb_svc, "LB_MASTER_RETRIES", 1)
    monkeypatch.setattr(lb_svc, "metrics_bundle", MagicMock())

    with pytest.raises(HTTPException) as exc_info:
        lb_svc.handle_request(_PAYLOAD)

    assert exc_info.value.status_code == 503
    assert "exhausted" in exc_info.value.detail
    m1.release.assert_called_once()
    m2.release.assert_called_once()


def test_pending_tasks_released_on_failure(monkeypatch):
    m1 = _master("master-1", fail=True)
    lb = MagicMock()
    lb.select_worker.return_value = m1

    monkeypatch.setattr(lb_svc, "load_balancer", lb)
    monkeypatch.setattr(lb_svc, "LB_MASTER_RETRIES", 0)
    monkeypatch.setattr(lb_svc, "metrics_bundle", MagicMock())

    with pytest.raises(HTTPException):
        lb_svc.handle_request(_PAYLOAD)

    m1.release.assert_called_once()


def test_no_retry_on_uninitialized_lb(monkeypatch):
    monkeypatch.setattr(lb_svc, "load_balancer", None)
    monkeypatch.setattr(lb_svc, "metrics_bundle", MagicMock())

    with pytest.raises(HTTPException) as exc_info:
        lb_svc.handle_request(_PAYLOAD)

    assert exc_info.value.status_code == 503
    assert "not initialised" in exc_info.value.detail


def test_503_when_select_worker_raises_mid_loop(monkeypatch):
    m1 = _master("master-1", fail=True)
    lb = MagicMock()
    lb.select_worker.side_effect = [m1, RuntimeError("No healthy GPU workers available.")]

    monkeypatch.setattr(lb_svc, "load_balancer", lb)
    monkeypatch.setattr(lb_svc, "LB_MASTER_RETRIES", 1)
    monkeypatch.setattr(lb_svc, "metrics_bundle", MagicMock())

    with pytest.raises(HTTPException) as exc_info:
        lb_svc.handle_request(_PAYLOAD)

    assert exc_info.value.status_code == 503
    m1.release.assert_called_once()
