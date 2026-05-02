"""Pydantic models for HTTP wire format.

The in-process dataclasses in `common.models` stay the canonical types used
by the scheduler / worker / LB. These Pydantic models exist only at the
FastAPI service boundary for validation + JSON (de)serialization. Conversion
is one line in each direction.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from common.models import Request, Response


class RequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    user_id: str
    prompt: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dataclass(self) -> Request:
        return Request(
            request_id=self.request_id,
            user_id=self.user_id,
            prompt=self.prompt,
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_dataclass(cls, request: Request) -> "RequestPayload":
        return cls(
            request_id=request.request_id,
            user_id=request.user_id,
            prompt=request.prompt,
            metadata=dict(request.metadata),
        )


class ResponsePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    worker_id: str
    answer: str
    context: str
    status: str = "completed"

    def to_dataclass(self) -> Response:
        return Response(
            request_id=self.request_id,
            worker_id=self.worker_id,
            answer=self.answer,
            context=self.context,
            status=self.status,
        )

    @classmethod
    def from_dataclass(cls, response: Response) -> "ResponsePayload":
        return cls(
            request_id=response.request_id,
            worker_id=response.worker_id,
            answer=response.answer,
            context=response.context,
            status=response.status,
        )


class ProcessRequest(BaseModel):
    """Body of POST /process on a worker service."""

    model_config = ConfigDict(extra="forbid")

    request: RequestPayload
    context: str = ""


class ProcessResponse(BaseModel):
    """Body returned by POST /process on a worker service."""

    model_config = ConfigDict(extra="forbid")

    worker_id: str
    answer: str
    latency_seconds: float


class WorkerHealth(BaseModel):
    """Body returned by GET /health on a worker service."""

    model_config = ConfigDict(extra="allow")  # tolerate extra metric fields

    worker_id: str
    gpu_name: str
    status: str
    active_tasks: int
    pending_tasks: int
    max_concurrent_tasks: int
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_latency_seconds: float = 0.0
    last_latency_seconds: float = 0.0
