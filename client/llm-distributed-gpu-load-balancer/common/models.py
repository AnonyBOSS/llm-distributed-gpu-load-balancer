from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Request:
    request_id: str
    user_id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Response:
    request_id: str
    worker_id: str
    answer: str
    context: str
    status: str = "completed"
