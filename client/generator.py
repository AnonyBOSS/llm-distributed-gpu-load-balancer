# client/generator.py
"""
ClientLoadGenerator
====================
Responsible for building well-formed Request objects that match the
common.models.Request dataclass exactly:

    Request(request_id: str, user_id: str, prompt: str, metadata: dict)

This class is the only place in the client that constructs Requests,
so if the model ever changes you only update one file.
"""
from __future__ import annotations

import random
import uuid
from typing import Any

from common import Request
from client.config import SAMPLE_PROMPTS


class ClientLoadGenerator:
    """
    Generates synthetic LLM requests for load testing.

    Usage
    -----
    generator = ClientLoadGenerator()
    requests  = generator.generate_requests(count=1000)
    """

    def __init__(
        self,
        prompts: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        prompts : custom prompt pool (falls back to config.SAMPLE_PROMPTS)
        seed    : fix for reproducible request sequences
        """
        self._prompts = prompts or SAMPLE_PROMPTS
        if seed is not None:
            random.seed(seed)

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_requests(self, count: int = 1) -> list[Request]:
        """
        Generate `count` Request objects, each with a unique request_id and user_id.

        Returns
        -------
        list[Request]  — ready to pass to LoadBalancer.select_worker() or
                          MasterScheduler.handle_request() / handle_batch()
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")

        return [self._make_request(index=i) for i in range(count)]

    def generate_single(self) -> Request:
        """Convenience wrapper — returns exactly one Request."""
        return self.generate_requests(count=1)[0]

    # ── Internal ───────────────────────────────────────────────────────────────

    def _make_request(self, index: int) -> Request:
        """Build one Request with all required fields populated."""
        return Request(
            request_id = str(uuid.uuid4()),          # globally unique
            user_id    = f"user-{index:05d}",        # e.g. "user-00042"
            prompt     = random.choice(self._prompts),
            metadata   = self._build_metadata(index),
        )

    @staticmethod
    def _build_metadata(index: int) -> dict[str, Any]:
        """
        Metadata travels with the request through the system and can be
        used for tracing, priority routing, or analytics.
        """
        return {
            "client_index"  : index,
            "priority"      : "high" if index % 10 == 0 else "normal",
            "source"        : "load_test",
        }
