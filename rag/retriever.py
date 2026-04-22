from __future__ import annotations

from common import Request


class RAGRetriever:
    def retrieve_context(self, request: Request) -> str:
        print(f"[rag] Retrieved placeholder context for {request.request_id}")
        return (
            "Knowledge-base stub: route requests across worker nodes, "
            "keep latency low, and preserve availability during scale-out."
        )
