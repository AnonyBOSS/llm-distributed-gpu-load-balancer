from __future__ import annotations

from common import Request


class LLMInferenceEngine:
    def generate(self, request: Request, context: str) -> str:
        print(f"[llm] Generating stub response for {request.request_id}")
        return (
            "Stub response: the request was processed by the prototype cluster. "
            f"Prompt='{request.prompt}' | Context='{context}'"
        )
