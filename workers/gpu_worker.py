from __future__ import annotations

from common import Request
from llm import LLMInferenceEngine


class GPUWorkerNode:
    def __init__(self, worker_id: str, gpu_name: str) -> None:
        self.worker_id = worker_id
        self.gpu_name = gpu_name
        self.active_tasks = 0

    def process(
        self,
        request: Request,
        context: str,
        inference_engine: LLMInferenceEngine,
    ) -> str:
        self.active_tasks += 1
        print(
            f"[worker:{self.worker_id}] Starting {request.request_id} on {self.gpu_name}"
        )
        try:
            return inference_engine.generate(request, context)
        finally:
            self.active_tasks -= 1
            print(f"[worker:{self.worker_id}] Finished {request.request_id}")
