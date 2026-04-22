from __future__ import annotations

from common import Request, Response
from llm import LLMInferenceEngine
from rag import RAGRetriever
from workers import GPUWorkerNode


class MasterScheduler:
    def __init__(
        self,
        retriever: RAGRetriever,
        inference_engine: LLMInferenceEngine,
    ) -> None:
        self._retriever = retriever
        self._inference_engine = inference_engine

    def handle_request(self, request: Request, worker: GPUWorkerNode) -> Response:
        print(
            f"[master] Scheduling {request.request_id} on worker {worker.worker_id}"
        )
        context = self._retriever.retrieve_context(request)
        answer = worker.process(request, context, self._inference_engine)
        print(f"[master] Completed {request.request_id}")
        return Response(
            request_id=request.request_id,
            worker_id=worker.worker_id,
            answer=answer,
            context=context,
        )
