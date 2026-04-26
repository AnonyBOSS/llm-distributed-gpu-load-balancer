from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

from common import Request, Response
from llm import LLMInferenceEngine
from rag import RAGRetriever
from workers import GPUWorkerNode


@dataclass(slots=True)
class SchedulerStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time_seconds: float = 0.0


class MasterScheduler:
    def __init__(
        self,
        retriever: RAGRetriever,
        inference_engine: LLMInferenceEngine,
        workers: Sequence[GPUWorkerNode] | None = None,
        max_retries: int = 1,
    ) -> None:
        self._retriever = retriever
        self._inference_engine = inference_engine
        self._workers = list(workers or [])
        self._max_retries = max_retries
        self._stats = SchedulerStats()
        self._worker_successes: dict[str, int] = {}
        self._worker_failures: dict[str, int] = {}

    @property
    def stats(self) -> SchedulerStats:
        return self._stats

    @property
    def worker_successes(self) -> dict[str, int]:
        return dict(self._worker_successes)

    @property
    def worker_failures(self) -> dict[str, int]:
        return dict(self._worker_failures)

    def handle_batch(self, requests: Sequence[Request]) -> list[Response]:
        responses: list[Response] = []

        for request in requests:
            responses.append(self.handle_request(request))

        return responses

    def handle_request(
        self,
        request: Request,
        worker: GPUWorkerNode | None = None,
    ) -> Response:
        # Prefer explicit worker (for compatibility), otherwise select the least busy known worker.
        candidate_workers = self._resolve_candidate_workers(worker)
        self._stats.total_requests += 1
        start_time = perf_counter()

        try:
            context = self._retriever.retrieve_context(request)
            answer, assigned_worker = self._process_with_retry(
                request=request,
                context=context,
                candidate_workers=candidate_workers,
            )
            self._stats.successful_requests += 1
            print(f"[master] Completed {request.request_id}")
            return Response(
                request_id=request.request_id,
                worker_id=assigned_worker.worker_id,
                answer=answer,
                context=context,
            )
        except Exception as exc:
            self._stats.failed_requests += 1
            print(f"[master] Failed {request.request_id}: {exc}")
            return Response(
                request_id=request.request_id,
                worker_id="unassigned",
                answer="",
                context="",
                status="failed",
            )
        finally:
            # If the caller pre-reserved a worker via LoadBalancer.select_worker(),
            # release that handoff reservation now. Per-attempt reserve/release
            # inside _process_with_retry handles its own bookkeeping.
            if worker is not None:
                worker.release()
            self._stats.total_processing_time_seconds += perf_counter() - start_time

    def _resolve_candidate_workers(
        self,
        worker: GPUWorkerNode | None,
    ) -> list[GPUWorkerNode]:
        if worker is not None:
            fallback_workers = [w for w in self._workers if w.worker_id != worker.worker_id]
            return [worker, *self._sort_by_active_tasks(fallback_workers)]

        if not self._workers:
            raise ValueError(
                "MasterScheduler requires either a worker argument or configured workers."
            )

        return self._sort_by_active_tasks(self._workers)

    def _sort_by_active_tasks(
        self,
        workers: Sequence[GPUWorkerNode],
    ) -> list[GPUWorkerNode]:
        # Use pending_tasks (reservations) so concurrent picks stay balanced
        # even before active_tasks moves.
        return sorted(workers, key=lambda node: node.pending_tasks)

    def _process_with_retry(
        self,
        request: Request,
        context: str,
        candidate_workers: Sequence[GPUWorkerNode],
    ) -> tuple[str, GPUWorkerNode]:
        max_attempts = min(len(candidate_workers), self._max_retries + 1)
        last_error: Exception | None = None

        for attempt, worker in enumerate(candidate_workers[:max_attempts], start=1):
            print(
                f"[master] Scheduling {request.request_id} on worker "
                f"{worker.worker_id} (attempt {attempt}/{max_attempts})"
            )

            worker.reserve()
            try:
                answer = worker.process(request, context, self._inference_engine)
                self._worker_successes[worker.worker_id] = (
                    self._worker_successes.get(worker.worker_id, 0) + 1
                )
                return answer, worker
            except Exception as exc:
                last_error = exc
                self._worker_failures[worker.worker_id] = (
                    self._worker_failures.get(worker.worker_id, 0) + 1
                )
                print(
                    f"[master] Worker {worker.worker_id} failed for "
                    f"{request.request_id}: {exc}"
                )
            finally:
                worker.release()

        assert last_error is not None
        raise RuntimeError(
            f"All scheduling attempts failed for {request.request_id}"
        ) from last_error
