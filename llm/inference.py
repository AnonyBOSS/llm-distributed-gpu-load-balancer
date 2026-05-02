from __future__ import annotations

import os
import random
import re
import threading
import time
from typing import Protocol

from common import Request


class LLMInferenceError(RuntimeError):
    pass


class LLMBackend(Protocol):
    def generate(self, prompt: str, context: str) -> str: ...


class SimulatedLLMBackend:
    """Sleep-based LLM stub.

    `serialise=True` adds a class-level lock held during the sleep, modelling
    a single physical GPU where calls genuinely cannot overlap. Without it,
    sleep releases the GIL and N concurrent calls of latency T finish in T,
    which inflates simulated throughput beyond what real hardware delivers.
    The bench-batching comparison uses serialise=True on both sides so the
    sim-vs-batched win is real, not an artefact of free GIL parallelism.
    """

    # Class-level lock shared across instances when serialise=True. One GPU
    # in the simulated world.
    _SERIAL_LOCK = threading.Lock()

    def __init__(
        self,
        base_latency_s: float = 0.15,
        per_token_latency_s: float = 0.004,
        jitter_s: float = 0.05,
        failure_rate: float = 0.0,
        rng_seed: int | None = None,
        serialise: bool = False,
    ) -> None:
        if not 0.0 <= failure_rate <= 1.0:
            raise ValueError("failure_rate must be in [0.0, 1.0]")
        if base_latency_s < 0 or per_token_latency_s < 0 or jitter_s < 0:
            raise ValueError("latency parameters must be non-negative")

        self._base_latency_s = base_latency_s
        self._per_token_latency_s = per_token_latency_s
        self._jitter_s = jitter_s
        self._failure_rate = failure_rate
        self._serialise = serialise
        self._rng = random.Random(rng_seed)

    def generate(self, prompt: str, context: str) -> str:
        if self._rng.random() < self._failure_rate:
            raise LLMInferenceError("simulated transient inference failure")

        approx_tokens = max(1, (len(prompt) + len(context)) // 4)
        jitter = self._rng.uniform(-self._jitter_s, self._jitter_s)
        latency = max(
            0.0,
            self._base_latency_s + self._per_token_latency_s * approx_tokens + jitter,
        )
        if self._serialise:
            with SimulatedLLMBackend._SERIAL_LOCK:
                time.sleep(latency)
        else:
            time.sleep(latency)

        keywords = _extract_keywords(context, limit=3)
        keyword_note = f"grounded in {', '.join(keywords)}" if keywords else "no retrieval hits"
        prompt_preview = prompt.strip().replace("\n", " ")[:120]
        return (
            f"[sim] Answer to '{prompt_preview}': "
            f"based on the retrieved context ({keyword_note}), "
            "the distributed cluster routes this request, generates a response, "
            "and returns it to the caller."
        )


class BatchedSimulatedLLMBackend:
    """Simulated LLM backend that models continuous batching.

    Real production transformer servers (vLLM, TGI, Orca) amortise per-token
    decode cost across many requests by interleaving them in one forward pass:
    instead of N requests x latency_per_request, you pay
    max(latency_per_request) once and serve the whole batch. This class
    captures that behaviour as a sleep-based simulation.

    References:
      - Yu et al., "Orca: A Distributed Serving System for Transformer-Based
        Generative Models", OSDI '22 -- introduces iteration-level scheduling
        and continuous batching.
      - Kwon et al., "Efficient Memory Management for Large Language Model
        Serving with PagedAttention" (vLLM), SOSP '23 -- paged-KV-cache that
        unblocks practical batching at scale.

    Concurrency model: every caller of generate() enqueues into a shared
    batch and waits on a per-call Event. A single executor thread flushes
    the batch when it reaches BATCH_MAX_SIZE *or* when BATCH_WINDOW_S has
    elapsed since the first arrival, whichever comes first. The flush
    sleeps once for the batch (modelling one forward pass) and then signals
    every waiter.
    """

    DEFAULT_BATCH_MAX_SIZE = 8
    DEFAULT_BATCH_WINDOW_S = 0.010

    def __init__(
        self,
        base_latency_s: float = 0.15,
        per_token_latency_s: float = 0.004,
        jitter_s: float = 0.05,
        failure_rate: float = 0.0,
        batch_max_size: int = DEFAULT_BATCH_MAX_SIZE,
        batch_window_s: float = DEFAULT_BATCH_WINDOW_S,
        rng_seed: int | None = None,
        serialise: bool = False,
    ) -> None:
        if not 0.0 <= failure_rate <= 1.0:
            raise ValueError("failure_rate must be in [0.0, 1.0]")
        if base_latency_s < 0 or per_token_latency_s < 0 or jitter_s < 0:
            raise ValueError("latency parameters must be non-negative")
        if batch_max_size < 1:
            raise ValueError("batch_max_size must be >= 1")
        if batch_window_s <= 0:
            raise ValueError("batch_window_s must be > 0")

        self._base_latency_s = base_latency_s
        self._per_token_latency_s = per_token_latency_s
        self._jitter_s = jitter_s
        self._failure_rate = failure_rate
        self._batch_max_size = batch_max_size
        self._batch_window_s = batch_window_s
        self._serialise = serialise

        self._rng = random.Random(rng_seed)
        self._lock = threading.Lock()
        # Each in-flight call appends a (event, result_box) tuple to this
        # list. The first arrival starts a window timer; the executor
        # thread holds the lock long enough to drain the queue and replace
        # it with an empty list.
        self._queue: list[tuple[threading.Event, list[object]]] = []
        self._first_arrival_at: float | None = None
        self._executor_active = False

    def generate(self, prompt: str, context: str) -> str:
        if self._rng.random() < self._failure_rate:
            raise LLMInferenceError("simulated transient inference failure")

        approx_tokens = max(1, (len(prompt) + len(context)) // 4)
        done = threading.Event()
        result_box: list[object] = [None]

        with self._lock:
            self._queue.append((done, result_box))
            if self._first_arrival_at is None:
                self._first_arrival_at = time.monotonic()
            should_dispatch = len(self._queue) >= self._batch_max_size and not self._executor_active
            if should_dispatch:
                self._executor_active = True
                threading.Thread(
                    target=self._flush_now,
                    args=(approx_tokens,),
                    daemon=True,
                ).start()
            elif not self._executor_active:
                self._executor_active = True
                threading.Thread(
                    target=self._flush_after_window,
                    args=(approx_tokens,),
                    daemon=True,
                ).start()

        # Block until the executor thread fills our result.
        done.wait(timeout=30.0)
        if isinstance(result_box[0], Exception):
            raise result_box[0]  # type: ignore[misc]
        return self._render_answer(prompt, context, approx_tokens)

    def _flush_after_window(self, approx_tokens: int) -> None:
        time.sleep(self._batch_window_s)
        self._flush_now(approx_tokens)

    def _flush_now(self, approx_tokens: int) -> None:
        with self._lock:
            batch = self._queue[: self._batch_max_size]
            self._queue = self._queue[self._batch_max_size :]
            self._first_arrival_at = None if not self._queue else self._first_arrival_at
            self._executor_active = False
            # If the residual queue is non-empty, kick another executor.
            residual = bool(self._queue)

        # One forward pass for the whole batch -- this is the win the
        # references describe. Latency scales with the longest sequence
        # in the batch, not with the number of requests.
        jitter = self._rng.uniform(-self._jitter_s, self._jitter_s)
        batch_latency = max(
            0.0,
            self._base_latency_s + self._per_token_latency_s * approx_tokens + jitter,
        )
        if self._serialise:
            # Share the same class-level lock as SimulatedLLMBackend so a
            # bench that mixes both classes models a single physical GPU.
            with SimulatedLLMBackend._SERIAL_LOCK:
                time.sleep(batch_latency)
        else:
            time.sleep(batch_latency)
        for event, _box in batch:
            event.set()

        if residual:
            # Kick the next batch immediately (continuous-batching style:
            # we don't wait for a full window when work is already queued).
            with self._lock:
                if not self._executor_active:
                    self._executor_active = True
                    threading.Thread(
                        target=self._flush_now,
                        args=(approx_tokens,),
                        daemon=True,
                    ).start()

    def _render_answer(self, prompt: str, context: str, approx_tokens: int) -> str:
        keywords = _extract_keywords(context, limit=3)
        keyword_note = f"grounded in {', '.join(keywords)}" if keywords else "no retrieval hits"
        prompt_preview = prompt.strip().replace("\n", " ")[:120]
        return (
            f"[batched-sim] Answer to '{prompt_preview}' "
            f"(~{approx_tokens} tokens, {keyword_note}): the distributed cluster "
            "interleaved this request with peers in one decode pass."
        )


class HuggingFaceLLMBackend:
    """Real LLM via HuggingFace transformers.

    `device="auto"` (the default) probes torch.cuda.is_available() and uses
    the GPU when present, CPU otherwise. Set LLM_DEVICE=cpu to force CPU
    even on a GPU host (useful for apples-to-apples sim-vs-cpu benchmarks).
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_new_tokens: int = 64,
        device: str = "auto",
    ) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import pipeline  # type: ignore[import-not-found]
        except ImportError as exc:
            raise LLMInferenceError(
                "transformers is not installed. Install with "
                "'pip install transformers torch' or leave LLM_BACKEND unset "
                "to use the simulated backend."
            ) from exc

        if device == "auto":
            resolved_device = 0 if torch.cuda.is_available() else -1
        elif device in ("cuda", "gpu"):
            if not torch.cuda.is_available():
                raise LLMInferenceError(
                    "device='cuda' requested but torch.cuda.is_available() is False"
                )
            resolved_device = 0
        elif device == "cpu":
            resolved_device = -1
        else:
            raise LLMInferenceError(f"Unknown device {device!r}; use 'auto', 'cuda', or 'cpu'.")

        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._device = "cuda:0" if resolved_device == 0 else "cpu"
        print(
            f"[llm] Loading HuggingFace model '{model_name}' "
            f"on {self._device} (cuda_available={torch.cuda.is_available()})"
        )
        self._pipeline = pipeline(
            "text-generation",
            model=model_name,
            device=resolved_device,
        )

    @property
    def device(self) -> str:
        return self._device

    def generate(self, prompt: str, context: str) -> str:
        composed = (
            f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
            if context
            else f"Question: {prompt}\nAnswer:"
        )
        outputs = self._pipeline(
            composed,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        if not outputs:
            raise LLMInferenceError("HuggingFace pipeline returned no output")
        return outputs[0]["generated_text"].strip()


class LLMInferenceEngine:
    def __init__(self, backend: LLMBackend | None = None) -> None:
        self._backend = backend if backend is not None else _default_backend_from_env()

    @property
    def backend(self) -> LLMBackend:
        return self._backend

    def generate(self, request: Request, context: str) -> str:
        print(f"[llm] Generating response for {request.request_id}")
        try:
            return self._backend.generate(request.prompt, context)
        except LLMInferenceError:
            raise
        except Exception as exc:
            raise LLMInferenceError(
                f"backend '{type(self._backend).__name__}' failed: {exc}"
            ) from exc


def _default_backend_from_env() -> LLMBackend:
    backend_name = os.environ.get("LLM_BACKEND", "sim").strip().lower()
    serialise = os.environ.get("LLM_SERIALISE", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )
    if backend_name in ("", "sim", "simulated", "fake"):
        return SimulatedLLMBackend(serialise=serialise)
    if backend_name in ("hf", "huggingface", "transformers"):
        model_name = os.environ.get("LLM_MODEL", "distilgpt2")
        device = os.environ.get("LLM_DEVICE", "auto").strip().lower()
        return HuggingFaceLLMBackend(model_name=model_name, device=device)
    if backend_name in ("batched", "batched_sim", "batched-sim"):
        return BatchedSimulatedLLMBackend(serialise=serialise)
    raise LLMInferenceError(
        f"Unknown LLM_BACKEND '{backend_name}'. " "Use 'sim', 'batched_sim', or 'hf'."
    )


_KEYWORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")
_STOPWORDS = frozenset(
    {
        "the",
        "that",
        "this",
        "with",
        "from",
        "have",
        "been",
        "they",
        "their",
        "there",
        "which",
        "when",
        "what",
        "into",
        "across",
        "because",
        "while",
        "other",
        "such",
        "some",
        "also",
        "still",
        "than",
        "these",
        "those",
        "each",
        "about",
        "after",
        "before",
    }
)


def _extract_keywords(context: str, limit: int = 3) -> list[str]:
    seen: list[str] = []
    for match in _KEYWORD_PATTERN.findall(context.lower()):
        if match in _STOPWORDS or match in seen:
            continue
        seen.append(match)
        if len(seen) >= limit:
            break
    return seen
