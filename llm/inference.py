from __future__ import annotations

import os
import random
import re
import time
from typing import Protocol

from common import Request


class LLMInferenceError(RuntimeError):
    pass


class LLMBackend(Protocol):
    def generate(self, prompt: str, context: str) -> str: ...


class SimulatedLLMBackend:
    def __init__(
        self,
        base_latency_s: float = 0.15,
        per_token_latency_s: float = 0.004,
        jitter_s: float = 0.05,
        failure_rate: float = 0.0,
        rng_seed: int | None = None,
    ) -> None:
        if not 0.0 <= failure_rate <= 1.0:
            raise ValueError("failure_rate must be in [0.0, 1.0]")
        if base_latency_s < 0 or per_token_latency_s < 0 or jitter_s < 0:
            raise ValueError("latency parameters must be non-negative")

        self._base_latency_s = base_latency_s
        self._per_token_latency_s = per_token_latency_s
        self._jitter_s = jitter_s
        self._failure_rate = failure_rate
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
        time.sleep(latency)

        keywords = _extract_keywords(context, limit=3)
        keyword_note = (
            f"grounded in {', '.join(keywords)}" if keywords else "no retrieval hits"
        )
        prompt_preview = prompt.strip().replace("\n", " ")[:120]
        return (
            f"[sim] Answer to '{prompt_preview}': "
            f"based on the retrieved context ({keyword_note}), "
            "the distributed cluster routes this request, generates a response, "
            "and returns it to the caller."
        )


class HuggingFaceLLMBackend:
    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_new_tokens: int = 64,
    ) -> None:
        try:
            from transformers import pipeline  # type: ignore[import-not-found]
        except ImportError as exc:
            raise LLMInferenceError(
                "transformers is not installed. Install with "
                "'pip install transformers torch' or leave LLM_BACKEND unset "
                "to use the simulated backend."
            ) from exc

        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        print(f"[llm] Loading HuggingFace model '{model_name}'")
        self._pipeline = pipeline("text-generation", model=model_name)

    def generate(self, prompt: str, context: str) -> str:
        composed = (
            f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:" if context
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
    if backend_name in ("", "sim", "simulated", "fake"):
        return SimulatedLLMBackend()
    if backend_name in ("hf", "huggingface", "transformers"):
        model_name = os.environ.get("LLM_MODEL", "distilgpt2")
        return HuggingFaceLLMBackend(model_name=model_name)
    raise LLMInferenceError(
        f"Unknown LLM_BACKEND '{backend_name}'. Use 'sim' or 'hf'."
    )


_KEYWORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")
_STOPWORDS = frozenset(
    {
        "the", "that", "this", "with", "from", "have", "been", "they",
        "their", "there", "which", "when", "what", "into", "across",
        "because", "while", "other", "such", "some", "also", "still",
        "than", "these", "those", "each", "about", "after", "before",
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
