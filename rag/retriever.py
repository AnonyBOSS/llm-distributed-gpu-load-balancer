from __future__ import annotations

import math
import threading
from collections import Counter
from typing import Sequence

from common import Request

from .corpus import DEFAULT_CORPUS, Document

# BM25 tuning parameters. k1 controls term-frequency saturation (1.2–2.0 is
# typical); b controls document-length normalisation (0.75 is standard).
_BM25_K1 = 1.5
_BM25_B = 0.75


class RAGRetriever:
    def __init__(
        self,
        corpus: Sequence[Document] | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
        use_stub: bool = False,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self._corpus: list[Document] = list(corpus or DEFAULT_CORPUS)
        if not self._corpus:
            raise ValueError("RAGRetriever requires a non-empty corpus")

        self._model_name = model_name
        self._top_k = min(top_k, len(self._corpus))
        self._use_stub = use_stub
        self._model = None
        self._index = None
        self._lock = threading.Lock()

        # BM25 index — built lazily on first stub retrieval.
        self._bm25: dict | None = None

    @property
    def corpus(self) -> tuple[Document, ...]:
        return tuple(self._corpus)

    def retrieve_context(self, request: Request) -> str:
        if self._use_stub:
            return self._stub_retrieve(request)

        self._ensure_index()
        assert self._model is not None and self._index is not None

        query_vec = self._model.encode(
            [request.prompt],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        _scores, idxs = self._index.search(query_vec, self._top_k)

        snippets: list[str] = []
        matched_ids: list[str] = []
        for row in idxs:
            for doc_idx in row:
                if doc_idx < 0:
                    continue
                doc = self._corpus[int(doc_idx)]
                snippets.append(f"{doc.title}\n{doc.text}")
                matched_ids.append(doc.doc_id)

        print(
            f"[rag] Retrieved {len(snippets)} doc(s) for {request.request_id} "
            f"({', '.join(matched_ids) or 'none'})"
        )
        return "\n---\n".join(snippets) if snippets else ""

    # ── BM25 stub retrieval ────────────────────────────────────────────────────

    def _stub_retrieve(self, request: Request) -> str:
        """BM25 retrieval — no model download needed; much better than raw keyword count."""
        self._ensure_bm25()
        assert self._bm25 is not None

        query_terms = _tokenize(request.prompt)
        scores = _bm25_scores(query_terms, self._bm25)

        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        top = [self._corpus[i] for i in ranked[: self._top_k]]
        ids = [d.doc_id for d in top]
        print(
            f"[rag] BM25 retrieved {len(top)} doc(s) for {request.request_id} "
            f"({', '.join(ids) or 'none'})"
        )
        return "\n---\n".join(f"{doc.title}\n{doc.text}" for doc in top) if top else ""

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        with self._lock:
            if self._bm25 is not None:
                return
            self._bm25 = _build_bm25_index(self._corpus)

    # ── FAISS index ────────────────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        with self._lock:
            if self._index is not None:
                return

            import faiss  # type: ignore[import-not-found]
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            print(f"[rag] Loading embedding model '{self._model_name}'")
            model = SentenceTransformer(self._model_name)
            corpus_texts = [f"{doc.title}. {doc.text}" for doc in self._corpus]
            embeddings = model.encode(
                corpus_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)

            self._model = model
            self._index = index
            print(f"[rag] Indexed {len(self._corpus)} documents (dim={embeddings.shape[1]})")


# ── Module-level BM25 helpers (pure functions, easy to unit-test) ──────────────

def _tokenize(text: str) -> list[str]:
    return [w for w in text.lower().split() if len(w) > 2]


def _build_bm25_index(corpus: list[Document]) -> dict:
    """Pre-compute per-document token frequencies and corpus-level stats."""
    tokenized = [_tokenize(f"{doc.title} {doc.text}") for doc in corpus]
    N = len(tokenized)
    avgdl = sum(len(t) for t in tokenized) / N if N else 1.0
    # df[term] = number of documents containing term
    df: Counter[str] = Counter(term for tokens in tokenized for term in set(tokens))
    # tf[i][term] = raw count of term in document i
    tf_list = [Counter(tokens) for tokens in tokenized]
    return {"N": N, "avgdl": avgdl, "df": df, "tf_list": tf_list, "tokenized": tokenized}


def _bm25_scores(query_terms: list[str], index: dict) -> list[float]:
    N: int = index["N"]
    avgdl: float = index["avgdl"]
    df: Counter = index["df"]
    tf_list: list[Counter] = index["tf_list"]
    tokenized: list[list[str]] = index["tokenized"]

    scores: list[float] = []
    for i, tokens in enumerate(tokenized):
        dl = len(tokens)
        tf_map = tf_list[i]
        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            n_t = df.get(term, 0)
            idf = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1.0)
            tf_norm = tf * (_BM25_K1 + 1) / (tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl))
            score += idf * tf_norm
        scores.append(score)
    return scores
