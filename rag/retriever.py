from __future__ import annotations

import threading
from typing import Sequence

from common import Request

from .corpus import DEFAULT_CORPUS, Document


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

    @property
    def corpus(self) -> tuple[Document, ...]:
        return tuple(self._corpus)

    def retrieve_context(self, request: Request) -> str:
        if self._use_stub:
            doc = self._corpus[abs(hash(request.request_id)) % len(self._corpus)]
            print(
                f"[rag] Retrieved stub context for {request.request_id} "
                f"(doc_id={doc.doc_id})"
            )
            return f"{doc.title}\n{doc.text}"

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
            print(
                f"[rag] Indexed {len(self._corpus)} documents "
                f"(dim={embeddings.shape[1]})"
            )
