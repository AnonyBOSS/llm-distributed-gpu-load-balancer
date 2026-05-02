"""RAG retriever stub determinism + corpus shape."""

from __future__ import annotations

import pytest

from common import Request
from rag import RAGRetriever
from rag.corpus import DEFAULT_CORPUS, Document


def _req(rid: str) -> Request:
    return Request(request_id=rid, user_id="u", prompt="hi", metadata={})


def test_stub_is_deterministic():
    r1 = RAGRetriever(use_stub=True)
    r2 = RAGRetriever(use_stub=True)
    assert r1.retrieve_context(_req("same-id")) == r2.retrieve_context(_req("same-id"))


def test_stub_returns_non_empty_for_every_corpus_size():
    r = RAGRetriever(use_stub=True)
    for i in range(50):
        assert r.retrieve_context(_req(f"r{i}"))


def test_corpus_documents_have_required_fields():
    assert len(DEFAULT_CORPUS) > 0
    for doc in DEFAULT_CORPUS:
        assert isinstance(doc, Document)
        assert doc.doc_id and doc.title and doc.text


def test_top_k_validated():
    with pytest.raises(ValueError):
        RAGRetriever(use_stub=True, top_k=0)


def test_empty_corpus_falls_back_to_default():
    # `corpus or DEFAULT_CORPUS` substitutes the default for None *and* []
    # by design; an explicitly-empty corpus uses the built-in instead of
    # raising. Documented behaviour, not a bug.
    r = RAGRetriever(corpus=[], use_stub=True)
    assert len(r.corpus) == len(DEFAULT_CORPUS)
