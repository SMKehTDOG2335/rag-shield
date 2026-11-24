"""Hybrid retrieval tests using a deterministic toy embedder (no model download).

The toy embedder maps words to concept dimensions, so synonyms that share no
characters ("payout" / "reward") land close together. That is exactly the
property real sentence embeddings provide and BM25 lacks.
"""
import re

import pytest

from rag_shield.chunking import split_into_chunks
from rag_shield.dense import DenseRetriever, HybridRetriever, reciprocal_rank_fusion
from rag_shield.pipeline import SecureRAG
from rag_shield.retrieval import BM25Retriever, ScoredChunk

CONCEPTS = {
    "money": {"payout", "reward", "rewarded", "bounty", "bounties", "usd", "paid", "pay"},
    "security": {"bug", "vulnerability", "vulnerabilities", "security", "disclosure"},
    "time": {"days", "weeks", "leave", "holiday", "vacation"},
    "api": {"api", "requests", "rate", "limit", "throttle"},
}


class ToyEmbedder:
    name = "toy"

    def _vec(self, text):
        words = set(re.findall(r"[a-z]+", text.lower()))
        return [float(len(words & vocab)) for vocab in CONCEPTS.values()]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def chunks():
    docs = {
        "security.md": "Valid vulnerabilities are rewarded with bounties of up to 5,000 USD.",
        "leave.md": "Employees receive 24 days of annual leave per year.",
        "api.md": "The API allows 100 requests per minute before returning 429.",
    }
    out = []
    for name, text in docs.items():
        out.extend(split_into_chunks(text, name))
    return out


def test_bm25_misses_vocabulary_mismatch(chunks):
    assert BM25Retriever(chunks).search("bug bounty payout") == []


def test_dense_retrieval_bridges_vocabulary_mismatch(chunks):
    hits = DenseRetriever(chunks, ToyEmbedder()).search("bug payout", k=1)
    assert hits[0].chunk.source == "security.md"


def test_hybrid_recovers_what_bm25_misses(chunks):
    hits = HybridRetriever(chunks, ToyEmbedder()).search("bug payout", k=1)
    assert hits[0].chunk.source == "security.md"


def test_hybrid_keeps_exact_token_matches(chunks):
    # "429" is an exact token: no concept dimension covers it, BM25 does.
    hits = HybridRetriever(chunks, ToyEmbedder()).search("what does 429 mean", k=1)
    assert hits[0].chunk.source == "api.md"


def test_rrf_rewards_agreement_between_rankers(chunks):
    a, b, c = chunks
    fused = reciprocal_rank_fusion(
        [[ScoredChunk(a, 9), ScoredChunk(b, 5)], [ScoredChunk(b, 0.9), ScoredChunk(c, 0.1)]]
    )
    # b appears in both lists, so it outranks a (top of one list only).
    assert [s.chunk.source for s in fused] == ["leave.md", "security.md", "api.md"]


def test_pipeline_accepts_hybrid_retriever(chunks):
    rag = SecureRAG(chunks, retriever=HybridRetriever(chunks, ToyEmbedder()))
    ans = rag.ask("What is the bug bounty payout?")
    assert ans.retrieved and ans.retrieved[0] == "security.md#0"
