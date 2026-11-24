"""Dense and hybrid retrieval.

Lexical retrieval (BM25) fails on vocabulary mismatch: "bug bounty payout"
shares no stems with "rewarded with bounties of up to 5,000 USD". Dense
embeddings capture meaning instead of exact words, but are weaker on rare
exact tokens (error codes, IDs, numbers). Hybrid retrieval fuses both rankings
with Reciprocal Rank Fusion (RRF), which needs no score calibration.

The embedder is pluggable. FastEmbedEmbedder (ONNX, CPU-only, no PyTorch) is
the default real backend and is an optional dependency: pip install rag-shield[dense].
"""
from __future__ import annotations

import math
from typing import Protocol, Sequence

from .chunking import Chunk
from .retrieval import BM25Retriever, ScoredChunk


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Sentence embeddings via fastembed (default: BAAI/bge-small-en-v1.5, 384-d)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "Dense retrieval needs the optional extra: pip install 'rag-shield[dense]'"
            ) from exc
        self.name = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def _normalise(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class DenseRetriever:
    """Cosine-similarity search over pre-computed chunk embeddings."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        if not chunks:
            raise ValueError("Cannot build an index over an empty corpus")
        self.chunks = chunks
        self.embedder = embedder
        texts = [f"{c.metadata.get('title') or ''}. {c.text}" for c in chunks]
        self._vectors = [_normalise(v) for v in embedder.embed_documents(texts)]

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        q = _normalise(self.embedder.embed_query(query))
        scored = [
            ScoredChunk(chunk, sum(a * b for a, b in zip(q, vec)))
            for chunk, vec in zip(self.chunks, self._vectors)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], k: int = 60
) -> list[ScoredChunk]:
    """Fuse ranked lists: score(d) = sum over lists of 1 / (k + rank(d))."""
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.chunk_id
            by_id[cid] = hit.chunk
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [ScoredChunk(by_id[cid], score) for cid, score in ordered]


class HybridRetriever:
    """BM25 + dense retrieval fused with RRF. Drop-in replacement for BM25Retriever."""

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        candidates: int = 20,
        rrf_k: int = 60,
    ):
        self.chunks = chunks
        self.lexical = BM25Retriever(chunks)
        self.dense = DenseRetriever(chunks, embedder)
        self.candidates = candidates
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        lexical = self.lexical.search(query, k=self.candidates)
        dense = self.dense.search(query, k=self.candidates)
        return reciprocal_rank_fusion([lexical, dense], k=self.rrf_k)[:k]
