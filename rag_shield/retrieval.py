"""Okapi BM25 retriever implemented from scratch (no external dependencies)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i in is it its
    of on or our that the their there this to was we what when where which who
    will with you your my me if not""".split()
)


def _stem(token: str) -> str:
    """Very light suffix stripping so 'refunds'/'refunded' match 'refund'."""
    for suffix in ("ing", "ed", "es", "s"):
        if suffix == "ed" and token.endswith("eed"):
            continue  # 'exceed', 'proceed' are not past tenses
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[ScoredChunk]: ...


class BM25Retriever:
    """Classic BM25 ranking (Robertson & Zaragoza) over a list of chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("Cannot build an index over an empty corpus")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        # Index the document title alongside the body so headings aid recall.
        self._doc_tokens = [
            tokenize(f"{c.metadata.get('title') or ''} {c.text}") for c in chunks
        ]
        self._tf = [Counter(toks) for toks in self._doc_tokens]
        self._doc_len = [len(toks) for toks in self._doc_tokens]
        self._avgdl = sum(self._doc_len) / len(self._doc_len) or 1.0
        df: Counter[str] = Counter()
        for toks in self._doc_tokens:
            df.update(set(toks))
        n = len(chunks)
        # BM25+ style floor keeps IDF non-negative for very common terms.
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def _score(self, query_tokens: list[str], idx: int) -> float:
        tf = self._tf[idx]
        dl = self._doc_len[idx]
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            score += self._idf[term] * freq * (self.k1 + 1) / denom
        return score

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        q = tokenize(query)
        if not q:
            return []
        scored = [ScoredChunk(c, self._score(q, i)) for i, c in enumerate(self.chunks)]
        scored = [s for s in scored if s.score > 0]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]
