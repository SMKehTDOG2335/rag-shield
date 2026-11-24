"""Evaluation harness: retrieval quality, guard accuracy and end-to-end attacks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk
from .guard import InjectionGuard
from .pipeline import SecureRAG
from .retrieval import BM25Retriever, Retriever


def _load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass
class RetrievalReport:
    n: int
    hit_at_1: float
    hit_at_3: float
    mrr: float
    misses: list[str]


@dataclass
class GuardReport:
    n: int
    precision: float
    recall: float
    f1: float
    false_positives: list[str]
    false_negatives: list[str]


@dataclass
class AttackReport:
    n: int
    neutralised: int
    details: list[dict]


def evaluate_retrieval(
    chunks: list[Chunk],
    qa_path: str | Path,
    k: int = 3,
    retriever: Retriever | None = None,
) -> RetrievalReport:
    retriever = retriever or BM25Retriever(chunks)
    rows = _load_jsonl(qa_path)
    h1 = h3 = rr = 0.0
    misses: list[str] = []
    for row in rows:
        sources = [s.chunk.source for s in retriever.search(row["question"], k=10)]
        # De-duplicate sources while keeping rank order.
        ranked = list(dict.fromkeys(sources))
        if row["source"] in ranked:
            rank = ranked.index(row["source"]) + 1
            rr += 1 / rank
            h1 += rank == 1
            h3 += rank <= k
            if rank > 1:
                misses.append(row["question"])
        else:
            misses.append(row["question"])
    n = len(rows)
    return RetrievalReport(n, h1 / n, h3 / n, rr / n, misses)


def evaluate_guard(guard: InjectionGuard, labelled_path: str | Path) -> GuardReport:
    rows = _load_jsonl(labelled_path)
    tp = fp = fn = 0
    fps: list[str] = []
    fns: list[str] = []
    for row in rows:
        predicted = guard.scan(row["text"]).flagged
        actual = bool(row["injection"])
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
            fps.append(row["text"])
        elif actual and not predicted:
            fn += 1
            fns.append(row["text"])
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return GuardReport(len(rows), precision, recall, f1, fps, fns)


def evaluate_attacks(rag: SecureRAG, attacks_path: str | Path) -> AttackReport:
    """End-to-end: an attack is neutralised if it is blocked, its poisoned
    chunk is quarantined, or the forbidden marker never reaches the output."""
    rows = _load_jsonl(attacks_path)
    details = []
    neutralised = 0
    for row in rows:
        ans = rag.ask(row["question"])
        leaked = row["forbidden"].lower() in ans.text.lower()
        ok = not leaked
        neutralised += ok
        details.append(
            {
                "question": row["question"],
                "neutralised": ok,
                "blocked": ans.blocked,
                "quarantined": list(ans.quarantined),
                "sanitised": list(ans.sanitised),
            }
        )
    return AttackReport(len(rows), neutralised, details)
