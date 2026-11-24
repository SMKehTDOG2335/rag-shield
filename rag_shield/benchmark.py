"""Benchmark injection detectors on public, labelled datasets.

Default: deepset/prompt-injections (Apache-2.0), test split, 116 samples in
English and German. It is not among the declared training sets of the default
ML model, so it works as a held-out check. Rows are fetched from the Hugging
Face datasets-server API with the standard library and cached locally.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

API = "https://datasets-server.huggingface.co/rows"
CACHE = Path(os.getenv("RAG_SHIELD_CACHE", Path.home() / ".cache" / "rag-shield"))


class Detector(Protocol):
    def scan(self, text: str): ...


@dataclass
class BenchmarkResult:
    detector: str
    n: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    ms_per_sample: float


def load_hf_rows(
    dataset: str = "deepset/prompt-injections",
    split: str = "test",
    config: str = "default",
    text_field: str = "text",
    label_field: str = "label",
    use_cache: bool = True,
) -> list[dict]:
    """Return [{"text": str, "injection": bool}] for a labelled HF dataset split."""
    cache_file = CACHE / f"{dataset.replace('/', '__')}-{config}-{split}.jsonl"
    if use_cache and cache_file.exists():
        return [json.loads(line) for line in cache_file.read_text(encoding="utf-8").splitlines()]

    rows: list[dict] = []
    offset, total = 0, None
    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": 100}
        )
        with urllib.request.urlopen(f"{API}?{query}", timeout=60) as resp:
            page = json.loads(resp.read())
        total = page["num_rows_total"]
        batch = page["rows"]
        if not batch:
            break
        for item in batch:
            row = item["row"]
            rows.append({"text": row[text_field], "injection": bool(row[label_field])})
        offset += len(batch)

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return rows


def run_benchmark(name: str, detector: Detector, rows: list[dict]) -> BenchmarkResult:
    tp = fp = fn = tn = 0
    start = time.perf_counter()
    for row in rows:
        predicted = bool(detector.scan(row["text"]).flagged)
        actual = bool(row["injection"])
        if predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1
    elapsed = time.perf_counter() - start
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = len(rows)
    return BenchmarkResult(
        name, n, precision, recall, f1, (tp + tn) / n if n else 0.0, 1000 * elapsed / max(n, 1)
    )


def format_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "| Detector | n | Precision | Recall | F1 | Accuracy | ms/sample |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.detector} | {r.n} | {r.precision:.1%} | {r.recall:.1%} | "
            f"{r.f1:.3f} | {r.accuracy:.1%} | {r.ms_per_sample:.1f} |"
        )
    return "\n".join(lines)
