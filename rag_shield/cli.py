"""Command-line interface: rag-shield [--hybrid] ask|scan|eval."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chunking import Chunk, load_corpus
from .evaluate import evaluate_attacks, evaluate_guard, evaluate_retrieval
from .guard import InjectionGuard
from .llm import llm_from_env
from .pipeline import SecureRAG
from .retrieval import BM25Retriever, Retriever

DATA = Path(__file__).resolve().parent.parent / "data"


def build_retriever(chunks: list[Chunk], hybrid: bool) -> Retriever:
    if not hybrid:
        return BM25Retriever(chunks)
    from .dense import FastEmbedEmbedder, HybridRetriever

    return HybridRetriever(chunks, FastEmbedEmbedder())


def _cmd_ask(args: argparse.Namespace) -> int:
    chunks = load_corpus(args.docs)
    rag = SecureRAG(
        chunks, llm=llm_from_env(), top_k=args.k, retriever=build_retriever(chunks, args.hybrid)
    )
    ans = rag.ask(args.question)
    print(f"\nAnswer ({rag.llm.name}):\n  {ans.text}\n")
    if ans.citations:
        print("Sources:", ", ".join(ans.citations))
    for event in ans.events:
        print("  [shield]", event)
    return 1 if ans.blocked else 0


def _cmd_scan(args: argparse.Namespace) -> int:
    result = InjectionGuard(threshold=args.threshold).scan(args.text)
    print("FLAGGED" if result.flagged else "clean", f"(score={result.score})")
    for name in result.matches + result.signals:
        print("  -", name)
    return 1 if result.flagged else 0


def _cmd_eval(args: argparse.Namespace) -> int:
    chunks = load_corpus(args.docs)
    eval_dir = Path(args.eval_dir)
    retriever = build_retriever(chunks, args.hybrid)
    label = "Hybrid BM25 + dense (RRF)" if args.hybrid else "BM25"
    r = evaluate_retrieval(chunks, eval_dir / "qa.jsonl", retriever=retriever)
    g = evaluate_guard(InjectionGuard(), eval_dir / "injections.jsonl")
    a = evaluate_attacks(SecureRAG(chunks, retriever=retriever), eval_dir / "attacks.jsonl")
    print(f"Retrieval ({label})")
    print(f"  questions : {r.n}")
    print(f"  hit@1     : {r.hit_at_1:.2%}")
    print(f"  hit@3     : {r.hit_at_3:.2%}")
    print(f"  MRR       : {r.mrr:.3f}")
    for q in r.misses:
        print("  not rank 1:", q)
    print("\nInjection guard")
    print(f"  samples   : {g.n}")
    print(f"  precision : {g.precision:.2%}")
    print(f"  recall    : {g.recall:.2%}")
    print(f"  F1        : {g.f1:.3f}")
    for t in g.false_positives:
        print("  FP:", t[:90])
    for t in g.false_negatives:
        print("  FN:", t[:90])
    print("\nEnd-to-end indirect-injection attacks")
    print(f"  neutralised: {a.neutralised}/{a.n}")
    for d in a.details:
        status = "OK  " if d["neutralised"] else "FAIL"
        how = "blocked" if d["blocked"] else f"sanitised={d['sanitised']} quarantined={d['quarantined']}"
        print(f"  {status} {d['question'][:58]!r} {how}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-shield", description=__doc__)
    parser.add_argument("--docs", default=str(DATA / "docs"), help="corpus directory")
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="use hybrid BM25 + dense retrieval (needs: pip install 'rag-shield[dense]')",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="ask a question over the corpus")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=3)
    p_ask.set_defaults(func=_cmd_ask)

    p_scan = sub.add_parser("scan", help="scan a piece of text for prompt injection")
    p_scan.add_argument("text")
    p_scan.add_argument("--threshold", type=float, default=0.6)
    p_scan.set_defaults(func=_cmd_scan)

    p_eval = sub.add_parser("eval", help="run the evaluation suite")
    p_eval.add_argument("--eval-dir", default=str(DATA / "eval"))
    p_eval.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
