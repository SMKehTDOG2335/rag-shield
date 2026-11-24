import pytest

from rag_shield.chunking import split_into_chunks
from rag_shield.retrieval import BM25Retriever, tokenize


def test_split_respects_window_and_overlap():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = split_into_chunks(text, "doc.md", max_words=100, overlap_words=20)
    assert [len(c.text.split()) for c in chunks] == [100, 100, 90]
    # Overlap: last 20 words of chunk 0 start chunk 1.
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]
    assert chunks[1].chunk_id == "doc.md#1"


def test_headings_become_metadata_not_body():
    chunks = split_into_chunks("# Title\n\nBody text here.", "d.md")
    assert chunks[0].metadata["title"] == "Title"
    assert "#" not in chunks[0].text


def test_split_validates_arguments():
    with pytest.raises(ValueError):
        split_into_chunks("x", "d", max_words=10, overlap_words=10)


def test_stemmer_is_consistent():
    assert tokenize("exceed") == tokenize("exceeds")
    assert tokenize("refunds") == tokenize("refunded") == tokenize("refund")


def test_bm25_ranks_relevant_chunk_first():
    a = split_into_chunks("Refunds are issued within 30 days of purchase.", "refund.md")
    b = split_into_chunks("The API allows 100 requests per minute.", "api.md")
    retriever = BM25Retriever(a + b)
    hits = retriever.search("how many API requests per minute?")
    assert hits[0].chunk.source == "api.md"
    assert retriever.search("refund window")[0].chunk.source == "refund.md"


def test_bm25_returns_nothing_for_unrelated_query():
    chunks = split_into_chunks("Refunds are issued within 30 days.", "r.md")
    assert BM25Retriever(chunks).search("quantum chromodynamics") == []


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        BM25Retriever([])
