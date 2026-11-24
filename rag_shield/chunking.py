"""Document loading and chunking."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with provenance."""

    chunk_id: str
    source: str
    text: str
    metadata: dict = field(default_factory=dict, compare=False, hash=False)


_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


def split_into_chunks(
    text: str,
    source: str,
    max_words: int = 120,
    overlap_words: int = 30,
) -> list[Chunk]:
    """Split text into overlapping word windows, respecting paragraph breaks.

    Paragraphs are packed greedily into windows of max_words; when a window
    is full, the last overlap_words words are carried into the next window
    so that facts spanning a boundary remain retrievable.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be in [0, max_words)")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    window: list[str] = []
    title = None
    match = _HEADING.search(text)
    if match:
        title = match.group(1).strip()

    def flush() -> None:
        if not window:
            return
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{source}#{idx}",
                source=source,
                text=" ".join(window),
                metadata={"title": title, "index": idx},
            )
        )

    for para in paragraphs:
        if _HEADING.fullmatch(para):
            continue  # headings live in metadata["title"], not in the body
        words = para.split()
        while words:
            room = max_words - len(window)
            window.extend(words[:room])
            words = words[room:]
            if len(window) >= max_words:
                flush()
                window = window[-overlap_words:] if overlap_words else []
    # Avoid emitting a trailing chunk that is pure overlap.
    if window and (not chunks or len(window) > overlap_words):
        flush()
    return chunks


def load_corpus(directory: str | Path, **chunk_kwargs) -> list[Chunk]:
    """Load every .md / .txt file under directory into chunks."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".md", ".txt"} and path.is_file():
            text = path.read_text(encoding="utf-8")
            chunks.extend(split_into_chunks(text, source=path.name, **chunk_kwargs))
    return chunks
