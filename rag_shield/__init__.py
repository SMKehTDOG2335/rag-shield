"""rag-shield: a retrieval-augmented generation pipeline hardened against prompt injection."""
from .chunking import Chunk, load_corpus, split_into_chunks
from .classifier import LayeredGuard, OnnxInjectionClassifier
from .dense import DenseRetriever, FastEmbedEmbedder, HybridRetriever
from .guard import GuardResult, InjectionGuard
from .llm import ExtractiveLLM, OpenAICompatibleLLM, llm_from_env
from .pipeline import Answer, SecureRAG
from .retrieval import BM25Retriever

__all__ = [
    "Answer",
    "BM25Retriever",
    "Chunk",
    "DenseRetriever",
    "ExtractiveLLM",
    "FastEmbedEmbedder",
    "GuardResult",
    "HybridRetriever",
    "InjectionGuard",
    "LayeredGuard",
    "OnnxInjectionClassifier",
    "OpenAICompatibleLLM",
    "SecureRAG",
    "llm_from_env",
    "load_corpus",
    "split_into_chunks",
]
__version__ = "0.3.0"
