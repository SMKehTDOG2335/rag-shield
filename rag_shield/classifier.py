"""ML classifier stage for the injection guard.

Rules are fast, explainable and precise, but only catch phrasings someone has
written a pattern for. A fine-tuned transformer generalises to paraphrases and
other languages, at the cost of latency and explainability. LayeredGuard runs
both: rules first (cheap, auditable), and the classifier on whatever the rules
let through.

Default model: protectai/deberta-v3-base-prompt-injection-v2 (Apache-2.0),
run through ONNX Runtime on CPU, so no PyTorch is needed. Optional extra:
pip install rag-shield[ml]
"""
from __future__ import annotations

import math
from typing import Protocol, Sequence

from .guard import _SEGMENT, HTML_COMMENT, GuardResult, InjectionGuard, SanitizeResult, normalize

DEFAULT_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


class InjectionClassifier(Protocol):
    name: str

    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        """Probability that each text is a prompt injection."""
        ...


class OnnxInjectionClassifier:
    """Sequence classifier exported to ONNX (label 1 = injection)."""

    def __init__(
        self,
        repo_id: str = DEFAULT_MODEL,
        subfolder: str = "onnx",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        try:
            import numpy as np
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "The ML guard needs the optional extra: pip install 'rag-shield[ml]'"
            ) from exc

        self._np = np
        self.name = f"onnx:{repo_id}"
        self.batch_size = batch_size
        model_path = hf_hub_download(repo_id, f"{subfolder}/model.onnx")
        tokenizer_path = hf_hub_download(repo_id, f"{subfolder}/tokenizer.json")
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=max_length)
        pad_id = self._tokenizer.token_to_id("[PAD]") or 0
        self._tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}

    def predict_proba(self, texts: Sequence[str]) -> list[float]:
        np = self._np
        probs: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = self._tokenizer.encode_batch(list(texts[start : start + self.batch_size]))
            feeds = {
                "input_ids": np.array([e.ids for e in batch], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in batch], dtype=np.int64),
            }
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
            logits = self._session.run(None, {k: v for k, v in feeds.items() if k in self._input_names})[0]
            for row in logits:
                safe, injection = float(row[0]), float(row[1])
                # Numerically stable two-class softmax.
                probs.append(1.0 / (1.0 + math.exp(safe - injection)))
        return probs


class LayeredGuard(InjectionGuard):
    """Rules first, then an ML classifier on what the rules did not flag."""

    def __init__(
        self,
        classifier: InjectionClassifier,
        ml_threshold: float = 0.5,
        **rule_kwargs,
    ):
        super().__init__(**rule_kwargs)
        self.classifier = classifier
        self.ml_threshold = ml_threshold

    def scan(self, text: str) -> GuardResult:
        result = super().scan(text)
        if result.flagged:
            return result  # rules are cheaper and explain themselves
        cleaned, _ = normalize(text)
        prob = self.classifier.predict_proba([cleaned])[0]
        result.ml_score = round(prob, 4)
        if prob >= self.ml_threshold:
            result.flagged = True
            result.signals.append("ml_classifier")
        return result

    def sanitize(self, text: str) -> SanitizeResult:
        """Rule-based sanitisation, then drop any remaining sentence the model flags."""
        first = super().sanitize(text)
        segments = [s.strip() for s in _SEGMENT.split(HTML_COMMENT.sub(" ", first.text)) if s.strip()]
        if not segments:
            return first
        probs = self.classifier.predict_proba(segments)
        kept, removed = [], list(first.removed)
        for segment, prob in zip(segments, probs):
            if prob >= self.ml_threshold:
                removed.append((segment, ["ml_classifier"]))
            else:
                kept.append(segment)
        return SanitizeResult(" ".join(kept), removed)
