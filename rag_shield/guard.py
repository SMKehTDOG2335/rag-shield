"""Prompt-injection guard.

Scans user queries (direct injection) and retrieved documents (indirect
injection) for patterns commonly used to hijack LLM behaviour. It is a
transparent, rule-based first line of defence: every decision comes with the
rules that fired, so it is auditable and cheap to run on every chunk.

It is *not* a complete defence on its own. In the pipeline it is layered with
context spotlighting, a system-prompt canary and citation checks.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    weight: float
    description: str


def _r(name: str, regex: str, weight: float, description: str) -> Rule:
    return Rule(name, re.compile(regex, re.IGNORECASE | re.DOTALL), weight, description)


RULES: tuple[Rule, ...] = (
    _r(
        "instruction_override",
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|any|your|system)\b.{0,30}\b(instructions?|prompts?|rules|guidelines|directions)\b",
        1.0,
        "Attempts to cancel the model's existing instructions",
    ),
    _r(
        "new_instructions",
        r"\b(new|updated|real|actual)\s+(instructions?|system\s+prompt|rules)\s*[:\-]",
        0.8,
        "Declares replacement instructions",
    ),
    _r(
        "role_hijack",
        r"\b(you are now|from now on,? you|act as|pretend to be|roleplay as|you must now)\b",
        0.6,
        "Tries to assign the model a new persona or mode",
    ),
    _r(
        "jailbreak_mode",
        r"\b(DAN|developer mode|jailbreak|god mode|unfiltered mode|no restrictions)\b",
        0.9,
        "Known jailbreak vocabulary",
    ),
    _r(
        "prompt_exfiltration",
        r"\b(reveal|print|show|output|repeat|leak|tell me)\b.{0,30}\b(system prompt|hidden prompt|initial instructions|your instructions|the prompt above)\b",
        1.0,
        "Asks the model to disclose its system prompt",
    ),
    _r(
        "data_exfiltration",
        r"\b(send|post|forward|upload|exfiltrate|transmit|email)\b.{0,60}(https?://|\b[\w.-]+@[\w-]+\.\w+)",
        0.9,
        "Instructs sending data to an external destination",
    ),
    _r(
        "markdown_image_beacon",
        r"!\[[^\]]*\]\(https?://[^)]*\?[^)]*=",
        0.9,
        "Markdown image with query string (classic exfiltration beacon)",
    ),
    _r(
        "hidden_instruction_markup",
        r"<!--.{0,400}?(instruction|assistant|ai|model|llm|ignore|system).{0,400}?-->",
        0.7,
        "Instructions hidden in HTML comments",
    ),
    _r(
        "fake_chat_turn",
        r"(^|\n)\s*(system|assistant)\s*:|<\|?(system|im_start)\|?>|\[/?INST\]",
        0.7,
        "Forged chat-role delimiters",
    ),
    _r(
        "addressed_to_ai",
        r"\b(note|message|attention|instructions?)\s+(to|for)\s+(the\s+)?(ai|assistant|llm|language model|chatbot|model)\b",
        0.6,
        "Text explicitly addressed to an AI reader",
    ),
    _r(
        "encoded_payload",
        r"\b(base64|rot13|decode (this|the following))\b",
        0.4,
        "Mentions an encoding often used to smuggle payloads",
    ),
)


@dataclass
class GuardResult:
    flagged: bool
    score: float
    matches: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)

    def explain(self) -> str:
        if not self.flagged:
            return "clean"
        return f"score={self.score:.2f} rules={', '.join(self.matches + self.signals)}"


def normalize(text: str) -> tuple[str, list[str]]:
    """NFKC-normalise and strip zero-width characters, reporting what was found."""
    signals: list[str] = []
    if ZERO_WIDTH.search(text):
        signals.append("zero_width_chars")
    cleaned = ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))
    return cleaned, signals


HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_SEGMENT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class SanitizeResult:
    text: str
    removed: list[tuple[str, list[str]]] = field(default_factory=list)


class InjectionGuard:
    def __init__(self, threshold: float = 0.6, rules: tuple[Rule, ...] = RULES):
        self.threshold = threshold
        self.rules = rules

    def sanitize(self, text: str) -> SanitizeResult:
        """Remove injected segments while keeping the legitimate content.

        Hidden HTML comments are dropped wholesale (a human reader never sees
        them, so they carry no legitimate content for the answer). The rest is
        split into sentences and any sentence that trips a rule on its own is
        removed. Callers should re-scan the result before trusting it.
        """
        removed: list[tuple[str, list[str]]] = []
        cleaned, _ = normalize(text)
        for comment in HTML_COMMENT.findall(cleaned):
            removed.append((comment, ["html_comment"]))
        cleaned = HTML_COMMENT.sub(" ", cleaned)
        kept: list[str] = []
        for segment in _SEGMENT.split(cleaned):
            segment = segment.strip()
            if not segment:
                continue
            hits = [r.name for r in self.rules if r.pattern.search(segment)]
            if hits:
                removed.append((segment, hits))
            else:
                kept.append(segment)
        return SanitizeResult(" ".join(kept), removed)

    def scan(self, text: str) -> GuardResult:
        cleaned, signals = normalize(text)
        score = 0.3 * len(signals)
        matches: list[str] = []
        for rule in self.rules:
            if rule.pattern.search(cleaned):
                matches.append(rule.name)
                score += rule.weight
        return GuardResult(
            flagged=score >= self.threshold,
            score=round(score, 3),
            matches=matches,
            signals=signals,
        )
