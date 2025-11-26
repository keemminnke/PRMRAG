"""Utilities for extracting and normalizing answers from model outputs."""

import re
import string
from collections import Counter


def _clean_answer_span(span: str) -> str:
    """Trim whitespace/brackets and trailing periods."""
    span = span.strip()
    # Remove wrapping parentheses/spaces
    span = re.sub(r'^[()\s]+|[()\s]+$', '', span)
    # Drop trailing period but keep abbreviations intact (they'll be handled by regexes)
    return span.rstrip('.').strip()


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text.strip()


def extract_answer_from_text(text: str) -> str:
    """Extract the most plausible answer span from model output.

    Priority:
    1) After explicit "Final Answer:" marker (supports multiline)
    2) Common answer phrasings ("the answer is", "therefore, the answer is", etc.)
    3) First non-empty line that is not an intermediate/next-query marker
    """
    if not text:
        return ""

    text = text.strip()

    # 1) Explicit Final Answer (multi-line safe)
    final_match = re.search(r'final\s+answer\s*:?\s*(.+)', text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        remainder = final_match.group(1)
        for line in remainder.splitlines():
            candidate = _clean_answer_span(line)
            if candidate:
                return candidate
        candidate = _clean_answer_span(remainder)
        if candidate:
            return candidate

    # 2) Common phrasings (cross-line)
    answer_patterns = [
        r'therefore,?\s+(?:the\s+answer\s+is\s*:?\s*)?(.+?)(?:(?<![A-Z])\.(?=\s+[A-Z])|[;]|\s+(?:because|since)\s+|$)',
        r'(?:the\s+)?answer\s+is\s*:?\s*(.+?)(?:(?<![A-Z])\.(?=\s+[A-Z])|[;]|\s+(?:because|since)\s+|$)',
        r'(?:in\s+)?conclusion,?\s+(.+?)(?:(?<![A-Z])\.(?=\s+[A-Z])|[;]|\s+(?:because|since)\s+|$)',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidate = _clean_answer_span(match.group(1))
            if candidate:
                return candidate

    # 3) Parenthetical tail
    paren_match = re.search(r'\(([^)]+)\)[.,;]?\s*$', text, flags=re.DOTALL)
    if paren_match:
        candidate = _clean_answer_span(paren_match.group(1))
        if candidate:
            return candidate

    # 4) Fallback: first non-empty line that is not an intermediate marker
    skip_prefixes = ("intermediate answer", "missing info", "next query target")
    for line in text.splitlines():
        if not line.strip():
            continue
        lower_line = line.strip().lower()
        if lower_line.startswith(skip_prefixes):
            continue
        candidate = _clean_answer_span(line)
        if candidate:
            return candidate

    return ""


def check_answer_match(predicted: str, gold: str, threshold: float = 0.8) -> bool:
    """Check if predicted answer matches gold answer using token overlap.

    Args:
        predicted: Predicted answer (normalized)
        gold: Gold answer (normalized)
        threshold: Minimum token overlap ratio (default: 0.8 = 80%)

    Returns:
        True if token overlap >= threshold
    """
    # Exact match first (fast path)
    if predicted == gold:
        return True

    # Token-level matching
    gold_tokens = gold.split()
    pred_tokens = predicted.split()

    if not gold_tokens:
        return False

    # Count how many gold tokens appear in prediction
    matches = sum(1 for token in gold_tokens if token in pred_tokens)
    accuracy = matches / len(gold_tokens)

    return accuracy >= threshold
