"""Language detection transform.

This module implements a lightweight heuristic language detector.
It classifies records for metadata filtering in phase-one workflows.
"""

from __future__ import annotations

from typing import Iterable

from core.constants import DEFAULT_LANGUAGE_CODE, ENGLISH_LANGUAGE_CODE

_ENGLISH_STOPWORDS = {
    "the",
    "and",
    "is",
    "in",
    "to",
    "of",
    "that",
    "for",
    "with",
    "on",
    "as",
    "this",
    "it",
    "by",
    "an",
}


def detect_language(text: str) -> str:
    """Detect language code for a text sample.

    Args:
        text: Input text to classify.

    Returns:
        ISO-like language code, currently "en" or "unknown".
    """
    tokens = _tokenize(text)
    if not tokens:
        return DEFAULT_LANGUAGE_CODE
    ascii_ratio = _compute_ascii_ratio(text)
    stopword_hits = _count_stopword_hits(tokens)
    if ascii_ratio < 0.9:
        return DEFAULT_LANGUAGE_CODE
    if stopword_hits / len(tokens) >= 0.05:
        return ENGLISH_LANGUAGE_CODE
    return DEFAULT_LANGUAGE_CODE


def detect_languages(texts: Iterable[str]) -> list[str]:
    """Detect language codes for multiple texts.

    Args:
        texts: Iterable of text documents.

    Returns:
        List of language codes aligned to the input order.
    """
    return [detect_language(text) for text in texts]


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word-like tokens.

    Args:
        text: Input text.

    Returns:
        Token list.
    """
    cleaned = "".join(character if character.isalpha() else " " for character in text)
    return [token for token in cleaned.lower().split() if token]


def _compute_ascii_ratio(text: str) -> float:
    """Compute the fraction of ASCII characters.

    Args:
        text: Input text.

    Returns:
        Ratio between 0 and 1.
    """
    if not text:
        return 0.0
    ascii_count = sum(1 for character in text if ord(character) < 128)
    return ascii_count / len(text)


def _count_stopword_hits(tokens: list[str]) -> int:
    """Count English stopword matches in token list.

    Args:
        tokens: Tokenized words.

    Returns:
        Number of stopword tokens.
    """
    return sum(1 for token in tokens if token in _ENGLISH_STOPWORDS)
