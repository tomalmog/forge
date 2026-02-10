"""Perplexity-based quality scoring transform.

This module builds a unigram language model from the ingest batch.
It assigns perplexity and normalized quality scores to each document.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable


def score_texts_with_perplexity(texts: Iterable[str]) -> list[tuple[float, float]]:
    """Compute perplexity and normalized quality scores.

    Args:
        texts: Input texts in snapshot order.

    Returns:
        List of ``(perplexity, quality_score)`` tuples.
    """
    text_list = list(texts)
    tokenized_texts = [_tokenize_text(text) for text in text_list]
    token_probabilities, unknown_probability = _build_unigram_model(tokenized_texts)
    perplexities = [
        _compute_perplexity(tokens, token_probabilities, unknown_probability)
        for tokens in tokenized_texts
    ]
    return _normalize_perplexity_scores(perplexities)


def _tokenize_text(text: str) -> list[str]:
    """Tokenize text into simple lowercase terms.

    Args:
        text: Raw input text.

    Returns:
        Ordered token list.
    """
    cleaned = "".join(character if character.isalnum() else " " for character in text)
    return [token for token in cleaned.lower().split() if token]


def _build_unigram_model(tokenized_texts: list[list[str]]) -> tuple[dict[str, float], float]:
    """Build unigram probabilities with add-one smoothing.

    Args:
        tokenized_texts: Tokenized documents.

    Returns:
        Token probability map and default unknown token probability.
    """
    token_counter: Counter[str] = Counter()
    for tokens in tokenized_texts:
        token_counter.update(tokens)
    vocabulary_size = max(len(token_counter), 1)
    total_tokens = sum(token_counter.values()) + vocabulary_size
    probability_map = {
        token: (count + 1) / total_tokens for token, count in token_counter.items()
    }
    unknown_probability = 1 / total_tokens
    return probability_map, unknown_probability


def _compute_perplexity(
    tokens: list[str],
    token_probabilities: dict[str, float],
    unknown_probability: float,
) -> float:
    """Compute text perplexity from a unigram model.

    Args:
        tokens: Tokenized text.
        token_probabilities: Unigram probabilities.
        unknown_probability: Smoothed fallback probability.

    Returns:
        Positive perplexity value.
    """
    if not tokens:
        return 1.0
    negative_log_likelihood = 0.0
    for token in tokens:
        probability = token_probabilities.get(token, unknown_probability)
        negative_log_likelihood += -math.log(probability)
    average_neg_log_likelihood = negative_log_likelihood / len(tokens)
    return math.exp(average_neg_log_likelihood)


def _normalize_perplexity_scores(perplexities: list[float]) -> list[tuple[float, float]]:
    """Convert perplexities to 0-1 quality scores.

    Args:
        perplexities: Perplexity scores in document order.

    Returns:
        Pairs of raw perplexity and normalized quality score.
    """
    if not perplexities:
        return []
    min_perplexity = min(perplexities)
    max_perplexity = max(perplexities)
    if math.isclose(min_perplexity, max_perplexity):
        return [(value, 1.0) for value in perplexities]
    score_pairs: list[tuple[float, float]] = []
    denominator = max_perplexity - min_perplexity
    for perplexity in perplexities:
        normalized = 1 - ((perplexity - min_perplexity) / denominator)
        score_pairs.append((perplexity, round(normalized, 6)))
    return score_pairs
