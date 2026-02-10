"""Unit tests for perplexity-based quality scoring."""

from __future__ import annotations

from transforms.perplexity_quality import score_texts_with_perplexity


def test_score_texts_with_perplexity_returns_pair_per_text() -> None:
    """Scorer should return one pair per input text."""
    scores = score_texts_with_perplexity([
        "The model is trained with clean data",
        "Clean data helps model quality",
    ])

    assert len(scores) == 2


def test_score_texts_with_perplexity_normalizes_scores_to_unit_interval() -> None:
    """Quality score should stay within 0 and 1."""
    _, quality = score_texts_with_perplexity([
        "alpha beta gamma",
        "alpha alpha alpha",
    ])[0]

    assert 0.0 <= quality <= 1.0
