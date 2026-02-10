"""Unit tests for language detection transform."""

from __future__ import annotations

from transforms.language_detection import detect_language


def test_detect_language_returns_en_for_english_text() -> None:
    """Detector should classify common English text as en."""
    language = detect_language("The dataset is ready for training and evaluation")

    assert language == "en"


def test_detect_language_returns_unknown_for_non_ascii_text() -> None:
    """Detector should classify low-ascii text as unknown."""
    language = detect_language("こんにちは 世界")

    assert language == "unknown"
