"""Unit tests for vocabulary tokenizer behavior."""

from __future__ import annotations

from serve.tokenization import VocabularyTokenizer


def test_vocabulary_tokenizer_respects_max_vocabulary_size() -> None:
    """Tokenizer fit should stop adding tokens at configured max size."""
    tokenizer = VocabularyTokenizer.create()
    tokenizer.fit(["one two three four"], max_vocabulary_size=4)

    assert len(tokenizer.vocabulary) == 4


def test_vocabulary_tokenizer_encodes_omitted_tokens_as_unknown() -> None:
    """Tokenizer encode should map unseen tokens to unknown id."""
    tokenizer = VocabularyTokenizer.create()
    tokenizer.fit(["one two three"], max_vocabulary_size=3)
    encoded = tokenizer.encode("one three", max_token_length=10)

    assert encoded == [2, 1]
