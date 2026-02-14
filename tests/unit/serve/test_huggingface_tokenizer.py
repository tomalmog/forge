"""Unit tests for HuggingFace tokenizer wrapper."""

from __future__ import annotations

from core.chat_types import ChatTokenizer
from serve.huggingface_tokenizer import HuggingFaceTokenizer


class _FakeEncoding:
    """Minimal encoding result stub."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTokenizer:
    """Minimal HuggingFace tokenizer stub."""

    def get_vocab(self) -> dict[str, int]:
        return {"hello": 0, "world": 1}

    def encode(self, text: str) -> _FakeEncoding:
        _ = text
        return _FakeEncoding(ids=[0, 1, 0])

    def decode(self, ids: list[int]) -> str:
        _ = ids
        return "hello world hello"


def test_huggingface_tokenizer_encode_respects_max_length() -> None:
    """Encode should truncate to max_token_length."""
    tokenizer = HuggingFaceTokenizer(_FakeTokenizer())

    result = tokenizer.encode("anything", max_token_length=2)

    assert result == [0, 1]


def test_huggingface_tokenizer_decode_returns_string() -> None:
    """Decode should return the decoded text string."""
    tokenizer = HuggingFaceTokenizer(_FakeTokenizer())

    result = tokenizer.decode([0, 1])

    assert result == "hello world hello"


def test_huggingface_tokenizer_exposes_vocabulary() -> None:
    """Vocabulary property should reflect the underlying tokenizer vocab."""
    tokenizer = HuggingFaceTokenizer(_FakeTokenizer())

    assert tokenizer.vocabulary == {"hello": 0, "world": 1}


def test_huggingface_tokenizer_satisfies_chat_tokenizer_protocol() -> None:
    """HuggingFaceTokenizer should be a valid ChatTokenizer."""
    tokenizer = HuggingFaceTokenizer(_FakeTokenizer())

    assert isinstance(tokenizer, ChatTokenizer)
