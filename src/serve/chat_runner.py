"""PyTorch chat inference runner for Forge-trained models.

This module loads a trained model checkpoint and generates text
responses from prompt input using training-compatible settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from core.chat_types import ChatOptions, ChatResult
from core.errors import ForgeDependencyError, ForgeServeError
from core.types import DataRecord
from serve.architecture_loader import load_training_model
from serve.chat_option_resolver import (
    resolve_chat_model_vocab_size,
    resolve_chat_tokenizer,
    resolve_chat_training_options,
)
from serve.device_selection import resolve_execution_device
from serve.model_weights import load_initial_weights, read_model_state_dict
from serve.training_setup import validate_training_options


@dataclass
class ChatRuntimeContext:
    """Runtime objects needed for chat text generation."""

    torch_module: Any
    model: Any
    tokenizer: Any
    options: ChatOptions
    device: Any
    max_context_tokens: int


def run_chat(records: list[DataRecord], options: ChatOptions) -> ChatResult:
    """Run one model chat inference call.

    Args:
        records: Dataset records used to reconstruct tokenizer vocabulary.
        options: Chat inference options.

    Returns:
        Generated response text.

    Raises:
        ForgeServeError: If model loading or generation fails.
    """
    context = _build_runtime_context(records, options)
    response_text = _generate_response_text(context)
    return ChatResult(response_text=response_text)


def _build_runtime_context(
    records: list[DataRecord],
    options: ChatOptions,
) -> ChatRuntimeContext:
    """Build chat runtime context from dataset records and options."""
    torch_module = _import_torch()
    _validate_chat_options(options)
    device = _resolve_inference_device(torch_module)
    model_state = read_model_state_dict(torch_module, options.model_path, device)
    training_options = resolve_chat_training_options(options, model_state)
    validate_training_options(training_options)
    tokenizer = resolve_chat_tokenizer(records, options, training_options)
    model = load_training_model(
        torch_module,
        training_options,
        resolve_chat_model_vocab_size(tokenizer.vocabulary, model_state, training_options),
    )
    model = model.to(device)
    load_initial_weights(
        torch_module=torch_module,
        model=model,
        initial_weights_path=options.model_path,
        device=device,
    )
    model.eval()
    max_context_tokens = _resolve_runtime_context_limit(model, training_options.max_token_length)
    return ChatRuntimeContext(
        torch_module=torch_module,
        model=model,
        tokenizer=tokenizer,
        options=options,
        device=device,
        max_context_tokens=max_context_tokens,
    )


def _import_torch() -> Any:
    """Import torch dependency."""
    try:
        import torch
    except ImportError as error:
        raise ForgeDependencyError(
            "Chat inference requires torch, but it is not installed. Install torch to run forge chat."
        ) from error
    return torch


def _resolve_inference_device(torch_module: Any) -> Any:
    """Select torch device for inference."""
    return resolve_execution_device(torch_module)


def _validate_chat_options(options: ChatOptions) -> None:
    """Validate chat-specific option fields."""
    if not options.prompt.strip():
        raise ForgeServeError(
            "Invalid prompt: expected non-empty input text. Provide --prompt with message content."
        )
    if options.max_new_tokens < 1:
        raise ForgeServeError(
            f"Invalid max_new_tokens {options.max_new_tokens}: expected value >= 1."
        )
    if options.temperature < 0:
        raise ForgeServeError(f"Invalid temperature {options.temperature}: expected value >= 0.")
    if options.top_k < 0:
        raise ForgeServeError(f"Invalid top_k {options.top_k}: expected value >= 0.")


def _generate_response_text(context: ChatRuntimeContext) -> str:
    """Generate response text from prompt with autoregressive decoding."""
    options = context.options
    prompt_ids = context.tokenizer.encode(options.prompt, context.max_context_tokens)
    if not prompt_ids:
        prompt_ids = [1]
    generated_ids: list[int] = []
    context_ids = list(prompt_ids)
    for _ in range(options.max_new_tokens):
        next_token_id = _sample_next_token(context, context_ids)
        if next_token_id == 0:
            break
        context_ids.append(next_token_id)
        generated_ids.append(next_token_id)
    if not generated_ids:
        return ""
    decoded = cast(str, context.tokenizer.decode(generated_ids))
    return decoded.strip()


def _sample_next_token(context: ChatRuntimeContext, context_ids: list[int]) -> int:
    """Sample the next token id using configured decoding settings."""
    torch_module = context.torch_module
    options = context.options
    input_ids = context_ids[-context.max_context_tokens :]
    input_tensor = torch_module.tensor([input_ids], dtype=torch_module.long).to(context.device)
    with torch_module.no_grad():
        logits = context.model(input_tensor)
    next_logits = logits[0, -1, :]
    if options.temperature == 0:
        return int(torch_module.argmax(next_logits).item())
    scaled_logits = next_logits / options.temperature
    if options.top_k > 0:
        top_k = min(options.top_k, int(scaled_logits.shape[-1]))
        values, indices = torch_module.topk(scaled_logits, top_k)
        probabilities = torch_module.softmax(values, dim=-1)
        sampled_position = int(torch_module.multinomial(probabilities, num_samples=1).item())
        return int(indices[sampled_position].item())
    probabilities = torch_module.softmax(scaled_logits, dim=-1)
    return int(torch_module.multinomial(probabilities, num_samples=1).item())


def _resolve_runtime_context_limit(model: Any, fallback_limit: int) -> int:
    """Resolve the usable max context length for inference."""
    position_embedding = getattr(model, "position_embedding", None)
    if position_embedding is not None:
        num_embeddings = getattr(position_embedding, "num_embeddings", None)
        if isinstance(num_embeddings, int) and num_embeddings > 0:
            return num_embeddings
    sinusoidal = getattr(model, "sinusoidal_position_encoding", None)
    if sinusoidal is not None:
        shape = getattr(sinusoidal, "shape", None)
        if shape is not None and len(shape) > 0 and int(shape[0]) > 0:
            return int(shape[0])
    return fallback_limit
