"""PyTorch training runner for Forge datasets.

This module executes the default training cycle and supports
user-defined custom loops with shared runtime context.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from core.errors import ForgeDependencyError, ForgeServeError
from core.types import BatchLossMetric, DataRecord, EpochMetric, TrainingOptions, TrainingRunResult
from serve.architecture_loader import load_training_model
from serve.custom_loop_loader import load_custom_training_loop
from serve.model_weights import load_initial_weights
from serve.tokenization import (
    SequenceBatch,
    build_sequence_batches,
    build_training_sequences,
    split_sequences,
)
from serve.training_artifacts import (
    ensure_training_output_dir,
    save_model_weights,
    save_training_history,
    save_training_plot,
)
from serve.training_context import TrainingRuntimeContext
from serve.training_setup import fit_training_tokenizer, validate_training_options


def run_training(
    records: list[DataRecord],
    options: TrainingOptions,
    random_seed: int,
) -> TrainingRunResult:
    """Run training workflow and persist artifacts.

    Args:
        records: Dataset records used for training.
        options: Training options.
        random_seed: Deterministic random seed.

    Returns:
        Training artifact summary.

    Raises:
        ForgeServeError: If training setup or loop fails.
    """
    context = _build_runtime_context(records, options, random_seed)
    epoch_metrics, batch_metrics = _run_training_loop(context)
    return _persist_training_outputs(context, epoch_metrics, batch_metrics)


def _build_runtime_context(
    records: list[DataRecord],
    options: TrainingOptions,
    random_seed: int,
) -> TrainingRuntimeContext:
    """Build training runtime context from records and options."""
    torch_module = _import_torch()
    validate_training_options(options)
    output_dir = ensure_training_output_dir(options.output_dir)
    tokenizer = fit_training_tokenizer(records, options)
    sequences = build_training_sequences(records, tokenizer, options.max_token_length)
    if not sequences:
        raise ForgeServeError(
            "No trainable sequences were generated from dataset records. "
            "Check dataset content and max token length."
        )
    random.Random(random_seed).shuffle(sequences)
    train_batches, validation_batches = _build_batches(sequences, options)
    model = load_training_model(torch_module, options, len(tokenizer.vocabulary))
    device = _resolve_training_device(torch_module)
    model = model.to(device)
    load_initial_weights(
        torch_module=torch_module,
        model=model,
        initial_weights_path=options.initial_weights_path,
        device=device,
    )
    optimizer = torch_module.optim.Adam(model.parameters(), lr=options.learning_rate)
    loss_function = torch_module.nn.CrossEntropyLoss(ignore_index=0)
    return TrainingRuntimeContext(
        torch_module=torch_module,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_batches=train_batches,
        validation_batches=validation_batches,
        options=options,
        output_dir=output_dir,
        device=device,
    )


def _persist_training_outputs(
    context: TrainingRuntimeContext,
    epoch_metrics: list[EpochMetric],
    batch_metrics: list[BatchLossMetric],
) -> TrainingRunResult:
    """Persist model/history/plot outputs and return summary."""
    model_path = save_model_weights(context.output_dir, context.torch_module, context.model)
    history_path = save_training_history(context.output_dir, epoch_metrics, batch_metrics)
    plot_path = _try_save_plot(context.output_dir, epoch_metrics, batch_metrics)
    return TrainingRunResult(
        model_path=str(model_path),
        history_path=str(history_path),
        plot_path=str(plot_path) if plot_path else None,
        epochs_completed=len(epoch_metrics),
    )


def _import_torch() -> Any:
    """Import torch dependency.

    Returns:
        Imported torch module.

    Raises:
        ForgeDependencyError: If torch is unavailable.
    """
    try:
        import torch
    except ImportError as error:
        raise ForgeDependencyError(
            "Training requires torch, but it is not installed. Install torch to run forge train."
        ) from error
    return torch


def _build_batches(
    sequences: list[list[int]],
    options: TrainingOptions,
) -> tuple[list[SequenceBatch], list[SequenceBatch]]:
    """Build train and validation sequence batches."""
    train_sequences, validation_sequences = split_sequences(
        sequences,
        options.validation_split,
    )
    train_batches = build_sequence_batches(train_sequences, options.batch_size)
    validation_batches = build_sequence_batches(validation_sequences, options.batch_size)
    return train_batches, validation_batches


def _resolve_training_device(torch_module: Any) -> Any:
    """Select torch device for training."""
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def _run_training_loop(
    context: TrainingRuntimeContext,
) -> tuple[list[EpochMetric], list[BatchLossMetric]]:
    """Run default or custom training loop."""
    custom_loop = load_custom_training_loop(context.options.custom_loop_path)
    if custom_loop is None:
        return _run_default_training_loop(context)
    metrics = custom_loop(context)
    _validate_metric_rows(metrics)
    return metrics, []


def _run_default_training_loop(
    context: TrainingRuntimeContext,
) -> tuple[list[EpochMetric], list[BatchLossMetric]]:
    """Run the built-in epoch training loop."""
    epoch_rows: list[EpochMetric] = []
    batch_rows: list[BatchLossMetric] = []
    global_step = 0
    for epoch_index in range(1, context.options.epochs + 1):
        train_loss, global_step = _run_epoch_pass(
            context,
            context.train_batches,
            training=True,
            epoch_index=epoch_index,
            global_step=global_step,
            batch_rows=batch_rows,
        )
        validation_loss, global_step = _run_epoch_pass(
            context,
            context.validation_batches,
            training=False,
            epoch_index=epoch_index,
            global_step=global_step,
            batch_rows=batch_rows,
        )
        epoch_rows.append(
            EpochMetric(
                epoch=epoch_index,
                train_loss=round(train_loss, 6),
                validation_loss=round(validation_loss, 6),
            )
        )
    return epoch_rows, batch_rows


def _run_epoch_pass(
    context: TrainingRuntimeContext,
    batches: list[SequenceBatch],
    training: bool,
    epoch_index: int,
    global_step: int,
    batch_rows: list[BatchLossMetric],
) -> tuple[float, int]:
    """Run one full pass over batches."""
    if not batches:
        return 0.0, global_step
    total_loss = 0.0
    context.model.train(mode=training)
    for batch_index, batch in enumerate(batches, start=1):
        inputs, targets = _tensorize_batch(context, batch)
        logits = context.model(inputs)
        loss = context.loss_function(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        loss_value = float(loss.item())
        total_loss += loss_value
        if training:
            context.optimizer.zero_grad()
            loss.backward()
            context.optimizer.step()
            global_step += 1
            batch_rows.append(
                BatchLossMetric(
                    epoch=epoch_index,
                    batch_index=batch_index,
                    global_step=global_step,
                    train_loss=round(loss_value, 6),
                )
            )
    return total_loss / len(batches), global_step


def _tensorize_batch(
    context: TrainingRuntimeContext,
    batch: SequenceBatch,
) -> tuple[Any, Any]:
    """Convert batch lists to padded torch tensors."""
    torch_module = context.torch_module
    max_length = max(len(sequence) for sequence in batch.inputs)
    padded_inputs = [_pad_sequence(sequence, max_length) for sequence in batch.inputs]
    padded_targets = [_pad_sequence(sequence, max_length) for sequence in batch.targets]
    input_tensor = torch_module.tensor(padded_inputs, dtype=torch_module.long).to(context.device)
    target_tensor = torch_module.tensor(padded_targets, dtype=torch_module.long).to(context.device)
    return input_tensor, target_tensor


def _pad_sequence(sequence: list[int], max_length: int) -> list[int]:
    """Pad sequence to max length with pad token id 0."""
    if len(sequence) >= max_length:
        return sequence
    return sequence + ([0] * (max_length - len(sequence)))


def _validate_metric_rows(metrics: list[EpochMetric]) -> None:
    """Validate metric list from custom loop."""
    if not metrics:
        raise ForgeServeError(
            "Custom loop returned no metrics. Return at least one EpochMetric row."
        )


def _try_save_plot(
    output_dir: Path,
    epoch_metrics: list[EpochMetric],
    batch_metrics: list[BatchLossMetric],
) -> Path | None:
    """Save training plot, skipping when matplotlib is missing."""
    try:
        return save_training_plot(output_dir, epoch_metrics, batch_metrics)
    except ForgeDependencyError:
        return None
