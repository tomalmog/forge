"""Train command wiring for Forge CLI.

This module isolates train command parser and execution logic.
It keeps the top-level CLI module focused and within size constraints.
"""

from __future__ import annotations

import argparse
from typing import Any

from core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_TOKEN_LENGTH,
    DEFAULT_TRAIN_ATTENTION_HEADS,
    DEFAULT_TRAIN_DROPOUT,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_HIDDEN_DIM,
    DEFAULT_TRAIN_LEARNING_RATE,
    DEFAULT_TRAIN_MLP_HIDDEN_DIM,
    DEFAULT_TRAIN_MLP_LAYERS,
    DEFAULT_TRAIN_NUM_LAYERS,
    DEFAULT_TRAIN_VALIDATION_SPLIT,
)
from core.types import TrainingOptions
from store.dataset_sdk import ForgeClient


def run_train_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Handle train command invocation."""
    options = TrainingOptions(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        version_id=args.version_id,
        architecture_path=args.architecture_file,
        custom_loop_path=args.custom_loop_file,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_token_length=args.max_token_length,
        validation_split=args.validation_split,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        attention_heads=args.attention_heads,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout,
        vocabulary_size=args.vocabulary_size,
        initial_weights_path=args.initial_weights_path,
    )
    result = client.train(options)
    print(f"model_path={result.model_path}")
    print(f"history_path={result.history_path}")
    print(f"plot_path={result.plot_path or '-'}")
    print(f"epochs_completed={result.epochs_completed}")
    return 0


def add_train_command(subparsers: Any) -> None:
    """Register train subcommand."""
    parser = subparsers.add_parser(
        "train", help="Train a PyTorch language model on a dataset version"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--output-dir", required=True, help="Training artifact output directory")
    parser.add_argument("--version-id", help="Optional specific version id")
    parser.add_argument("--architecture-file", help="Optional .py or .json model architecture file")
    parser.add_argument("--custom-loop-file", help="Optional .py custom training loop file")
    parser.add_argument(
        "--initial-weights-path", help="Optional model.pt path used as initial weights"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS, help="Training epochs")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_TRAIN_LEARNING_RATE,
        help="Optimizer learning rate",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=DEFAULT_MAX_TOKEN_LENGTH,
        help="Maximum token length per record",
    )
    parser.add_argument(
        "--vocabulary-size", type=int, help="Optional maximum tokenizer vocabulary size"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_TRAIN_VALIDATION_SPLIT,
        help="Validation data fraction in [0,1)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=DEFAULT_TRAIN_HIDDEN_DIM, help="Default model hidden size"
    )
    parser.add_argument(
        "--num-layers", type=int, default=DEFAULT_TRAIN_NUM_LAYERS, help="Default model layer count"
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=DEFAULT_TRAIN_ATTENTION_HEADS,
        help="Attention heads per transformer layer",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=DEFAULT_TRAIN_MLP_HIDDEN_DIM,
        help="Hidden width of transformer feed-forward block",
    )
    parser.add_argument(
        "--mlp-layers",
        type=int,
        default=DEFAULT_TRAIN_MLP_LAYERS,
        help="Number of MLP layers before vocabulary projection",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_TRAIN_DROPOUT,
        help="Dropout probability in default model",
    )
