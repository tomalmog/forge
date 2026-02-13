"""Unit tests for training progress reporting helpers."""

from __future__ import annotations

from serve.training_progress import TrainingProgressTracker, read_optimizer_learning_rate


class _FakeLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


class _FakeOptimizer:
    def __init__(self, lr: float) -> None:
        self.param_groups = [{"lr": lr}]


def test_training_progress_tracker_logs_periodic_batch_updates(monkeypatch) -> None:
    """Progress tracker should emit first/interval/last batch updates."""
    fake_logger = _FakeLogger()
    monkeypatch.setattr("serve.training_progress._LOGGER", fake_logger)
    tracker = TrainingProgressTracker(
        dataset_name="demo",
        total_epochs=3,
        start_epoch=1,
        train_batch_count=5,
        validation_batch_count=2,
        batch_log_interval_steps=2,
    )

    tracker.log_training_started()
    tracker.log_epoch_started(1)
    for batch_index in [1, 2, 3, 4, 5]:
        tracker.log_batch_progress("train", 1, batch_index, 5, batch_index, 0.5)
    tracker.log_epoch_completed(1, train_loss=0.5, validation_loss=0.4, learning_rate=0.001)

    batch_events = [event for event, _ in fake_logger.events if event == "training_batch_progress"]

    assert batch_events == [
        "training_batch_progress",
        "training_batch_progress",
        "training_batch_progress",
        "training_batch_progress",
    ]


def test_read_optimizer_learning_rate_reads_first_param_group() -> None:
    """Learning-rate reader should return the first optimizer param-group value."""
    learning_rate = read_optimizer_learning_rate(_FakeOptimizer(lr=0.003))

    assert learning_rate == 0.003
