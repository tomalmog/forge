"""Unit tests for optimizer and scheduler builders."""

from __future__ import annotations

from core.types import TrainingOptions
from serve.training_optimization import build_training_optimization


class _FakeModel:
    def parameters(self) -> list[object]:
        return [object()]


class _FakeRuntimeOptimizer:
    def __init__(self, kind: str, kwargs: dict[str, float]) -> None:
        self.kind = kind
        self.kwargs = kwargs
        self.param_groups = [{"lr": kwargs["lr"]}]


class _FakeRuntimeScheduler:
    def __init__(self, kind: str, kwargs: dict[str, float | int]) -> None:
        self.kind = kind
        self.kwargs = kwargs


class _FakeLRSchedulerNamespace:
    def StepLR(self, optimizer: _FakeRuntimeOptimizer, step_size: int, gamma: float) -> object:
        _ = optimizer
        return _FakeRuntimeScheduler("step", {"step_size": step_size, "gamma": gamma})

    def CosineAnnealingLR(
        self,
        optimizer: _FakeRuntimeOptimizer,
        T_max: int,
        eta_min: float,
    ) -> object:
        _ = optimizer
        return _FakeRuntimeScheduler("cosine", {"t_max": T_max, "eta_min": eta_min})


class _FakeOptimNamespace:
    def __init__(self) -> None:
        self.lr_scheduler = _FakeLRSchedulerNamespace()

    def Adam(self, params: list[object], lr: float, weight_decay: float) -> object:
        _ = params
        return _FakeRuntimeOptimizer("adam", {"lr": lr, "weight_decay": weight_decay})

    def AdamW(self, params: list[object], lr: float, weight_decay: float) -> object:
        _ = params
        return _FakeRuntimeOptimizer("adamw", {"lr": lr, "weight_decay": weight_decay})

    def SGD(
        self,
        params: list[object],
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> object:
        _ = params
        return _FakeRuntimeOptimizer(
            "sgd",
            {"lr": lr, "momentum": momentum, "weight_decay": weight_decay},
        )


class _FakeTorch:
    def __init__(self) -> None:
        self.optim = _FakeOptimNamespace()


def test_build_training_optimization_defaults_to_adam_without_scheduler(tmp_path) -> None:
    """Default options should create Adam optimizer and no scheduler."""
    options = TrainingOptions(dataset_name="demo", output_dir=str(tmp_path))

    optimization = build_training_optimization(_FakeTorch(), _FakeModel(), options)

    assert optimization.optimizer.kind == "adam" and optimization.scheduler is None


def test_build_training_optimization_supports_sgd_with_cosine_scheduler(tmp_path) -> None:
    """Configured SGD + cosine should create both optimizer and scheduler."""
    options = TrainingOptions(
        dataset_name="demo",
        output_dir=str(tmp_path),
        optimizer_type="sgd",
        weight_decay=0.01,
        sgd_momentum=0.95,
        scheduler_type="cosine",
        scheduler_t_max_epochs=7,
        scheduler_eta_min=0.0001,
    )

    optimization = build_training_optimization(_FakeTorch(), _FakeModel(), options)

    assert (
        optimization.optimizer.kind == "sgd"
        and optimization.optimizer.kwargs == {"lr": 0.001, "momentum": 0.95, "weight_decay": 0.01}
        and optimization.scheduler is not None
        and optimization.scheduler.kind == "cosine"
        and optimization.scheduler.kwargs == {"t_max": 7, "eta_min": 0.0001}
    )
