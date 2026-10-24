import math
from typing import Optional

import torch

from src.callbacks.base import Callback, TrainContext, TrainerState


class LRSchedulerCallback(Callback):
    """Wraps a torch LR scheduler. `step_kind` controls when `.step()` is
    called:
      "step"       — every optimizer step (StepLR, CosineAnnealingLR, ...)
      "validation" — every validation pass (ReduceLROnPlateau, which needs the
                     monitored metric value)
    """

    def __init__(
        self,
        scheduler,
        step_kind: str = "step",
        monitor: str = "val_loss",
        start_epoch: int = 0,
    ):
        assert step_kind in ("step", "validation")
        self.scheduler = scheduler
        self.step_kind = step_kind
        self.monitor = monitor
        # The reference holds the LR constant for the first `start_epoch`
        # epochs, then anneals (config.yaml: start_scheduler_epoch=20).
        self.start_epoch = start_epoch

    def on_step_end(self, ctx: TrainContext, state: TrainerState) -> None:
        if self.step_kind == "step" and state.epoch >= self.start_epoch:
            self.scheduler.step()

    def on_validation_end(self, ctx: TrainContext, state: TrainerState) -> None:
        if self.step_kind == "validation" and state.epoch >= self.start_epoch:
            value = state.get(self.monitor)
            if value is not None:
                self.scheduler.step(value)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Optional[dict]
) -> Optional[LRSchedulerCallback]:
    """cfg: {"type": "step"|"cosine"|"warmup_cosine"|"plateau"|"none", ...}.

    step          : StepLR(step_size, gamma)
    cosine        : CosineAnnealingLR(t_max, eta_min) — the reference's choice
    warmup_cosine : linear warmup then cosine decay (transformers convention)
    plateau       : ReduceLROnPlateau(mode, factor, patience), needs a metric
    """
    if not cfg:
        return None
    sched_type = cfg.get("type", "none")
    if sched_type == "none":
        return None

    if sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.get("step_size", 50), gamma=cfg.get("gamma", 0.5)
        )
        return LRSchedulerCallback(
            scheduler, step_kind="step", start_epoch=cfg.get("start_epoch", 0)
        )

    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.get("t_max", 200), eta_min=cfg.get("eta_min", 0.0)
        )
        return LRSchedulerCallback(
            scheduler, step_kind="step", start_epoch=cfg.get("start_epoch", 0)
        )

    if sched_type == "warmup_cosine":
        warmup_steps = max(cfg.get("warmup_steps", 20), 1)
        total_steps = max(cfg.get("t_max", 200), warmup_steps + 1)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return LRSchedulerCallback(
            scheduler, step_kind="step", start_epoch=cfg.get("start_epoch", 0)
        )

    if sched_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.get("mode", "min"),
            factor=cfg.get("factor", 0.5),
            patience=cfg.get("patience", 3),
        )
        return LRSchedulerCallback(
            scheduler,
            step_kind="validation",
            monitor=cfg.get("monitor", "val_loss"),
            start_epoch=cfg.get("start_epoch", 0),
        )

    raise ValueError(f"unknown lr_scheduler type: {sched_type!r}")
