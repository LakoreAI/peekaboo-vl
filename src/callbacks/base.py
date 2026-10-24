from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class TrainerState:
    """Snapshot passed to callbacks at a validation/logging point."""

    step: int
    train_loss: float
    epoch: int = 0
    val_loss: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def get(self, key: str) -> Optional[float]:
        """Look up a monitored quantity by name — checks the named fields
        first (train_loss/val_loss/epoch/step), then falls back to `extra`.
        """
        if key in ("train_loss", "val_loss", "epoch", "step"):
            return getattr(self, key)
        return self.extra.get(key)


@dataclass
class TrainContext:
    """References a callback needs to act (save a checkpoint, read config)
    without the training loop having to know which callbacks care about what.
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    ckpt_dir: Path
    cfg_dict: dict = field(default_factory=dict)
    direction: str = ""
    wandb_run_id: Optional[str] = None


class Callback:
    """Base class. Override any subset of hooks — no-ops by default.

    Hook order within the training loop (see src/pipelines/train.py):
      on_train_start()                once, before the loop
      on_step_end(ctx, state)         after every optimizer.step()
      on_validation_end(ctx, state)   after every periodic validation pass
      on_train_end(ctx)               once, after the loop (incl. early stop)
    """

    def on_train_start(self, ctx: TrainContext) -> None:
        pass

    def on_step_end(self, ctx: TrainContext, state: TrainerState) -> None:
        pass

    def on_validation_end(self, ctx: TrainContext, state: TrainerState) -> None:
        pass

    def on_train_end(self, ctx: TrainContext) -> None:
        pass
