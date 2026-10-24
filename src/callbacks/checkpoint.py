from typing import Optional

from src.callbacks.base import Callback, TrainContext, TrainerState
from src.utils.io_utils import save_checkpoint


class BestCheckpoint(Callback):
    """Save `<ckpt_dir>/best.pt` whenever `monitor` improves. Uses the same
    {"model","optimizer","step","extra"} schema as the periodic checkpoints so
    src.pipelines.eval / infer can load it identically.

    Default monitors `val_loss` (min); the training pipeline overrides this to
    monitor validation accuracy (max).
    """

    def __init__(self, monitor: str = "val_loss", mode: str = "min"):
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got {mode!r}"
        self.monitor = monitor
        self.mode = mode
        self.best: Optional[float] = None

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        return value < self.best if self.mode == "min" else value > self.best

    def on_validation_end(self, ctx: TrainContext, state: TrainerState) -> None:
        value = state.get(self.monitor)
        if value is None or not self._is_improvement(value):
            return

        self.best = value
        path = ctx.ckpt_dir / "best.pt"
        # Adapter weights only: best.pt is for evaluation, not resuming.
        save_checkpoint(
            ctx.model,
            None,
            state.step,
            path,
            extra={
                "cfg": ctx.cfg_dict,
                "monitor": self.monitor,
                "monitor_value": value,
                "epoch": state.epoch,
                "wandb_run_id": ctx.wandb_run_id,
            },
        )
        print(f"  [BestCheckpoint] new best {self.monitor}={value:.4f} -> {path}")
