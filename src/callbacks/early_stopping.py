from typing import Optional

from src.callbacks.base import Callback, TrainContext, TrainerState


class EarlyStopping(Callback):
    """Stop training when `monitor` hasn't improved for `patience` validation
    checks in a row. Set `enabled: false` (or omit the block) to disable.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got {mode!r}"
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.num_bad_checks = 0
        self.should_stop = False

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def on_validation_end(self, ctx: TrainContext, state: TrainerState) -> None:
        value = state.get(self.monitor)
        if value is None:
            return

        if self._is_improvement(value):
            self.best = value
            self.num_bad_checks = 0
        else:
            self.num_bad_checks += 1
            if self.num_bad_checks >= self.patience:
                self.should_stop = True
                print(
                    f"  [EarlyStopping] {self.monitor} did not improve for "
                    f"{self.patience} checks (best={self.best:.4f}) — stopping"
                )
