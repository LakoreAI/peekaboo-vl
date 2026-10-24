"""Optional Weights & Biases logging callback.

Opt-in via YAML (`wandb: {enabled: true, ...}`). Never breaks a run that
doesn't want it: no-ops if `wandb` isn't installed, and no-ops if
`wandb.init()` fails (no API key / offline), logging a message instead of
raising.
"""

from pathlib import Path
from typing import Optional

from src.callbacks.base import Callback, TrainContext, TrainerState

try:
    import wandb
except ImportError:
    wandb = None


class WandbCallback(Callback):
    """Logs train/val loss and any `state.extra` metrics, and uploads
    `<ckpt_dir>/best.pt` as an artifact when `monitor` improves. Run AFTER
    BestCheckpoint so the file exists when this fires.
    """

    def __init__(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        entity: Optional[str] = None,
        monitor: str = "val_loss",
        mode: str = "min",
        log_artifacts: bool = True,
        group: Optional[str] = None,
    ):
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got {mode!r}"
        self.monitor = monitor
        self.mode = mode
        self.log_artifacts = log_artifacts
        self.best: Optional[float] = None
        self.enabled = wandb is not None
        self.run = None

        if not self.enabled:
            print(
                "  [WandbCallback] wandb not installed — logging disabled (pip install wandb to enable)"
            )
            return

        try:
            self.run = wandb.init(
                project=project_name,
                name=run_name,
                config=config or {},
                entity=entity,
                group=group,
                reinit="finish_previous",
            )
        except Exception as e:  # auth/network failure shouldn't kill a training run
            print(
                f"  [WandbCallback] could not start a wandb run ({e}) — logging disabled"
            )
            self.enabled = False

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        return value < self.best if self.mode == "min" else value > self.best

    def log_metrics(self, metrics: dict, step: int, prefix: str = "eval") -> None:
        if not self.enabled:
            return
        log_dict = {f"{prefix}/{k}": v for k, v in metrics.items() if v is not None}
        self.run.log(log_dict, step=step)

        if self.monitor in metrics and metrics[self.monitor] is not None:
            value = metrics[self.monitor]
            if self._is_improvement(value):
                self.best = value
                self.run.summary[f"best_{self.monitor}"] = value
                print(f"  [WandbCallback] new best {self.monitor}: {value:.4f}")

    def log_artifact(self, model_path, name: str = "model-checkpoint") -> None:
        if not self.enabled:
            return
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(str(model_path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()

    # ---- Callback hooks ----

    def on_train_start(self, ctx: TrainContext) -> None:
        if self.enabled:
            self.run.log({"train/begin": True})

    def on_step_end(self, ctx: TrainContext, state: TrainerState) -> None:
        metrics = {"loss": state.train_loss}
        if ctx is not None and ctx.optimizer.param_groups:
            metrics["lr"] = ctx.optimizer.param_groups[0]["lr"]
        self.log_metrics(metrics, step=state.step, prefix="train")

    def on_validation_end(self, ctx: TrainContext, state: TrainerState) -> None:
        metrics = dict(state.extra)
        if state.val_loss is not None:
            metrics["val_loss"] = state.val_loss
        self.log_metrics(metrics, step=state.step, prefix="eval")

    def on_train_end(self, ctx: TrainContext) -> None:
        # One artifact per run: the final best checkpoint (never epoch_*.pt).
        if self.enabled and self.log_artifacts and ctx is not None:
            best_path = Path(ctx.ckpt_dir) / "best.pt"
            if best_path.exists():
                run_label = self.run.name if self.run is not None else "mlp"
                self.log_artifact(best_path, name=f"{run_label}-best")
        self.finish()


def flat_metrics(result: dict) -> dict:
    """ReVA metrics dict -> flat {name: float} for W&B (per_task/per_category
    expanded to `task/<name>`, `category/<name>`)."""
    flat = {}
    for key in ("accuracy", "test_weighted_accuracy", "pred_letter_kl", "n"):
        if key in result:
            flat[key] = result[key]
    for task, acc in (result.get("per_task") or {}).items():
        flat[f"task/{task}"] = acc
    for cat, acc in (result.get("per_category") or {}).items():
        flat[f"category/{cat}"] = acc
    return flat


def log_eval_run(
    wandb_cfg: Optional[dict], name: str, metrics: dict, config: dict
) -> None:
    """One-off W&B run for a scoring job (predict.py). No-op when disabled,
    when wandb is missing, or when init fails (offline / no key)."""
    if not wandb_cfg or not wandb_cfg.get("enabled", False) or wandb is None:
        return
    try:
        run = wandb.init(
            project=wandb_cfg.get("project", "peekaboo-vl"),
            entity=wandb_cfg.get("entity"),
            group=wandb_cfg.get("group"),
            name=name,
            job_type="eval",
            config=config,
            reinit="finish_previous",
        )
    except Exception as e:  # never fail a scoring job over logging
        print(f"  [wandb] could not start an eval run ({e}) — skipped")
        return
    run.log({f"eval/{k}": v for k, v in metrics.items()})
    for k, v in metrics.items():
        run.summary[f"eval/{k}"] = v
    run.finish()
