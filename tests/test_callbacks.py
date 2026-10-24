import torch

from src.callbacks import (
    BestCheckpoint,
    EarlyStopping,
    LRSchedulerCallback,
    TrainContext,
    TrainerState,
    build_lr_scheduler,
)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)


def make_ctx(tmp_path):
    model = Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, opt, TrainContext(model=model, optimizer=opt, ckpt_dir=tmp_path)


def test_best_checkpoint_saves_and_improves(tmp_path):
    model, opt, ctx = make_ctx(tmp_path)
    cb = BestCheckpoint(monitor="auc", mode="max")
    cb.on_validation_end(ctx, TrainerState(step=1, train_loss=1.0, extra={"auc": 0.5}))
    assert (tmp_path / "best.pt").exists()
    first_mtime = (tmp_path / "best.pt").stat().st_mtime_ns
    # worse value -> no rewrite
    cb.on_validation_end(ctx, TrainerState(step=2, train_loss=1.0, extra={"auc": 0.4}))
    assert (tmp_path / "best.pt").stat().st_mtime_ns == first_mtime
    # better value -> rewrite
    cb.on_validation_end(ctx, TrainerState(step=3, train_loss=1.0, extra={"auc": 0.9}))
    assert cb.best == 0.9


def test_early_stopping_triggers(tmp_path):
    _, _, ctx = make_ctx(tmp_path)
    es = EarlyStopping(monitor="auc", mode="max", patience=2)
    es.on_validation_end(ctx, TrainerState(step=1, train_loss=1.0, extra={"auc": 0.5}))
    assert not es.should_stop
    es.on_validation_end(ctx, TrainerState(step=2, train_loss=1.0, extra={"auc": 0.4}))
    assert not es.should_stop
    es.on_validation_end(ctx, TrainerState(step=3, train_loss=1.0, extra={"auc": 0.3}))
    assert es.should_stop


def test_cosine_scheduler_respects_start_epoch(tmp_path):
    model, opt, ctx = make_ctx(tmp_path)
    cb = build_lr_scheduler(
        opt, {"type": "cosine", "t_max": 10, "eta_min": 0.0, "start_epoch": 2}
    )
    assert isinstance(cb, LRSchedulerCallback)
    initial = opt.param_groups[0]["lr"]
    opt.step()
    cb.on_step_end(ctx, TrainerState(step=1, train_loss=1.0, epoch=0))
    assert opt.param_groups[0]["lr"] == initial
    opt.step()
    cb.on_step_end(ctx, TrainerState(step=2, train_loss=1.0, epoch=2))
    assert opt.param_groups[0]["lr"] < initial


def test_build_lr_scheduler_none():
    model = Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    assert build_lr_scheduler(opt, None) is None
    assert build_lr_scheduler(opt, {"type": "none"}) is None


def test_flat_metrics_and_disabled_eval_run():
    from src.callbacks.wandb_callback import flat_metrics, log_eval_run

    res = {
        "accuracy": 0.7,
        "n": 10,
        "per_task": {"TU_Temporal": 0.5},
        "per_category": {"Causal Reasoning": 0.9},
    }
    flat = flat_metrics(res)
    assert flat == {
        "accuracy": 0.7,
        "n": 10,
        "task/TU_Temporal": 0.5,
        "category/Causal Reasoning": 0.9,
    }
    log_eval_run({"enabled": False}, "x", flat, {})  # must be a silent no-op
    log_eval_run(None, "x", flat, {})
