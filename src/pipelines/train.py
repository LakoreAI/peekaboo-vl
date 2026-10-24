"""LoRA fine-tuning entrypoint for the video MCQ scorer.

Cross-entropy over the 4 answer-letter logits, one question per sample,
option order re-shuffled every time a sample is drawn. Validation is dev
accuracy (official option order; TTA is applied once at the end on the best
checkpoint). Controlled via a YAML config (see configs/train.yaml) with
individual CLI overrides. Callbacks (LR schedule, early stopping, best
checkpoint, W&B) are wired from that file.

Usage:
    uv run python -m src.pipelines.train --config configs/train.yaml
    uv run python -m src.pipelines.train --config configs/<run>.yaml --lr 5e-5
"""

import argparse
from collections import defaultdict
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.callbacks import (
    BestCheckpoint,
    EarlyStopping,
    TrainContext,
    TrainerState,
    build_lr_scheduler,
)
from src.callbacks.wandb_callback import WandbCallback, flat_metrics
from src.config import ModelConfig
from src.data import ReVADataset, ReVAExample, VideoGroupDataset, load_split
from src.modules.collate import BranchedMCQCollator, MCQCollator, PackedMCQCollator
from src.modules.cues import CueStore
from src.modules.cues.memory import VideoMemory
from src.modules.loss import ClassificationLoss, unlikelihood
from src.modules.model import VideoMCQModel
from src.pipelines._utils import announce_training
from src.pipelines.config import TrainingConfig, load_training_config
from src.pipelines.eval import evaluate, format_report
from src.utils.io_utils import (
    load_checkpoint,
    load_env,
    load_trainable,
    save_checkpoint,
    save_json,
)
from src.utils.model_utils import count_parameters, detect_device, get_run_name

REPO_ROOT = Path(__file__).resolve().parents[2]


def _forward_cf(model, cf: dict, device, branched: bool) -> torch.Tensor:
    """Letter logits of the counterfactual (reversed) view of a batch."""
    cf["inputs"] = {
        k: v.to(device) if hasattr(v, "to") else v for k, v in cf["inputs"].items()
    }
    if branched:
        return model.forward_branched(cf)
    return model(cf["inputs"])[0]


def load_examples(paths, cap: int = 0, seed: int = 0) -> List[ReVAExample]:
    """Concatenate splits; with `cap`, keep a seeded random sample of `cap`
    examples (in file order). Files are ordered by clip, so taking the first
    `cap` would drop whole clips, and test asks about every clip."""
    examples: List[ReVAExample] = []
    for path in paths:
        examples.extend(load_split(path))
    if cap and cap < len(examples):
        keep = sorted(random.Random(seed).sample(range(len(examples)), cap))
        examples = [examples[i] for i in keep]
    return examples


def build_model_config(train_cfg: TrainingConfig) -> ModelConfig:
    return ModelConfig(**(train_cfg.arch or {}))


def build_cue_store(
    model_cfg: ModelConfig,
    cues_root: Optional[str],
    memory_pool: Optional[List[ReVAExample]] = None,
) -> CueStore:
    memory = None
    if "memory" in model_cfg.cues:
        if not memory_pool:
            raise ValueError("the memory cue needs labelled examples (memory_files)")
        memory = VideoMemory(memory_pool, k=model_cfg.memory_k)
    return CueStore(model_cfg.cues, cues_root=cues_root, memory=memory)


def build_callbacks(
    train_cfg: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    run_name: str,
    wandb_config: dict,
):
    callbacks = []

    lr_cb = build_lr_scheduler(optimizer, train_cfg.lr_scheduler)
    if lr_cb is not None:
        callbacks.append(lr_cb)

    if train_cfg.early_stopping and train_cfg.early_stopping.get("enabled", True):
        es_kwargs = {
            k: v for k, v in train_cfg.early_stopping.items() if k != "enabled"
        }
        callbacks.append(EarlyStopping(**es_kwargs))

    if train_cfg.save_best:
        callbacks.append(
            BestCheckpoint(monitor=train_cfg.best_metric, mode=train_cfg.best_mode)
        )

    # after BestCheckpoint: relies on <ckpt_dir>/best.pt already existing
    if train_cfg.wandb and train_cfg.wandb.get("enabled", False):
        wb = train_cfg.wandb
        callbacks.append(
            WandbCallback(
                project_name=wb.get("project", "peekaboo-vl"),
                run_name=run_name,
                config=wandb_config,
                entity=wb.get("entity"),
                monitor=wb.get("monitor", train_cfg.best_metric),
                mode=wb.get("mode", train_cfg.best_mode),
                log_artifacts=wb.get("log_artifacts", False),
                group=wb.get("group"),
            )
        )

    return callbacks


def total_optimizer_steps(n_examples: int, train_cfg: TrainingConfig) -> int:
    per_epoch = math.ceil(
        math.ceil(n_examples / train_cfg.batch_size) / train_cfg.accum_steps
    )
    return max(per_epoch * train_cfg.epochs, 1)


def _validate(
    model, collator, dev, device, train_cfg, step, epoch, train_loss, ctx, callbacks
):
    result = evaluate(
        model,
        collator,
        dev,
        device,
        tta=train_cfg.eval_tta,
        batch_size=train_cfg.eval_batch_size,
        num_workers=train_cfg.num_workers,
    )
    print(format_report(result))
    extra = {
        "accuracy": result["accuracy"],
        "test_weighted_accuracy": result["test_weighted_accuracy"],
        "pred_letter_kl": result["pred_letter_kl"],
        **{f"task/{k}": v for k, v in result["per_task"].items()},
    }
    state = TrainerState(step=step, train_loss=train_loss, epoch=epoch, extra=extra)
    for cb in callbacks:
        cb.on_validation_end(ctx, state)
    model.train()
    return extra


def train(train_cfg: TrainingConfig):
    load_env(REPO_ROOT / ".env")  # HF_TOKEN / WANDB_API_KEY
    device = detect_device()
    print(f"device: {device}")

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    # Frame-phase jitter and other augmentation draw from the stdlib RNG.
    random.seed(train_cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_cfg.seed)

    fit = load_examples(
        train_cfg.train_files, train_cfg.max_train_examples, train_cfg.seed
    )
    dev = (
        load_examples([train_cfg.dev_file], train_cfg.max_dev_examples)
        if train_cfg.dev_file and Path(train_cfg.dev_file).exists()
        else []
    )
    if dev:
        overlap = {ex.qa_id for ex in fit} & {ex.qa_id for ex in dev}
        if overlap:
            raise ValueError(f"{len(overlap)} dev qa_ids are also in the train files")
    print(f"fit: {len(fit)} questions   dev: {len(dev)} questions")

    model_cfg = build_model_config(train_cfg)
    memory_pool = (
        load_examples(train_cfg.memory_files) if train_cfg.memory_files else fit
    )
    cue_store = build_cue_store(model_cfg, train_cfg.cues_root, memory_pool)

    model = VideoMCQModel.from_config(model_cfg, device)
    if train_cfg.init_adapter:
        raw = torch.load(train_cfg.init_adapter, map_location="cpu", weights_only=False)
        load_trainable(model, raw["model"])
        print(f"initialized adapter from {train_cfg.init_adapter}")
    # Keep the adapter in fp32 (bf16 Adam updates lose small steps); the
    # frozen backbone stays bf16 and compute runs under autocast.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    count_parameters(model)

    collator = (BranchedMCQCollator if train_cfg.branch_questions > 1 else MCQCollator)(
        model.processor,
        model_cfg,
        cue_store,
        train_cfg.frames_root,
        train_cfg.reva_root,
    )
    if train_cfg.branch_questions > 1:
        collator.group_size = train_cfg.branch_questions
    # Separate batch builder for training so augmentation never reaches eval.
    packed = train_cfg.pack_questions > 1
    branched = train_cfg.branch_questions > 1
    if packed and branched:
        raise ValueError("pack_questions and branch_questions are exclusive")
    collator_cls = (
        PackedMCQCollator if packed else BranchedMCQCollator if branched else MCQCollator
    )
    train_collator = collator_cls(
        model.processor,
        model_cfg,
        cue_store,
        train_cfg.frames_root,
        train_cfg.reva_root,
    )
    train_collator.random_phase = train_cfg.train_frame_phase_jitter
    if train_cfg.cue_dropout:
        unknown = set(train_cfg.cue_dropout) - set(model_cfg.cues)
        if unknown:
            raise ValueError(f"cue_dropout for cues not enabled: {sorted(unknown)}")
        train_collator.cue_dropout = dict(train_cfg.cue_dropout)
        print(f"cue dropout (train only): {train_cfg.cue_dropout}")
    if train_cfg.fact_weight > 0:
        if not branched:
            raise ValueError("fact_weight > 0 needs branch_questions > 1")
        facts = defaultdict(list)
        for ex in fit:
            facts[ex.video_path].append((ex.question, ex.options[ex.label]))
        train_collator.facts = dict(facts)
        train_collator.max_facts = train_cfg.max_facts
        print(f"fact rows: {len(facts)} clips, weight {train_cfg.fact_weight}")
    if train_cfg.cf_tasks:
        if packed:
            raise ValueError("cf_tasks is not supported with pack_questions")
        train_collator.cf_tasks = tuple(train_cfg.cf_tasks)
        print(f"counterfactual reversed view on {train_cfg.cf_tasks}, weight {train_cfg.cf_weight}")
    if packed:
        if train_cfg.batch_size != 1:
            raise ValueError("pack_questions > 1 needs batch_size: 1")
        train_set = VideoGroupDataset(
            fit,
            train_cfg.pack_questions,
            shuffle_options=train_cfg.shuffle_options,
            seed=train_cfg.seed,
        )
        print(
            f"packing: {train_set.num_questions} questions in {len(train_set)} "
            f"sequences (<= {train_cfg.pack_questions} per video chunk)"
        )
    elif branched:
        if train_cfg.batch_size != 1:
            raise ValueError("branch_questions > 1 needs batch_size: 1")
        train_set = VideoGroupDataset(
            fit,
            train_cfg.branch_questions,
            shuffle_options=train_cfg.shuffle_options,
            seed=train_cfg.seed,
            split_tasks=model_cfg.overlay_time_tasks if model_cfg.overlay_time else (),
        )
        print(
            f"branching: {train_set.num_questions} questions in {len(train_set)} "
            f"groups (<= {train_cfg.branch_questions} per video chunk, shared prefix)"
        )
    else:
        train_set = ReVADataset(
            fit, shuffle_options=train_cfg.shuffle_options, seed=train_cfg.seed
        )
    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory and device.type == "cuda",
        collate_fn=train_collator,
        # Packed groups are re-cut every epoch, so workers must re-read the set.
        # With option shuffling the permutation stream is keyed on the epoch, so
        # persistent workers (which keep a stale epoch) must be disabled.
        persistent_workers=train_cfg.num_workers > 0
        and not packed
        and not branched
        and not train_cfg.shuffle_options,
    )
    criterion = ClassificationLoss(train_cfg.label_smoothing)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters; set a `lora` block in arch")
    optimizer = torch.optim.AdamW(
        trainable, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    sched = dict(train_cfg.lr_scheduler or {})
    if sched.get("type") in ("cosine", "warmup_cosine") and not sched.get("t_max"):
        sched["t_max"] = total_optimizer_steps(len(train_set), train_cfg)
        if sched["type"] == "warmup_cosine" and "warmup_ratio" in sched:
            sched["warmup_steps"] = int(sched.pop("warmup_ratio") * sched["t_max"])
    train_cfg.lr_scheduler = sched or None

    run_name = train_cfg.run_name or get_run_name(
        Path(model_cfg.backbone).name,
        "reva",
        train_cfg.lr,
        train_cfg.batch_size * train_cfg.accum_steps,
    )
    ckpt_dir = Path(train_cfg.ckpt_dir) / run_name
    result_dir = Path(train_cfg.result_dir) / run_name

    callbacks = build_callbacks(
        train_cfg, optimizer, run_name, {**asdict(model_cfg), **asdict(train_cfg)}
    )
    early_stoppers = [cb for cb in callbacks if isinstance(cb, EarlyStopping)]
    wandb_run_id = None
    for cb in callbacks:
        if isinstance(cb, WandbCallback) and cb.enabled and cb.run is not None:
            wandb_run_id = cb.run.id

    ctx = TrainContext(
        model=model,
        optimizer=optimizer,
        ckpt_dir=ckpt_dir,
        cfg_dict=asdict(model_cfg),
        wandb_run_id=wandb_run_id,
    )
    for cb in callbacks:
        cb.on_train_start(ctx)

    if train_cfg.resume_from:
        raw = load_checkpoint(
            train_cfg.resume_from, model, optimizer, map_location="cpu"
        )
        step = int(raw.get("step", 0))
        start_epoch = int(raw.get("extra", {}).get("epoch", 0)) + 1
        print(f"resumed at step {step}, continuing from epoch {start_epoch}")
    else:
        step, start_epoch = 0, 1

    announce_training(
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        collator=collator,
        fit=fit,
        dev=dev,
        device=device,
        run_name=run_name,
        ckpt_dir=ckpt_dir,
        result_dir=result_dir,
    )

    total_steps = total_optimizer_steps(len(train_set), train_cfg)
    t_start, step_start = time.time(), step
    autocast_dtype = (
        torch.bfloat16 if model_cfg.torch_dtype == "bfloat16" else torch.float16
    )
    use_autocast = device.type == "cuda"
    history = []
    stopped_early = False
    model.train()
    for epoch in range(start_epoch, train_cfg.epochs + 1):
        epoch_losses, pending = [], []
        if packed or branched:
            train_set.regroup(epoch)
        else:
            train_set.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch["inputs"].items()}
            labels = batch["labels"].to(device)
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=use_autocast
            ):
                fact_loss = None
                if branched and "fact_start" in batch:
                    batch["inputs"] = inputs
                    logits, fact_loss = model.forward_branched_facts(batch)
                elif branched:
                    batch["inputs"] = inputs
                    logits = model.forward_branched(batch)
                elif packed:
                    logits = model.forward_positions(
                        inputs, batch["positions"].to(device)
                    )
                else:
                    logits, _ = model(inputs)
                loss = criterion(logits, labels)
                if fact_loss is not None:
                    loss = loss + train_cfg.fact_weight * fact_loss
                if "cf" in batch:
                    loss = loss + train_cfg.cf_weight * unlikelihood(
                        _forward_cf(model, batch["cf"], device, branched),
                        batch["cf"]["labels"].to(device),
                    )
            last_batch = i + 1 == len(train_loader)
            # A forced step at the end of an epoch may accumulate fewer than
            # `accum_steps` batches; average over those actually accumulated
            # instead of scaling by the full count (which would shrink the
            # gradient at every epoch boundary).
            n_in_step = len(pending) + 1
            denom = (
                n_in_step
                if last_batch and n_in_step < train_cfg.accum_steps
                else train_cfg.accum_steps
            )
            (loss / denom).backward()
            pending.append(loss.item())

            if len(pending) < train_cfg.accum_steps and not last_batch:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, train_cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            avg = sum(pending) / len(pending)
            epoch_losses.append(avg)
            pending = []
            state = TrainerState(step=step, train_loss=avg, epoch=epoch)
            for cb in callbacks:
                cb.on_step_end(ctx, state)
            if step % train_cfg.log_every == 0 or step == 1:
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_start
                rate = elapsed / max(step - step_start, 1)
                eta_h = rate * max(total_steps - step, 0) / 3600
                print(
                    f"  epoch {epoch:2d}  step {step:6d}/{total_steps}  loss={avg:.4f}  "
                    f"lr={lr:.2e}  {rate:.1f}s/step  eta={eta_h:.2f}h"
                )
            if train_cfg.save_every_steps and step % train_cfg.save_every_steps == 0:
                save_checkpoint(
                    model,
                    None,
                    step,
                    ckpt_dir / "last.pt",
                    extra={"cfg": asdict(model_cfg), "epoch": epoch, "step": step},
                )
                print(f"  saved adapter: {ckpt_dir / 'last.pt'} (step {step})")

            if (
                dev
                and train_cfg.eval_every_steps
                and step % train_cfg.eval_every_steps == 0
            ):
                extra = _validate(
                    model,
                    collator,
                    dev,
                    device,
                    train_cfg,
                    step,
                    epoch,
                    avg,
                    ctx,
                    callbacks,
                )
                history.append({"epoch": epoch, "step": step, "loss": avg, **extra})
                if any(es.should_stop for es in early_stoppers):
                    stopped_early = True
                    break

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        record = {"epoch": epoch, "step": step, "loss": train_loss}
        print(f"epoch {epoch:2d}/{train_cfg.epochs}  train_loss={train_loss:.4f}")
        if dev and not train_cfg.eval_every_steps and not stopped_early:
            record.update(
                _validate(
                    model,
                    collator,
                    dev,
                    device,
                    train_cfg,
                    step,
                    epoch,
                    train_loss,
                    ctx,
                    callbacks,
                )
            )
            if any(es.should_stop for es in early_stoppers):
                stopped_early = True
        history.append(record)

        if (
            epoch % train_cfg.ckpt_every == 0
            or epoch == train_cfg.epochs
            or stopped_early
        ):
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            save_checkpoint(
                model,
                optimizer,
                step,
                ckpt_path,
                extra={
                    "cfg": asdict(model_cfg),
                    "epoch": epoch,
                    "wandb_run_id": ctx.wandb_run_id,
                },
            )
            print(f"  saved checkpoint: {ckpt_path}")
        if stopped_early:
            break

    # Without dev there is no best.pt; the last epoch is the model.
    best = ckpt_dir / "best.pt"
    if dev and best.exists():
        load_checkpoint(best, model, map_location="cpu")
        final = evaluate(
            model,
            collator,
            dev,
            device,
            tta=train_cfg.tta,
            batch_size=train_cfg.eval_batch_size,
            pride_fraction=train_cfg.pride_fraction,
            num_workers=train_cfg.num_workers,
        )
        print(
            f"\nbest checkpoint on dev (tta={train_cfg.tta}):\n{format_report(final)}"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        np.save(result_dir / f"dev_scores_{train_cfg.tta}.npy", final.pop("scores"))
        save_json(final, result_dir / f"dev_metrics_{train_cfg.tta}.json")
        for cb in callbacks:
            if isinstance(cb, WandbCallback):
                cb.log_metrics(
                    flat_metrics(final), step=step, prefix=f"final_{train_cfg.tta}"
                )
                if cb.enabled:
                    for k, v in flat_metrics(final).items():
                        cb.run.summary[f"final_{train_cfg.tta}/{k}"] = v

    for cb in callbacks:
        cb.on_train_end(ctx)
    save_json(history, ckpt_dir / "train_log.json")
    save_json(history, result_dir / "train_log.json")
    print(
        f"\n{'stopped early' if stopped_early else 'finished'} at epoch {epoch} (step {step})"
    )
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML, see configs/train.yaml"
    )
    parser.add_argument("--reva_root", type=str, default=None)
    parser.add_argument("--frames_root", type=str, default=None)
    parser.add_argument("--cues_root", type=str, default=None)
    parser.add_argument("--dev_file", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--accum_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--eval_every_steps", type=int, default=None)
    parser.add_argument("--save_every_steps", type=int, default=None)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_dev_examples", type=int, default=None)
    parser.add_argument(
        "--tta", type=str, default=None, choices=["none", "cyclic", "pride"]
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    train_cfg = load_training_config(args.config, **overrides)
    train(train_cfg)
