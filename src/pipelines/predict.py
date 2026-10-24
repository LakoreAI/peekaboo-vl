"""Score a split with a checkpoint (or a zero-shot backbone) and save outputs.

Writes to `--out_dir`:
    scores.npy        (N, 4) log-probs in the original option order
    qa_ids.json       row order of scores.npy
    predictions.json  {qa_id: letter}, the Codabench payload
    metrics.json      only when the split is labelled (dev / val)

Keeping the log-probs makes ensembling and
TTA ablations free after the fact.

Usage:
    # fine-tuned adapter; model/prompt config comes from the checkpoint
    # (a `.pt` file, or the PEFT folder released on HF LakoreAI/peekaboo-vl)
    uv run python -m src.pipelines.predict --ckpt checkpoints/<run>/best.pt \
        --split data/raw/reva/test.json --tta cyclic --out_dir results/<run>/test

    # zero-shot baseline; config from a YAML's `arch` block
    uv run python -m src.pipelines.predict --config configs/<zero-shot>.yaml \
        --split data/processed/splits/dev.json --out_dir results/zeroshot/dev
"""

import argparse
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.callbacks.wandb_callback import flat_metrics, log_eval_run
from src.config import ModelConfig
from src.data import load_split
from src.modules.collate import BranchedMCQCollator, MCQCollator
from src.modules.model import VideoMCQModel
from src.pipelines.config import load_training_config
from src.pipelines.eval import evaluate, format_report
from src.pipelines.train import build_cue_store, load_examples
from src.utils.io_utils import load_env, load_trainable, read_adapter, save_json
from src.utils.model_utils import detect_device


REPO_ROOT = Path(__file__).resolve().parents[2]


def config_from_checkpoint(ckpt: dict) -> ModelConfig:
    """Rebuild ModelConfig from a checkpoint's saved `extra["cfg"]`."""
    saved = ckpt.get("extra", {}).get("cfg", {})
    known = {f.name for f in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in saved.items() if k in known})


def load_model(
    ckpt_path: Optional[Path], model_cfg: Optional[ModelConfig], device: torch.device
):
    """Either rebuild from a checkpoint (adapter + saved config) or build
    `model_cfg` as-is (zero-shot)."""
    raw = None
    if ckpt_path is not None:
        raw = read_adapter(ckpt_path)
        model_cfg = config_from_checkpoint(raw)
    if model_cfg is None:
        raise ValueError("need --ckpt or a config with an arch block")
    model = VideoMCQModel.from_config(model_cfg, device)
    if raw is not None:
        load_trainable(model, raw["model"])
    model.eval()
    return model, model_cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="training YAML (paths, arch for zero-shot)",
    )
    parser.add_argument(
        "--split", type=Path, required=True, help="annotation JSON to score"
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--tta", choices=["none", "cyclic", "pride"], default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument(
        "--cue_mode",
        choices=["asis", "none", "shuffled"],
        default="asis",
        help="cue ablation: as trained, removed, or taken from a random other clip",
    )
    parser.add_argument(
        "--overlay_time",
        choices=["asis", "on", "off"],
        default="asis",
        help="VTimeCoT-style progress bar + timestamp on frames (inference only)",
    )
    parser.add_argument(
        "--add_cues",
        nargs="*",
        default=None,
        help="append cues to the checkpoint's set (e.g. timeline) for a zero-shot test",
    )
    parser.add_argument(
        "--overlay_time_tasks",
        nargs="*",
        default=None,
        help="restrict --overlay_time to these tasks (e.g. 'Temporal Grounding')",
    )
    parser.add_argument(
        "--frame_phase",
        type=float,
        default=0.5,
        help="position inside each temporal segment to sample frames from (TTA)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="override the model's frame count at inference (must be even)",
    )
    parser.add_argument(
        "--branch",
        type=int,
        default=0,
        help="score up to N questions per video in one shared-prefix pass "
        "(Qwen3.5; identical scores, faster). 0 = one question at a time",
    )
    args = parser.parse_args()

    load_env(REPO_ROOT / ".env")  # WANDB_API_KEY / HF_TOKEN, as in training
    train_cfg = load_training_config(args.config)
    tta = args.tta or train_cfg.tta
    device = detect_device()

    arch_cfg = ModelConfig(**(train_cfg.arch or {})) if args.ckpt is None else None
    model, model_cfg = load_model(args.ckpt, arch_cfg, device)

    examples = load_split(args.split)
    if args.max_examples:
        examples = examples[: args.max_examples]
    memory_pool = (
        load_examples(train_cfg.memory_files or train_cfg.train_files)
        if "memory" in model_cfg.cues
        else None
    )
    if args.num_frames:
        if args.num_frames % 2:
            raise ValueError("--num_frames must be even")
        model_cfg.num_frames = args.num_frames
    if args.cue_mode == "none":
        model_cfg.cues = ()
    if args.add_cues:
        model_cfg.cues = tuple(dict.fromkeys(list(model_cfg.cues) + list(args.add_cues)))
    if args.overlay_time != "asis":
        model_cfg.overlay_time = args.overlay_time == "on"
    if args.overlay_time_tasks:
        model_cfg.overlay_time_tasks = tuple(args.overlay_time_tasks)
    cue_store = build_cue_store(model_cfg, train_cfg.cues_root, memory_pool)
    if args.cue_mode == "shuffled":
        cue_store.shuffle(seed=0)
    collator = (BranchedMCQCollator if args.branch > 1 else MCQCollator)(
        model.processor,
        model_cfg,
        cue_store,
        train_cfg.frames_root,
        train_cfg.reva_root,
    )
    collator.frame_phase = args.frame_phase
    if args.branch > 1:
        collator.group_size = args.branch
    result = evaluate(
        model,
        collator,
        examples,
        device,
        tta=tta,
        batch_size=args.batch_size or train_cfg.eval_batch_size,
        pride_fraction=train_cfg.pride_fraction,
        num_workers=train_cfg.num_workers,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "scores.npy", result.pop("scores"))
    save_json([ex.qa_id for ex in examples], args.out_dir / "qa_ids.json")
    save_json(result.pop("predictions"), args.out_dir / "predictions.json")
    if result.get("n"):
        save_json(
            {**result, "tta": tta, "split": str(args.split)},
            args.out_dir / "metrics.json",
        )
        print(format_report(result))
        log_eval_run(
            train_cfg.wandb,
            name=f"{train_cfg.run_name or args.out_dir.parent.name}-{args.split.stem}-{tta}",
            metrics=flat_metrics(result),
            config={
                **asdict(model_cfg),
                "split": str(args.split),
                "tta": tta,
                "ckpt": str(args.ckpt) if args.ckpt else None,
            },
        )
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
