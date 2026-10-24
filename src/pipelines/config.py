"""Training-loop hyperparameters — separate from `src.config.ModelConfig`
(model, video input, prompt cues, adapter). Controllable via a YAML file (see
`configs/train.yaml`) with individual fields overridable from the CLI.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.utils.io_utils import read_yaml


@dataclass
class TrainingConfig:
    # --- data ---
    # Official annotation files live in `reva_root` (train.json, val.json,
    # test.json, and the video folders if not deleted after extraction).
    reva_root: str = "data/raw/reva"
    # Pre-extracted frames (scripts/data/extract_frames.py) and per-clip cues
    # (scripts/data/compute_*_cues.py).
    frames_root: str = "data/processed/frames"
    cues_root: str = "data/processed/cues"
    # Fit/dev files from scripts/data/make_splits.py. `dev` mirrors the test
    # regime (seen video, unseen question). Set `train_files` to
    # [train.json, val.json] under reva_root for the final fit on everything.
    train_files: List[str] = ("data/processed/splits/fit.json",)
    dev_file: Optional[str] = "data/processed/splits/dev.json"
    # Labelled pool for the "memory" cue at train/eval time (defaults to the
    # train files). Never include the dev file here.
    memory_files: Optional[List[str]] = None
    # Debug: cap examples per split (0 = all).
    max_train_examples: int = 0
    max_dev_examples: int = 0
    seed: int = 42

    # --- output ---
    ckpt_dir: str = "checkpoints"
    result_dir: str = "results"
    run_name: Optional[str] = None

    # --- optimization ---
    epochs: int = 2
    batch_size: int = 1
    eval_batch_size: int = 4
    accum_steps: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    # Re-draw the option order every time an example is seen (train only),
    # so the model cannot learn letter priors.
    shuffle_options: bool = True
    log_every: int = 10
    # Validate every `eval_every_steps` optimizer steps (0 = once per epoch).
    eval_every_steps: int = 0
    ckpt_every: int = 1
    # Save the adapter to <ckpt_dir>/last.pt every N optimizer steps (0 = off),
    # so a long run that crashes or runs out of time still leaves a usable model.
    save_every_steps: int = 0

    # Optional ModelConfig overrides, e.g. {"backbone": "Qwen/Qwen3.5-4B",
    # "num_frames": 16, "cues": ["motion"], "lora": {"r": 64}}.
    arch: Optional[dict] = None

    # --- evaluation-time decoding ---
    # "none": official option order only. "cyclic": average over the 4 cyclic
    # option shifts (4x cost). "pride": PriDe debiasing with the prior
    # estimated on `pride_fraction` of the split.
    tta: str = "none"
    # Decoding used for in-loop validation and best-checkpoint selection. Kept
    # separate from `tta` so the (often 4x more expensive) serving-time TTA is
    # not paid on every validation pass, while the selector can still match it.
    eval_tta: str = "none"
    pride_fraction: float = 0.05

    # --- callbacks ---
    lr_scheduler: Optional[dict] = None
    early_stopping: Optional[dict] = None
    save_best: bool = True
    best_metric: str = "accuracy"
    best_mode: str = "max"
    wandb: Optional[dict] = None

    resume_from: Optional[str] = None
    # Start from the adapter weights of a checkpoint (fresh optimizer, step 0).
    init_adapter: Optional[str] = None
    # Random frame phase per training example (evaluation stays at 0.5).
    train_frame_phase_jitter: bool = False
    # Questions per packed training sequence (1 = one question per
    # sequence, the original recipe). With k > 1, batch_size must be 1 and an
    # optimizer step covers ~k * accum_steps questions.
    pack_questions: int = 1
    # Leak-free alternative: k questions per video share the prefix and are
    # scored as independent branches (`src.modules.branched`). 1 = off.
    branch_questions: int = 1
    # Counterfactual temporal loss (BPO-inspired, arXiv 2610.03141): for these
    # tasks, also score the clip played backwards (no text cues) and add
    # `cf_weight * -log(1 - p(original answer))`. Training only.
    cf_tasks: List[str] = field(default_factory=list)
    cf_weight: float = 0.5
    # Mixed-format training (arXiv 2505.16178): each branched group also
    # generates its clip's training Q/A pairs as text; loss weight on the
    # mean token CE. 0 = off. Needs branch_questions > 1.
    fact_weight: float = 0.0
    max_facts: int = 24
    # Training-only cue dropout, {cue: p}: each listed cue is left out of a
    # question's prompt with probability p (e.g. {caption: 0.25}).
    cue_dropout: Dict[str, float] = field(default_factory=dict)


def load_training_config(
    config_path: Optional[Union[str, Path]] = None, **cli_overrides
) -> TrainingConfig:
    """Build a TrainingConfig from defaults, a YAML file, then CLI overrides
    (highest precedence, but only applied for keys the caller actually passed —
    argparse defaults of None are treated as "not set").
    """
    cfg = TrainingConfig()

    if config_path is not None:
        yaml_data = read_yaml(config_path)
        known_fields = set(asdict(cfg).keys())
        for key, value in yaml_data.items():
            if key not in known_fields:
                raise ValueError(
                    f"unknown training config key {key!r} in {config_path}"
                )
            setattr(cfg, key, value)

    for key, value in cli_overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    if cfg.tta not in ("none", "cyclic", "pride"):
        raise ValueError(f"tta must be none|cyclic|pride, got {cfg.tta!r}")
    if cfg.eval_tta not in ("none", "cyclic", "pride"):
        raise ValueError(f"eval_tta must be none|cyclic|pride, got {cfg.eval_tta!r}")
    cfg.train_files = list(cfg.train_files)
    return cfg
