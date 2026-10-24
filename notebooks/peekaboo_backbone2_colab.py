# %% [markdown]
# # Peekaboo-VL: second backbone + ensemble (self-contained, Colab)
#
# Drone-video multiple-choice QA on **ReVA** (MAVU @ WACV 2027). This notebook
# trains a **second, different-family backbone** with the H20 recipe and
# ensembles it with our best Qwen3.5-9B model (H20, 83.55 on test). Our earlier
# ensembles failed because the partner was a Qwen relative (Cosmos-Reason2,
# Qwen3.5-27B) that made the same mistakes; a different vision encoder and
# pre-training mix is the point here. Nothing is imported from the project
# repository: every piece is defined in the cells below.
#
# | Family | Checkpoint | Video input | Timestamps |
# |---|---|---|---|
# | `internvl` (default) | `OpenGVLab/InternVL3_5-8B-HF` | one 448x448 tile per frame (256 tokens) | written by us: `Frame3 (1.2s): <image>` |
# | `glm4v` | `zai-org/GLM-4.1V-9B-Thinking` | Qwen2-VL-style video, native 640x360 | GLM's own (whole seconds) |
# | `qwen` | `Qwen/Qwen3.5-9B` (+ H20 adapter) | Qwen video, native 640x360 | Qwen's own `<t seconds>` |
#
# The rest is the H20 recipe: 32 frames with a random sampling phase during
# training, the camera-motion text cue in the answer vocabulary, the progress
# bar on Temporal Grounding frames, LoRA r=64 on every LM linear + the
# vision projector, cross-entropy over the letters `A B C D`, all 14,068 fit
# questions, 1 epoch, PriDe decoding.
#
# **Presets** (`PRESET` in section 1):
#
# | Preset | What it does | A100 time (rough) |
# |---|---|---|
# | `internvl_screen` | InternVL3.5-8B **zero-shot** on dev. Go/no-go before paying for training | ~1 h |
# | `glm_screen` | GLM-4.1V-9B zero-shot on dev | ~1 h |
# | `caption` | **Round-2 S1.** Replay ReVA's own captioner (Qwen3-VL-30B-A3B via vLLM: FP8 on 48 GB cards, bf16 on 80 GB, else AWQ 4-bit) on every clip; writes `cues/caption.*.jsonl`. Shard over 2 sessions with `PEEKABOO_SHARD=0/2` / `1/2`. Fresh runtime (installs vLLM) | ~1-2 h per shard |
# | `caption_check` | S1 go/no-go: Qwen3.5-9B zero-shot, **no video**, caption + question only. Go if dev >= 72 (blind floor 55.5) | ~20 min |
# | `blind_check` / `caption_check_shuffle` | paper controls for `caption_check`: no caption (the blind floor, same code path), and every clip given another clip's caption | ~20 min each |
# | `caption_8b` / `caption_check_8b` | paper control: the same prompts with a different captioner (Qwen3-VL-8B), written to `cues/caption_qwen3vl8b.jsonl`, then its text-only check | ~1 h + ~20 min |
# | `h20_caption` | S1 winner: the H20 recipe (Qwen3.5-9B) + caption cue, dev + test, `submission.zip` | ~10-12 h |
# | `h21` / `h21_caption` | 2 epochs of the H20 recipe (+ caption cue), fit only, best epoch on dev | ~16 h (A100 80 GB) |
# | `h21_final` / `h21_caption_final` | same on fit + dev (all labelled data), last epoch, `submission.zip` | ~18-20 h |
# | `qwen3vl30b_train` | S2 fine-tune of the 30B (LoRA on attention), **80 GB only** | ~8-12 h |
# | `qwen3vl30b_screen` | **Round-2 S2.** Qwen3-VL-30B-A3B (the model that wrote ReVA's QAs) zero-shot, bf16 with the overflow layers in CPU RAM (40-48 GB GPUs); dev, then test scores for the ensemble | ~2 h dev + ~4 h test |
# | `internvl` | fine-tune InternVL3.5-8B (H20 recipe), dev + test, `submission.zip` | ~12-16 h |
# | `glm` | fine-tune GLM-4.1V-9B | ~10-14 h |
# | `h20_score` | re-score our H20 adapter (from HF) on dev + test, for the ensemble | ~3 h |
# | `smoke` | InternVL3.5-1B on a handful of questions, minutes | ~10 min |
#
# **Round 2** (`docs/research/round2-suggestions.md`): ReVA's questions were
# written by Qwen3-VL-30B-A3B from a caption it wrote of each clip (paper App.
# I). Two sessions in parallel: A runs `caption` with `PEEKABOO_SHARD=0/2`, B
# with `1/2`; then A runs `caption_check` (merges the shards) and B runs
# `qwen3vl30b_screen`. If `caption_check` passes, run `h20_caption`.
#
# **Plan.** (1) Run a `*_screen` preset; fine-tune only if zero-shot dev is
# within ~3 points of Qwen3.5-9B zero-shot (79.6 dev, 16 frames). (2) Run the
# fine-tune preset. (3) Run `h20_score` (or reuse its saved scores). (4)
# Section 16 fits one ensemble weight per task out-of-fold on dev and writes
# the ensemble `submission.zip`; ship it only if test-weighted dev gains
# >= 0.8 points over H20 alone.
#
# **Runtime.** A100 40/80 GB (8-9B bf16 weights ~17-20 GB). InternVL at 32
# frames is ~8.2k visual tokens per question (Qwen ~3.5k), so it is the
# slowest; `num_frames=16` halves that if time is short. Checkpoints and
# scores go to Google Drive, and training resumes from `last.pt` after a
# disconnect.
#
# **Data.** ReVA is research / non-commercial only and may not be
# redistributed (arXiv 2609.35507). The full release is ~30 GB; mp4s are
# deleted after their frames are cached.
#
# Sections: 0 setup · 1 config · 2 labels · 3 data & splits · 4 download ·
# 5 frame cache · 6 camera-motion cue · 7 prompts · 8 dataset & collator ·
# 9 model · 10 decoding & metrics · 11 evaluation · 12 trainer · 13 build &
# train · 14 dev evaluation · 15 test predictions · 16 ensemble

# %% [markdown]
# ## 0. Setup

# %%
import os
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
# The preset decides what to install (vLLM only for `caption`), so it is read here.
PRESET = os.environ.get("PEEKABOO_PRESET", "internvl_screen")
# Same library versions as the H20 run (Colab keeps its own torch build).
# flash-linear-attention only matters for Qwen3.5 (h20_score); harmless otherwise.
PACKAGES = [
    "transformers==5.18.0",
    "peft==0.21.1",
    "accelerate==1.15.0",
    "huggingface_hub>=1.0",
    "opencv-python-headless>=4.10",
    "flash-linear-attention==0.5.2",  # fast Gated-DeltaNet kernels for Qwen3.5
    "wandb>=0.19",  # optional experiment tracking (see `wandb` in the config)
]
if PRESET in ("caption", "caption_8b"):
    # vLLM (released 2026-09-22) runs the FP8 / AWQ MoE captioner on a 40-48 GB GPU. It
    # installs its own torch, so use a fresh runtime and run this cell before anything
    # imports torch.
    PACKAGES.append("vllm==0.30.0")
# Optional ~8 min build; speeds up the short convolution in Qwen3.5's linear layers.
BUILD_CAUSAL_CONV1D = False

if IN_COLAB or os.environ.get("PEEKABOO_INSTALL") == "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)
    if BUILD_CAUSAL_CONV1D:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "causal-conv1d>=1.5"],
            check=True,
            env={**os.environ, "CAUSAL_CONV1D_FORCE_BUILD": "TRUE"},
        )

# %%
import json
import math
import random
import re
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import transformers
from tqdm.auto import tqdm

cv2.setNumThreads(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print("torch", torch.__version__, "| transformers", transformers.__version__, "| device", DEVICE)
if DEVICE.type == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"GPU {p.name}, {p.total_memory / 2**30:.0f} GB, bf16 {torch.cuda.is_bf16_supported()}")

# %% [markdown]
# ## 1. Configuration
#
# One dataclass holds every knob. `PRESET` picks a starting point; edit any
# field afterwards. `mode` decides what section 13 does:
#
# * `"screen"`: no training; score dev with the frozen backbone (zero-shot).
# * `"train"`: fine-tune, then score dev (and test if `score_test`).
# * `"score"`: no training; load `init_adapter` (e.g. H20 from HF) and score.
#
# Outputs (cues, checkpoints, scores) go to `OUT_DIR`, which should be on
# Google Drive so a disconnect does not lose work.

# %%
# PRESET is read in section 0 (it decides whether vLLM is installed).


@dataclass
class Config:
    # model
    backbone: str = "OpenGVLab/InternVL3_5-8B-HF"
    family: str = ""  # "qwen" | "glm4v" | "internvl"; "" = detect from the name
    # Text after the generation prompt, before the letter (GLM-4.1V-Thinking
    # answers as "<think>...</think><answer>X").
    answer_prefix: str = ""
    torch_dtype: str = "bfloat16"
    # > 0: keep at most this many GB of weights on the GPU, the rest in CPU RAM
    # (moved in per forward pass). Scoring only; for models bigger than the GPU.
    # -1 = the GPU's memory minus 6 GB (34 on a 40 GB A100, 42 on a 48 GB card).
    gpu_max_memory_gb: int = 0
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_merger: bool = True  # also adapt the vision -> LLM projector
    gradient_checkpointing: bool = True
    mode: str = "train"  # "screen" | "train" | "score"
    # "user/repo/path/best.pt" on the HF Hub, or a local path; loaded before
    # scoring ("score") or as the starting point of training ("train").
    init_adapter: str = ""
    score_test: bool = True
    # video input
    num_frames: int = 32
    cache_frames: int = 64  # frames decoded once per clip; training subsamples them
    max_pixels_per_frame: int = 640 * 360  # qwen / glm4v; InternVL tiles are 448x448
    overlay_time_tasks: Tuple[str, ...] = ("Temporal Grounding",)
    train_phase_jitter: bool = True
    motion_frames: int = 32  # frames used to estimate camera motion
    use_video: bool = True  # False = text-only probe (no frames in the prompt)
    motion_cue: bool = True  # the camera-motion line
    caption_cue: bool = False  # the replayed generator caption (section 6b)
    # generator-replay captions (preset `caption`, section 6b)
    # "" = by GPU memory: bf16 (~61 GB, the generator exactly) on >= 70 GB, the
    # official FP8 checkpoint (~32 GB, close to bf16) on >= 44 GiB (A40, L40S and A6000 report 45-48), else AWQ 4-bit (~18 GB). vLLM runs FP8 natively on Ada/Hopper and as
    # weight-only FP8 on Ampere (A100, A6000).
    caption_model: str = ""
    caption_shard: str = os.environ.get("PEEKABOO_SHARD", "0/1")  # "i/n": every n-th clip from i
    caption_segment_frames: int = 16  # per segment (3 segments per clip)
    caption_consolidate_frames: int = 32
    caption_max_tokens: int = 400
    caption_chunk: int = 32  # clips per vLLM call; rows are saved after each
    caption_limit: int = int(os.environ.get("PEEKABOO_CAPTION_LIMIT", "0"))  # > 0: smoke test on N clips
    # Caption files are cues/<caption_name>.jsonl (merged) and <caption_name>.<i>of<n>.jsonl
    # (shards), so a second captioner (a control) never mixes with the generator's.
    caption_name: str = "caption"
    caption_shuffle: bool = False  # control: every clip gets another clip's caption
    # data
    holdout_per_task: int = 150
    max_train: int = 0  # 0 = all fit questions
    # Final fit: train on fit + dev (all labelled data). No held-out selection then:
    # the last epoch is saved as best.pt and the dev score printed is on seen questions.
    train_on_dev: bool = False
    max_dev: int = 0
    max_test: int = 0
    seed: int = 42
    # optimisation
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    accum_steps: int = 16  # questions per optimizer step (batch size 1)
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    save_every_steps: int = 100
    log_every: int = 10
    num_workers: int = 4
    # Weights & Biases: needs the Colab secret / env var WANDB_API_KEY; without it
    # the notebook prints a note and carries on without tracking.
    wandb: bool = True
    wandb_project: str = "peekaboo-vl"
    wandb_group: str = "backbone2"
    # decoding
    tta: str = "pride"  # "none" | "cyclic" | "pride"
    pride_fraction: float = 0.05
    eval_batch_size: int = 2
    # paths
    work_dir: str = "/content/peekaboo"  # fast local disk: videos, frames
    out_dir: str = "/content/drive/MyDrive/peekaboo_ensemble"  # persistent outputs
    run_name: str = "internvl35_8b_screen"


INTERNVL = dict(backbone="OpenGVLab/InternVL3_5-8B-HF")
GLM = dict(backbone="zai-org/GLM-4.1V-9B-Thinking", answer_prefix="<think></think><answer>")
QWEN9B = dict(backbone="Qwen/Qwen3.5-9B")
PRESETS = {
    "internvl_screen": dict(**INTERNVL, mode="screen", score_test=False, run_name="internvl35_8b_screen"),
    "glm_screen": dict(**GLM, mode="screen", score_test=False, run_name="glm41v_9b_screen"),
    # Round 2 (docs/research/round2-suggestions.md). S1: replay ReVA's captioner.
    "caption": dict(mode="caption", score_test=False, run_name="caption_replay"),
    "caption_check": dict(
        **QWEN9B, mode="screen", score_test=False, use_video=False, motion_cue=False,
        caption_cue=True, eval_batch_size=8, run_name="caption_check_qwen35_9b_textonly",
    ),
    # Controls for the paper: the blind floor through the same code path, captions
    # shuffled across clips, and the same prompts with a smaller, different captioner.
    "blind_check": dict(
        **QWEN9B, mode="screen", score_test=False, use_video=False, motion_cue=False,
        eval_batch_size=8, run_name="blind_check_qwen35_9b_textonly",
    ),
    "caption_check_shuffle": dict(
        **QWEN9B, mode="screen", score_test=False, use_video=False, motion_cue=False,
        caption_cue=True, caption_shuffle=True, eval_batch_size=8,
        run_name="caption_check_shuffle_qwen35_9b_textonly",
    ),
    "caption_8b": dict(mode="caption", score_test=False, caption_model="Qwen/Qwen3-VL-8B-Instruct",
                       caption_name="caption_qwen3vl8b", run_name="caption_replay_qwen3vl8b"),
    "caption_check_8b": dict(
        **QWEN9B, mode="screen", score_test=False, use_video=False, motion_cue=False,
        caption_cue=True, caption_name="caption_qwen3vl8b", eval_batch_size=8,
        run_name="caption_check_8b_qwen35_9b_textonly",
    ),
    "h20_caption": dict(**QWEN9B, mode="train", caption_cue=True,
                        run_name="h23_qwen35_9b_32f_bar_caption_full_1ep"),
    # Round 2, longer runs (A100 80 GB times): 2 epochs of the H20 recipe; select on dev
    # (~16 h), or train on fit + dev for the submission (`*_final`, ~18-20 h).
    "h21": dict(**QWEN9B, mode="train", epochs=2, run_name="h21a_qwen35_9b_32f_bar_fit_2ep"),
    "h21_caption": dict(**QWEN9B, mode="train", epochs=2, caption_cue=True,
                        run_name="h21_qwen35_9b_32f_bar_caption_fit_2ep"),
    "h21_final": dict(**QWEN9B, mode="train", epochs=2, train_on_dev=True,
                      run_name="h21_qwen35_9b_32f_bar_all_2ep"),
    "h21_caption_final": dict(**QWEN9B, mode="train", epochs=2, train_on_dev=True, caption_cue=True,
                              run_name="h21_qwen35_9b_32f_bar_caption_all_2ep"),
    # S2 fine-tune (80 GB GPUs only): bf16 30B MoE, LoRA on attention + projector (the
    # experts are 3-D parameters, which LM_PATTERN does not match). Run only if
    # qwen3vl30b_screen reached ~76.6 dev.
    "qwen3vl30b_train": dict(
        backbone="Qwen/Qwen3-VL-30B-A3B-Instruct", mode="train", lora_r=32, lora_alpha=64,
        eval_batch_size=1, num_workers=4, run_name="h24_qwen3vl_30b_a3b_32f_bar_full_1ep",
    ),
    # S2: ReVA's QA generator itself (arXiv 2609.35507, Sec. 3.3); MoE, 3B active, bf16
    # ~61 GB. bitsandbytes cannot shrink it (its experts are 3-D tensors, not Linear
    # layers), so GPU memory minus 6 GB stays on the GPU and the rest is streamed from CPU RAM.
    # Dev is printed first: stop the run if it is far below 76.6.
    "qwen3vl30b_screen": dict(
        backbone="Qwen/Qwen3-VL-30B-A3B-Instruct", mode="screen", score_test=True,
        gpu_max_memory_gb=-1, eval_batch_size=1, num_workers=2, run_name="qwen3vl_30b_a3b_screen",
    ),
    "internvl": dict(**INTERNVL, mode="train", run_name="h22_internvl35_8b_32f_bar_full_1ep"),
    "glm": dict(**GLM, mode="train", run_name="h22_glm41v_9b_32f_bar_full_1ep"),
    # Our best model (83.55 test): Qwen3.5-9B + the H20 adapter. The HF repo is
    # private: the HF_TOKEN secret must have read access to it.
    "h20_score": dict(
        backbone="Qwen/Qwen3.5-9B", mode="score", run_name="h20_32f_bar_full_1ep_9b",
        init_adapter="LakoreAI/peekaboo-storage/h20_32f_bar_full_1ep_9b/best.pt",
    ),
    "smoke": dict(
        backbone="OpenGVLab/InternVL3_5-1B-HF", num_frames=8, cache_frames=16, motion_frames=8,
        max_train=16, max_dev=8, max_test=8, accum_steps=4, lora_r=8, lora_alpha=16,
        save_every_steps=2, log_every=1, num_workers=0, wandb=False, run_name="smoke_internvl_1b",
    ),
}
cfg = replace(Config(), **PRESETS[PRESET])
if not IN_COLAB:  # local runs: keep everything under ./notebook_runs
    cfg = replace(cfg, work_dir=os.environ.get("PEEKABOO_WORK", "notebook_runs/work"),
                  out_dir=os.environ.get("PEEKABOO_OUT", "notebook_runs/out"))
if DEVICE.type != "cuda":
    cfg = replace(cfg, num_workers=0)
else:  # "auto" settings that depend on the GPU's memory
    GPU_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
    if cfg.gpu_max_memory_gb < 0:
        cfg = replace(cfg, gpu_max_memory_gb=int(GPU_GB - 6))
    if cfg.mode == "caption" and not cfg.caption_model:  # bf16 = the generator exactly
        cfg = replace(cfg, caption_model="Qwen/Qwen3-VL-30B-A3B-Instruct" if GPU_GB >= 70
                      else "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8" if GPU_GB >= 44
                      else "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ")
    if "30B" in cfg.backbone and cfg.mode == "train":
        assert GPU_GB >= 70, "fine-tuning the 30B model needs an 80 GB GPU"


def detect_family(backbone: str, family: str = "") -> str:
    if family:
        assert family in ("qwen", "glm4v", "internvl"), family
        return family
    name = backbone.lower()
    if "internvl" in name:
        return "internvl"
    if "glm-4" in name or "glm4" in name:
        return "glm4v"
    return "qwen"


FAMILY = detect_family(cfg.backbone, cfg.family)
assert FAMILY == "internvl" or cfg.num_frames % 2 == 0, "Qwen/GLM merge 2 frames per temporal patch"
print("family:", FAMILY)
print(json.dumps(asdict(cfg), indent=1))

# %%
# Google Drive for persistent outputs (Colab only). Hugging Face token is
# optional for the public dataset and backbones, REQUIRED for `h20_score` (private
# LakoreAI/peekaboo-storage adapter): add it as the Colab secret HF_TOKEN.
if IN_COLAB and cfg.out_dir.startswith("/content/drive"):
    from google.colab import drive

    drive.mount("/content/drive")
if IN_COLAB:
    try:
        from google.colab import userdata

        os.environ.setdefault("HF_TOKEN", userdata.get("HF_TOKEN"))
    except Exception:
        print("no HF_TOKEN secret; downloading anonymously")
    try:
        os.environ.setdefault("WANDB_API_KEY", userdata.get("WANDB_API_KEY"))
    except Exception:
        pass  # no W&B secret: tracking is skipped below

WORK = Path(cfg.work_dir)
OUT = Path(cfg.out_dir)
REVA_ROOT = Path(os.environ.get("PEEKABOO_REVA_ROOT", WORK / "reva"))
FRAMES_ROOT = WORK / "frames"
CUES_DIR = OUT / "cues"
CKPT_DIR = OUT / "checkpoints" / cfg.run_name
RESULTS_DIR = OUT / "results" / cfg.run_name
for d in (WORK, OUT, REVA_ROOT, FRAMES_ROOT, CUES_DIR, CKPT_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

# Weights & Biases, started on the first log call. The run id is the run name, so
# a Colab restart that resumes from `last.pt` continues the same W&B run.
_WB = {"run": None, "tried": False}


def wb_run():
    if _WB["tried"] or not cfg.wandb:
        return _WB["run"]
    _WB["tried"] = True
    if not os.environ.get("WANDB_API_KEY"):
        print("W&B off: add the Colab secret WANDB_API_KEY (or set cfg.wandb=False to silence this)")
        return None
    try:
        import wandb

        _WB["run"] = wandb.init(project=cfg.wandb_project, group=cfg.wandb_group, name=cfg.run_name,
                                id=cfg.run_name, resume="allow", config=asdict(cfg),
                                dir=str(WORK), reinit=True)
    except Exception as e:  # tracking must never break a 16 h run
        print("W&B disabled:", repr(e))
    return _WB["run"]


def wb_log(data: dict, step: Optional[int] = None) -> None:
    run = wb_run()
    if run is not None:
        run.log(data, step=step)


def wb_eval(prefix: str, m: dict, step: Optional[int] = None, summary: bool = False) -> None:
    """Log accuracy, test-weighted accuracy, per-task accuracy and the letter mix."""
    if not m.get("n"):
        return
    flat = {f"{prefix}/accuracy": m["accuracy"], f"{prefix}/test_weighted_accuracy": m["test_weighted_accuracy"]}
    flat.update({f"{prefix}/task/{k}": v for k, v in m["per_task"].items()})
    flat.update({f"{prefix}/letter_{k}": v for k, v in m["pred_letter_dist"].items()})
    run = wb_run()
    if run is None:
        return
    if summary:
        run.summary.update(flat)
    else:
        run.log(flat, step=step)


def wb_finish() -> None:
    if _WB["run"] is not None:
        _WB["run"].finish()
        _WB["run"], _WB["tried"] = None, False

# %% [markdown]
# ## 2. Labels and task taxonomy
#
# The four answer letters are the classes. The 11 tasks / 4 categories and
# their short names match the Codabench leaderboard columns.
# `TEST_TASK_COUNTS` (the hidden test set's task mix) gives the
# **test-weighted dev accuracy**, which predicted the leaderboard score within
# 0.2 points; select models on it, not on raw dev accuracy.

# %%
LETTERS = ("A", "B", "C", "D")
LETTER_TO_INDEX = {l: i for i, l in enumerate(LETTERS)}
TASKS = {
    "General Understanding": ("Factual Perception", "FP_General"),
    "Object and Land Cover Recognition": ("Factual Perception", "FP_ObjectLand"),
    "Change Detection": ("Factual Perception", "FP_Change"),
    "Temporal Grounding": ("Temporal Understanding", "TU_Temporal"),
    "Trend and Pattern": ("Temporal Understanding", "TU_TrendPattern"),
    "Geometric Relation": ("Spatial Reasoning", "SR_Geometric"),
    "Structural Layout": ("Spatial Reasoning", "SR_Structural"),
    "Perspective and Viewpoint": ("Spatial Reasoning", "SR_Viewpoint"),
    "Causation Reasoning": ("Causal Reasoning", "CR_Causation"),
    "Consequence Reasoning": ("Causal Reasoning", "CR_Consequence"),
    "Hypothetical Reasoning": ("Causal Reasoning", "CR_Hypothetical"),
}
TEST_TASK_COUNTS = {
    "General Understanding": 180, "Object and Land Cover Recognition": 660,
    "Change Detection": 500, "Temporal Grounding": 640, "Trend and Pattern": 360,
    "Geometric Relation": 400, "Structural Layout": 340, "Perspective and Viewpoint": 480,
    "Causation Reasoning": 180, "Consequence Reasoning": 100, "Hypothetical Reasoning": 160,
}


def task_short(task: str) -> str:
    return TASKS[task][1] if task in TASKS else task

# %% [markdown]
# ## 3. Data model and the leak-free dev split
#
# Facts about the release that shape the split (checked on the data):
#
# * **Test = new questions about seen videos.** 1,012 of 1,014 test clips are
#   also train clips. Dev must hold out *questions*, not videos.
# * **Val is 79% duplicates of train** (same video + same question), so raw
#   val accuracy after training on train measures memorisation.
#
# `make_dev_split` builds dev from the val non-duplicates plus 150 held-out
# train questions per task, and removes every record that asks a dev question
# from the training set (`fit`). Result: fit 14,068 / dev 2,078 questions.

# %%
@dataclass(frozen=True)
class ReVAExample:
    qa_id: str
    video_path: str
    task: str
    category: str
    question: str
    options: Tuple[str, ...]  # options[i] is shown as LETTERS[i]
    label: int = -1  # index of the correct option; -1 = unlabeled (test)
    perm: Tuple[int, ...] = (0, 1, 2, 3)  # display i shows original option perm[i]

    @property
    def answer_letter(self) -> Optional[str]:
        return LETTERS[self.label] if self.label >= 0 else None


def example_from_record(rec: dict) -> ReVAExample:
    answer = (rec.get("correct_answer") or "").strip()
    return ReVAExample(
        qa_id=rec["qa_id"], video_path=rec["video_path"], task=rec.get("task", ""),
        category=rec.get("category", ""), question=rec["question"],
        options=tuple(str(rec["options"][k]) for k in LETTERS),
        label=LETTER_TO_INDEX[answer] if answer else -1,
    )


def load_split(path: Path) -> List[ReVAExample]:
    data = json.loads(Path(path).read_text())
    return [example_from_record(r) for r in (data["QA"] if isinstance(data, dict) else data)]


def permute(ex: ReVAExample, perm: Sequence[int]) -> ReVAExample:
    """Display position i shows current option perm[i]; composes with ex.perm."""
    perm = tuple(int(p) for p in perm)
    return replace(
        ex,
        options=tuple(ex.options[p] for p in perm),
        label=perm.index(ex.label) if ex.label >= 0 else -1,
        perm=tuple(ex.perm[p] for p in perm),
    )


def cyclic_perms(n: int = 4) -> List[Tuple[int, ...]]:
    return [tuple((i + s) % n for i in range(n)) for s in range(n)]


def question_key(ex: ReVAExample) -> Tuple[str, str]:
    return ex.video_path, re.sub(r"\s+", " ", ex.question.strip().lower())


def make_dev_split(train, val, holdout_per_task=150, seed=42):
    rng = random.Random(seed)
    train_keys = {question_key(ex) for ex in train}
    dev = [ex for ex in val if question_key(ex) not in train_keys]
    by_task = defaultdict(list)
    for ex in train:
        by_task[ex.task].append(ex)
    taken = {question_key(ex) for ex in dev}
    for task in sorted(by_task):
        pool = [ex for ex in by_task[task] if question_key(ex) not in taken]
        rng.shuffle(pool)
        for ex in pool[:holdout_per_task]:
            dev.append(ex)
            taken.add(question_key(ex))
    dev_keys = {question_key(ex) for ex in dev}
    dev_ids = {ex.qa_id for ex in dev}
    fit = [ex for ex in list(train) + list(val)
           if ex.qa_id not in dev_ids and question_key(ex) not in dev_keys]
    seen, deduped = set(), []
    for ex in fit:  # drop repeated (video, question) records
        if question_key(ex) not in seen:
            seen.add(question_key(ex))
            deduped.append(ex)
    return deduped, sorted(dev, key=lambda ex: ex.qa_id)

# %% [markdown]
# ## 4. Download ReVA and build the splits

# %%
from huggingface_hub import snapshot_download

REPO_ID = "ReVA-Benchmark/ReVA"
ANNOTATIONS = ["train.json", "val.json", "test.json"]

if not all((REVA_ROOT / f).exists() for f in ANNOTATIONS):
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=REVA_ROOT, allow_patterns=ANNOTATIONS)

train_all = load_split(REVA_ROOT / "train.json")
val_all = load_split(REVA_ROOT / "val.json")
test_all = load_split(REVA_ROOT / "test.json")
fit, dev = make_dev_split(train_all, val_all, cfg.holdout_per_task, cfg.seed)
test = list(test_all)
print(f"train {len(train_all)}  val {len(val_all)}  test {len(test_all)}  ->  fit {len(fit)}  dev {len(dev)}")

# Optional subsets (presets "small" / "smoke"). Smoke keeps a few videos only.
rng = random.Random(cfg.seed)
if cfg.max_train:
    fit = rng.sample(fit, min(cfg.max_train, len(fit)))
if cfg.max_dev:
    dev = rng.sample(dev, min(cfg.max_dev, len(dev)))
if cfg.max_test:
    test = test[: cfg.max_test]
if cfg.train_on_dev:
    fit = fit + dev
    print("train_on_dev: training on fit + dev; dev scores below are on SEEN questions")
print(f"using fit {len(fit)}  dev {len(dev)}  test {len(test)}")
print("dev tasks:", dict(sorted(Counter(task_short(e.task) for e in dev).items())))

# %%
# Videos: only the clips referenced by the splits we use. With the full data
# this is ~30 GB; set DELETE_VIDEOS to remove each mp4 after its frames are cached.
DELETE_VIDEOS = os.environ.get("PEEKABOO_DELETE_VIDEOS", "1" if IN_COLAB else "0") == "1"
needed = sorted({ex.video_path for ex in fit + dev + test})
missing = [v for v in needed if not (REVA_ROOT / v).exists()
           and not (FRAMES_ROOT / Path(v).with_suffix("") / "meta.json").exists()]
print(f"{len(needed)} clips needed, {len(missing)} not on disk")
if missing:
    # allow_patterns with exact paths downloads just these files.
    for i in range(0, len(missing), 200):
        snapshot_download(REPO_ID, repo_type="dataset", local_dir=REVA_ROOT,
                          allow_patterns=missing[i : i + 200], max_workers=16)

# %% [markdown]
# ## 5. Frame cache with true timestamps
#
# Each clip is decoded once into `cache_frames` uniformly sampled JPEGs plus a
# `meta.json` with the source **fps** and the sampled **frame indices**.
# Qwen3.5 writes a `<t seconds>` timestamp before each temporal patch; without
# fps + indices transformers assumes 24 fps and every timestamp is wrong (ReVA
# clips are 24-30 fps and 0.1-40 s long).
#
# Sampling takes the frame at relative position `phase` inside each of N equal
# segments (0.5 = centre). Training draws a random phase per question
# (H20), so the model sees many timestamp layouts.

# %%
@dataclass
class FrameSet:
    frames: np.ndarray  # (T, H, W, 3) RGB uint8
    frame_indices: List[int]
    fps: float
    total_frames: int

    @property
    def timestamps(self) -> List[float]:
        return [i / self.fps for i in self.frame_indices]


def uniform_indices(total: int, num: int, phase: float = 0.5) -> List[int]:
    seg = total / num
    return [min(int(seg * i + seg * phase), total - 1) for i in range(num)]


def read_video(path: Path, num_frames: int) -> FrameSet:
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if total <= 0:
            raise ValueError(f"cannot read {path}")
        wanted = set(uniform_indices(total, num_frames))
        frames, kept = [], []
        for idx in range(max(wanted) + 1):  # sequential decode beats seeking on short clips
            if idx in wanted:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                kept.append(idx)
            elif not cap.grab():
                break
    finally:
        cap.release()
    if len(set(kept)) < 0.9 * len(wanted):
        raise ValueError(f"{path}: decoded {len(kept)} of {len(wanted)} frames")
    return FrameSet(np.stack(frames), kept, fps, total)


class FrameCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def dir(self, video_path: str) -> Path:
        return self.root / Path(video_path).with_suffix("")

    def has(self, video_path: str) -> bool:
        return (self.dir(video_path) / "meta.json").exists()

    def save(self, video_path: str, fs: FrameSet) -> None:
        d = self.dir(video_path)
        d.mkdir(parents=True, exist_ok=True)
        for k, frame in enumerate(fs.frames):
            cv2.imwrite(str(d / f"{k:03d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        (d / "meta.json").write_text(json.dumps({
            "fps": fs.fps, "total_frames": fs.total_frames, "frame_indices": fs.frame_indices}))

    def load(self, video_path: str, num_frames: int, phase: float = 0.5) -> FrameSet:
        d = self.dir(video_path)
        meta = json.loads((d / "meta.json").read_text())
        n = len(meta["frame_indices"])
        pos = list(range(n)) if num_frames >= n else uniform_indices(n, num_frames, phase)
        frames = [cv2.cvtColor(cv2.imread(str(d / f"{k:03d}.jpg")), cv2.COLOR_BGR2RGB) for k in pos]
        return FrameSet(np.stack(frames), [meta["frame_indices"][k] for k in pos],
                        float(meta["fps"]), int(meta["total_frames"]))


def overlay_time_bar(frames: np.ndarray, timestamps: Sequence[float], bar_height: int = 16) -> np.ndarray:
    """VTimeCoT-style visual time cue: a filling progress bar + the frame's time."""
    out = np.array(frames, copy=True)
    t, h, w = out.shape[:3]
    t_end = max(float(timestamps[-1]), 1e-6)
    for i in range(t):
        top = h - bar_height
        cv2.rectangle(out[i], (0, top), (w, h), (0, 0, 0), -1)
        x = int(min(max(float(timestamps[i]) / t_end, 0.0), 1.0) * w)
        cv2.rectangle(out[i], (0, top), (x, h), (235, 64, 52), -1)
        cv2.putText(out[i], f"{float(timestamps[i]):.1f}s", (4, h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


cache = FrameCache(FRAMES_ROOT)


def extract_one(video_path: str) -> Optional[str]:
    if cache.has(video_path):
        return None
    try:
        cache.save(video_path, read_video(REVA_ROOT / video_path, cfg.cache_frames))
        if DELETE_VIDEOS:
            (REVA_ROOT / video_path).unlink(missing_ok=True)
        return None
    except Exception as e:  # report, keep going
        return f"{video_path}: {e}"


t0 = time.time()
with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
    errors = [e for e in tqdm(pool.map(extract_one, needed), total=len(needed), desc="frames", unit="clip") if e]
print(f"frames cached for {sum(cache.has(v) for v in needed)}/{len(needed)} clips in {time.time() - t0:.0f}s")
for e in errors[:10]:
    print("  ERROR", e)

# %% [markdown]
# ## 6. Camera-motion cue ("tell, don't learn")
#
# Drone footage is close to planar, so background motion between frames is
# well described by a 2-D similarity transform. Per frame pair:
# ORB keypoints -> Hamming matches (Lowe ratio) -> RANSAC similarity `M`; at the
# image centre read the shift (pan / forward flight), scale (approach / pull
# away) and rotation (yaw). Steps are labelled with per-second thresholds,
# merged into segments, and written **in the dataset's answer vocabulary**
# ("moving forward", "panning right", "stationary") plus the dominant motion.
# Wording matters: the same estimates phrased in image terms ("pans toward the
# top", "zooms in") *lowered* accuracy on camera-motion questions (H2 vs H2.1).
#
# Example: `Camera motion: 0.0-6.2s moving forward slowly; 6.2-15.0s panning
# right slowly. Mostly moving forward (41% of the clip).`

# %%
@dataclass
class MotionConfig:
    n_features: int = 1500
    ratio: float = 0.75
    min_matches: int = 25
    ransac_threshold: float = 3.0
    static_shift: float = 0.015  # frame diagonals per second
    fast_shift: float = 0.08
    zoom_rate: float = 0.015  # |log scale| per second
    yaw_rate: float = 1.5  # degrees per second
    min_segment: float = 0.8  # seconds


@dataclass
class MotionStep:
    t0: float
    t1: float
    ok: bool
    dx: float = 0.0  # scene shift of the image centre (px); camera moves opposite
    dy: float = 0.0
    scale: float = 1.0
    rot_ccw: float = 0.0  # degrees


class CameraMotionCue:
    names = {"forward": "moving forward", "backward": "moving backward",
             "pan_right": "panning right", "pan_left": "panning left",
             "rotating": "rotating", "stationary": "stationary", "unclear": "unclear"}

    def __init__(self, mcfg: MotionConfig = MotionConfig()):
        self.cfg = mcfg

    def estimate_pair(self, orb, prev: np.ndarray, nxt: np.ndarray) -> Optional[dict]:
        c = self.cfg
        g0, g1 = (cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in (prev, nxt))
        k0, d0 = orb.detectAndCompute(g0, None)
        k1, d1 = orb.detectAndCompute(g1, None)
        if d0 is None or d1 is None or len(k0) < c.min_matches or len(k1) < c.min_matches:
            return None
        good = [p[0] for p in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0, d1, k=2)
                if len(p) == 2 and p[0].distance < c.ratio * p[1].distance]
        if len(good) < c.min_matches:
            return None
        p0 = np.float32([k0[m.queryIdx].pt for m in good])
        p1 = np.float32([k1[m.trainIdx].pt for m in good])
        M, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                           ransacReprojThreshold=c.ransac_threshold)
        if M is None:
            return None
        h, w = g0.shape[:2]
        centre = np.array([w / 2.0, h / 2.0, 1.0])
        moved = M @ centre
        return {"dx": float(moved[0] - centre[0]), "dy": float(moved[1] - centre[1]),
                "scale": float(math.hypot(M[0, 0], M[1, 0])),
                "rot_ccw": float(-math.degrees(math.atan2(M[1, 0], M[0, 0])))}

    def steps(self, fs: FrameSet) -> List[MotionStep]:
        orb = cv2.ORB_create(nfeatures=self.cfg.n_features)
        ts, out = fs.timestamps, []
        for i in range(len(fs.frames) - 1):
            est = self.estimate_pair(orb, fs.frames[i], fs.frames[i + 1])
            out.append(MotionStep(ts[i], ts[i + 1], ok=False) if est is None
                       else MotionStep(ts[i], ts[i + 1], ok=True, **est))
        return out

    def label(self, s: MotionStep, diag: float) -> Tuple[str, str]:
        c = self.cfg
        if not s.ok:
            return "unclear", "unclear"
        dt = max(s.t1 - s.t0, 1e-6)
        shift = math.hypot(s.dx, s.dy) / diag / dt
        zoom = math.log(max(s.scale, 1e-6)) / dt
        yaw = s.rot_ccw / dt
        cam_x, cam_y = -s.dx, -s.dy
        moving = shift >= c.static_shift
        speed = "quickly" if shift >= c.fast_shift else "slowly"
        if moving and abs(cam_x) >= abs(cam_y):
            primary = "pan_right" if cam_x > 0 else "pan_left"
            phrase = f"panning {'right' if cam_x > 0 else 'left'} {speed}"
            if zoom >= c.zoom_rate:
                phrase += " while moving forward"
            elif zoom <= -c.zoom_rate:
                phrase += " while moving backward"
        elif (moving and cam_y < 0) or zoom >= c.zoom_rate:
            primary = "forward"
            phrase = f"moving forward {speed}" if moving else "moving closer (descending or zooming in)"
        elif (moving and cam_y > 0) or zoom <= -c.zoom_rate:
            primary = "backward"
            phrase = f"moving backward {speed}" if moving else "pulling away (ascending or zooming out)"
        else:
            primary, phrase = "stationary", "stationary"
        if abs(yaw) >= c.yaw_rate:
            turn = "clockwise" if yaw > 0 else "counter-clockwise"
            if primary == "stationary":
                primary, phrase = "rotating", f"rotating {turn}"
            else:
                phrase += f" while rotating {turn}"
        return primary, phrase

    def segments(self, steps: List[MotionStep], labels: List[str]) -> List[dict]:
        segs: List[dict] = []
        for s, lab in zip(steps, labels):
            if segs and segs[-1]["label"] == lab:
                segs[-1]["t1"] = s.t1
            else:
                segs.append({"t0": s.t0, "t1": s.t1, "label": lab})
        changed = True
        while changed and len(segs) > 1:  # short segments go to the longer neighbour
            changed = False
            for i, s in enumerate(segs):
                if s["t1"] - s["t0"] < self.cfg.min_segment:
                    nbrs = [n for n in (segs[i - 1] if i > 0 else None,
                                        segs[i + 1] if i + 1 < len(segs) else None) if n]
                    tgt = max(nbrs, key=lambda n: n["t1"] - n["t0"])
                    tgt["t0"], tgt["t1"] = min(tgt["t0"], s["t0"]), max(tgt["t1"], s["t1"])
                    segs.pop(i)
                    changed = True
                    break
        merged: List[dict] = []
        for s in segs:
            if merged and merged[-1]["label"] == s["label"]:
                merged[-1]["t1"] = s["t1"]
            else:
                merged.append(dict(s))
        return merged

    def describe(self, fs: FrameSet) -> str:
        steps = self.steps(fs)
        if not steps:
            return ""
        h, w = fs.frames.shape[1:3]
        labelled = [self.label(s, math.hypot(w, h)) for s in steps]
        segs = self.segments(steps, [p for _, p in labelled])
        body = "; ".join(f"{s['t0']:.1f}-{s['t1']:.1f}s {s['label']}" for s in segs)
        share: Dict[str, float] = defaultdict(float)
        for s, (primary, _) in zip(steps, labelled):
            share[primary] += s.t1 - s.t0
        top, dur = max(share.items(), key=lambda kv: kv[1])
        text = (f"Camera motion: {body}. Mostly {self.names[top]} "
                f"({100 * dur / (sum(share.values()) or 1.0):.0f}% of the clip).")
        if sum(s.ok for s in steps) / len(steps) < 0.5:
            text += " (Low-confidence estimate.)"
        return text


motion_cue = CameraMotionCue()
cue_file = CUES_DIR / "motion_camera.jsonl"
motion_text: Dict[str, str] = {}
if cue_file.exists():
    motion_text = {r["video_path"]: r["text"] for r in map(json.loads, cue_file.read_text().splitlines())}
todo = [v for v in needed if v not in motion_text and cache.has(v)]
t0 = time.time()
with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
    texts = list(tqdm(pool.map(lambda v: motion_cue.describe(cache.load(v, cfg.motion_frames)), todo),
                      total=len(todo), desc="motion cue", unit="clip"))
motion_text.update(zip(todo, texts))
cue_file.write_text("".join(json.dumps({"video_path": v, "text": t}) + "\n" for v, t in sorted(motion_text.items())))
print(f"motion cue for {len(motion_text)} clips ({len(todo)} new, {time.time() - t0:.0f}s)")
for v in needed[:3]:
    print(f"  {v}\n    {motion_text.get(v)}")

# %% [markdown]
# ## 6b. Generator-replay captions (round 2, S1)
#
# ReVA's questions, options and answers were written by
# **Qwen3-VL-30B-A3B-Instruct** from the video plus a *consolidated caption*
# it wrote first (arXiv 2609.35507, Sec. 3.3, App. I):
#
# 1. three segment captions with the Fig. 11 prompt;
# 2. one consolidated caption from the full clip and the three captions (Fig. 12).
#
# Preset `caption` replays both steps with the same public model (FP8 or AWQ
# 4-bit by GPU memory, vLLM, greedy decoding) and writes `cues/caption.<i>of<n>.jsonl`. Preset
# `caption_check` / `h20_caption` merge the shards into `cues/caption.jsonl`
# and add the line `Video description: ...` after the motion cue. No ReVA
# question or answer is read, so this is not the disallowed video memory.
# Rows are appended per chunk, so a disconnect loses at most one chunk.
# Same prompts and row format as `scripts/data/compute_caption_cues.py`.

# %%
# ReVA paper, App. I, Fig. 11 and Fig. 12 (verbatim).
CAPTION_PROMPT = (
    "Provide a detailed and objective description of the video. Focus only on "
    "information directly observable in the video and avoid speculative "
    "interpretations.\n"
    "Describe the content in the following order:\n"
    "(1) Scene and environment\n"
    "(2) Main objects or entities\n"
    "(3) Actions and interactions\n"
    "(4) Temporal progression of events\n"
    "(5) Notable visual details\n"
    "Ensure that the description follows the chronological order of events and "
    "contains approximately 100–200 words."
)
CONSOLIDATE_PROMPT = (
    "Produce a single comprehensive and coherent description of the video by "
    "consolidating the information from the provided captions while verifying "
    "their consistency with the visual content of the video.\n"
    "When generating the consolidated caption:\n"
    "(1) Carefully review all provided captions\n"
    "(2) Combine complementary information from different descriptions\n"
    "(3) Remove redundant or contradictory content\n"
    "(4) Preserve all important visual details and events\n"
    "(5) Organize the description in a clear chronological order\n"
    "The captions are: {captions}"
)
CAPTION_FILE = CUES_DIR / f"{cfg.caption_name}.jsonl"  # merged; the cue reads this one


def read_rows(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    return {r["video_path"]: r for r in (json.loads(ln) for ln in path.read_text().splitlines() if ln.strip())}


def merge_caption_shards() -> Dict[str, dict]:
    rows = read_rows(CAPTION_FILE)
    for f in sorted(CUES_DIR.glob(f"{cfg.caption_name}.*of*.jsonl")):
        rows.update(read_rows(f))
    CAPTION_FILE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for _, r in sorted(rows.items())))
    return rows


def caption_segments(n: int, k: int = 3) -> List[List[int]]:
    """Frame positions 0..n-1 cut into k contiguous segments (one if too short)."""
    if n < 2 * k:
        return [list(range(n))]
    b = np.linspace(0, n, k + 1).round().astype(int)
    return [list(range(x, y)) for x, y in zip(b[:-1], b[1:])]


def caption_video(fs: FrameSet, positions: Sequence[int], num: int):
    """(frames, metadata) for vLLM: `num` frames of `positions`, true source indices,
    even length (Qwen video patches span two frames)."""
    positions = list(positions)
    if len(positions) > num:
        positions = [positions[k] for k in uniform_indices(len(positions), num)]
    frames, idx = fs.frames[positions], [fs.frame_indices[k] for k in positions]
    if len(idx) % 2:
        frames, idx = np.concatenate([frames, frames[-1:]]), idx + idx[-1:]
    meta = {"fps": fs.fps, "frames_indices": idx, "total_num_frames": fs.total_frames,
            "duration": fs.total_frames / fs.fps, "video_backend": "opencv", "do_sample_frames": False}
    return frames, meta


def run_captioner(videos: Sequence[str], out_file: Path) -> None:
    # CUDA is already initialised above, so vLLM would spawn its engine process, which
    # re-runs this file as a script and fails. Keep the engine in this process instead.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # Greedy decoding: skip FlashInfer's sampler, which JIT-compiles a kernel (needs ninja
    # and a CUDA toolkit matching vLLM's torch) during warm-up.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    proc = AutoProcessor.from_pretrained(cfg.caption_model)
    llm = LLM(model=cfg.caption_model, max_model_len=16384, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 0, "video": 1}, seed=cfg.seed)
    params = SamplingParams(temperature=0.0, max_tokens=cfg.caption_max_tokens)

    def request(frames: np.ndarray, meta: dict, text: str) -> dict:
        prompt = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}],
            tokenize=False, add_generation_prompt=True)
        n = len(frames)  # keep ~640x360 per frame (the processor's budget is per video)
        return {"prompt": prompt, "multi_modal_data": {"video": (frames, meta)},
                "mm_processor_kwargs": {"do_sample_frames": False, "size": {
                    "longest_edge": cfg.max_pixels_per_frame * n, "shortest_edge": 64 * 32 * 32 * n}}}

    def generate(reqs: List[dict]) -> List[str]:
        return [o.outputs[0].text.strip() for o in llm.generate(reqs, params, use_tqdm=False)]

    t_start, n_done = time.time(), 0
    for b in tqdm(range(0, len(videos), cfg.caption_chunk), desc="caption chunks"):
        chunk = videos[b : b + cfg.caption_chunk]
        sets = [cache.load(v, cfg.cache_frames) for v in chunk]
        reqs, owner = [], []
        for i, fs in enumerate(sets):
            for seg in caption_segments(len(fs.frames)):
                reqs.append(request(*caption_video(fs, seg, cfg.caption_segment_frames), CAPTION_PROMPT))
                owner.append(i)
        segs: List[List[str]] = [[] for _ in sets]
        for i, c in zip(owner, generate(reqs)):
            segs[i].append(c)
        joined = [" ".join(f"<Caption {k + 1}> {c}" for k, c in enumerate(s)) for s in segs]
        cons = generate([request(*caption_video(fs, range(len(fs.frames)), cfg.caption_consolidate_frames),
                                 CONSOLIDATE_PROMPT.format(captions=j)) for fs, j in zip(sets, joined)])
        with open(out_file, "a") as f:
            for v, c, s in zip(chunk, cons, segs):
                f.write(json.dumps({"video_path": v, "text": "Video description: " + " ".join(c.split()),
                                    "caption": c, "segments": s, "model": cfg.caption_model},
                                   ensure_ascii=False) + "\n")
        n_done += len(chunk)
        wb_log({"caption/clips_done": n_done, "caption/clips_total": len(videos),
                "caption/sec_per_clip": (time.time() - t_start) / n_done,
                "caption/mean_words": float(np.mean([len(c.split()) for c in cons])),
                "caption/mean_segment_words": float(np.mean([len(x.split()) for s in segs for x in s]))})


if cfg.mode == "caption":
    shard_i, shard_n = (int(x) for x in cfg.caption_shard.split("/"))
    shard_file = CUES_DIR / f"{cfg.caption_name}.{shard_i}of{shard_n}.jsonl"
    mine = [v for v in needed if cache.has(v)][shard_i::shard_n]
    if cfg.caption_limit:
        mine = mine[: cfg.caption_limit]
    done = set(read_rows(shard_file))
    todo = [v for v in mine if v not in done]
    print(f"shard {shard_i}/{shard_n}: {len(mine)} clips, {len(done)} done, {len(todo)} to do -> {shard_file}")
    t0 = time.time()
    if todo:
        run_captioner(todo, shard_file)
    rows = read_rows(shard_file)
    print(f"captioned {len(rows)} clips in {(time.time() - t0) / 60:.0f} min; example:")
    for v in list(rows)[:2]:
        print(f"  {v}\n    {rows[v]['text'][:600]}")
    merged = merge_caption_shards()
    print(f"merged: {len(merged)} of {len(needed)} clips captioned -> {CAPTION_FILE}")
    run = wb_run()
    if run is not None:  # the captions as a browsable table and a versioned file
        import wandb

        words = [len(r["caption"].split()) for r in rows.values()]
        run.summary.update({"caption/clips": len(rows), "caption/merged_clips": len(merged),
                            "caption/words_mean": float(np.mean(words)), "caption/words_min": min(words),
                            "caption/words_max": max(words), "caption/minutes": (time.time() - t0) / 60})
        run.log({"caption/table": wandb.Table(
            columns=["video_path", "words", "caption", "segments"],
            data=[[v, len(r["caption"].split()), r["caption"], "\n---\n".join(r["segments"])]
                  for v, r in sorted(rows.items())])})
        art = wandb.Artifact(f"{cfg.caption_name}-{shard_i}of{shard_n}", type="cue",
                             metadata={"model": cfg.caption_model, "clips": len(rows)})
        art.add_file(str(shard_file))
        run.log_artifact(art)
        wb_finish()
    raise SystemExit("caption preset finished; nothing below runs. Restart the session "
                     "(frees vLLM's GPU memory) and run `caption_check` once all shards are done.")

caption_text: Dict[str, str] = {}
if cfg.caption_cue:
    rows = merge_caption_shards()
    caption_text = {v: r["text"] for v, r in rows.items()}
    lacking = [v for v in needed if v not in caption_text]
    print(f"caption cue for {len(caption_text)} clips; {len(lacking)} needed clips have none")
    assert len(lacking) <= 0.01 * len(needed), "run the `caption` preset (all shards) first"
    if cfg.caption_shuffle:  # a derangement: no clip keeps its own caption
        keys = sorted(caption_text)
        perm, rng_shuffle = keys[:], random.Random(cfg.seed)
        while len(keys) > 1 and any(a == b for a, b in zip(keys, perm)):
            rng_shuffle.shuffle(perm)
        caption_text = {k: caption_text[v] for k, v in zip(keys, perm)}
        print("caption_shuffle: every clip shows another clip's caption")
cue_text = {v: "\n".join(t for t in ((motion_text.get(v, "") if cfg.motion_cue else ""),
                                      caption_text.get(v, "")) if t) for v in needed}

# %% [markdown]
# ## 7. Prompts
#
# The user turn is `[video] + text`:
#
# ```
# Camera motion: ... Mostly moving forward (41% of the clip).
# Question: <question>
# Options:
# A. ...
# B. ...
# C. ...
# D. ...
# Answer with the option's letter from the given choices directly.
# ```
#
# How `[video]` is placed depends on the family:
#
# * **qwen / glm4v**: one `video` entry; the processor expands it and writes
#   the timestamps itself (from the fps + frame indices we pass).
# * **internvl**: the processor's video format is `Frame1: <image>` with no
#   times, so we lay the clip out as one image per frame and write the true
#   time ourselves: `Frame1 (0.2s): <image>`.
#
# No text is generated: we read the logits of `A`, `B`, `C`, `D` right after
# the generation prompt (plus `answer_prefix`, which GLM-Thinking needs).
# Qwen3.5's thinking is disabled (`enable_thinking=False`); the other
# templates ignore that flag.

# %%
INSTRUCTION = "Answer with the option's letter from the given choices directly."
CHAT_KWARGS = {"enable_thinking": False}


def question_text(ex: ReVAExample, cue: str = "") -> str:
    lines = [cue] if cue else []
    lines += [f"Question: {ex.question.strip()}", "Options:"]
    lines += [f"{l}. {o.strip()}" for l, o in zip(LETTERS, ex.options)]
    lines.append(INSTRUCTION)
    return "\n".join(lines)


def video_content(frame_times: Optional[Sequence[float]]) -> List[dict]:
    if frame_times is None:
        return [{"type": "video"}]
    out = []
    for i, t in enumerate(frame_times):
        out += [{"type": "text", "text": f"Frame{i + 1} ({float(t):.1f}s): "}, {"type": "image"}]
    return out


class PromptBuilder:
    def __init__(self, processor, cues: Dict[str, str], answer_prefix: str = ""):
        self.processor, self.cues, self.answer_prefix = processor, cues, answer_prefix

    def __call__(self, ex: ReVAExample, frame_times: Optional[Sequence[float]] = None,
                 video: bool = True) -> str:
        messages = [{"role": "user", "content": (video_content(frame_times) if video else []) + [
            {"type": "text", "text": question_text(ex, self.cues.get(ex.video_path, ""))},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **CHAT_KWARGS)
        return text + self.answer_prefix

# %% [markdown]
# ## 8. Dataset and collator
#
# * **Option shuffling.** Every time a training question is drawn its options
#   are re-ordered (a pure function of seed, epoch and index), so no letter
#   prior can be learned; dev/test keep the official order.
# * **Pixel budget (qwen / glm4v).** The video processor's `longest_edge` is a
#   budget for *all frames together*; we scale it with the frame count to keep
#   640x360. InternVL resizes each frame to one 448x448 tile
#   (`crop_to_patches=False`, i.e. no dynamic tiling into up to 12 crops).
# * **Left padding**, so the answer position is the last token of every row.

# %%
class ReVADataset(Dataset):
    def __init__(self, examples: Sequence[ReVAExample], shuffle_options: bool, seed: int = 0):
        self.examples, self.shuffle_options, self.seed, self.epoch = list(examples), shuffle_options, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> ReVAExample:
        ex = self.examples[i]
        if self.shuffle_options:
            perm = list(range(4))
            random.Random(f"{self.seed}:{self.epoch}:{i}").shuffle(perm)
            ex = permute(ex, perm)
        return ex


MIN_PIXELS_PER_FRAME = 64 * 32 * 32


class MCQCollator:
    def __init__(self, processor, prompts: PromptBuilder, cache: FrameCache, cfg: Config, train: bool):
        self.processor, self.prompts, self.cache, self.cfg = processor, prompts, cache, cfg
        self.random_phase = train and cfg.train_phase_jitter
        self.frame_phase = 0.5
        processor.tokenizer.padding_side = "left"

    def video(self, ex: ReVAExample):
        phase = random.random() if self.random_phase else self.frame_phase
        fs = self.cache.load(ex.video_path, self.cfg.num_frames, phase)
        frames = overlay_time_bar(fs.frames, fs.timestamps) if ex.task in self.cfg.overlay_time_tasks else fs.frames
        meta = {"fps": fs.fps, "frames_indices": fs.frame_indices, "total_num_frames": fs.total_frames}
        return frames, meta, fs.timestamps

    def __call__(self, examples: List[ReVAExample]) -> dict:
        if not self.cfg.use_video:  # text-only probe: cue + question, no frames
            inputs = self.processor(text=[self.prompts(ex, video=False) for ex in examples],
                                    padding=True, return_tensors="pt")
            return {"inputs": inputs, "labels": torch.tensor([ex.label for ex in examples]),
                    "qa_ids": [ex.qa_id for ex in examples], "tasks": [ex.task for ex in examples]}
        clips = [self.video(ex) for ex in examples]
        if FAMILY == "internvl":
            inputs = self.processor(
                text=[self.prompts(ex, times) for ex, (_, _, times) in zip(examples, clips)],
                images=[list(frames) for frames, _, _ in clips], crop_to_patches=False,
                padding=True, return_tensors="pt")
        else:
            n = self.cfg.num_frames
            extra = {"cap_pixels_per_frame": True} if FAMILY == "qwen" else {}  # pins transformers >= 5.22
            inputs = self.processor(
                text=[self.prompts(ex) for ex in examples], videos=[c[0] for c in clips],
                video_metadata=[c[1] for c in clips], do_sample_frames=False, padding=True, return_tensors="pt",
                size={"longest_edge": self.cfg.max_pixels_per_frame * n, "shortest_edge": MIN_PIXELS_PER_FRAME * n},
                **extra)
        return {"inputs": inputs, "labels": torch.tensor([ex.label for ex in examples]),
                "qa_ids": [ex.qa_id for ex in examples], "tasks": [ex.task for ex in examples]}

# %% [markdown]
# ## 9. Model: a VLM read out as a 4-way classifier
#
# `AutoModelForImageTextToText` + LoRA. The class logits are the LM-head
# logits of the single tokens `A B C D` at the last position
# (`logits_to_keep=1`). LoRA covers every language-model linear layer plus
# the vision -> LLM projector; the vision encoder stays frozen. Layer names
# per family:
#
# | Family | LM linears | Projector |
# |---|---|---|
# | qwen (Qwen3.5) | q/k/v/o, Gated DeltaNet `in_proj_qkv / in_proj_z / out_proj`, MLP | `visual.merger.linear_fc1/2` |
# | glm4v | q/k/v/o, `gate_up_proj` (fused), `down_proj` | `visual.merger.proj / gate_proj / up_proj / down_proj` |
# | internvl (Qwen3 LM) | q/k/v/o, `gate_proj / up_proj / down_proj` | `multi_modal_projector.linear_1/2` |

# %%
LM_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
              "out_proj", "gate_proj", "up_proj", "gate_up_proj", "down_proj")
LM_PATTERN = r".*language_model\.layers\.\d+\.(?:self_attn|linear_attn|mlp)\.(?:{})".format("|".join(LM_LINEARS))
PROJECTOR_PATTERNS = {
    "qwen": r".*visual\.(?:merger|deepstack_merger_list\.\d+)\.linear_fc[12]",
    "glm4v": r".*visual\.merger\.(?:proj|gate_proj|up_proj|down_proj)",
    "internvl": r".*multi_modal_projector\.linear_[12]",
}


def fetch_adapter(spec: str) -> Path:
    """Local path, or "user/repo/path/in/repo.pt" on the HF Hub."""
    if Path(spec).exists():
        return Path(spec)
    from huggingface_hub import hf_hub_download

    user, repo, *rest = spec.split("/")
    return Path(hf_hub_download(f"{user}/{repo}", "/".join(rest)))


class VideoMCQModel(nn.Module):
    def __init__(self, cfg: Config, device: torch.device):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.torch_dtype]
        self.processor = AutoProcessor.from_pretrained(cfg.backbone)
        kwargs = dict(dtype=dtype, attn_implementation="sdpa")
        if cfg.gpu_max_memory_gb > 0:
            # Bigger than the GPU: the first layers stay on it, the rest live in CPU RAM
            # and are moved in for each forward pass (slow, scoring only).
            assert cfg.mode == "screen" and not cfg.init_adapter, "CPU offload: zero-shot scoring only"
            kwargs.update(device_map="auto", max_memory={0: f"{cfg.gpu_max_memory_gb}GiB", "cpu": "200GiB"})
        backbone = AutoModelForImageTextToText.from_pretrained(cfg.backbone, **kwargs)
        if cfg.gpu_max_memory_gb <= 0:
            backbone = backbone.to(device)
        for p in backbone.parameters():
            p.requires_grad_(False)
        target = f"(?:{LM_PATTERN})" + (f"|(?:{PROJECTOR_PATTERNS[FAMILY]})" if cfg.lora_merger else "")
        if cfg.mode == "screen" and not cfg.init_adapter:
            self.backbone = backbone  # zero-shot: the plain backbone, no adapter
        else:
            self.backbone = get_peft_model(backbone, LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                target_modules=target, bias="none"))
        if cfg.mode == "screen":
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        elif cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.backbone.enable_input_require_grads()
            self.backbone.config.use_cache = False
        tok = self.processor.tokenizer
        self.letter_ids = []
        for l in LETTERS:
            ids = tok.encode(l, add_special_tokens=False)
            assert len(ids) == 1, f"letter {l} is not one token"
            self.letter_ids.append(ids[0])

    def forward(self, inputs) -> torch.Tensor:
        out = self.backbone(**inputs, logits_to_keep=1, use_cache=False)
        idx = torch.as_tensor(self.letter_ids, device=out.logits.device)
        return out.logits[:, -1, :].index_select(-1, idx).float()  # (B, 4)

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        return {n: p.detach().cpu() for n, p in self.named_parameters() if "lora_" in n or p.requires_grad}

    def load_trainable(self, state: Dict[str, torch.Tensor]) -> None:
        # Notebook checkpoints keep the adapter under "adapter"; the repo's
        # training pipeline (H20 on HF) under "model". Same parameter names.
        state = state.get("adapter", state.get("model", state))
        result = self.load_state_dict(state, strict=False)
        assert not result.unexpected_keys, f"{len(result.unexpected_keys)} tensors match no parameter, e.g. {result.unexpected_keys[:2]}"
        lora = [n for n, _ in self.named_parameters() if "lora_" in n]
        missing = [n for n in lora if n not in state]
        assert not missing, f"checkpoint lacks {len(missing)} adapter tensors, e.g. {missing[:2]}"

# %% [markdown]
# ## 10. Decoding (option-order TTA, PriDe) and ReVA metrics
#
# Scores live in *display order* (what letter the model saw); `to_original`
# maps them back to the official option order. **PriDe** (Zheng et al., ICLR
# 2024) estimates the model's letter prior from a 5% subset scored under all 4
# cyclic option orders, then divides it out of single-pass scores elsewhere.
# After option-shuffled training the prior is ~uniform, so PriDe changes
# little, but it costs only ~1.15x.

# %%
def log_softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def to_original(display: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    out = np.empty_like(display)
    for i, p in enumerate(perm):
        out[..., p] = display[..., i]
    return out


def aggregate_tta(per_perm: Sequence[np.ndarray], perms) -> np.ndarray:
    return log_softmax_np(np.mean([to_original(log_softmax_np(s), p) for s, p in zip(per_perm, perms)], axis=0))


def pride_subset(n: int, fraction: float, seed: int = 0) -> np.ndarray:
    idx = np.random.default_rng(seed).permutation(n)
    return np.sort(idx[: max(1, int(round(n * fraction)))])


def decode(display: np.ndarray, perms, tta: str, pride_fraction: float) -> np.ndarray:
    """(N, P, 4) display logits -> (N, 4) log-probs in the original option order."""
    if tta == "none":
        return np.stack([to_original(log_softmax_np(d[0]), perms[0]) for d in display])
    if tta == "cyclic":
        return np.stack([aggregate_tta(list(d), perms) for d in display])
    est = pride_subset(len(display), pride_fraction)
    per_q = log_softmax_np(log_softmax_np(display[est]).mean(axis=1))
    prior = np.log(np.exp(per_q).mean(axis=0))
    out = log_softmax_np(log_softmax_np(display[:, 0]) - prior)
    for i in est:  # these already paid for all shifts
        out[i] = aggregate_tta(list(display[i]), perms)
    return out


def reva_metrics(pred: Sequence[str], examples: Sequence[ReVAExample]) -> dict:
    hits = defaultdict(list)
    for p, ex in zip(pred, examples):
        if ex.label >= 0:
            ok = float(p == ex.answer_letter)
            hits["__all__"].append(ok)
            hits[ex.task].append(ok)
    if not hits:
        return {"n": 0}
    weighted = [(TEST_TASK_COUNTS[t], np.mean(hits[t])) for t in TASKS if hits.get(t)]
    dist = Counter(pred)
    return {
        "accuracy": float(np.mean(hits["__all__"])),
        "test_weighted_accuracy": float(sum(w * a for w, a in weighted) / sum(w for w, _ in weighted)),
        "n": len(hits["__all__"]),
        "per_task": {task_short(t): float(np.mean(v)) for t, v in hits.items() if t in TASKS},
        "pred_letter_dist": {l: dist[l] / max(len(pred), 1) for l in LETTERS},
    }


def report(m: dict) -> str:
    if not m.get("n"):
        return "no labels"
    lines = [f"accuracy {m['accuracy'] * 100:.2f}  test-weighted {m['test_weighted_accuracy'] * 100:.2f}  n={m['n']}"]
    lines += [f"  {k:16s} {v * 100:6.2f}" for k, v in sorted(m["per_task"].items())]
    lines.append("  letters " + "  ".join(f"{k}:{v:.2f}" for k, v in m["pred_letter_dist"].items()))
    return "\n".join(lines)

# %% [markdown]
# ## 11. Evaluation

# %%
@torch.no_grad()
def score_examples(model: VideoMCQModel, collator: MCQCollator, examples, perms, batch_size: int) -> np.ndarray:
    """(N, P, 4) display-order letter logits, one slice per permutation."""
    model.eval()
    jobs = [(n, p, permute(ex, perm)) for n, ex in enumerate(examples) for p, perm in enumerate(perms)]
    out = np.zeros((len(examples), len(perms), 4), dtype=np.float32)
    loader = DataLoader([j[2] for j in jobs], batch_size=batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collator)
    k = 0
    bar = tqdm(total=len(jobs), desc="scoring", unit="q", leave=False)
    for batch in loader:
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            logits = model(batch["inputs"].to(DEVICE))
        for row in logits.cpu().numpy():
            n, p, _ = jobs[k]
            out[n, p] = row
            k += 1
        bar.update(len(logits))
    bar.close()
    return out


def evaluate(model, collator, examples, tta: str = "none", batch_size: int = 2) -> dict:
    perms = cyclic_perms()
    if tta == "none":
        perms = perms[:1]
        display = score_examples(model, collator, examples, perms, batch_size)
    elif tta == "cyclic":
        display = score_examples(model, collator, examples, perms, batch_size)
    else:  # pride: identity everywhere, all 4 shifts on the 5% estimation subset
        display = np.zeros((len(examples), 4, 4), dtype=np.float32)
        display[:, :1] = score_examples(model, collator, examples, perms[:1], batch_size)
        est = pride_subset(len(examples), cfg.pride_fraction)
        display[est] = score_examples(model, collator, [examples[i] for i in est], perms, batch_size)
    scores = decode(display, perms, tta, cfg.pride_fraction)
    preds = [LETTERS[i] for i in scores.argmax(-1)]
    m = reva_metrics(preds, examples)
    m["scores"], m["predictions"] = scores, dict(zip((e.qa_id for e in examples), preds))
    return m

# %% [markdown]
# ## 12. Trainer
#
# Batch size 1, gradient accumulation over `accum_steps` questions, AdamW,
# linear warmup + cosine decay, gradient clipping, bf16 autocast. Only the
# adapter is saved (~650 MB for the 9B, LoRA r=64). `last.pt` (adapter +
# optimizer + step) is written every `save_every_steps` optimizer steps so a
# Colab disconnect can resume; `best.pt` keeps the best **test-weighted** dev.

# %%
class Trainer:
    def __init__(self, model: VideoMCQModel, train_set: ReVADataset, collator: MCQCollator,
                 dev_collator: MCQCollator, dev: Sequence[ReVAExample], cfg: Config):
        self.model, self.train_set, self.collator, self.dev_collator, self.dev, self.cfg = (
            model, train_set, collator, dev_collator, dev, cfg)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.total_steps = math.ceil(len(train_set) / cfg.accum_steps) * cfg.epochs
        warmup = max(int(cfg.warmup_ratio * self.total_steps), 1)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(self.total_steps - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        self.step, self.start_epoch, self.skip, self.best = 0, 1, 0, -1.0
        self.history: List[dict] = []

    # -- checkpoints -------------------------------------------------------
    def save(self, name: str, with_optimizer: bool, epoch: int, batches_done: int) -> None:
        state = {"adapter": self.model.trainable_state_dict(), "step": self.step, "epoch": epoch,
                 "batches_done": batches_done, "best": self.best, "cfg": asdict(self.cfg),
                 "history": self.history}
        if with_optimizer:
            state["optimizer"], state["scheduler"] = self.opt.state_dict(), self.sched.state_dict()
        tmp = CKPT_DIR / f"{name}.tmp"
        torch.save(state, tmp)
        tmp.replace(CKPT_DIR / f"{name}.pt")

    def resume(self) -> None:
        path = CKPT_DIR / "last.pt"
        if not path.exists():
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_trainable(state["adapter"])
        self.opt.load_state_dict(state["optimizer"])
        self.sched.load_state_dict(state["scheduler"])
        self.step, self.best, self.history = state["step"], state["best"], state.get("history", [])
        self.start_epoch, self.skip = state["epoch"], state["batches_done"]
        print(f"resumed at step {self.step} (epoch {self.start_epoch}, {self.skip} batches done)")

    # -- loop --------------------------------------------------------------
    def fit(self) -> None:
        cfg, model = self.cfg, self.model
        use_amp = DEVICE.type == "cuda"
        for epoch in range(self.start_epoch, cfg.epochs + 1):
            self.train_set.set_epoch(epoch)
            # Same order on resume; already-trained questions are skipped without loading them.
            gen = torch.Generator().manual_seed(cfg.seed + epoch)
            order = torch.randperm(len(self.train_set), generator=gen).tolist()
            skip, self.skip = self.skip, 0
            loader = DataLoader(self.train_set, batch_size=1, sampler=order[skip:],
                                num_workers=cfg.num_workers, collate_fn=self.collator)
            model.train()
            self.opt.zero_grad(set_to_none=True)
            pending, t0, steps0 = [], time.time(), self.step
            bar = tqdm(total=len(order) - skip, desc=f"epoch {epoch}/{cfg.epochs}", unit="q")
            for j, batch in enumerate(loader):
                bar.update(1)
                i = skip + j  # position in the epoch
                with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(batch["inputs"].to(DEVICE))
                    loss = F.cross_entropy(logits, batch["labels"].to(DEVICE))
                last = i + 1 == len(order)
                n_in_step = len(pending) + 1
                (loss / (n_in_step if last and n_in_step < cfg.accum_steps else cfg.accum_steps)).backward()
                pending.append(loss.item())
                if len(pending) < cfg.accum_steps and not last:
                    continue
                torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.opt.step()
                self.sched.step()
                self.opt.zero_grad(set_to_none=True)
                self.step += 1
                avg, pending = sum(pending) / len(pending), []
                wb_data = {"train/loss": avg, "train/lr": self.sched.get_last_lr()[0], "epoch": epoch}
                if self.step % cfg.log_every == 0 or self.step == 1:
                    rate = (time.time() - t0) / max(self.step - steps0, 1)
                    eta = rate * (self.total_steps - self.step) / 3600
                    bar.set_postfix(step=f"{self.step}/{self.total_steps}", loss=f"{avg:.4f}",
                                    lr=f"{self.sched.get_last_lr()[0]:.1e}", s_per_step=f"{rate:.1f}", eta_h=f"{eta:.2f}")
                    wb_data.update({"train/sec_per_step": rate, "train/eta_h": eta,
                                    "gpu/peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if DEVICE.type == "cuda" else 0.0})
                wb_log(wb_data, step=self.step)
                if cfg.save_every_steps and self.step % cfg.save_every_steps == 0:
                    self.save("last", with_optimizer=True, epoch=epoch, batches_done=i + 1)
            bar.close()
            if cfg.train_on_dev:  # dev is in the training data: keep the latest epoch
                self.history.append({"epoch": epoch, "step": self.step})
                self.save("best", with_optimizer=False, epoch=epoch + 1, batches_done=0)
                self.save("last", with_optimizer=True, epoch=epoch + 1, batches_done=0)
                print(f"epoch {epoch} done (train_on_dev: no dev pass) -> {CKPT_DIR / 'best.pt'}")
                continue
            # end of epoch: dev pass (no TTA), keep the best by test-weighted accuracy
            m = evaluate(model, self.dev_collator, self.dev, tta="none", batch_size=cfg.eval_batch_size)
            print(f"epoch {epoch} dev:\n{report(m)}")
            wb_eval("dev_epoch", m, step=self.step)
            self.history.append({"epoch": epoch, "step": self.step, "accuracy": m.get("accuracy"),
                                 "test_weighted_accuracy": m.get("test_weighted_accuracy")})
            if m.get("test_weighted_accuracy", -1) > self.best:
                self.best = m["test_weighted_accuracy"]
                self.save("best", with_optimizer=False, epoch=epoch + 1, batches_done=0)
                print(f"  new best test-weighted {self.best * 100:.2f} -> {CKPT_DIR / 'best.pt'}")
            self.save("last", with_optimizer=True, epoch=epoch + 1, batches_done=0)
            model.train()

# %% [markdown]
# ## 13. Build everything and train
#
# A quick look at one prompt and one batch first. With `mode="train"` training
# resumes automatically from `last.pt` if a previous session was interrupted;
# with `"screen"` / `"score"` this section only builds the model.
#
# `unload_model()` frees the model (and the trainer) from GPU and CPU memory.
# Run it before building another model in the same session, e.g. when switching
# from a screen preset to a train preset, or before section 16.

# %%
import ctypes
import gc


def unload_model(names=("model", "trainer", "optimizer", "batch", "loss", "logits")) -> None:
    """Free the model from GPU and CPU RAM. Also drops the Trainer (it holds the
    model and the optimizer state) and leftover tensors from interactive cells."""
    g = globals()
    for n in names:
        if n in g:
            obj = g.pop(n)
            if isinstance(obj, torch.nn.Module):
                obj.to("cpu")  # release the VRAM even if something else still references it
            del obj
    # a failed cell keeps its traceback, and the traceback keeps that frame's model alive
    sys.last_traceback = sys.last_value = sys.last_type = None
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # hand freed CPU memory back to the OS
    except OSError:
        pass
    gpu = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    try:
        import psutil

        ram = f", process RAM {psutil.Process().memory_info().rss / 2**30:.2f} GB"
    except ImportError:
        ram = ""
    print(f"GPU allocated {gpu:.2f} GB{ram}")

# %%
model = VideoMCQModel(cfg, DEVICE)
if cfg.init_adapter:
    model.load_trainable(torch.load(fetch_adapter(cfg.init_adapter), map_location="cpu", weights_only=False))
    print("loaded adapter", cfg.init_adapter)
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_all:,} total, {n_train:,} trainable ({100 * n_train / n_all:.2f}%)")

prompts = PromptBuilder(model.processor, cue_text, cfg.answer_prefix)
train_collator = MCQCollator(model.processor, prompts, cache, cfg, train=True)
eval_collator = MCQCollator(model.processor, prompts, cache, cfg, train=False)
train_set = ReVADataset(fit, shuffle_options=True, seed=cfg.seed)

example = next((e for e in dev if e.task == "Temporal Grounding"), dev[0])
batch = eval_collator([example])
shown = model.processor.tokenizer.decode(batch["inputs"]["input_ids"][0])
shown = re.sub(r"(<IMG_CONTEXT>|<\|image\|>|<\|video_pad\|>)\1+", lambda m: f"[{len(m.group(0)) // len(m.group(1))} x {m.group(1)}]", shown)
print(shown[-1500:])
print({k: tuple(v.shape) for k, v in batch["inputs"].items() if hasattr(v, "shape")})
print("visual + text tokens per question:", batch["inputs"]["input_ids"].shape[1])

# %%
if cfg.mode == "train":
    trainer = Trainer(model, train_set, train_collator, eval_collator, dev, cfg)
    trainer.resume()
    print(f"{len(train_set)} questions, {trainer.total_steps} optimizer steps of {cfg.accum_steps} questions")
    trainer.fit()
else:
    print(f"mode={cfg.mode}: no training")

# %% [markdown]
# ## 14. Dev evaluation
#
# Trained runs reload `best.pt`; `screen` scores the frozen backbone and
# `score` the loaded adapter. Scores are saved next to the dev `qa_id`s so
# section 16 can align runs.

# %%
def save_scores(split: str, result: dict, examples: Sequence[ReVAExample]) -> None:
    np.save(RESULTS_DIR / f"{split}_scores.npy", result.pop("scores"))
    (RESULTS_DIR / f"{split}_qa_ids.json").write_text(json.dumps([e.qa_id for e in examples]))
    (RESULTS_DIR / f"{split}_predictions.json").write_text(json.dumps(result.pop("predictions"), indent=1))
    (RESULTS_DIR / f"{split}_metrics.json").write_text(json.dumps(result, indent=1))


if cfg.mode == "train":
    model.load_trainable(torch.load(CKPT_DIR / "best.pt", map_location="cpu", weights_only=False))
dev_result = evaluate(model, eval_collator, dev, tta=cfg.tta, batch_size=cfg.eval_batch_size)
print(f"{cfg.run_name} dev ({cfg.tta}):\n{report(dev_result)}")
save_scores("dev", dev_result, dev)
wb_eval("dev", dev_result, summary=True)
if cfg.mode == "screen":
    acc = dev_result["accuracy"] * 100
    if not cfg.use_video:
        print(f"\ncaption check: text-only dev {acc:.2f} (blind floor 64.49, preset blind_check). Go (run "
              "h20_caption) if >= ~73, or >= blind + 15 on ObjectLand / Viewpoint / Trend.")
    else:
        print(f"\nscreen: zero-shot dev {acc:.2f}. Fine-tune if >= ~76.6 "
              "(Qwen3.5-9B zero-shot 79.6 minus 3); compare per task above.")

# %% [markdown]
# ## 15. Test predictions and `submission.zip`
#
# Codabench expects `submission.zip` containing `predictions.json` laid out
# like `test.json` (a `metadata` block and a `QA` list) with `correct_answer`
# filled for all 4,000 questions; a bare `{qa_id: letter}` dict is rejected.

# %%
def write_submission(pred: Dict[str, str], out_dir: Path) -> None:
    raw = json.loads((REVA_ROOT / "test.json").read_text())
    if len(pred) != len(raw["QA"]):
        print(f"scored {len(pred)} of {len(raw['QA'])} test questions (subset preset): no submission written")
        return
    payload = {"metadata": raw.get("metadata", {}),
               "QA": [{**r, "correct_answer": pred[r["qa_id"]]} for r in raw["QA"]]}
    assert all(r["correct_answer"] in LETTERS for r in payload["QA"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(json.dumps(payload))
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_dir / "predictions.json", arcname="predictions.json")
    print(f"{len(pred)} predictions -> {out_dir / 'submission.zip'}")
    print("letters:", dict(Counter(pred.values())))


if cfg.score_test:  # screen presets set score_test=False unless test scores are wanted
    test_result = evaluate(model, eval_collator, test, tta=cfg.tta, batch_size=cfg.eval_batch_size)
    pred = dict(test_result["predictions"])
    save_scores("test", test_result, test)
    wb_eval("test", test_result, summary=True)
    write_submission(pred, RESULTS_DIR)
else:
    print("test scoring skipped (score_test=False)")
wb_finish()

# %% [markdown]
# ## 16. Ensemble with H20
#
# Needs two runs' saved scores under `OUT/results/<run>/` (`dev_scores.npy`,
# `test_scores.npy` and their `qa_id` lists): the new backbone and H20 (preset
# `h20_score`, run once). Each run's scores are log-probs over the original
# option order (after PriDe), so they combine as
#
#     score = (1 - w_task) * log p_H20 + w_task * log p_new
#
# with one weight per task from a grid. The weight is chosen **out of fold**
# (5 folds grouped by clip): each dev question is scored with a weight fitted
# on the other folds, so the reported dev number is not tuned on itself. The
# test ensemble uses weights fitted on all of dev. Ship it only if the
# out-of-fold test-weighted dev gain over H20 alone is >= 0.8 points.
#
# Run `unload_model()` (section 13) first if you are in the same session; the
# section itself needs no GPU.

# %%
ENSEMBLE_RUNS = ("h20_32f_bar_full_1ep_9b", "h22_internvl35_8b_32f_bar_full_1ep")  # (base, new)
GRID = np.round(np.arange(0.0, 1.0001, 0.1), 2)


def load_run(run: str, split: str) -> Optional[Dict[str, np.ndarray]]:
    d = OUT / "results" / run
    if not (d / f"{split}_scores.npy").exists():
        return None
    ids = json.loads((d / f"{split}_qa_ids.json").read_text())
    return dict(zip(ids, np.load(d / f"{split}_scores.npy")))


def tw_accuracy(correct: np.ndarray, tasks: Sequence[str]) -> float:
    by = defaultdict(list)
    for c, t in zip(correct, tasks):
        by[t].append(c)
    pairs = [(TEST_TASK_COUNTS[t], np.mean(v)) for t, v in by.items() if t in TEST_TASK_COUNTS]
    return float(sum(w * a for w, a in pairs) / sum(w for w, _ in pairs))


def fit_weights(base, new, labels, tasks) -> Dict[str, float]:
    out = {}
    for t in set(tasks):
        m = np.array([x == t for x in tasks])
        accs = [((1 - w) * base[m] + w * new[m]).argmax(-1).__eq__(labels[m]).mean() for w in GRID]
        out[t] = float(GRID[int(np.argmax(accs))])  # ties -> the smaller weight (closer to H20)
    return out


base_dev, new_dev = (load_run(r, "dev") for r in ENSEMBLE_RUNS)
if base_dev is None or new_dev is None:
    print("missing dev scores for", [r for r, s in zip(ENSEMBLE_RUNS, (base_dev, new_dev)) if s is None])
else:
    exs = [e for e in dev if e.qa_id in base_dev and e.qa_id in new_dev and e.label >= 0]
    B = np.stack([base_dev[e.qa_id] for e in exs])
    N = np.stack([new_dev[e.qa_id] for e in exs])
    y = np.array([e.label for e in exs])
    tasks = [e.task for e in exs]
    clips = sorted({e.video_path for e in exs})
    fold_of = {v: i % 5 for i, v in enumerate(random.Random(0).sample(clips, len(clips)))}
    folds = np.array([fold_of[e.video_path] for e in exs])
    oof = np.zeros_like(B)
    for k in range(5):
        tr, te = folds != k, folds == k
        w = fit_weights(B[tr], N[tr], y[tr], [t for t, m in zip(tasks, tr) if m])
        wv = np.array([w.get(t, 0.0) for t, m in zip(tasks, te) if m])[:, None]
        oof[te] = (1 - wv) * B[te] + wv * N[te]
    for name, S in (("H20 alone", B), ("new alone", N), ("ensemble (out-of-fold)", oof)):
        c = S.argmax(-1) == y
        print(f"{name:24s} dev {c.mean() * 100:6.2f}   test-weighted {tw_accuracy(c, tasks) * 100:6.2f}")
    gain = (tw_accuracy(oof.argmax(-1) == y, tasks) - tw_accuracy(B.argmax(-1) == y, tasks)) * 100
    b_ok, o_ok = B.argmax(-1) == y, oof.argmax(-1) == y
    print(f"gain {gain:+.2f} TW; fixed {int((~b_ok & o_ok).sum())}, broke {int((b_ok & ~o_ok).sum())} dev questions")
    weights = fit_weights(B, N, y, tasks)
    print("per-task weight on the new model:", {task_short(t): w for t, w in sorted(weights.items())})

    base_test, new_test = (load_run(r, "test") for r in ENSEMBLE_RUNS)
    if base_test is None or new_test is None:
        print("no test scores for both runs yet: ensemble submission not written")
    else:
        pred = {}
        for e in test:
            w = weights.get(e.task, 0.0)
            s = (1 - w) * base_test[e.qa_id] + w * new_test[e.qa_id]
            pred[e.qa_id] = LETTERS[int(np.argmax(s))]
        write_submission(pred, OUT / "results" / f"ensemble_{ENSEMBLE_RUNS[1]}")
        if gain < 0.8:
            print(f"NOTE: out-of-fold gain {gain:+.2f} < 0.8 TW: below the ship bar")
