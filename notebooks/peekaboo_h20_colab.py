# %% [markdown]
# # Peekaboo-VL: H20 full pipeline (self-contained, Colab)
#
# Drone-video multiple-choice QA on **ReVA** (MAVU @ WACV 2027) with a
# fine-tuned **Qwen3.5-9B**, built from scratch in this notebook. Nothing is
# imported from the project repository: every piece (data splits, frame cache,
# camera-motion cue, prompts, collator, model, training loop, PriDe decoding,
# metrics, submission) is defined in the cells below.
#
# **H20 recipe** (dev 85.37 / test-weighted 83.04 on the A40 run):
#
# | | |
# |---|---|
# | Backbone | `Qwen/Qwen3.5-9B`, LoRA r=64 / alpha=128 / dropout 0.05 on all LM linears + vision merger |
# | Video | 32 frames at native 640x360, true timestamps (fps + frame indices), random sampling phase during training |
# | Text cue | camera ego-motion from ORB + RANSAC similarity, written in the dataset's answer vocabulary |
# | Visual cue | progress bar + time drawn on the frames of *Temporal Grounding* questions only |
# | Readout | next-token logits of `A B C D`; cross-entropy over the 4 letters; options re-shuffled every draw |
# | Training | all 14,068 fit questions, 1 epoch, AdamW lr 1e-4, warmup 3% + cosine, 16 questions per step, bf16 |
# | Decoding | PriDe letter-prior debiasing (5% of questions scored under the 4 cyclic option orders) |
#
# **Runtime.** The full preset needs an **A100 (40 or 80 GB)** or similar
# (9B bf16 weights ~19 GB; 32 frames ~ 7k visual tokens). Training took ~17 h
# on an A40; expect roughly 8-12 h on an A100, so keep checkpoints on Google
# Drive and resume across sessions. For a quick look use `PRESET = "small"`
# (Qwen3.5-2B, 16 frames, a 3k-question subset) or `"smoke"` (tiny, minutes).
#
# **Data.** ReVA is research / non-commercial only and may not be
# redistributed (arXiv 2609.35507). The full release is ~30 GB; the notebook
# can download one source folder at a time and delete the mp4s after
# extracting frames.
#
# Sections: 0 setup · 1 config · 2 labels · 3 data & splits · 4 download ·
# 5 frame cache · 6 camera-motion cue · 7 prompts · 8 dataset & collator ·
# 9 model · 10 decoding & metrics · 11 evaluation · 12 trainer · 13 train ·
# 14 dev evaluation · 15 test submission

# %% [markdown]
# ## 0. Setup

# %%
import os
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
# Same library versions as the H20 run (Colab keeps its own torch build).
PACKAGES = [
    "transformers==5.18.0",
    "peft==0.21.1",
    "accelerate==1.15.0",
    "huggingface_hub>=1.0",
    "opencv-python-headless>=4.10",
    "flash-linear-attention==0.5.2",  # fast Gated-DeltaNet kernels for Qwen3.5
]
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
# field afterwards. Outputs (cues, checkpoints, predictions) go to `OUT_DIR`,
# which should be on Google Drive so a disconnect does not lose training.

# %%
PRESET = os.environ.get("PEEKABOO_PRESET", "h20")  # "h20" | "small" | "smoke"


@dataclass
class Config:
    # model
    backbone: str = "Qwen/Qwen3.5-9B"
    torch_dtype: str = "bfloat16"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_merger: bool = True
    gradient_checkpointing: bool = True
    # video input
    num_frames: int = 32
    cache_frames: int = 64  # frames decoded once per clip; training subsamples them
    max_pixels_per_frame: int = 640 * 360
    overlay_time_tasks: Tuple[str, ...] = ("Temporal Grounding",)
    train_phase_jitter: bool = True
    motion_frames: int = 32  # frames used to estimate camera motion
    # data
    holdout_per_task: int = 150
    max_train: int = 0  # 0 = all fit questions
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
    # decoding
    tta: str = "pride"  # "none" | "cyclic" | "pride"
    pride_fraction: float = 0.05
    eval_batch_size: int = 2
    # paths
    work_dir: str = "/content/peekaboo"  # fast local disk: videos, frames
    out_dir: str = "/content/drive/MyDrive/peekaboo_h20"  # persistent outputs
    run_name: str = "h20_32f_bar_full_1ep"


PRESETS = {
    "h20": {},
    "small": dict(
        backbone="Qwen/Qwen3.5-2B", num_frames=16, max_train=3000, max_dev=400,
        accum_steps=8, run_name="small_2b_16f",
    ),
    "smoke": dict(
        backbone="Qwen/Qwen3.5-0.8B", num_frames=8, cache_frames=16, motion_frames=8,
        max_train=16, max_dev=8, max_test=8, accum_steps=4, lora_r=8, lora_alpha=16,
        save_every_steps=2, log_every=1, num_workers=0, run_name="smoke_0p8b",
    ),
}
cfg = replace(Config(), **PRESETS[PRESET])
if not IN_COLAB:  # local runs: keep everything under ./notebook_runs
    cfg = replace(cfg, work_dir=os.environ.get("PEEKABOO_WORK", "notebook_runs/work"),
                  out_dir=os.environ.get("PEEKABOO_OUT", "notebook_runs/out"))
if DEVICE.type != "cuda":
    cfg = replace(cfg, num_workers=0)
print(json.dumps(asdict(cfg), indent=1))

# %%
# Google Drive for persistent outputs (Colab only). Hugging Face token is
# optional (the dataset and Qwen weights are public; a token raises rate limits).
if IN_COLAB and cfg.out_dir.startswith("/content/drive"):
    from google.colab import drive

    drive.mount("/content/drive")
if IN_COLAB:
    try:
        from google.colab import userdata

        os.environ.setdefault("HF_TOKEN", userdata.get("HF_TOKEN"))
    except Exception:
        print("no HF_TOKEN secret; downloading anonymously")

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
print(f"using fit {len(fit)}  dev {len(dev)}  test {len(test)}")
print("dev tasks:", dict(sorted(Counter(task_short(e.task) for e in dev).items())))

# %%
# Videos: only the clips referenced by the splits we use. With the full data
# this is ~30 GB; set DELETE_VIDEOS to remove each mp4 after its frames are cached.
DELETE_VIDEOS = IN_COLAB
needed = sorted({ex.video_path for ex in fit + dev + test})
missing = [v for v in needed if not (REVA_ROOT / v).exists()]
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
    errors = [e for e in pool.map(extract_one, needed) if e]
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
    texts = list(pool.map(lambda v: motion_cue.describe(cache.load(v, cfg.motion_frames)), todo))
motion_text.update(zip(todo, texts))
cue_file.write_text("".join(json.dumps({"video_path": v, "text": t}) + "\n" for v, t in sorted(motion_text.items())))
print(f"motion cue for {len(motion_text)} clips ({len(todo)} new, {time.time() - t0:.0f}s)")
for v in needed[:3]:
    print(f"  {v}\n    {motion_text.get(v)}")

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
# Thinking is disabled (`enable_thinking=False`; Qwen3.5 then emits an empty
# think block), so the very next token is the answer letter. No text is
# generated: we read the logits of `A`, `B`, `C`, `D` at that position.

# %%
INSTRUCTION = "Answer with the option's letter from the given choices directly."
CHAT_KWARGS = {"enable_thinking": False}


def question_text(ex: ReVAExample, cue: str = "") -> str:
    lines = [cue] if cue else []
    lines += [f"Question: {ex.question.strip()}", "Options:"]
    lines += [f"{l}. {o.strip()}" for l, o in zip(LETTERS, ex.options)]
    lines.append(INSTRUCTION)
    return "\n".join(lines)


class PromptBuilder:
    def __init__(self, processor, cues: Dict[str, str]):
        self.processor, self.cues = processor, cues

    def __call__(self, ex: ReVAExample) -> str:
        messages = [{"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": question_text(ex, self.cues.get(ex.video_path, ""))},
        ]}]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **CHAT_KWARGS)

# %% [markdown]
# ## 8. Dataset and collator
#
# * **Option shuffling.** Every time a training question is drawn its options
#   are re-ordered (a pure function of seed, epoch and index), so no letter
#   prior can be learned; dev/test keep the official order.
# * **Pixel budget.** The Qwen3-VL/3.5 video processor's `longest_edge` is a
#   budget for *all frames together*. Its default would shrink 32 frames of
#   640x360 to ~208x117; we scale it with the frame count to keep native size.
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
        return frames, meta

    def __call__(self, examples: List[ReVAExample]) -> dict:
        videos, metas = zip(*(self.video(ex) for ex in examples))
        n = self.cfg.num_frames
        inputs = self.processor(
            text=[self.prompts(ex) for ex in examples], videos=list(videos), video_metadata=list(metas),
            do_sample_frames=False, padding=True, return_tensors="pt",
            size={"longest_edge": self.cfg.max_pixels_per_frame * n, "shortest_edge": MIN_PIXELS_PER_FRAME * n},
            cap_pixels_per_frame=True,  # no-op at this budget; pins transformers >= 5.22 behaviour
        )
        return {"inputs": inputs, "labels": torch.tensor([ex.label for ex in examples]),
                "qa_ids": [ex.qa_id for ex in examples], "tasks": [ex.task for ex in examples]}

# %% [markdown]
# ## 9. Model: a VLM read out as a 4-way classifier
#
# `AutoModelForImageTextToText` + LoRA. The class logits are the LM-head
# logits of the single tokens `A B C D` at the last position (`logits_to_keep=1`
# keeps the 250k-vocab projection to one position). LoRA covers every
# language-model linear layer of Qwen3.5's hybrid stack (gated full attention
# q/k/v/o, Gated DeltaNet `in_proj_qkv / in_proj_z / out_proj`, MLP) plus the
# vision->LLM merger; the vision encoder stays frozen.

# %%
LM_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
              "out_proj", "gate_proj", "up_proj", "down_proj")
LM_PATTERN = r".*language_model\.layers\.\d+\.(?:self_attn|linear_attn|mlp)\.(?:{})".format("|".join(LM_LINEARS))
MERGER_PATTERN = r".*visual\.(?:merger|deepstack_merger_list\.\d+)\.linear_fc[12]"


class VideoMCQModel(nn.Module):
    def __init__(self, cfg: Config, device: torch.device):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.torch_dtype]
        self.processor = AutoProcessor.from_pretrained(cfg.backbone)
        backbone = AutoModelForImageTextToText.from_pretrained(
            cfg.backbone, dtype=dtype, attn_implementation="sdpa").to(device)
        for p in backbone.parameters():
            p.requires_grad_(False)
        target = f"(?:{LM_PATTERN})" + (f"|(?:{MERGER_PATTERN})" if cfg.lora_merger else "")
        self.backbone = get_peft_model(backbone, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=target, bias="none"))
        if cfg.gradient_checkpointing:
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
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, state: Dict[str, torch.Tensor]) -> None:
        missing = set(self.trainable_state_dict()) - set(state)
        assert not missing, f"checkpoint lacks {len(missing)} adapter tensors"
        self.load_state_dict(state, strict=False)

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
    k, t0 = 0, time.time()
    for b, batch in enumerate(loader, 1):
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            logits = model(batch["inputs"].to(DEVICE))
        for row in logits.cpu().numpy():
            n, p, _ = jobs[k]
            out[n, p] = row
            k += 1
        if b % max(len(loader) // 10, 1) == 0:
            rate = (time.time() - t0) / k
            print(f"  scored {k}/{len(jobs)}  {rate:.2f}s/item  eta {rate * (len(jobs) - k) / 60:.1f} min", flush=True)
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
            for j, batch in enumerate(loader):
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
                if self.step % cfg.log_every == 0 or self.step == 1:
                    rate = (time.time() - t0) / max(self.step - steps0, 1)
                    eta = rate * (self.total_steps - self.step) / 3600
                    print(f"epoch {epoch} step {self.step}/{self.total_steps} loss {avg:.4f} "
                          f"lr {self.sched.get_last_lr()[0]:.2e} {rate:.1f}s/step eta {eta:.2f}h", flush=True)
                if cfg.save_every_steps and self.step % cfg.save_every_steps == 0:
                    self.save("last", with_optimizer=True, epoch=epoch, batches_done=i + 1)
            # end of epoch: dev pass (no TTA), keep the best by test-weighted accuracy
            m = evaluate(model, self.dev_collator, self.dev, tta="none", batch_size=cfg.eval_batch_size)
            print(f"epoch {epoch} dev:\n{report(m)}")
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
# A quick look at one prompt and one training batch first. Training resumes
# automatically from `last.pt` if a previous session was interrupted.

# %%
model = VideoMCQModel(cfg, DEVICE)
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_all:,} total, {n_train:,} trainable ({100 * n_train / n_all:.2f}%)")

prompts = PromptBuilder(model.processor, motion_text)
train_collator = MCQCollator(model.processor, prompts, cache, cfg, train=True)
eval_collator = MCQCollator(model.processor, prompts, cache, cfg, train=False)
train_set = ReVADataset(fit, shuffle_options=True, seed=cfg.seed)

example = next(e for e in dev if e.task == "Temporal Grounding") if any(
    e.task == "Temporal Grounding" for e in dev) else dev[0]
print(prompts(example)[-900:])
batch = train_collator([train_set[0]])
print({k: tuple(v.shape) for k, v in batch["inputs"].items() if hasattr(v, "shape")})
frames, _ = eval_collator.video(example)
print("frame shape", frames.shape, "| time bar drawn:", example.task in cfg.overlay_time_tasks)

# %%
trainer = Trainer(model, train_set, train_collator, eval_collator, dev, cfg)
trainer.resume()
print(f"{len(train_set)} questions, {trainer.total_steps} optimizer steps of {cfg.accum_steps} questions")
trainer.fit()

# %% [markdown]
# ## 14. Final dev evaluation (best checkpoint, PriDe)

# %%
best = torch.load(CKPT_DIR / "best.pt", map_location="cpu", weights_only=False)
model.load_trainable(best["adapter"])
dev_result = evaluate(model, eval_collator, dev, tta=cfg.tta, batch_size=cfg.eval_batch_size)
print(f"dev ({cfg.tta}):\n{report(dev_result)}")
np.save(RESULTS_DIR / "dev_scores.npy", dev_result.pop("scores"))
(RESULTS_DIR / "dev_predictions.json").write_text(json.dumps(dev_result.pop("predictions"), indent=1))
(RESULTS_DIR / "dev_metrics.json").write_text(json.dumps(dev_result, indent=1))

# %% [markdown]
# ## 15. Test predictions and `submission.zip`
#
# Codabench expects `submission.zip` containing `predictions.json` laid out
# like `test.json` (a `metadata` block and a `QA` list) with `correct_answer`
# filled for all 4,000 questions; a bare `{qa_id: letter}` dict is rejected.

# %%
test_result = evaluate(model, eval_collator, test, tta=cfg.tta, batch_size=cfg.eval_batch_size)
pred = test_result["predictions"]
raw = json.loads((REVA_ROOT / "test.json").read_text())
if len(pred) == len(raw["QA"]):
    payload = {"metadata": raw.get("metadata", {}),
               "QA": [{**r, "correct_answer": pred[r["qa_id"]]} for r in raw["QA"]]}
    assert all(r["correct_answer"] in LETTERS for r in payload["QA"])
    (RESULTS_DIR / "predictions.json").write_text(json.dumps(payload))
    with zipfile.ZipFile(RESULTS_DIR / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(RESULTS_DIR / "predictions.json", arcname="predictions.json")
    print(f"{len(pred)} predictions -> {RESULTS_DIR / 'submission.zip'}")
    print("letters:", dict(Counter(pred.values())))
    if IN_COLAB:
        from google.colab import files

        files.download(str(RESULTS_DIR / "submission.zip"))
else:
    print(f"scored {len(pred)} of {len(raw['QA'])} test questions (subset preset): no submission written")
