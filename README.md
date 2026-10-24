# Peekaboo-VL — ReVA Drone Video Understanding Challenge

> Camera-motion text hints and LoRA letter scoring for drone video question
> answering, on top of Qwen3.5-9B. Entry for the
> [ReVA challenge](https://www.codabench.org/competitions/18274/) @ MAVU, WACV 2027.

- **Test accuracy:** 83.55% on the ReVA leaderboard (organizers' ReMoSense: 80.1%).
- **Weights:** [`LakoreAI/peekaboo-vl`](https://huggingface.co/LakoreAI/peekaboo-vl)
  (LoRA adapter, PEFT format).
- **Paper:** [`docs/reports/peekaboo_report.pdf`](docs/reports/peekaboo_report.pdf).

## How it works

```
clip (15 s, 640x360) --extract_frames--> 64-frame cache + fps/indices
   |                                      |
   |                     ORB + RANSAC similarity per frame pair
   |                     -> "Camera motion: 0.0-4.1s pans right, slow; ..."
   v                                      v
32 frames @ 640x352  +  cue lines + question + A-D options
   -> Qwen3.5-9B (+ LoRA)  -> next-token logits of A/B/C/D
   -> cyclic-permutation TTA / PriDe  -> predictions.json
```

- **Scoring, not generation.** The four class logits are the LM-head logits
  of `A B C D` at the answer position, and training is cross-entropy over
  them. Options are re-shuffled every time a question is drawn.
- **True timestamps.** Qwen3.5/Qwen3-VL write `<t seconds>` before each
  temporal patch. The frame cache keeps the source fps and frame indices so
  those timestamps are correct.
- **Native resolution.** The processor's pixel budget is per video, not per
  frame. `max_pixels_per_frame` keeps 640×352 instead of ~208×117.
- **Honest dev split.** Test is new questions on train videos, and val is 79%
  duplicates of train. `make_splits.py` builds a seen-video / unseen-question
  dev set (2,078 Q).

## Quickstart

Requires Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra rich                 # core (CPU/MPS on macOS, CUDA on Linux)
uv sync --extra llm --extra rich     # + peft / bitsandbytes / fla kernels for training
uv run pytest                        # offline test suite
uv run python scripts/training/smoke_test.py                              # no GPU, no data
uv run python scripts/training/smoke_test.py --model Qwen/Qwen3.5-0.8B    # + real VLM forward
```

Data and training (GPU box):

```bash
uv run python scripts/data/fetch_data.py            # HF ReVA-Benchmark/ReVA, 29.9 GB
uv run python scripts/data/make_splits.py           # fit / dev
uv run python scripts/data/extract_frames.py --workers 16 [--delete_videos]
uv run python scripts/data/compute_motion_cues.py --workers 16

uv run python scripts/training/train.py --config configs/train.yaml
uv run python -m src.pipelines.predict --ckpt checkpoints/<run>/best.pt \
    --config configs/train.yaml \
    --split data/raw/reva/test.json --tta cyclic --out_dir results/<run>/test
uv run python scripts/submission/make_submission.py \
    --pred results/<run>/test/predictions.json --out results/submissions/<run>
```

The ReVA data is for non-commercial research only and may not be
redistributed. `data/` is gitignored.

## Repository layout

```
src/
├── labels.py            # A-D letters, 11 tasks / 4 categories, test task counts
├── data.py              # ReVAExample, JSON I/O, option permutation, fit/dev split
├── video.py             # uniform sampling, OpenCV decode, frame cache (+fps, indices)
├── prompting.py         # cue lines + question + options -> chat messages
├── decoding.py          # permutation TTA, PriDe prior estimation / debiasing
├── config.py            # ModelConfig: backbone, frames, pixels, cues, LoRA (saved in ckpt)
├── modules/
│   ├── model.py         # VideoMCQModel: VLM -> letter logits; LoRA targets
│   ├── collate.py       # MCQCollator: prompts + videos + video_metadata
│   ├── loss.py          # CE over the 4 letter logits
│   └── cues/            # motion.py (camera ego-motion), counts.py (detector), memory.py
├── pipelines/
│   ├── config.py        # TrainingConfig: data paths, optimization, TTA, callbacks
│   ├── train.py         # LoRA fine-tuning loop + callback wiring
│   ├── eval.py          # scoring under permutations, decoding, ReVA metrics
│   └── predict.py       # checkpoint / zero-shot scoring -> scores, predictions.json
├── callbacks/           # best checkpoint, early stopping, LR schedule, W&B
└── utils/               # io (adapter-only checkpoints, jsonl), device, run names
configs/                 # train.yaml: the Peekaboo-VL training config
scripts/                 # data/, training/, submission/ (see scripts/README.md)
tests/                   # pytest suite (offline)
docs/reports/            # the paper (LaTeX sources + PDF)
```

## Configuration

Two configs, as in the template:

- **`src/config.py` → `ModelConfig`** is everything that defines the model's
  input: backbone, frame count, pixel budget, cue list, chat-template kwargs,
  and the LoRA shape. It is saved into every checkpoint, so `predict.py`
  rebuilds the exact prompt the model was trained with.
- **`src/pipelines/config.py` → `TrainingConfig`** is the loop: data paths,
  lr, epochs, accumulation, TTA mode, and callbacks. It loads from YAML with
  CLI overrides; the `arch:` block overrides `ModelConfig` fields.

## License

Code: see [LICENSE](LICENSE). Dataset: ReVA terms (research only, cite
arXiv 2609.35507).
