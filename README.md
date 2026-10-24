# Peekaboo-VL: Camera-Motion Text Hints for Drone Video Question Answering

**Peekaboo-VL** is a drone (UAV) video question-answering model built on
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). It reaches **83.55%
accuracy on the ReVA test leaderboard**, 3.5 points above the organizers'
ReMoSense model (80.1%), using one LoRA adapter, a one-sentence camera-motion
hint, and letter scoring. It was built for the
[ReVA Drone Video Understanding Challenge](https://www.codabench.org/competitions/18274/)
at the MAVU Workshop, WACV 2027.

| | |
|---|---|
| **Code** | [github.com/LakoreAI/peekaboo-vl](https://github.com/LakoreAI/peekaboo-vl) |
| **Weights** | [huggingface.co/LakoreAI/peekaboo-vl](https://huggingface.co/LakoreAI/peekaboo-vl) (LoRA adapter, PEFT format) |
| **Report** | [peekaboo_report.pdf](https://github.com/LakoreAI/peekaboo-vl/blob/main/docs/reports/peekaboo_report.pdf) |
| **Benchmark** | ReVA ([paper](https://arxiv.org/abs/2609.35507), [data](https://huggingface.co/datasets/ReVA-Benchmark/ReVA)) |
| **Base model** | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |

## Results

| Split | Questions | Accuracy |
|---|---|---|
| ReVA test leaderboard | 4,000 | **83.55%** |
| Development split (seen videos, unseen questions) | 2,078 | 85.37% |
| ReMoSense (organizers' baseline), test | 4,000 | 80.1% |

The hardest task is Temporal Grounding (about 70% on the development split).
Most of its errors miss the right answer by about one second, the gap between
two sampled frames.

## Why drone video is hard, and what Peekaboo-VL does about it

Drone footage is hard for video language models: the camera never stops
moving, objects are a few pixels wide, and many questions ask *when*
something happens. ReMoSense handles camera motion by learning new motion
tokens. Peekaboo-VL takes a simpler route: it measures the motion and **tells
the model in words**.

- **Camera-motion hint.** A CPU probe (ORB features + RANSAC) estimates the
  frame-to-frame camera motion in milliseconds and writes it as one sentence in
  the benchmark's own answer vocabulary, for example *"Camera motion:
  0.0-4.1s moving forward, slow; 4.1-15.0s panning left"*. No new network
  parts are added.
- **True timestamps.** 32 frames per clip at 640×352, each tagged with its real
  time in seconds. During training, frames are sampled at a random position
  inside each time segment.
- **Time bar.** On Temporal Grounding questions, a progress bar is drawn on each
  frame so the model can see where in the clip it is.
- **Letter scoring, not text generation.** The answer is read from the model's
  next-token scores for `A`, `B`, `C` and `D` in a single forward pass, and
  training is cross-entropy over those four scores.
- **PriDe debiasing.** At test time, [PriDe](https://arxiv.org/abs/2309.03882)
  estimates the model's preference for each letter on 5% of the questions with
  rotated option orders, then removes it.

```
drone clip (15 s, 640x360)
   ├─> frame cache: 32 frames + true timestamps
   └─> camera-motion probe (ORB + RANSAC) -> "Camera motion: 0.0-4.1s moving forward ..."
          │
          v
   frames + motion sentence + question + options A-D
          -> Qwen3.5-9B + LoRA -> scores for A / B / C / D -> PriDe -> answer
```

## Use the model

### With PEFT

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

base = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, "LakoreAI/peekaboo-vl")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
```

The model was trained to pick a letter, not to write free text. To get the
reported scores, build the prompt the same way (32 frames with timestamps, the
camera-motion sentence, the question, options A–D) and compare the next-token
scores of `A`, `B`, `C` and `D`. The project code below does this for you.

### With the project code

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/LakoreAI/peekaboo-vl && cd peekaboo-vl
uv sync --extra llm --extra rich

# data: download ReVA, build the frame cache and the camera-motion hints
uv run python scripts/data/fetch_data.py
uv run python scripts/data/make_splits.py
uv run python scripts/data/extract_frames.py --workers 16
uv run python scripts/data/compute_motion_cues.py --workers 16

# score the test set with the released weights, then package for Codabench
hf download LakoreAI/peekaboo-vl --local-dir weights/peekaboo-vl
uv run python -m src.pipelines.predict --ckpt weights/peekaboo-vl \
    --config configs/train.yaml --split data/raw/reva/test.json \
    --tta pride --branch 8 --out_dir results/peekaboo-vl/test
uv run python scripts/submission/make_submission.py \
    --pred results/peekaboo-vl/test/predictions.json --out results/submissions/peekaboo-vl
```

### Train it yourself

```bash
uv run python scripts/training/train.py --config configs/train.yaml
```

This trains the released recipe: LoRA rank 64 (alpha 128) on all language-model
linear layers plus the vision-language merger, with the vision tower frozen.
It runs 1 epoch over 14,068 ReVA training questions at learning rate 1e-4
(warm-up and cosine decay), in bf16. That took about 17 hours on one NVIDIA
A40 (48 GB). Run `uv run pytest` for the offline test suite, and
`uv run python scripts/training/smoke_test.py` for a quick end-to-end check
that needs no GPU and no data.

## Repository layout

```
src/            model, prompting, frame cache, cues, decoding, training and inference pipelines
configs/        train.yaml: the released training recipe
scripts/        data download and preparation, training, evaluation, Codabench packaging
tests/          offline pytest suite
notebooks/      self-contained Colab notebooks
docs/reports/   the technical report (LaTeX sources + PDF)
```

## FAQ

**What is ReVA?** A benchmark of multiple-choice questions about short drone
videos (VisDrone, UAVDT, ERA and others), covering 11 tasks such as Temporal
Grounding, Change Detection, Object and Land Cover Recognition, and Trend and
Pattern analysis.

**Does it work on drone videos outside ReVA?** It has not been tested there.
The ReVA test set asks new questions about videos that also appear in
training, so the scores show performance in that setting, not on unseen
footage.

**Why a text hint instead of a motion module?** It needs no new parameters,
runs on a CPU, and can be read and checked by a person. In our tests, how the
hint is *worded* mattered more than how precise the measurement behind it
was: writing it in the benchmark's answer words helped, and writing it in
image terms hurt.

**How low can a model score without the video?** A text-only model that never
sees a frame scores 55.9% on ReVA, so the practical floor is much higher than
the 25% chance level.

## Limitations

- Results are for the ReVA setting, where test questions are about videos
  seen in training.
- Temporal Grounding remains the weakest task.
- PriDe uses 5% of the test questions to estimate the letter preference.
  Scoring each question on its own works too, but is slightly less accurate.

## Citation

```bibtex
@misc{le2026peekaboovl,
  title        = {Peekaboo-VL: Camera-Motion Text Hints and LoRA Letter Scoring
                  for Drone Video Question Answering},
  author       = {Le, Duc Minh},
  year         = {2026},
  howpublished = {\url{https://github.com/LakoreAI/peekaboo-vl}},
  note         = {ReVA Drone Video Understanding Challenge, MAVU Workshop @ WACV 2027}
}
```

## License

The code is released under the
[MIT License](https://github.com/LakoreAI/peekaboo-vl/blob/main/LICENSE). The
adapter weights follow the license of the base model,
[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). The ReVA videos
and annotations are for non-commercial research only and may not be
redistributed; please cite the
[ReVA paper](https://arxiv.org/abs/2609.35507) if you use them.
