"""End-to-end smoke test without a GPU or the real dataset.

Synthesizes a few clips with known camera motion, then runs every stage that
does not need VLM weights: mp4 decode -> frame cache -> motion cue text ->
prompt -> TTA/PriDe decoding of fake letter logits -> metrics -> submission
validation. With `--model <hf id>` (e.g. Qwen/Qwen3.5-0.8B) it also runs a
real forward pass of the VLM on the synthetic clips, which needs the weights
(and is slow on CPU).

Usage:
    uv run python scripts/training/smoke_test.py
    uv run python scripts/training/smoke_test.py --model Qwen/Qwen3.5-0.8B
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import ReVAExample, cyclic_perms, permute  # noqa: E402
from src.decoding import predict_letters  # noqa: E402
from src.modules.cues import CueStore  # noqa: E402
from src.modules.cues.motion import motion_record  # noqa: E402
from src.pipelines.eval import decode, format_report, reva_metrics  # noqa: E402
from src.prompting import question_text  # noqa: E402
from src.utils.io_utils import write_jsonl  # noqa: E402
from src.video import cache_dir_for, cache_frames, load_cached_frames, read_video  # noqa: E402


def texture(h: int, w: int, seed: int) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(img, (3, 3), 0)


def write_clip(path: Path, motion: str, n: int = 48, fps: float = 12.0) -> None:
    """A 4 s clip of a large texture seen through a moving 320x180 window."""
    import cv2

    h, w = 180, 320
    world = texture(h * 3, w * 3, seed=len(motion))
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for t in range(n):
        cx, cy, s = w * 1.5, h * 1.5, 1.0
        if motion == "pan_right":
            cx += 3.0 * t  # window moves right -> scene shifts left
        elif motion == "zoom_in":
            s = 1.0 + 0.01 * t
        M = cv2.getRotationMatrix2D((cx, cy), 0, s)
        M[0, 2] += w / 2 - cx
        M[1, 2] += h / 2 - cy
        frame = cv2.warpAffine(world, M, (w, h))
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", default=None, help="HF id for an optional real forward pass"
    )
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="peekaboo_smoke_"))
    try:
        videos, frames_root, cues_root = tmp / "videos", tmp / "frames", tmp / "cues"
        videos.mkdir()
        rows, examples = [], []
        for k, motion in enumerate(["pan_right", "zoom_in", "static"]):
            vp = f"SYN/{motion}.mp4"
            (videos / "SYN").mkdir(exist_ok=True)
            write_clip(videos / vp, motion)
            fs = read_video(videos / vp, 16)
            cache_frames(fs, cache_dir_for(frames_root, vp))
            cached = load_cached_frames(frames_root, vp, 8)
            rec = motion_record(list(cached.frames), cached.timestamps)
            rows.append({"video_path": vp, "text": rec["text"]})
            print(f"{motion:9s} -> {rec['text']}")
            examples.append(
                ReVAExample(
                    qa_id=f"test_{k:06d}",
                    video_path=vp,
                    task="Perspective and Viewpoint",
                    category="Spatial Reasoning",
                    question="How does the camera move?",
                    options=(
                        "It pans right.",
                        "It zooms in.",
                        "It stays still.",
                        "It rotates.",
                    ),
                    label=k,
                )
            )
        assert "pans right" in rows[0]["text"], rows[0]["text"]
        assert "zooms in" in rows[1]["text"], rows[1]["text"]
        assert "static" in rows[2]["text"], rows[2]["text"]
        write_jsonl(rows, cues_root / "motion.jsonl")

        store = CueStore(["motion"], cues_root=cues_root)
        print("\n" + question_text(examples[0], store.text(examples[0])) + "\n")

        # Fake scorer: right answer + a strong bias toward letter A.
        perms = cyclic_perms()
        display = np.zeros((len(examples), len(perms), 4))
        for n, ex in enumerate(examples):
            for p, perm in enumerate(perms):
                shown = permute(ex, perm)
                display[n, p, shown.label] += 1.0
                display[n, p, 0] += 1.5
        none = reva_metrics(
            predict_letters(decode(display[:, :1], perms[:1], "none")), examples
        )
        cyc = reva_metrics(predict_letters(decode(display, perms, "cyclic")), examples)
        print(
            f"letter-biased fake scorer: none={none['accuracy']:.2f}  cyclic TTA={cyc['accuracy']:.2f}"
        )
        assert cyc["accuracy"] == 1.0 and none["accuracy"] < 1.0
        print(format_report(cyc))

        if args.model:
            import torch

            from src.config import ModelConfig
            from src.modules.collate import MCQCollator
            from src.modules.model import VideoMCQModel
            from src.pipelines.eval import evaluate
            from src.utils.model_utils import detect_device

            device = detect_device()
            cfg = ModelConfig(
                backbone=args.model,
                num_frames=8,
                cues=("motion",),
                torch_dtype="bfloat16" if device.type == "cuda" else "float32",
                gradient_checkpointing=False,
            )
            model = VideoMCQModel.from_config(cfg, device)
            collator = MCQCollator(model.processor, cfg, store, frames_root)
            with torch.no_grad():
                result = evaluate(
                    model, collator, examples, device, tta="cyclic", batch_size=2
                )
            print(f"\nreal forward ({args.model}):\n{format_report(result)}")

        print("\nsmoke test: PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
