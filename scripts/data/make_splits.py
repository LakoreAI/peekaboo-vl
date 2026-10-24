"""Build the fit/dev split that mirrors the test regime.

Test = new questions about videos that are (almost all) in train. Raw val is
79% exact duplicates of train questions, so it cannot rank models. This
writes:

    data/processed/splits/dev.json   val non-duplicates + N held-out train
                                     questions per task (videos stay in fit)
    data/processed/splits/fit.json   train + val minus every record asking a
                                     dev question, de-duplicated

Usage:
    uv run python scripts/data/make_splits.py
    uv run python scripts/data/make_splits.py --holdout_per_task 200 --seed 1
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import dedupe, load_split, make_dev_split, question_key, save_split  # noqa: E402
from src.labels import task_short  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT / "data" / "raw" / "reva"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data" / "processed" / "splits"
    )
    parser.add_argument("--holdout_per_task", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = load_split(args.root / "train.json")
    val = load_split(args.root / "val.json")
    fit, dev = make_dev_split(train, val, args.holdout_per_task, args.seed)
    fit = dedupe(fit)

    dev_keys = {question_key(ex) for ex in dev}
    assert not any(question_key(ex) in dev_keys for ex in fit), (
        "dev question leaked into fit"
    )
    fit_videos = {ex.video_path for ex in fit}
    seen = sum(ex.video_path in fit_videos for ex in dev)

    meta = {"seed": args.seed, "holdout_per_task": args.holdout_per_task}
    save_split(fit, args.out / "fit.json", split="fit", **meta)
    save_split(dev, args.out / "dev.json", split="dev", **meta)
    print(
        f"fit: {len(fit)} questions, {len(fit_videos)} videos -> {args.out / 'fit.json'}"
    )
    print(
        f"dev: {len(dev)} questions ({seen} on fit videos) -> {args.out / 'dev.json'}"
    )
    print(
        "dev tasks:", dict(sorted(Counter(task_short(ex.task) for ex in dev).items()))
    )


if __name__ == "__main__":
    main()
