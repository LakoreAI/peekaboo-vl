"""Validate predictions against test.json and package `submission.zip`.

Codabench rejects missing, duplicate, or unknown `qa_id`s and any answer
outside A/B/C/D, so all of that is checked locally first.

`--format` writes one of:
    testjson test.json with `correct_answer` filled in               (default)
    dict     {"test_000001": "A", ...}
    list     [{"qa_id": "test_000001", "answer": "A"}, ...]
The Codabench scorer requires `testjson`: it rejects anything without a
top-level "QA" list ("Submission must contain a top-level 'QA' list").
The other layouts are kept for reference.

Usage:
    uv run python scripts/submission/make_submission.py \
        --pred results/<run>/test/predictions.json --out results/submissions/<name>
"""

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.decoding import letter_distribution  # noqa: E402
from src.labels import LETTERS  # noqa: E402
from src.utils.io_utils import load_json, save_json  # noqa: E402


def validate(pred: dict, test_ids: list) -> None:
    errors = []
    unknown = sorted(set(pred) - set(test_ids))
    missing = sorted(set(test_ids) - set(pred))
    bad = sorted(k for k, v in pred.items() if v not in LETTERS)
    if unknown:
        errors.append(f"{len(unknown)} unknown qa_ids, e.g. {unknown[:3]}")
    if missing:
        errors.append(f"{len(missing)} missing qa_ids, e.g. {missing[:3]}")
    if bad:
        errors.append(
            f"{len(bad)} answers not in A-D, e.g. {[(k, pred[k]) for k in bad[:3]]}"
        )
    if len(test_ids) != len(set(test_ids)):
        errors.append("test.json itself has duplicate qa_ids")
    if errors:
        raise ValueError("invalid submission:\n  " + "\n  ".join(errors))


def render(pred: dict, test: dict, fmt: str):
    ids = [r["qa_id"] for r in test["QA"]]
    if fmt == "dict":
        return {i: pred[i] for i in ids}
    if fmt == "list":
        return [{"qa_id": i, "answer": pred[i]} for i in ids]
    if fmt == "testjson":
        return {
            **test,
            "QA": [{**r, "correct_answer": pred[r["qa_id"]]} for r in test["QA"]],
        }
    raise ValueError(f"unknown format {fmt!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pred", type=Path, required=True, help="{qa_id: letter} from predict.py"
    )
    parser.add_argument(
        "--test", type=Path, default=REPO_ROOT / "data" / "raw" / "reva" / "test.json"
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--format", choices=["testjson", "dict", "list"], default="testjson"
    )
    args = parser.parse_args()

    pred = load_json(args.pred)
    test = load_json(args.test)
    validate(pred, [r["qa_id"] for r in test["QA"]])

    args.out.mkdir(parents=True, exist_ok=True)
    payload = args.out / "predictions.json"
    save_json(render(pred, test, args.format), payload, save_pretty=False)
    zpath = args.out / "submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(payload, arcname="predictions.json")
    dist = "  ".join(
        f"{k}:{v:.3f}" for k, v in letter_distribution(list(pred.values())).items()
    )
    print(f"{len(pred)} predictions OK ({args.format})  letters {dist}")
    print(f"saved: {zpath}")


if __name__ == "__main__":
    main()
