"""ReVA evaluation: scoring passes with optional TTA, and the official metrics.

`reva_metrics` mirrors the leaderboard: overall accuracy plus one accuracy
per task (leaderboard column names), and adds what the leaderboard hides:
per-category accuracy, a test-weighted accuracy (dev's task mix differs from
test's), and the predicted-letter distribution (position-bias check).

`score_examples` runs the model over examples under a set of option
permutations and returns log-probs in the *original* option order, so every
decoding mode (none / cyclic / pride) shares one code path.
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import ReVAExample, cyclic_perms, permute
from src.decoding import (
    aggregate_tta,
    debias,
    estimate_letter_prior,
    kl_from_uniform,
    letter_distribution,
    log_softmax,
    predict_letters,
    split_for_prior,
    to_original,
)
from src.labels import LETTERS, TASKS, TEST_TASK_COUNTS, task_category, task_short
from src.utils.io_utils import save_json

IDENTITY = tuple(range(len(LETTERS)))


def reva_metrics(
    pred_letters: Sequence[str], examples: Sequence[ReVAExample]
) -> Dict[str, object]:
    """Accuracy overall / per task / per category, plus letter skew."""
    if len(pred_letters) != len(examples):
        raise ValueError("one prediction per example")
    labelled = [(p, ex) for p, ex in zip(pred_letters, examples) if ex.label >= 0]
    if not labelled:
        return {"accuracy": float("nan"), "n": 0}

    hits = defaultdict(list)
    for p, ex in labelled:
        ok = float(p == ex.answer_letter)
        hits["__all__"].append(ok)
        hits[ex.task].append(ok)
        hits["cat:" + task_category(ex.task)].append(ok)

    per_task = {task_short(t): float(np.mean(v)) for t, v in hits.items() if t in TASKS}
    per_category = {
        k[4:]: float(np.mean(v)) for k, v in hits.items() if k.startswith("cat:")
    }
    weighted = [
        (TEST_TASK_COUNTS[t], float(np.mean(hits[t]))) for t in TASKS if hits.get(t)
    ]
    test_weighted = sum(w * a for w, a in weighted) / sum(w for w, _ in weighted)
    dist = letter_distribution([p for p, _ in labelled])
    return {
        "accuracy": float(np.mean(hits["__all__"])),
        "test_weighted_accuracy": test_weighted,
        "n": len(labelled),
        "per_task": per_task,
        "per_task_n": {task_short(t): len(v) for t, v in hits.items() if t in TASKS},
        "per_category": per_category,
        "pred_letter_dist": dist,
        "gold_letter_dist": letter_distribution(
            [ex.answer_letter for _, ex in labelled]
        ),
        "pred_letter_kl": kl_from_uniform(dist),
    }


@torch.no_grad()
def score_examples(
    model,
    collator,
    examples: Sequence[ReVAExample],
    device: torch.device,
    perms: Sequence[Sequence[int]] = (IDENTITY,),
    batch_size: int = 4,
    num_workers: int = 0,
) -> np.ndarray:
    """Display-position logits, shape `(N, P, 4)`, for each permutation.

    Row n, slice p holds the letter logits of `examples[n]` shown with
    permutation `perms[p]` (display order, not yet mapped back).
    """
    model.eval()
    jobs = [
        (n, p, permute(ex, perm))
        for n, ex in enumerate(examples)
        for p, perm in enumerate(perms)
    ]
    out = np.zeros((len(examples), len(perms), len(LETTERS)), dtype=np.float32)
    if getattr(collator, "group_size", 0) > 1:
        return _score_branched(model, collator, jobs, out, device, num_workers)
    loader = DataLoader(
        [job[2] for job in jobs],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    k = 0
    t0 = time.time()
    report_every = max(len(loader) // 20, 1)
    for b, batch in enumerate(loader, 1):
        inputs = batch["inputs"].to(device)
        logits, _ = model(inputs)
        for row in logits.float().cpu().numpy():
            n, p, _ = jobs[k]
            out[n, p] = row
            k += 1
        if b % report_every == 0 or b == len(loader):
            rate = (time.time() - t0) / k
            print(
                f"  scored {k}/{len(jobs)}  {rate:.2f}s/item  "
                f"eta={rate * (len(jobs) - k) / 60:.1f}min",
                flush=True,
            )
    return out


def _score_branched(model, collator, jobs, out, device, num_workers) -> np.ndarray:
    """`score_examples` with a `BranchedMCQCollator`: jobs on the same video
    (and time-bar setting) share one prefix pass, in chunks of
    `collator.group_size`. Scores equal the one-at-a-time ones."""
    by_key: Dict[tuple, List[int]] = {}
    for j, (_, _, ex) in enumerate(jobs):
        by_key.setdefault((ex.video_path, collator._overlay_time(ex)), []).append(j)
    size = collator.group_size
    chunks = [
        idx[i : i + size] for idx in by_key.values() for i in range(0, len(idx), size)
    ]
    loader = DataLoader(
        [[jobs[j][2] for j in c] for c in chunks],
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    done, t0 = 0, time.time()
    report_every = max(len(loader) // 20, 1)
    with torch.no_grad():
        for b, (chunk, batch) in enumerate(zip(chunks, loader), 1):
            batch["inputs"] = {k: v.to(device) for k, v in batch["inputs"].items()}
            logits = model.forward_branched(batch).float().cpu().numpy()
            for j, row in zip(chunk, logits):
                n, p, _ = jobs[j]
                out[n, p] = row
            done += len(chunk)
            if b % report_every == 0 or b == len(loader):
                rate = (time.time() - t0) / done
                print(
                    f"  scored {done}/{len(jobs)}  {rate:.2f}s/item  "
                    f"eta={rate * (len(jobs) - done) / 60:.1f}min",
                    flush=True,
                )
    return out


def decode(
    display_logits: np.ndarray,
    perms: Sequence[Sequence[int]],
    tta: str = "none",
    pride_fraction: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """`(N, P, 4)` display logits -> `(N, 4)` log-probs in original option order.

    none   : P must be 1 (identity), plain log-softmax.
    cyclic : P = 4 cyclic shifts, mean log-prob in the original frame.
    pride  : the first P = 4 cyclic shifts are only needed for the estimation
             subset; the rest use slice 0 (identity), debiased by the prior.
    """
    n = display_logits.shape[0]
    if tta == "none":
        return np.stack(
            [to_original(log_softmax(display_logits[i, 0]), perms[0]) for i in range(n)]
        )
    if tta == "cyclic":
        return np.stack(
            [aggregate_tta(list(display_logits[i]), perms) for i in range(n)]
        )
    if tta == "pride":
        est, _ = split_for_prior(n, pride_fraction, seed)
        prior = estimate_letter_prior(display_logits[est])
        out = debias(display_logits[:, 0], prior)
        # Questions in the estimation subset already paid for all shifts.
        for i in est:
            out[i] = aggregate_tta(list(display_logits[i]), perms)
        return out
    raise ValueError(f"unknown tta mode {tta!r}")


def evaluate(
    model,
    collator,
    examples: Sequence[ReVAExample],
    device: torch.device,
    tta: str = "none",
    batch_size: int = 4,
    pride_fraction: float = 0.05,
    num_workers: int = 0,
    save_json_path: Optional[Union[str, Path]] = None,
    scorer: Optional[Callable] = None,
) -> Dict[str, object]:
    """Score + decode + metrics. Returns the metrics dict with `scores`
    (N, 4 original-order log-probs) and `predictions` attached.

    `scorer(examples, perms) -> (N, P, 4)` display logits replaces the torch
    model (e.g. a vLLM scorer); decoding is shared."""
    if scorer is None:

        def scorer(exs, ps):
            return score_examples(
                model, collator, exs, device, ps, batch_size, num_workers
            )

    if tta == "none":
        perms = [IDENTITY]
        display = scorer(examples, perms)
    elif tta == "cyclic":
        perms = cyclic_perms()
        display = scorer(examples, perms)
    else:  # pride: all shifts on the estimation subset, identity elsewhere
        perms = cyclic_perms()
        est, _ = split_for_prior(len(examples), pride_fraction)
        display = np.zeros((len(examples), len(perms), len(LETTERS)), dtype=np.float32)
        display[:, :1] = scorer(examples, perms[:1])
        display[est] = scorer([examples[i] for i in est], perms)
    scores = decode(display, perms, tta, pride_fraction)
    preds = predict_letters(scores)
    result = reva_metrics(preds, examples)
    if save_json_path is not None:
        save_json(result, save_json_path)
    result["scores"] = scores
    result["predictions"] = dict(zip((ex.qa_id for ex in examples), preds))
    return result


def format_report(result: Dict[str, object]) -> str:
    if not result.get("n"):
        return "no labelled examples"
    lines = [
        f"accuracy={result['accuracy'] * 100:6.2f}  "
        f"test-weighted={result['test_weighted_accuracy'] * 100:6.2f}  "
        f"n={result['n']}  letter-KL={result['pred_letter_kl']:.4f}"
    ]
    for name, acc in result["per_task"].items():
        lines.append(
            f"  {name:<18s} acc={acc * 100:6.2f}  (n={result['per_task_n'][name]})"
        )
    dist = "  ".join(f"{k}:{v:.2f}" for k, v in result["pred_letter_dist"].items())
    lines.append(f"  predicted letters  {dist}")
    return "\n".join(lines)


def per_task_table(results: Dict[str, Dict[str, object]]) -> List[str]:
    """Markdown rows comparing several runs' per-task accuracies."""
    cols = [short for _, short in TASKS.values()]
    lines = [
        "| run | overall | " + " | ".join(cols) + " |",
        "|---|--:|" + "--:|" * len(cols),
    ]
    for name, res in results.items():
        cells = [f"{res['per_task'].get(c, float('nan')) * 100:.1f}" for c in cols]
        lines.append(
            f"| {name} | {res['accuracy'] * 100:.1f} | " + " | ".join(cells) + " |"
        )
    return lines
