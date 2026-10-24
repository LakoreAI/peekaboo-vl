"""Presentation helpers for the training entrypoint.

Kept separate from `src.pipelines.train` so the training loop stays readable.
Nothing here affects training — it only prints. The most useful part is the
rendered prompt of one real example: cue text, option layout, and the chat
template are where silent mistakes hide.

Output is rendered with `rich` when it is installed and falls back to plain
text otherwise (`uv sync --extra rich`).
"""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from src.config import ModelConfig
from src.data import ReVAExample
from src.labels import task_short
from src.pipelines.config import TrainingConfig

try:  # optional pretty-printing dependency
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    _HAS_RICH = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _HAS_RICH = False

WIDTH = 72


def banner(title: str) -> None:
    print(f"\n{title}\n{'-' * WIDTH}")


def indent(text: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def task_counts(examples: Sequence[ReVAExample]) -> str:
    counts = Counter(task_short(ex.task) for ex in examples)
    return "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def parameter_summary(model: torch.nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"parameters: {total:,d} total, {trainable:,d} trainable ({100 * trainable / max(total, 1):.2f}%)"


def sample_prompt(collator, ex: ReVAExample, max_chars: int = 2500) -> str:
    """The chat-templated prompt of one example (video tokens not expanded)."""
    try:
        text = collator.prompt(ex)
    except Exception as e:  # noqa: BLE001 - a preview must never block training
        return f"prompt preview skipped: {e}"
    return text if len(text) <= max_chars else text[:max_chars] + " ..."


def announce_training(
    *,
    model: torch.nn.Module,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    collator,
    fit: Sequence[ReVAExample],
    dev: Sequence[ReVAExample],
    device: torch.device,
    run_name: str,
    ckpt_dir: Path,
    result_dir: Path,
) -> None:
    """Print configs, the data mix, one rendered prompt, and parameter counts."""
    sections = [
        (
            f"RUN {run_name}",
            f"device: {device}   seed: {train_cfg.seed}\n"
            f"checkpoints: {ckpt_dir}\nresults: {result_dir}",
        ),
        (
            "model config (ModelConfig)",
            json.dumps(asdict(model_cfg), indent=2, default=str),
        ),
        (
            "training config (TrainingConfig)",
            json.dumps(asdict(train_cfg), indent=2, default=str),
        ),
        (
            f"data: fit {len(fit)} / dev {len(dev)}",
            f"fit  {task_counts(fit)}\ndev  {task_counts(dev)}",
        ),
        (
            f"sample prompt ({fit[0].qa_id}, answer {fit[0].answer_letter})"
            if fit
            else "sample prompt",
            sample_prompt(collator, fit[0]) if fit else "(no examples)",
        ),
        ("model", parameter_summary(model)),
    ]
    if _HAS_RICH:
        console = Console(highlight=False)
        for title, body in sections:
            # Text(): prompts contain [brackets] that rich would parse as markup.
            console.print(Panel(Text(body), title=title, border_style="blue"))
    else:
        for title, body in sections:
            banner(title)
            print(indent(body))
        print("\n" + "=" * WIDTH + "\n")
