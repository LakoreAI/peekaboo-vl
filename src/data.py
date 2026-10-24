"""ReVA annotation loading, option permutation, and leakage-safe splits.

One example is a 4-option multiple-choice question about one ~15 s UAV clip.
The official files (`train.json`, `val.json`, `test.json`) are dicts with
`metadata` and a `QA` list; each record carries:

    qa_id, video_path, subdir, dataset_name, category, task,
    question, options {"A".."D": text}, correct_answer ("" in test)

Two properties of the release shape the splits:

* 1,012 of 1,014 test videos are also train videos. The test set asks *new
  questions about seen videos*, so dev must hold out questions, not videos.
* 1,572 of 2,000 val questions are exact (video, question) duplicates of train
  questions with the same answer. Raw val accuracy after training on train is
  inflated; `make_dev_split` keeps only the non-duplicates.
"""

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset

from src.labels import LETTER_TO_INDEX, LETTERS, NUM_OPTIONS
from src.utils.io_utils import load_json, save_json


@dataclass(frozen=True)
class ReVAExample:
    qa_id: str
    video_path: str
    task: str
    category: str
    question: str
    # Option texts in display order: options[i] is shown as LETTERS[i].
    options: Tuple[str, ...]
    # Index into `options` of the correct answer; -1 when unlabeled (test).
    label: int = -1
    dataset_name: str = ""
    subdir: str = ""
    # Display position i shows original option `perm[i]`. Identity for
    # unpermuted examples; used to map scores back after permutation TTA.
    perm: Tuple[int, ...] = field(default=tuple(range(NUM_OPTIONS)))

    @property
    def answer_letter(self) -> Optional[str]:
        return LETTERS[self.label] if self.label >= 0 else None

    @property
    def answer_text(self) -> Optional[str]:
        return self.options[self.label] if self.label >= 0 else None


def example_from_record(rec: dict) -> ReVAExample:
    opts = rec["options"]
    missing = [k for k in LETTERS if k not in opts]
    if missing:
        raise ValueError(f"{rec.get('qa_id')}: options missing letters {missing}")
    answer = (rec.get("correct_answer") or "").strip()
    if answer and answer not in LETTER_TO_INDEX:
        raise ValueError(f"{rec.get('qa_id')}: bad correct_answer {answer!r}")
    return ReVAExample(
        qa_id=rec["qa_id"],
        video_path=rec["video_path"],
        task=rec.get("task", ""),
        category=rec.get("category", ""),
        question=rec["question"],
        options=tuple(str(opts[k]) for k in LETTERS),
        label=LETTER_TO_INDEX[answer] if answer else -1,
        dataset_name=rec.get("dataset_name", ""),
        subdir=rec.get("subdir", ""),
    )


def load_split(path: str | Path) -> List[ReVAExample]:
    """Read an official annotation file (or one written by `save_split`)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"annotation file not found: {path}")
    data = load_json(path)
    records = data["QA"] if isinstance(data, dict) else data
    examples = [example_from_record(r) for r in records]
    ids = [ex.qa_id for ex in examples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate qa_id values")
    return examples


def to_record(ex: ReVAExample) -> dict:
    """Inverse of `example_from_record` for an unpermuted example."""
    return {
        "qa_id": ex.qa_id,
        "video_path": ex.video_path,
        "subdir": ex.subdir,
        "dataset_name": ex.dataset_name,
        "category": ex.category,
        "task": ex.task,
        "question": ex.question,
        "options": dict(zip(LETTERS, ex.options)),
        "correct_answer": ex.answer_letter or "",
    }


def save_split(examples: Iterable[ReVAExample], path: str | Path, **metadata) -> None:
    examples = list(examples)
    save_json(
        {
            "metadata": {"total_questions": len(examples), **metadata},
            "QA": [to_record(ex) for ex in examples],
        },
        path,
    )


# --- option permutation -------------------------------------------------------


def permute(ex: ReVAExample, perm: Sequence[int]) -> ReVAExample:
    """Reorder the options so display position i shows current option perm[i].

    Composes with an existing permutation, so `ex.perm` always maps display
    positions back to the *original* option order.
    """
    perm = tuple(int(p) for p in perm)
    if sorted(perm) != list(range(NUM_OPTIONS)):
        raise ValueError(f"not a permutation of {NUM_OPTIONS}: {perm}")
    options = tuple(ex.options[p] for p in perm)
    label = perm.index(ex.label) if ex.label >= 0 else -1
    return replace(
        ex, options=options, label=label, perm=tuple(ex.perm[p] for p in perm)
    )


def cyclic_perms(n: int = NUM_OPTIONS) -> List[Tuple[int, ...]]:
    """The n cyclic shifts: every option visits every letter exactly once."""
    return [tuple((i + s) % n for i in range(n)) for s in range(n)]


def random_perm(rng: random.Random, n: int = NUM_OPTIONS) -> Tuple[int, ...]:
    perm = list(range(n))
    rng.shuffle(perm)
    return tuple(perm)


# --- dataset ------------------------------------------------------------------


class ReVADataset(Dataset):
    """List-backed dataset of `ReVAExample`s.

    `shuffle_options` re-draws a random option order on every access, so the
    model cannot learn letter priors (train-time augmentation against the
    A/B/C/D position bias). Evaluation datasets keep the official order.
    """

    def __init__(
        self,
        examples: Sequence[ReVAExample],
        shuffle_options: bool = False,
        seed: int = 0,
    ):
        self.examples = list(examples)
        self.shuffle_options = shuffle_options
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the permutation stream for this epoch (see `__getitem__`).

        The permutation is a pure function of `(seed, epoch, index)`, so it does
        not depend on access order or on how DataLoader workers are scheduled.
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ReVAExample:
        ex = self.examples[index]
        if self.shuffle_options:
            rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
            ex = permute(ex, random_perm(rng))
        return ex

    @property
    def num_classes(self) -> int:
        return NUM_OPTIONS

    @property
    def labels(self) -> List[int]:
        return [ex.label for ex in self.examples]


class VideoGroupDataset(Dataset):
    """Training items that are *groups* of questions about one video.

    Each epoch every video's questions are shuffled and cut into chunks of at
    most `max_per_group`; one chunk becomes one packed sequence (video once,
    then the questions as consecutive chat turns, see `PackedMCQCollator`).
    Call `regroup(epoch)` at the start of every epoch for fresh chunks.
    """

    def __init__(
        self,
        examples: Sequence[ReVAExample],
        max_per_group: int,
        shuffle_options: bool = False,
        seed: int = 0,
        split_tasks: Sequence[str] = (),
    ):
        if max_per_group < 1:
            raise ValueError(f"max_per_group must be >= 1, got {max_per_group}")
        # `split_tasks`: questions of these tasks never share a group with the
        # others (their video input differs, e.g. the task-gated time bar).
        self.by_video: Dict[Tuple[str, bool], List[ReVAExample]] = defaultdict(list)
        for ex in examples:
            self.by_video[(ex.video_path, ex.task in split_tasks)].append(ex)
        self.max_per_group = max_per_group
        self.shuffle_options = shuffle_options
        self.seed = seed
        self.epoch = 0
        self.groups: List[List[ReVAExample]] = []
        self.regroup(0)

    def set_epoch(self, epoch: int) -> None:
        """Select the permutation stream for this epoch (see `__getitem__`)."""
        self.epoch = int(epoch)

    def regroup(self, epoch: int) -> None:
        self.set_epoch(epoch)
        rng = random.Random(self.seed * 1000 + epoch)
        groups = []
        for video in sorted(self.by_video):
            exs = self.by_video[video][:]
            rng.shuffle(exs)
            k = self.max_per_group
            groups.extend(exs[i : i + k] for i in range(0, len(exs), k))
        self.groups = groups

    @property
    def num_questions(self) -> int:
        return sum(len(g) for g in self.groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> List[ReVAExample]:
        group = self.groups[index]
        if self.shuffle_options:
            group = [
                permute(
                    ex,
                    random_perm(
                        random.Random(f"{self.seed}:{self.epoch}:{index}:{j}")
                    ),
                )
                for j, ex in enumerate(group)
            ]
        return group


# --- leakage-safe splits ------------------------------------------------------


def question_key(ex: ReVAExample) -> Tuple[str, str]:
    """Two records ask "the same question" when video and wording match."""
    return ex.video_path, _normalize(ex.question)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def source_sequence(video_path: str) -> str:
    """`VisDrone/uav0000009_03358_v_01.mp4` -> `VisDrone/uav0000009_03358_v`.

    Clips cut from one source sequence share a stem with a `_NN` suffix.
    """
    stem = video_path.rsplit(".", 1)[0]
    return re.sub(r"_\d{2}$", "", stem)


def make_dev_split(
    train: Sequence[ReVAExample],
    val: Sequence[ReVAExample],
    holdout_per_task: int = 150,
    seed: int = 42,
) -> Tuple[List[ReVAExample], List[ReVAExample]]:
    """Return `(fit, dev)` mirroring the test regime: seen video, new question.

    dev = val questions that are not (video, question) duplicates of train
          + `holdout_per_task` train questions per task, drawn at random.
    fit = everything else from train + val, minus every record that asks a dev
          question (so no duplicate of a dev question leaks into training).

    Videos stay shared between fit and dev, as they are between train and test.
    """
    rng = random.Random(seed)
    train_keys = {question_key(ex) for ex in train}
    dev = [ex for ex in val if question_key(ex) not in train_keys]

    by_task: Dict[str, List[ReVAExample]] = defaultdict(list)
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
    fit = [
        ex
        for ex in list(train) + list(val)
        if ex.qa_id not in dev_ids and question_key(ex) not in dev_keys
    ]
    return fit, sorted(dev, key=lambda ex: ex.qa_id)


def dedupe(examples: Sequence[ReVAExample]) -> List[ReVAExample]:
    """Drop repeated (video, question) records, keeping the first."""
    seen, out = set(), []
    for ex in examples:
        key = question_key(ex)
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def group_by_video(examples: Iterable[ReVAExample]) -> Dict[str, List[ReVAExample]]:
    groups: Dict[str, List[ReVAExample]] = defaultdict(list)
    for ex in examples:
        groups[ex.video_path].append(ex)
    return dict(groups)
