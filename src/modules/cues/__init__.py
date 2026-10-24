"""Text cues prepended to the prompt: what the VLM cannot see well.

Per-clip cues (motion, counts) are precomputed once into JSONL files keyed by
`video_path` and looked up by `CueStore`. The per-question `memory` cue is
computed on the fly from labelled examples (see `src.modules.cues.memory`).
"""

from pathlib import Path
from typing import Dict, Optional, Sequence

from src.data import ReVAExample
from src.modules.cues.counts import classes_in_question, describe_timeline
from src.modules.cues.memory import VideoMemory
from src.utils.io_utils import read_jsonl

SUPPORTED_CUES = (
    "motion", "motion_camera", "objmotion", "counts", "timeline", "caption", "memory"
)
# Precomputed per-clip cue -> file name under `cues_root`.
# motion_camera: same probe, rendered in camera vocabulary (MotionConfig.phrasing).
# objmotion: residual flow after removing the camera transform (objmotion.py).
# timeline: object-appearance timeline rendered per question from counts.jsonl
#   (needs the detector run with more frames: `compute_counts_cues.py --num_times 8`).
# caption: the ReVA QA generator's consolidated caption, replayed with the same
#   public model and prompts (`compute_caption_cues.py`).
CUE_FILES = {
    "caption": "caption.jsonl",
    "motion": "motion.jsonl",
    "motion_camera": "motion_camera.jsonl",
    "objmotion": "objmotion.jsonl",
    "counts": "counts.jsonl",
    "timeline": "counts.jsonl",
}
# Tasks the query-aware timeline cue is shown for (the "when" question types).
TIMELINE_TASKS = {"Temporal Grounding"}


class CueStore:
    """Looks up the enabled cues for an example and joins them into text."""

    def __init__(
        self,
        cues: Sequence[str] = (),
        cues_root: Optional[str | Path] = None,
        memory: Optional[VideoMemory] = None,
        strict: bool = True,
    ):
        unknown = [c for c in cues if c not in SUPPORTED_CUES]
        if unknown:
            raise ValueError(f"unknown cues {unknown}; choose from {SUPPORTED_CUES}")
        if "memory" in cues and memory is None:
            raise ValueError("the 'memory' cue needs a VideoMemory")
        self.cues = tuple(cues)
        self.memory = memory
        self.strict = strict
        self.tables: Dict[str, Dict[str, str]] = {}
        for cue in self.cues:
            if cue not in CUE_FILES:
                continue
            if cues_root is None:
                raise ValueError(f"cue {cue!r} needs cues_root")
            path = Path(cues_root) / CUE_FILES[cue]
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing; precompute the {cue!r} cue first "
                    "(scripts/data/compute_motion_cues.py for the camera-motion cue)"
                )
            rows = read_jsonl(path)
            if cue == "timeline":
                # Keep the raw per-frame counts so the line can be rendered per
                # question (query-aware, task-gated) at prompt time.
                self.tables[cue] = {r["video_path"]: r for r in rows}
            else:
                self.tables[cue] = {r["video_path"]: r["text"] for r in rows}

    def shuffle(self, seed: int = 0) -> "CueStore":
        """Give every clip the per-clip cue text of a different random clip
        (a derangement), for cue-reliance ablations."""
        import random

        rng = random.Random(seed)
        for cue, table in self.tables.items():
            keys = sorted(table)
            perm = keys[:]
            while len(keys) > 1 and any(a == b for a, b in zip(keys, perm)):
                rng.shuffle(perm)
            self.tables[cue] = {k: table[v] for k, v in zip(keys, perm)}
        return self

    def text(self, ex: ReVAExample, skip: Sequence[str] = ()) -> str:
        """The enabled cues for `ex`, one per line; cues in `skip` are left
        out (training-time cue dropout)."""
        lines = []
        for cue in self.cues:
            if cue in skip:
                continue
            if cue == "memory":
                line = self.memory.describe(ex)
            elif cue == "timeline":
                if ex.task not in TIMELINE_TASKS:
                    continue
                row = self.tables["timeline"].get(ex.video_path)
                if row is None:
                    if self.strict:
                        raise KeyError(f"no timeline cue for {ex.video_path}")
                    continue
                line = describe_timeline(
                    row.get("per_frame", []),
                    row.get("timestamps", []),
                    classes_in_question(ex.question),
                )
            else:
                line = self.tables[cue].get(ex.video_path)
                if line is None and self.strict:
                    raise KeyError(f"no {cue} cue for {ex.video_path}")
            if line:
                lines.append(line)
        return "\n".join(lines)
