"""Object counts as text: do not trust the VLM to count small, dense objects.

A VisDrone-trained detector counts objects per sampled frame; the counts are
rendered as one prompt line (GroundCount-style, arXiv 2603.10978):

    Detector counts: 0.0s car 23, pedestrian 4; 7.5s car 19; 14.9s car 31.

ReVA's counting answers are bins ("10-20", "20-30"), so a +/-20% detector
error is usually harmless. Only a few timestamps are rendered (first, middle,
last by default), because most counting questions anchor at "the 00:00 mark"
or compare start vs end.

`run_detector` wraps ultralytics (optional extra `detect`) and is only used
to precompute the counts; everything else works on plain dicts.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import re

# VisDrone-DET class names. A VisDrone-trained YOLO uses these ids 0..9.
VISDRONE_CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
# Merge near-synonyms so the text matches ReVA's vocabulary ("people", "cars").
DEFAULT_MERGE = {
    "pedestrian": "person",
    "people": "person",
    "motor": "motorcycle",
    "awning-tricycle": "tricycle",
}


def merge_counts(
    counts: Dict[str, int], merge: Optional[Dict[str, str]] = None
) -> Dict[str, int]:
    merge = DEFAULT_MERGE if merge is None else merge
    out: Dict[str, int] = {}
    for name, n in counts.items():
        key = merge.get(name, name)
        out[key] = out.get(key, 0) + int(n)
    return out


def pick_frames(n: int, num: int = 3) -> List[int]:
    """Positions of the first, last, and evenly spaced frames in between."""
    if n <= 0:
        return []
    if num <= 1 or n == 1:
        return [0]
    return sorted({round(i * (n - 1) / (num - 1)) for i in range(num)})


def describe_counts(
    per_frame: Sequence[Dict[str, int]],
    timestamps: Sequence[float],
    num_times: int = 3,
    min_count: int = 1,
    max_classes: int = 5,
) -> str:
    """Render per-frame `{class: count}` dicts as one prompt line."""
    if not per_frame:
        return ""
    parts = []
    for i in pick_frames(len(per_frame), num_times):
        counts = {k: v for k, v in merge_counts(per_frame[i]).items() if v >= min_count}
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:max_classes]
        body = ", ".join(f"{k} {v}" for k, v in top) if top else "nothing detected"
        parts.append(f"{timestamps[i]:.1f}s {body}")
    return "Detector counts: " + "; ".join(parts) + "."


# Words in a question that map to the detector's class names.
QUESTION_CLASS_SYNONYMS = {
    "car": {"car", "cars", "vehicle", "vehicles", "automobile", "automobiles"},
    "van": {"van", "vans"},
    "truck": {"truck", "trucks"},
    "bus": {"bus", "buses"},
    "person": {"person", "people", "pedestrian", "pedestrians", "human", "humans"},
    "bicycle": {"bicycle", "bicycles", "bike", "bikes"},
    "motorcycle": {"motorcycle", "motorcycles", "motorbike", "motorbikes"},
    "tricycle": {"tricycle", "tricycles"},
}


def classes_in_question(question: str) -> set:
    """Detector classes named in a question (empty set = no constraint)."""
    tokens = set(re.findall(r"[a-z]+", question.lower()))
    return {
        cls for cls, words in QUESTION_CLASS_SYNONYMS.items() if tokens & words
    }


def describe_timeline(
    per_frame: Sequence[Dict[str, int]],
    timestamps: Sequence[float],
    classes: Optional[set] = None,
    max_classes: int = 4,
) -> str:
    """Compact object-appearance timeline, e.g.

        Object timeline: car first 1.5s, peak 5 at 3.0s, last 6.0s; bus first 2.0s.

    `classes` restricts the output to the objects a question names (query-aware);
    None keeps every class that appears. Directly supports "when does X first
    appear / how long is X visible" questions.
    """
    if not per_frame:
        return ""
    merged = [merge_counts(pf) for pf in per_frame]
    present_any = {k for pf in merged for k, v in pf.items() if v > 0}
    wanted = (classes or present_any) & present_any
    if not wanted:
        return ""
    lines = []
    for cls in sorted(wanted):
        pts = [(float(timestamps[i]), merged[i].get(cls, 0)) for i in range(len(merged))]
        seen = [(t, c) for t, c in pts if c > 0]
        first, last = seen[0][0], seen[-1][0]
        peak_t, peak_c = max(seen, key=lambda tc: tc[1])
        lines.append(
            f"{cls} first {first:.1f}s, peak {peak_c} at {peak_t:.1f}s, last {last:.1f}s"
        )
    return "Object timeline: " + "; ".join(lines[:max_classes]) + "."


def run_detector(
    model,
    frames: Sequence[np.ndarray],
    conf: float = 0.25,
    imgsz: int = 1280,
    class_names: Sequence[str] = VISDRONE_CLASSES,
) -> List[Dict[str, int]]:
    """Per-frame class counts from an ultralytics YOLO model.

    `imgsz` above the 640 px source width upsamples, which helps tiny objects.
    """
    results = model.predict(list(frames), conf=conf, imgsz=imgsz, verbose=False)
    out = []
    for res in results:
        cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else []
        counts: Dict[str, int] = {}
        for c in cls:
            name = class_names[c] if c < len(class_names) else str(c)
            counts[name] = counts.get(name, 0) + 1
        out.append(counts)
    return out
