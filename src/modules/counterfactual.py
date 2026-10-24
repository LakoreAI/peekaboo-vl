"""Which questions change their answer when the clip is played backwards.

The counterfactual temporal loss (`TrainingConfig.cf_tasks`) pushes the model
away from the original answer on the reversed clip. That is only right when
reversal changes the answer, so each task gets a conservative text rule:

- Temporal Grounding: time points / intervals ("at what time", "first
  appear", "disappear"), not durations ("how long", "for most of the video").
- Change Detection: changes and directions, not "remains the same".
- Perspective and Viewpoint: camera *motion* (forward, panning, tilting),
  not viewing angle or a stationary camera.

Trend and Pattern is mostly periodicity / frequency (time-symmetric) and is
never eligible.
"""

import re

from src.data import ReVAExample

_TG_POINT = re.compile(
    r"\b(when|what time|at what|first|appear|appears|disappear|disappears|"
    r"begin|begins|start|starts|end|ends|enter|enters|exit|exits|leave|leaves)\b",
    re.I,
)
_TG_DURATION = re.compile(r"\b(how long|duration|how many seconds)\b", re.I)
_SYMMETRIC_ANSWER = re.compile(
    r"\b(most of the video|entire|whole video|throughout|all the time|"
    r"remain|remains|remaining|same|unchanged|no change|does not|doesn't|"
    r"stationary|static|hover|hovering|constant|consistent|steady)\b",
    re.I,
)
_CAMERA_MOTION = re.compile(r"\b(motion|moving|move|moves|movement)\b", re.I)


def reversal_sensitive(ex: ReVAExample) -> bool:
    """True if reversing the clip should change the gold answer."""
    gold = ex.options[ex.label]
    if _SYMMETRIC_ANSWER.search(gold):
        return False
    if ex.task == "Temporal Grounding":
        return bool(_TG_POINT.search(ex.question)) and not _TG_DURATION.search(
            ex.question
        )
    if ex.task == "Change Detection":
        return True
    if ex.task == "Perspective and Viewpoint":
        return bool(_CAMERA_MOTION.search(ex.question))
    return False
