"""Video memory: answered training questions about the same clip, as text.

1,012 of the 1,014 test videos also appear in train, with ~15 labelled
questions each. A test question often overlaps a training fact about its
clip (a count at 00:00, the camera viewpoint, when an object appears). This
cue retrieves the `k` most similar *other* labelled questions on the same
video and writes them as facts:

    Known facts about this video: Q: What is the camera viewpoint? A: top-down
    (90 degrees). | Q: ...

OFF by default (`cues: [motion]`). It only uses the released training labels,
but whether that counts as "using the video" under the challenge rules is for
the organizers to say. Leakage rules when it is on:

* never retrieve the question itself (same qa_id) or any record asking the
  same question on the same video (that would copy the label);
* at training time the memory comes from the fit split only, so dev questions
  never see their own (or duplicate) labels.
"""

import re
from collections import defaultdict
from typing import Dict, List, Sequence

from src.data import ReVAExample, question_key


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class VideoMemory:
    def __init__(
        self, labelled: Sequence[ReVAExample], k: int = 5, max_chars: int = 900
    ):
        self.k = k
        self.max_chars = max_chars
        self.by_video: Dict[str, List[ReVAExample]] = defaultdict(list)
        for ex in labelled:
            if ex.label >= 0:
                self.by_video[ex.video_path].append(ex)

    def retrieve(self, query: ReVAExample) -> List[ReVAExample]:
        qkey = question_key(query)
        qtok = _tokens(query.question + " " + " ".join(query.options))
        pool = [
            ex
            for ex in self.by_video.get(query.video_path, [])
            if ex.qa_id != query.qa_id and question_key(ex) != qkey
        ]

        def score(ex: ReVAExample) -> float:
            tok = _tokens(ex.question + " " + ex.answer_text)
            same_task = 0.1 if ex.task == query.task else 0.0
            return len(qtok & tok) / max(1, len(qtok | tok)) + same_task

        return sorted(pool, key=lambda ex: (-score(ex), ex.qa_id))[: self.k]

    def describe(self, query: ReVAExample) -> str:
        facts = [f"Q: {ex.question} A: {ex.answer_text}" for ex in self.retrieve(query)]
        if not facts:
            return ""
        text = "Known facts about this video: " + " | ".join(facts)
        return text[: self.max_chars]
