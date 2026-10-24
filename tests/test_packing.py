from src.config import ModelConfig
from src.data import ReVAExample, VideoGroupDataset
from src.modules.collate import PackedMCQCollator, _find_ends


def _ex(i, video):
    return ReVAExample(
        qa_id=f"q{i}",
        video_path=video,
        task="t",
        category="c",
        question=f"question {i}?",
        options=("a", "b", "c", "d"),
        label=i % 4,
    )


class _FakeTokenizer:
    padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        return [1]


class _FakeProcessor:
    """Minimal stand-in that renders the last user text and a gen marker."""

    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        text = messages[-1]["content"][-1]["text"]
        return f"<user>{text}" + ("<gen>" if add_generation_prompt else "")


class _FakeCueStore:
    def text(self, ex, skip=()):
        return f"CUE({ex.qa_id})"


def test_packed_text_keeps_cue_for_every_question():
    """Regression: later questions in a packed sequence must keep the cue.

    A packed run trains one question as Q1 and the rest as follow-up turns; the
    follow-ups used to be rendered with an empty cue, so 7 of every 8 questions
    trained without the motion cue.
    """
    group = [_ex(i, "v.mp4") for i in range(3)]
    collator = PackedMCQCollator(
        _FakeProcessor(), ModelConfig(), cue_store=_FakeCueStore()
    )
    text = collator.packed_text(group)
    for i in range(3):
        assert text.count(f"CUE(q{i})") == 1
        assert f"CUE(q{i})\nQuestion: question {i}?" in text


def test_groups_cover_every_question_once_per_epoch():
    exs = [_ex(i, f"v{i % 3}.mp4") for i in range(20)]
    ds = VideoGroupDataset(exs, max_per_group=4, shuffle_options=True, seed=1)
    for epoch in (1, 2):
        ds.regroup(epoch)
        seen = [ex.qa_id for g in ds.groups for ex in g]
        assert sorted(seen) == sorted(e.qa_id for e in exs)
        assert all(len(g) <= 4 and len({ex.video_path for ex in g}) == 1 for g in ds.groups)
    g = ds[0]
    assert all(ex.label == ex.perm.index(int(ex.qa_id[1:]) % 4) for ex in g)


def test_regroup_changes_chunks():
    exs = [_ex(i, "v.mp4") for i in range(12)]
    ds = VideoGroupDataset(exs, max_per_group=4, seed=0)
    ds.regroup(1)
    a = [[e.qa_id for e in g] for g in ds.groups]
    ds.regroup(2)
    b = [[e.qa_id for e in g] for g in ds.groups]
    assert a != b


def test_find_ends():
    assert _find_ends([1, 2, 3, 9, 1, 2, 3], [1, 2, 3]) == [2, 6]
    assert _find_ends([1, 2], [1, 2, 3]) == []


# --- branched scoring (shared prefix, leak-free) ------------------------------

def test_shared_prefix_len_leaves_a_suffix_token():
    from src.modules.branched import shared_prefix_len

    assert shared_prefix_len([[1, 2, 3, 4], [1, 2, 5]]) == 2
    assert shared_prefix_len([[1, 2, 3], [1, 2, 3]]) == 2
    assert shared_prefix_len([[7, 8, 9]]) == 2


def test_right_pad_keeps_rows_left_aligned():
    from src.modules.branched import right_pad

    ids, mask = right_pad([[1, 2, 3], [4]], pad=0)
    assert ids.tolist() == [[1, 2, 3], [4, 0, 0]]
    assert mask.tolist() == [[1, 1, 1], [1, 0, 0]]


def test_video_groups_split_by_task():
    from dataclasses import replace

    tasks = ["Temporal Grounding", "Change Detection"] * 2
    exs = [replace(_ex(i, "v.mp4"), task=t) for i, t in enumerate(tasks)]
    ds = VideoGroupDataset(exs, max_per_group=8, split_tasks=("Temporal Grounding",))
    tasks = sorted(sorted({ex.task for ex in g}) for g in ds.groups)
    assert tasks == [["Change Detection"], ["Temporal Grounding"]]


def test_unlikelihood_matches_definition():
    import torch

    from src.modules.loss import unlikelihood

    logits = torch.tensor([[2.0, 0.0, -1.0, 0.5], [0.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([0, 3])
    p = logits.softmax(-1)[torch.arange(2), labels]
    assert torch.allclose(unlikelihood(logits, labels), -(1 - p).log().mean())


def test_reversal_sensitive_rules():
    from dataclasses import replace

    from src.modules.counterfactual import reversal_sensitive

    def q(task, question, gold):
        return replace(_ex(0, "v.mp4"), task=task, question=question,
                       options=(gold, "x", "y", "z"), label=0)

    assert reversal_sensitive(q("Temporal Grounding", "When does the red car first appear?", "At 3.5 seconds"))
    assert not reversal_sensitive(q("Temporal Grounding", "How long is the bridge visible?", "6.0 seconds"))
    assert not reversal_sensitive(q("Temporal Grounding", "When is the pool visible?", "For most of the video"))
    assert reversal_sensitive(q("Change Detection", "How does the car move?", "From left to right"))
    assert not reversal_sensitive(q("Change Detection", "How does the building change?", "It remains the same"))
    assert reversal_sensitive(q("Perspective and Viewpoint", "How does the camera move?", "Moving forward"))
    assert not reversal_sensitive(q("Perspective and Viewpoint", "What is the viewpoint angle?", "60° downward view"))
    assert not reversal_sensitive(q("Trend and Pattern", "Is traffic periodic?", "Vehicles pass every 2 s"))
