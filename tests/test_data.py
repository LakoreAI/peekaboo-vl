import random

import pytest

from src.data import (
    ReVADataset,
    cyclic_perms,
    dedupe,
    example_from_record,
    load_split,
    make_dev_split,
    permute,
    question_key,
    random_perm,
    save_split,
    source_sequence,
    to_record,
)
from tests.conftest import make_example


def test_record_roundtrip(official_record, tmp_path):
    ex = example_from_record(official_record)
    assert (
        ex.label == 1 and ex.answer_letter == "B" and ex.answer_text == "At 4.2 seconds"
    )
    assert to_record(ex) == official_record
    save_split([ex], tmp_path / "s.json")
    assert load_split(tmp_path / "s.json") == [ex]


def test_unlabelled_test_record(official_record):
    ex = example_from_record({**official_record, "correct_answer": ""})
    assert ex.label == -1 and ex.answer_letter is None


def test_bad_records_raise(official_record):
    with pytest.raises(ValueError):
        example_from_record({**official_record, "correct_answer": "E"})
    opts = dict(official_record["options"])
    del opts["D"]
    with pytest.raises(ValueError):
        example_from_record({**official_record, "options": opts})


def test_duplicate_ids_rejected(official_record, tmp_path):
    ex = example_from_record(official_record)
    save_split([ex, ex], tmp_path / "dup.json")
    with pytest.raises(ValueError, match="duplicate"):
        load_split(tmp_path / "dup.json")


def test_permute_keeps_answer_and_tracks_origin(example):
    for perm in cyclic_perms() + [random_perm(random.Random(s)) for s in range(5)]:
        shown = permute(example, perm)
        assert shown.answer_text == example.answer_text
        # display position i shows original option perm[i]
        assert all(
            shown.options[i] == example.options[p] for i, p in enumerate(shown.perm)
        )
    twice = permute(permute(example, (1, 2, 3, 0)), (3, 0, 1, 2))
    assert twice.options == example.options and twice.perm == (0, 1, 2, 3)


def test_cyclic_perms_cover_every_letter():
    perms = cyclic_perms()
    for option in range(4):
        assert sorted(p.index(option) for p in perms) == [0, 1, 2, 3]


def test_dataset_shuffles_only_when_asked(example):
    fixed = ReVADataset([example] * 20)
    assert all(fixed[i].options == example.options for i in range(20))
    shuffled = ReVADataset([example] * 50, shuffle_options=True, seed=0)
    seen = {shuffled[i].label for i in range(50)}
    assert len(seen) > 1
    assert all(shuffled[i].answer_text == example.answer_text for i in range(50))


def test_option_shuffle_is_deterministic_per_epoch(example):
    """The permutation must depend only on (seed, epoch, index), not on the
    order examples are drawn or on DataLoader worker scheduling."""
    a = ReVADataset([example] * 8, shuffle_options=True, seed=7)
    b = ReVADataset([example] * 8, shuffle_options=True, seed=7)
    a.set_epoch(2)
    b.set_epoch(2)
    assert [a[i].options for i in range(8)] == [b[i].options for i in range(8)]
    order = [5, 0, 3, 1, 7]
    assert [a[i].options for i in order] == [b[i].options for i in order]
    # A new epoch re-draws the permutations.
    a.set_epoch(3)
    assert [a[i].options for i in range(8)] != [b[i].options for i in range(8)]


def test_source_sequence():
    assert (
        source_sequence("VisDrone/uav0000009_03358_v_01.mp4")
        == "VisDrone/uav0000009_03358_v"
    )
    assert source_sequence("UAVDT/S0302.mp4") == "UAVDT/S0302"


def test_dev_split_is_leak_free():
    vids = [f"V/clip_{v:02d}.mp4" for v in range(6)]
    train = [
        make_example(
            i, video=vids[i % 6], task=("Temporal Grounding", "Change Detection")[i % 2]
        )
        for i in range(60)
    ]
    # val: 3 exact duplicates of train questions + 2 new questions
    dup = [
        make_example(100 + i, video=train[i].video_path, question=train[i].question)
        for i in range(3)
    ]
    new = [
        make_example(200 + i, video=vids[0], question=f"brand new {i}?")
        for i in range(2)
    ]
    fit, dev = make_dev_split(train, dup + new, holdout_per_task=4, seed=0)

    dev_ids = {ex.qa_id for ex in dev}
    assert {"train_000200", "train_000201"} <= dev_ids  # val non-duplicates kept
    assert not dev_ids & {
        "train_000100",
        "train_000101",
        "train_000102",
    }  # val dups dropped
    assert len(dev) == 2 + 4 * 2
    dev_keys = {question_key(ex) for ex in dev}
    assert not any(question_key(ex) in dev_keys for ex in fit)
    assert not dev_ids & {ex.qa_id for ex in fit}
    # seen-video regime: every dev video is still in fit
    assert {ex.video_path for ex in dev} <= {ex.video_path for ex in fit}


def test_dedupe():
    a = make_example(0, question="Same?")
    b = make_example(1, question="  same? ")
    assert dedupe([a, b]) == [a]
