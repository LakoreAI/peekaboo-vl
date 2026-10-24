import pytest

from src.modules.cues import CueStore
from src.modules.cues.counts import (
    classes_in_question,
    describe_counts,
    describe_timeline,
    merge_counts,
    pick_frames,
)
from src.modules.cues.memory import VideoMemory
from src.prompting import build_messages, question_text
from src.utils.io_utils import write_jsonl
from tests.conftest import make_example


def test_pick_frames():
    assert pick_frames(64, 3) == [0, 32, 63]
    assert pick_frames(1, 3) == [0]
    assert pick_frames(0, 3) == []


def test_describe_counts_merges_synonyms():
    per_frame = [{"pedestrian": 2, "people": 3, "car": 10}, {"car": 4}, {}]
    text = describe_counts(per_frame, [0.0, 7.5, 15.0])
    assert (
        text
        == "Detector counts: 0.0s car 10, person 5; 7.5s car 4; 15.0s nothing detected."
    )
    assert merge_counts({"motor": 2, "awning-tricycle": 1, "tricycle": 1}) == {
        "motorcycle": 2,
        "tricycle": 2,
    }


def test_memory_never_copies_the_label():
    v = "V/clip.mp4"
    q = make_example(0, video=v, question="How many cars at 00:00?")
    same_q = make_example(
        1, video=v, question="how many cars at 00:00?  "
    )  # duplicate wording
    other = make_example(2, video=v, question="What is the camera viewpoint?")
    elsewhere = make_example(3, video="V/other.mp4", question="How many cars at 00:00?")
    mem = VideoMemory([q, same_q, other, elsewhere], k=5)
    got = mem.retrieve(q)
    assert [ex.qa_id for ex in got] == ["train_000002"]
    assert mem.describe(q).startswith(
        "Known facts about this video: Q: What is the camera viewpoint?"
    )


def test_cue_store(tmp_path):
    ex = make_example()
    write_jsonl(
        [{"video_path": ex.video_path, "text": "Camera motion: 0.0-15.0s static."}],
        tmp_path / "motion.jsonl",
    )
    store = CueStore(["motion"], cues_root=tmp_path)
    assert store.text(ex) == "Camera motion: 0.0-15.0s static."
    with pytest.raises(KeyError):
        store.text(make_example(video="Other/x.mp4"))
    assert (
        CueStore(["motion"], cues_root=tmp_path, strict=False).text(
            make_example(video="Other/x.mp4")
        )
        == ""
    )
    with pytest.raises(FileNotFoundError):
        CueStore(["counts"], cues_root=tmp_path)
    with pytest.raises(ValueError):
        CueStore(["memory"])
    assert CueStore().text(ex) == ""


def test_prompt_layout(example):
    text = question_text(example, cue_text="Camera motion: static.")
    lines = text.splitlines()
    assert lines[0] == "Camera motion: static."
    assert lines[1] == f"Question: {example.question}"
    assert lines[3:7] == [f"{k}. {o}" for k, o in zip("ABCD", example.options)]
    assert lines[-1].startswith("Answer with the option's letter")

    msgs = build_messages(example)
    assert msgs[0]["content"][0] == {"type": "video"}
    assert build_messages(example, use_video=False)[0]["content"][0]["type"] == "text"
    assert build_messages(example, system_prompt="S")[0]["role"] == "system"


def test_cue_store_shuffle_is_a_derangement(tmp_path):
    rows = [{"video_path": f"V/{i}.mp4", "text": f"motion {i}"} for i in range(6)]
    write_jsonl(rows, tmp_path / "motion.jsonl")
    store = CueStore(["motion"], cues_root=tmp_path).shuffle(seed=0)
    table = store.tables["motion"]
    assert sorted(table.values()) == sorted(r["text"] for r in rows)
    assert all(table[f"V/{i}.mp4"] != f"motion {i}" for i in range(6))


def test_classes_in_question():
    assert classes_in_question("How many cars are parked?") == {"car"}
    assert classes_in_question("When does the bus first appear?") == {"bus"}
    assert classes_in_question("Is there a body of water?") == set()
    assert {"car", "truck"} <= classes_in_question("Count cars and trucks")


def test_describe_timeline_is_query_aware():
    per_frame = [{"car": 0}, {"car": 3}, {"car": 5}, {"car": 5}]
    ts = [0.0, 1.0, 2.0, 3.0]
    full = describe_timeline(per_frame, ts)
    assert full == "Object timeline: car first 1.0s, peak 5 at 2.0s, last 3.0s."
    per_frame2 = [{"car": 3, "bus": 0}, {"car": 4, "bus": 2}]
    only_bus = describe_timeline(per_frame2, [0.0, 1.0], classes={"bus"})
    assert "bus" in only_bus and "car" not in only_bus
    assert describe_timeline(per_frame2, [0.0, 1.0], classes={"boat"}) == ""


def test_timeline_cue_is_task_gated(tmp_path):
    write_jsonl(
        [{"video_path": "VisDrone/a.mp4", "per_frame": [{"car": 0}, {"car": 4}],
          "timestamps": [0.0, 1.0]}],
        tmp_path / "counts.jsonl",
    )
    store = CueStore(["timeline"], cues_root=tmp_path)
    tg = make_example(
        0, video="VisDrone/a.mp4", task="Temporal Grounding",
        question="When does the car first appear?",
    )
    assert store.text(tg) == "Object timeline: car first 1.0s, peak 4 at 1.0s, last 1.0s."
    other = make_example(
        1, video="VisDrone/a.mp4", task="Object and Land Cover Recognition",
        question="How many cars are visible?",
    )
    assert store.text(other) == ""
