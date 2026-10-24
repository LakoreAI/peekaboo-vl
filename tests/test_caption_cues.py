"""Caption cue: CueStore lookup, cue skipping, train-only cue dropout."""

from src.modules.cues import CueStore
from src.utils.io_utils import write_jsonl
from tests.conftest import make_example


def test_caption_store(tmp_path):
    text = "Video description: A road. Cars pass."
    ex = make_example()
    write_jsonl([{"video_path": ex.video_path, "text": text}], tmp_path / "caption.jsonl")
    assert CueStore(["caption"], cues_root=tmp_path).text(ex) == text


def test_store_skips_cues(tmp_path):
    ex = make_example()
    write_jsonl([{"video_path": ex.video_path, "text": "CAP"}], tmp_path / "caption.jsonl")
    write_jsonl([{"video_path": ex.video_path, "text": "MOT"}], tmp_path / "motion.jsonl")
    store = CueStore(["motion", "caption"], cues_root=tmp_path)
    assert store.text(ex) == "MOT\nCAP"
    assert store.text(ex, skip=["caption"]) == "MOT"


def test_collator_cue_dropout_is_train_only(tmp_path):
    from src.config import ModelConfig
    from src.modules.collate import MCQCollator
    from tests.test_packing import _FakeProcessor

    ex = make_example()
    write_jsonl([{"video_path": ex.video_path, "text": "CAP"}], tmp_path / "caption.jsonl")
    write_jsonl([{"video_path": ex.video_path, "text": "MOT"}], tmp_path / "motion.jsonl")
    store = CueStore(["motion", "caption"], cues_root=tmp_path)
    cfg = ModelConfig(use_video=False, cues=("motion", "caption"))
    collator = MCQCollator(_FakeProcessor(), cfg, store)
    assert collator._cue_text(ex) == "MOT\nCAP"  # default: no dropout (eval)
    collator.cue_dropout = {"caption": 1.0}
    assert collator._cue_text(ex) == "MOT"
    assert "CAP" not in collator.prompt(ex) and "MOT" in collator.prompt(ex)
    collator.cue_dropout = {"caption": 0.0}
    assert collator._cue_text(ex) == "MOT\nCAP"
