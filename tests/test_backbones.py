import re

import numpy as np
import pytest

from src.config import ModelConfig
from src.data import ReVAExample
from src.modules.backbones import detect_family, frame_prefix
from src.modules.collate import BranchedMCQCollator, MCQCollator
from src.modules.model import lora_target_regex
from src.prompting import build_messages


def _ex():
    return ReVAExample(
        qa_id="q1",
        video_path="v.mp4",
        task="Temporal Grounding",
        category="c",
        question="When?",
        options=("a", "b", "c", "d"),
        label=1,
    )


@pytest.mark.parametrize(
    "backbone,family",
    [
        ("Qwen/Qwen3.5-9B", "qwen"),
        ("nvidia/Cosmos-Reason2-8B", "qwen"),
        ("OpenGVLab/InternVL3_5-8B-HF", "internvl"),
        ("zai-org/GLM-4.1V-9B-Thinking", "glm4v"),
    ],
)
def test_detect_family(backbone, family):
    assert detect_family(backbone) == family
    assert ModelConfig(backbone=backbone).family_name == family


def test_explicit_family_is_validated():
    assert ModelConfig(backbone="my/finetune", family="internvl").family_name == "internvl"
    with pytest.raises(ValueError):
        ModelConfig(family="llava")


# Module names checked against the real checkpoints (meta-device load).
@pytest.mark.parametrize(
    "family,name,lm,projector",
    [
        ("internvl", "model.language_model.layers.0.self_attn.q_proj", True, True),
        ("internvl", "model.language_model.layers.35.mlp.gate_proj", True, True),
        ("internvl", "model.multi_modal_projector.linear_1", False, True),
        ("internvl", "model.vision_tower.encoder.layer.0.attention.q_proj", False, False),
        ("internvl", "model.vision_tower.encoder.layer.0.mlp.fc1", False, False),
        ("glm4v", "model.language_model.layers.3.mlp.gate_up_proj", True, True),
        ("glm4v", "model.language_model.layers.3.mlp.down_proj", True, True),
        ("glm4v", "model.visual.merger.proj", False, True),
        ("glm4v", "model.visual.merger.gate_proj", False, True),
        ("glm4v", "model.visual.blocks.0.mlp.gate_proj", False, False),
        ("glm4v", "model.visual.blocks.0.attn.qkv", False, False),
        ("qwen", "model.visual.merger.linear_fc1", False, True),
        ("internvl", "lm_head", False, False),
    ],
)
def test_lora_targets_per_family(family, name, lm, projector):
    assert bool(re.fullmatch(lora_target_regex(False, family=family), name)) == lm
    assert bool(re.fullmatch(lora_target_regex(True, family=family), name)) == projector


def test_vision_blocks_lora_is_qwen_only():
    with pytest.raises(ValueError):
        lora_target_regex(True, vision_blocks=2, family="internvl")


def test_frame_times_lay_out_timestamped_images():
    content = build_messages(_ex(), frame_times=[0.0, 1.25])[0]["content"]
    assert [c["type"] for c in content] == ["text", "image", "text", "image", "text"]
    assert content[0]["text"] == frame_prefix(0, 0.0) == "Frame1 (0.0s): "
    assert content[2]["text"] == "Frame2 (1.2s): "
    assert content[-1]["text"].startswith("Question: When?")
    # Default: one video entry (Qwen / GLM write their own timestamps).
    assert build_messages(_ex())[0]["content"][0] == {"type": "video"}


class _Tok:
    padding_side = "right"
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1]


class _Processor:
    """Records what the collator passes; renders entries as tags."""

    def __init__(self):
        self.tokenizer = _Tok()
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        tags = {"video": "<video>", "image": "<image>"}
        parts = [
            c["text"] if c["type"] == "text" else tags[c["type"]]
            for c in messages[-1]["content"]
        ]
        return "<user>" + "".join(parts) + ("<gen>" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {}


def _collate(cfg, monkeypatch):
    from src.modules import collate

    frames = np.zeros((4, 8, 8, 3), np.uint8)

    class FS:
        pass

    fs = FS()
    fs.frames, fs.fps, fs.frame_indices, fs.total_frames = frames, 10.0, [0, 5, 10, 15], 20
    fs.timestamps = [0.0, 0.5, 1.0, 1.5]
    monkeypatch.setattr(collate, "load_frames", lambda *a, **k: fs)
    proc = _Processor()
    MCQCollator(proc, cfg)([_ex()])
    return proc.calls[0]


def test_internvl_collator_passes_frames_as_timestamped_images(monkeypatch):
    cfg = ModelConfig(backbone="OpenGVLab/InternVL3_5-8B-HF", num_frames=4)
    (text,), kwargs = _collate(cfg, monkeypatch)
    assert "Frame1 (0.0s): <image>Frame2 (0.5s): <image>" in text
    assert "<video>" not in text
    assert len(kwargs["images"][0]) == 4
    assert kwargs["crop_to_patches"] is False
    assert "videos" not in kwargs and "size" not in kwargs


def test_glm_collator_uses_video_and_answer_prefix(monkeypatch):
    cfg = ModelConfig(
        backbone="zai-org/GLM-4.1V-9B-Thinking",
        num_frames=4,
        answer_prefix="<think></think><answer>",
    )
    (text,), kwargs = _collate(cfg, monkeypatch)
    assert text.endswith("<gen><think></think><answer>")
    assert "<video>" in text and "Frame1" not in text
    assert kwargs["video_metadata"][0]["fps"] == 10.0
    assert kwargs["size"]["longest_edge"] == cfg.max_pixels_per_frame * 4
    assert "cap_pixels_per_frame" not in kwargs  # Qwen3-VL processor only


def test_qwen_collator_unchanged(monkeypatch):
    (text,), kwargs = _collate(ModelConfig(num_frames=4), monkeypatch)
    assert text.startswith("<user><video>Question: When?")
    assert kwargs["cap_pixels_per_frame"] is True and kwargs["do_sample_frames"] is False


def test_branching_rejects_other_families():
    cfg = ModelConfig(backbone="OpenGVLab/InternVL3_5-8B-HF", num_frames=4)
    with pytest.raises(ValueError, match="Qwen3.5 only"):
        BranchedMCQCollator(_Processor(), cfg)([[_ex()]])
