import re

import pytest
import torch

from src.config import ModelConfig
from src.modules.loss import ClassificationLoss
from src.modules.model import letter_token_ids, lora_target_regex, select_letter_logits
from src.utils.io_utils import load_trainable, trainable_state_dict


class FakeTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, text, add_special_tokens=False):
        return self.vocab[text]


def test_letter_token_ids():
    assert letter_token_ids(
        FakeTokenizer({"A": [32], "B": [33], "C": [34], "D": [35]})
    ) == [32, 33, 34, 35]
    with pytest.raises(ValueError, match="single token"):
        letter_token_ids(FakeTokenizer({"A": [1, 2], "B": [3], "C": [4], "D": [5]}))
    with pytest.raises(ValueError, match="share"):
        letter_token_ids(FakeTokenizer({"A": [1], "B": [1], "C": [4], "D": [5]}))


def test_select_letter_logits():
    last = torch.arange(20, dtype=torch.bfloat16).reshape(2, 10)
    out = select_letter_logits(last, [3, 1, 4, 9])
    assert out.dtype == torch.float32
    assert out.tolist() == [[3, 1, 4, 9], [13, 11, 14, 19]]
    loss = ClassificationLoss()(out, torch.tensor([3, 3]))
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "name,lm,merger",
    [
        ("model.language_model.layers.0.self_attn.q_proj", True, True),
        ("model.language_model.layers.3.linear_attn.in_proj_qkv", True, True),
        ("model.language_model.layers.3.linear_attn.out_proj", True, True),
        ("model.language_model.layers.31.mlp.down_proj", True, True),
        ("base_model.model.model.language_model.layers.2.mlp.up_proj", True, True),
        ("model.visual.merger.linear_fc1", False, True),
        ("model.visual.deepstack_merger_list.1.linear_fc2", False, True),
        ("model.visual.blocks.0.attn.qkv", False, False),
        ("model.visual.blocks.0.mlp.linear_fc1", False, False),
        ("lm_head", False, False),
    ],
)
def test_lora_targets_language_model_only(name, lm, merger):
    assert bool(re.fullmatch(lora_target_regex(False), name)) == lm
    assert bool(re.fullmatch(lora_target_regex(True), name)) == merger


def test_model_config_validation():
    assert ModelConfig(cues=["motion"]).cues == ("motion",)
    with pytest.raises(ValueError):
        ModelConfig(num_frames=31)
    with pytest.raises(ValueError):
        ModelConfig(cues=["telepathy"])
    # 4-bit is allowed without LoRA for quantized zero-shot inference.
    assert ModelConfig(load_in_4bit=True).lora is None
    assert ModelConfig(use_video=False, num_frames=0).use_video is False
    assert ModelConfig().overlay_time_tasks == ()
    assert ModelConfig(overlay_time_tasks=["Temporal Grounding"]).overlay_time_tasks == (
        "Temporal Grounding",
    )


def test_trainable_state_roundtrip():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    model[0].requires_grad_(False)
    state = trainable_state_dict(model)
    assert set(state) == {"1.weight", "1.bias"}
    fresh = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    load_trainable(fresh, state)
    assert torch.equal(fresh[1].weight, model[1].weight)
    with pytest.raises(KeyError):
        load_trainable(fresh, {"9.weight": torch.zeros(1)})
