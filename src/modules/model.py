"""Video MCQ scorer: a chat VLM read out at the answer-letter position.

    batch["inputs"] (left-padded prompts + video tensors)
      -> AutoModelForImageTextToText(..., logits_to_keep=1)
      -> last-position vocab logits (B, V)            == `feature`
      -> gather token ids of "A","B","C","D" -> (B, 4) == `logits`

`forward` keeps the template's `(logits, feature)` contract. Training is
cross-entropy over the 4 letter logits (`src.modules.loss`), i.e. SFT on
the single answer token with the softmax restricted to the valid letters.

Only the adapter is trainable (LoRA on the language model, optionally on the
vision->LLM merger/projector; per-family names in `src.modules.backbones`); the vision encoder stays frozen. Checkpoints store just
the trainable tensors (`src.utils.io_utils.trainable_state_dict`).
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.labels import LETTERS
from src.modules.backbones import LM_LINEARS, LM_PATTERN, PROJECTOR_PATTERNS  # noqa: F401

# Kept for older imports; the per-family patterns live in src.modules.backbones.
MERGER_PATTERN = PROJECTOR_PATTERNS["qwen"]


def vision_block_pattern(last_n: int, depth: int) -> str:
    """Linears of the last `last_n` vision-encoder blocks."""
    ids = "|".join(str(i) for i in range(max(depth - last_n, 0), depth))
    return rf".*visual\.blocks\.(?:{ids})\.(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12])"


def lora_target_regex(
    include_merger: bool,
    vision_blocks: int = 0,
    vision_depth: int = 27,
    family: str = "qwen",
) -> str:
    parts = [LM_PATTERN]
    if include_merger:
        parts.append(PROJECTOR_PATTERNS[family])
    if vision_blocks > 0:
        if family != "qwen":
            raise ValueError("lora_vision_blocks is implemented for Qwen only")
        parts.append(vision_block_pattern(vision_blocks, vision_depth))
    return "|".join(f"(?:{p})" for p in parts)


def letter_token_ids(tokenizer, letters: Sequence[str] = LETTERS) -> List[int]:
    """Single-token ids for the answer letters (fails loudly otherwise)."""
    ids = []
    for letter in letters:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"letter {letter!r} is not a single token: {enc}")
        ids.append(enc[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"answer letters share token ids: {ids}")
    return ids


def select_letter_logits(
    last_logits: torch.Tensor, letter_ids: Sequence[int]
) -> torch.Tensor:
    """(B, V) vocab logits -> (B, 4) float32 letter logits."""
    index = torch.as_tensor(list(letter_ids), device=last_logits.device)
    return last_logits.index_select(-1, index).float()


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


class VideoMCQModel(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        backbone: nn.Module,
        processor,
        letter_ids: Sequence[int],
    ):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.processor = processor
        self.letter_ids = list(letter_ids)

    @classmethod
    def from_config(
        cls, cfg: ModelConfig, device: Optional[torch.device] = None
    ) -> "VideoMCQModel":
        """Load the pretrained VLM + processor and attach the LoRA adapter."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(cfg.processor_name)
        kwargs = {"dtype": _dtype(cfg.torch_dtype)}
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_dtype(cfg.torch_dtype),
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = (
                {"": device.index or 0}
                if device is not None and device.type == "cuda"
                else "auto"
            )
        backbone = AutoModelForImageTextToText.from_pretrained(cfg.backbone, **kwargs)
        if device is not None and not cfg.load_in_4bit:
            backbone = backbone.to(device)

        for p in backbone.parameters():
            p.requires_grad_(False)
        if cfg.lora:
            backbone = attach_lora(backbone, cfg)
        if cfg.gradient_checkpointing and cfg.lora:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(backbone, "enable_input_require_grads"):
                backbone.enable_input_require_grads()
            backbone.config.use_cache = False

        tok = getattr(processor, "tokenizer", processor)
        return cls(cfg, backbone, processor, letter_token_ids(tok))

    def forward(self, inputs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """inputs: the processor output (left-padded). Returns
        (letter logits (B, 4) float32, last-position vocab logits (B, V))."""
        out = self.backbone(**inputs, logits_to_keep=1, use_cache=False)
        last = out.logits[:, -1, :]
        return select_letter_logits(last, self.letter_ids), last


    def forward_positions(
        self, inputs: dict, positions: torch.Tensor
    ) -> torch.Tensor:
        """Letter logits (P, 4) at several positions of ONE sequence (packed
        questions). Only those positions go through the LM head."""
        out = self.backbone(**inputs, logits_to_keep=positions, use_cache=False)
        return select_letter_logits(out.logits[0], self.letter_ids)

    def forward_branched_facts(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """`forward_branched` for a group that also carries a fact row (last
        row, `fact_start`): returns (letter logits (k, 4), mean token CE of the
        fact text). The fact row shares the video prefix but is its own branch,
        so the questions never see it."""
        from src.modules.branched import _parts, branched_forward

        inputs = batch["inputs"]
        ids = inputs["input_ids"]
        k, P = ids.shape[0] - 1, int(batch["prefix_len"])
        hq, hf = branched_forward(
            self.backbone,
            inputs,
            P,
            [list(range(k)), [k]],
            grad_ckpt=self.cfg.gradient_checkpointing and self.training,
        )
        head = _parts(self.backbone)[2]
        last = batch["last_index"].to(ids.device)
        letters = select_letter_logits(
            head(hq[torch.arange(k, device=ids.device), last - P]), self.letter_ids
        )
        # Fact tokens ids[k, s:end] are predicted from positions s-1 .. end-2.
        start, end = int(batch["fact_start"]), int(inputs["attention_mask"][k].sum())
        hidden = hf[0, start - 1 - P : end - 1 - P]
        targets = ids[k, start:end]
        total = hidden.new_zeros((), dtype=torch.float32)
        for i in range(0, hidden.shape[0], 256):  # chunked: vocab is ~250k
            logits = head(hidden[i : i + 256]).float()
            total = total + F.cross_entropy(logits, targets[i : i + 256], reduction="sum")
        return letters, total / max(end - start, 1)

    def forward_branched(self, batch: dict) -> torch.Tensor:
        """Letter logits (k, 4) for k questions sharing one video prefix
        (`BranchedMCQCollator` batch); equal to scoring each prompt alone."""
        from src.modules.branched import _parts, branched_hidden

        hidden = branched_hidden(
            self.backbone,
            batch["inputs"],
            int(batch["prefix_len"]),
            batch["last_index"].to(batch["inputs"]["input_ids"].device),
            grad_ckpt=self.cfg.gradient_checkpointing and self.training,
        )
        head = _parts(self.backbone)[2]
        return select_letter_logits(head(hidden), self.letter_ids)


def attach_lora(backbone: nn.Module, cfg: ModelConfig) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    if cfg.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        backbone = prepare_model_for_kbit_training(
            backbone, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    lora = dict(cfg.lora)
    config = LoraConfig(
        r=lora.get("r", 64),
        lora_alpha=lora.get("alpha", 2 * lora.get("r", 64)),
        lora_dropout=lora.get("dropout", 0.05),
        target_modules=lora.get("target_modules")
        or lora_target_regex(
            cfg.lora_merger, cfg.lora_vision_blocks, cfg.vision_depth, cfg.family_name
        ),
        bias="none",
    )
    return get_peft_model(backbone, config)
