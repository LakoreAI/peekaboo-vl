"""Per-family differences between the supported VLM backbones.

Everything else in the pipeline (letter read-out, cues, PriDe, metrics) is
backbone-agnostic. Three things are not:

    family    video input                           timestamps            LoRA projector
    qwen      videos= + video_metadata (2-frame      "<t seconds>" per     visual.merger
              temporal patches, budget = all frames) temporal patch
    glm4v     same as qwen (Qwen2-VL-style           int seconds after     visual.merger
              processor, 14 px patches)              each frame pair       (proj/gate/up/down)
    internvl  frames as images, one 448x448 tile     none: we write        multi_modal_projector
              each (256 tokens), "Frame1 (0.3s): "   "Frame{i} ({t}s): "
              before each                            ourselves

Shared-prefix branching (`src.modules.branched`) drives Qwen3.5's decoder
layers directly, so it stays Qwen3.5-only.
"""

from typing import Optional

FAMILIES = ("qwen", "glm4v", "internvl")

# Linear layers of the language model. Qwen3.5 / Qwen3-VL: full attention
# (q/k/v/o), Gated DeltaNet (in_proj_qkv, in_proj_z, out_proj), MLP.
# InternVL3.5 (Qwen3 LM) uses the attention + MLP subset; GLM-4.1V fuses the
# MLP's gate and up projections into gate_up_proj.
LM_LINEARS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "gate_up_proj",
    "down_proj",
)
LM_PATTERN = (
    r".*language_model\.layers\.\d+\.(?:self_attn|linear_attn|mlp)\.(?:{})".format(
        "|".join(LM_LINEARS)
    )
)
# Vision -> LLM projector per family. Qwen3-VL adds DeepStack mergers.
PROJECTOR_PATTERNS = {
    "qwen": r".*visual\.(?:merger|deepstack_merger_list\.\d+)\.linear_fc[12]",
    "glm4v": r".*visual\.merger\.(?:proj|gate_proj|up_proj|down_proj)",
    "internvl": r".*multi_modal_projector\.linear_[12]",
}


def detect_family(backbone: str, family: Optional[str] = None) -> str:
    """Family from an explicit setting, else from the checkpoint name."""
    if family:
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")
        return family
    name = backbone.lower()
    if "internvl" in name:
        return "internvl"
    if "glm-4" in name or "glm4" in name:
        return "glm4v"
    # Qwen3.5 / Qwen3-VL and their fine-tunes (e.g. Cosmos-Reason2).
    return "qwen"


def frame_prefix(index: int, seconds: float) -> str:
    """Text placed before frame `index` (0-based) for families without
    built-in timestamps (InternVL's own format is "Frame1: <image>")."""
    return f"Frame{index + 1} ({seconds:.1f}s): "
