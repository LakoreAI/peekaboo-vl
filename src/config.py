"""Model + input config for the video MCQ scorer.

This is the ARCHITECTURE-ONLY config: which VLM, how the video is sampled and
sized, which text cues the prompt carries, and the LoRA adapter shape. It is
saved into every checkpoint, because a model trained with 32 frames and motion
cues must be *served* with 32 frames and motion cues. Training-loop
hyperparameters (lr, epochs, data paths, TTA) live in
`src.pipelines.config.TrainingConfig`.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.labels import NUM_OPTIONS
from src.modules.backbones import detect_family
from src.modules.cues import SUPPORTED_CUES


@dataclass
class ModelConfig:
    """A chat VLM whose next-token logits over A/B/C/D are the class logits.

        frames + metadata ─┐
        cue text + Q + options ─> chat template ─> VLM ─> logits[-1, {A,B,C,D}]

    Supported families (`src.modules.backbones`): Qwen3.5 / Qwen3-VL (the
    main target; `<t seconds>` text timestamps per temporal patch), GLM-4.1V
    (same video interface) and InternVL3.5 (frames as images, timestamps
    written into the prompt as text).
    """

    backbone: str = "Qwen/Qwen3.5-9B"
    # Processor id/path; defaults to `backbone`.
    processor: Optional[str] = None
    # "qwen" | "glm4v" | "internvl"; None = detect from the backbone name.
    family: Optional[str] = None
    num_classes: int = NUM_OPTIONS
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "sdpa"

    # --- video input ---
    use_video: bool = True  # False = text-only probe for QA artifacts
    # Must be even: Qwen and GLM merge 2 consecutive frames per temporal patch.
    num_frames: int = 32
    # Pixel budget per frame (Qwen and GLM; InternVL tiles are 448x448). The
    # Qwen3-VL video processor's `size.longest_edge`
    # is a budget for ALL frames together (t * h * w), and its default
    # (786,432) would shrink 32 ReVA frames from 640x360 to ~208x117. The
    # collator sets longest_edge = max_pixels_per_frame * num_frames, so
    # 640x360 survives (snapped to 640x352). None = processor default.
    max_pixels_per_frame: Optional[int] = 640 * 360
    # Extra kwargs for the processor's video call (override anything above).
    video_kwargs: Optional[dict] = None
    # Draw a progress bar + the true time on each frame (VTimeCoT-style visual
    # time cue). Training-free; set it at training time to keep train/serve
    # matched (changing it only at inference is off-distribution).
    overlay_time: bool = False
    # Restrict the overlay to these tasks (empty = all). Gating it to the
    # temporal tasks avoids the bar occluding objects on the others.
    overlay_time_tasks: Tuple[str, ...] = ()

    # --- prompt ---
    cues: Tuple[str, ...] = ()  # subset of SUPPORTED_CUES, in prompt order
    memory_k: int = 5  # facts retrieved by the "memory" cue
    system_prompt: Optional[str] = None
    # Passed to apply_chat_template; Qwen3.5 is a hybrid thinker.
    chat_template_kwargs: dict = field(
        default_factory=lambda: {"enable_thinking": False}
    )
    # Text appended after the generation prompt, before the answer letter.
    # GLM-4.1V-Thinking answers as "<think>...</think><answer>X": with
    # "<think></think><answer>" the next token is the letter.
    answer_prefix: str = ""

    # --- adapter ---
    # PEFT LoRA on the language model, e.g. {"r": 64, "alpha": 128,
    # "dropout": 0.05}. None = frozen backbone (zero-shot scoring).
    lora: Optional[dict] = None
    # Also adapt the vision->LLM merger/projector.
    lora_merger: bool = True
    # Also adapt the last N vision-encoder blocks (attn qkv/proj, MLP);
    # 0 = frozen vision tower. Needs `vision_depth` = the tower's block count.
    lora_vision_blocks: int = 0
    vision_depth: int = 27
    # 4-bit NF4 base weights: QLoRA with a `lora` block, or plain quantized
    # inference without one (e.g. a 27B model zero-shot on a 46 GB GPU).
    load_in_4bit: bool = False
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        self.cues = tuple(self.cues or ())
        self.overlay_time_tasks = tuple(self.overlay_time_tasks or ())
        if not self.backbone:
            raise ValueError("backbone must be a model id or path")
        if self.num_classes != NUM_OPTIONS:
            raise ValueError(
                f"num_classes must be {NUM_OPTIONS}, got {self.num_classes}"
            )
        if self.use_video and (self.num_frames <= 0 or self.num_frames % 2):
            raise ValueError(
                f"num_frames must be a positive even number, got {self.num_frames}"
            )
        unknown = [c for c in self.cues if c not in SUPPORTED_CUES]
        if unknown:
            raise ValueError(f"unknown cues {unknown}; choose from {SUPPORTED_CUES}")
        detect_family(self.backbone, self.family)  # validates `family`
        if self.torch_dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"unsupported torch_dtype {self.torch_dtype!r}")

    @property
    def processor_name(self) -> str:
        return self.processor or self.backbone

    @property
    def family_name(self) -> str:
        return detect_family(self.backbone, self.family)
