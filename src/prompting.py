"""Prompt construction for 4-option video MCQ.

The user turn is `[video] + text`; the text is

    <cue lines, if any>
    Question: <question>
    Options:
    A. <option>
    B. <option>
    C. <option>
    D. <option>
    Answer with the option's letter from the given choices directly.

which is the MCQ instruction Qwen-VL models are evaluated with. The answer
is read from the next-token logits of A/B/C/D right after the assistant turn
opens (thinking disabled), so no text is generated.
"""

from typing import List, Optional, Sequence

from src.data import ReVAExample
from src.labels import OPTION_LETTERS
from src.modules.backbones import frame_prefix

DEFAULT_INSTRUCTION = "Answer with the option's letter from the given choices directly."


def question_text(
    ex: ReVAExample, cue_text: str = "", instruction: str = DEFAULT_INSTRUCTION
) -> str:
    lines = []
    if cue_text:
        lines.append(cue_text)
    lines.append(f"Question: {ex.question.strip()}")
    lines.append("Options:")
    lines.extend(
        f"{letter}. {opt.strip()}" for letter, opt in zip(OPTION_LETTERS, ex.options)
    )
    lines.append(instruction)
    return "\n".join(lines)


def build_messages(
    ex: ReVAExample,
    cue_text: str = "",
    use_video: bool = True,
    system_prompt: Optional[str] = None,
    instruction: str = DEFAULT_INSTRUCTION,
    frame_times: Optional[Sequence[float]] = None,
) -> List[dict]:
    """Chat messages in the transformers multimodal format.

    The video itself is passed to the processor separately (pre-sampled
    frames + metadata); the `{"type": "video"}` entry only places its tokens.
    With `frame_times` (families without built-in timestamps, e.g. InternVL)
    the clip is laid out as one image per frame instead, each preceded by
    "Frame{i} ({t}s): " with its true time.
    """
    content = video_content(frame_times) if use_video else []
    content.append({"type": "text", "text": question_text(ex, cue_text, instruction)})
    messages = []
    if system_prompt:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )
    messages.append({"role": "user", "content": content})
    return messages


def video_content(frame_times: Optional[Sequence[float]] = None) -> List[dict]:
    """Content entries that place the video: one `video` entry, or per frame
    a timestamp text + an `image` entry."""
    if frame_times is None:
        return [{"type": "video"}]
    out = []
    for i, t in enumerate(frame_times):
        out.append({"type": "text", "text": frame_prefix(i, float(t))})
        out.append({"type": "image"})
    return out
