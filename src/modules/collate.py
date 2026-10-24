"""Batch builder: examples -> chat prompts + video tensors for the VLM.

Left padding, so the answer position is the last token of every row and
`logits_to_keep=1` gives exactly the logits we need.

`MCQCollator` serves every backbone family (`src.modules.backbones`); the
packed and branched collators are Qwen-only.
"""

import random
from pathlib import Path
from typing import List, Optional

import torch

from src.config import ModelConfig
from src.data import ReVAExample
from src.modules.branched import right_pad, shared_prefix_len
from src.modules.counterfactual import reversal_sensitive
from src.modules.cues import CueStore
from src.prompting import build_messages, question_text
from src.video import load_frames, overlay_time_bar

FACT_INSTRUCTION = "List the questions and answers you know about this video."

# Closes the (expanded) video span in Qwen chat prompts.
VISION_END = "<|vision_end|>"

# Floor on per-frame pixels when we override the processor's budget.
MIN_PIXELS_PER_FRAME = 64 * 32 * 32


class MCQCollator:
    def __init__(
        self,
        processor,
        cfg: ModelConfig,
        cue_store: Optional[CueStore] = None,
        frames_root: Optional[str | Path] = None,
        video_root: Optional[str | Path] = None,
    ):
        self.processor = processor
        self.cfg = cfg
        self.cue_store = cue_store or CueStore()
        # Where in each temporal segment frames are taken from the cache
        # (0.5 = centre). Other values give jittered views for test-time
        # augmentation.
        self.frame_phase = 0.5
        # Training-time augmentation: draw a fresh phase in [0, 1) per example
        # so the model sees arbitrary timestamp layouts.
        self.random_phase = False
        # Counterfactual temporal view (training only): for questions of these
        # tasks the batch also carries `cf`, the same prompts on the clip
        # played backwards (timestamps / time bar still run forward) and
        # without text cues (they describe the original direction).
        self.cf_tasks: tuple = ()
        self._reverse = False
        # Fact rows (training only): {video_path: [(question, answer)]}.
        # A branched group gets one extra row asking the model to write out
        # these Q/A facts about the clip as text (no options, no letters),
        # i.e. the same facts in a second format ("mixed training").
        self.facts: dict = {}
        self.max_facts = 24
        # Cue dropout (training only): {cue: p}, each cue left out of a
        # question's prompt with probability p, so the model does not lean on
        # a cue where it is weak (the caption on Temporal / Change).
        self.cue_dropout: dict = {}
        self.frames_root = frames_root
        self.video_root = video_root
        self.family = cfg.family_name
        tok = getattr(processor, "tokenizer", processor)
        tok.padding_side = "left"

    def prompt(self, ex: ReVAExample, frame_times=None) -> str:
        """Chat-templated prompt ending right before the answer letter.
        `frame_times`: per-frame seconds for families that take the clip as
        timestamped images (InternVL); None renders one video entry."""
        messages = build_messages(
            ex,
            cue_text=self._cue_text(ex),
            use_video=self.cfg.use_video,
            system_prompt=self.cfg.system_prompt,
            frame_times=frame_times,
        )
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(self.cfg.chat_template_kwargs or {}),
        )
        return text + (self.cfg.answer_prefix or "")

    def _cue_text(self, ex: ReVAExample) -> str:
        """Cue lines for `ex`: none on the reversed view, minus dropped cues."""
        if self._reverse:
            return ""
        skip = [c for c, p in self.cue_dropout.items() if random.random() < p]
        return self.cue_store.text(ex, skip=skip)

    def _suffixes(self):
        if getattr(self, "_gen", None) is None:
            msgs = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
            kw = self.cfg.chat_template_kwargs or {}
            with_gen = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **kw
            )
            without = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, **kw
            )
            if not with_gen.startswith(without):
                raise ValueError("chat template: generation prompt is not a suffix")
            self._gen = with_gen[len(without) :]
            # "<|im_start|>user\n" ... "<|im_end|>\n" around a later user turn
            head, _, tail = without.rpartition("x")
            self._user_open = head[head.rfind("<|im_start|>") :]
            self._turn_close = tail
        return self._gen, self._user_open, self._turn_close

    def _overlay_time(self, ex: ReVAExample) -> bool:
        """Whether to draw the time bar for this example (optionally task-gated)."""
        if not self.cfg.overlay_time:
            return False
        tasks = self.cfg.overlay_time_tasks
        return not tasks or ex.task in tasks

    def _video(self, ex: ReVAExample):
        """(frames, video_metadata) for one example, incl. the time bar and,
        for the counterfactual view, reversed frame order."""
        fs = load_frames(
            ex.video_path,
            self.cfg.num_frames,
            self.frames_root,
            self.video_root,
            phase=random.random() if self.random_phase else self.frame_phase,
        )
        frames = fs.frames[::-1].copy() if self._reverse else fs.frames
        if self._overlay_time(ex):
            frames = overlay_time_bar(frames, fs.timestamps)
        meta = {
            "fps": fs.fps,
            "frames_indices": fs.frame_indices,
            "total_num_frames": fs.total_frames,
        }
        return frames, meta

    def fact_text(self, ex: ReVAExample) -> str:
        """Prompt + target for the fact row of `ex`'s clip (shuffled, capped)."""
        pairs = list(self.facts[ex.video_path])
        random.shuffle(pairs)
        target = "".join(
            f"Q: {q.strip()}\nA: {a.strip()}\n" for q, a in pairs[: self.max_facts]
        )
        cue = self._cue_text(ex)
        text = (cue + "\n" if cue else "") + FACT_INSTRUCTION
        content = [{"type": "video"}, {"type": "text", "text": text}]
        messages = [{"role": "user", "content": content}]
        if self.cfg.system_prompt:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.cfg.system_prompt}],
                },
            )
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(self.cfg.chat_template_kwargs or {}),
        )
        return prompt + target.rstrip("\n") + "<|im_end|>"

    def _counterfactual(self, examples, grouped: bool) -> dict:
        """{"cf": batch} for the examples in `cf_tasks` (empty otherwise)."""
        if self._reverse or not self.cf_tasks:
            return {}
        sub = [
            ex for ex in examples if ex.task in self.cf_tasks and reversal_sensitive(ex)
        ]
        if not sub:
            return {}
        self._reverse = True
        try:
            return {"cf": self([sub] if grouped else sub)}
        finally:
            self._reverse = False

    def video_kwargs(self) -> dict:
        kwargs = {}
        if self.family == "internvl":
            # One 448x448 tile per frame (InternVL's video setting), no
            # dynamic tiling into up to 12 crops.
            kwargs["crop_to_patches"] = False
        elif self.cfg.max_pixels_per_frame:
            kwargs["size"] = {
                "longest_edge": self.cfg.max_pixels_per_frame * self.cfg.num_frames,
                "shortest_edge": MIN_PIXELS_PER_FRAME * self.cfg.num_frames,
            }
            # qwen-vl-utils' per-frame cap (768 tokens = 786k px) is above our
            # 640x360 share, so this changes nothing but pins the behavior that
            # becomes the default in transformers 5.22 (and silences its warning).
            if self.family == "qwen":
                kwargs["cap_pixels_per_frame"] = True
        kwargs.update(self.cfg.video_kwargs or {})
        return kwargs

    def __call__(self, examples: List[ReVAExample]) -> dict:
        kwargs = {}
        if not self.cfg.use_video:
            texts = [self.prompt(ex) for ex in examples]
        elif self.family == "internvl":
            texts, images = [], []
            for ex in examples:
                frames, meta = self._video(ex)
                times = [i / meta["fps"] for i in meta["frames_indices"]]
                texts.append(self.prompt(ex, frame_times=times))
                images.append(list(frames))
            kwargs = {"images": images, **self.video_kwargs()}
        else:
            texts, videos, metadata = [], [], []
            for ex in examples:
                frames, meta = self._video(ex)
                texts.append(self.prompt(ex))
                videos.append(frames)
                metadata.append(meta)
            kwargs = {
                "videos": videos,
                "video_metadata": metadata,
                # Frames are already sampled (and their indices recorded).
                "do_sample_frames": False,
                **self.video_kwargs(),
            }
        inputs = self.processor(text=texts, padding=True, return_tensors="pt", **kwargs)
        return {
            "inputs": inputs,
            "labels": torch.tensor([ex.label for ex in examples], dtype=torch.long),
            "qa_ids": [ex.qa_id for ex in examples],
            "tasks": [ex.task for ex in examples],
            "perms": [ex.perm for ex in examples],
            **self._counterfactual(examples, grouped=False),
        }


class PackedMCQCollator(MCQCollator):
    """One video + several of its questions in a single sequence.

        [video] cue Q1 <gen> A1 <eot> <user> Q2 <gen> A2 <eot> ... <user> Qk <gen>

    `<gen>` is the exact assistant prefix of the single-question prompt
    (incl. the empty think block), so every answer position sees the same
    local format as at test time. The gold letters A1..A(k-1) are in the
    sequence (teacher forcing). The video tokens are encoded once for k
    questions, which is where the speed-up comes from.

    Takes a batch of exactly one group; returns `positions`, the indices
    whose next-token logits are read out (one per question).
    """

    def packed_text(self, group) -> str:
        """Full chat-formatted sequence for one group: video + Q1, then each
        later question as its own user turn after the previous gold answer.

        Every question carries its own cue text. The first question gets it via
        `self.prompt`; the later turns must pass it explicitly, or all but the
        first question in a packed sequence would silently lose the cue.
        """
        gen, user_open, close = self._suffixes()
        text = self.prompt(group[0])
        for prev, ex in zip(group, group[1:]):
            text += (
                prev.answer_letter
                + close
                + user_open
                + question_text(ex, cue_text=self._cue_text(ex))
                + close
                + gen
            )
        return text

    def __call__(self, groups) -> dict:
        if len(groups) != 1:
            raise ValueError("PackedMCQCollator needs batch_size=1 (one group)")
        group = groups[0]
        if not self.cfg.use_video:
            raise ValueError("question packing needs use_video=True")
        if self.family != "qwen":
            raise ValueError("question packing is implemented for Qwen only")
        text = self.packed_text(group)
        gen, _, _ = self._suffixes()
        inputs = self._encode_packed(text, group[0])
        ids = inputs["input_ids"][0]
        gen_ids = self.processor.tokenizer.encode(gen, add_special_tokens=False)
        positions = _find_ends(ids.tolist(), gen_ids)
        if len(positions) != len(group):
            raise ValueError(
                f"found {len(positions)} answer positions for {len(group)} questions"
            )
        return dict(
            inputs=inputs,
            positions=torch.tensor(positions, dtype=torch.long),
            labels=torch.tensor([ex.label for ex in group], dtype=torch.long),
            qa_ids=[ex.qa_id for ex in group],
            tasks=[ex.task for ex in group],
            perms=[ex.perm for ex in group],
        )

    def _encode_packed(self, text: str, ex: ReVAExample):
        fs = load_frames(
            ex.video_path,
            self.cfg.num_frames,
            self.frames_root,
            self.video_root,
            phase=random.random() if self.random_phase else self.frame_phase,
        )
        meta = {
            "fps": fs.fps,
            "frames_indices": fs.frame_indices,
            "total_num_frames": fs.total_frames,
        }
        frames = (
            overlay_time_bar(fs.frames, fs.timestamps)
            if self._overlay_time(ex)
            else fs.frames
        )
        return self.processor(
            text=[text],
            videos=[frames],
            video_metadata=[meta],
            do_sample_frames=False,
            return_tensors="pt",
            **self.video_kwargs(),
        )


class BranchedMCQCollator(MCQCollator):
    """One video + k of its questions as k single-question prompts that share
    a prefix (`src.modules.branched`). Leak-free replacement for packing:
    every question is scored on exactly its test-time prompt.

    Takes a batch of exactly one group (one video, one overlay setting);
    returns right-padded rows, ONE copy of the video tensors, `prefix_len`
    and each row's `last_index`. `group_size` is the chunk size evaluation
    uses (`src.pipelines.eval.score_examples`).
    """

    group_size = 8

    def __call__(self, groups) -> dict:
        if len(groups) != 1:
            raise ValueError("BranchedMCQCollator needs batch_size=1 (one group)")
        group = groups[0]
        if not self.cfg.use_video:
            raise ValueError("branched scoring needs use_video=True")
        if self.family != "qwen":
            raise ValueError("branched scoring is implemented for Qwen3.5 only")
        if len({ex.video_path for ex in group}) != 1:
            raise ValueError("a branched group must share one video")
        overlay = {self._overlay_time(ex) for ex in group}
        if len(overlay) != 1:
            raise ValueError("a branched group must share the time-bar setting")
        frames, meta = self._video(group[0])
        texts = [self.prompt(ex) for ex in group]
        with_fact = (
            bool(self.facts) and not self._reverse and group[0].video_path in self.facts
        )
        if with_fact:
            texts.append(self.fact_text(group[0]))
        enc, rows, types = self._encode_shared_video(texts, frames, meta)
        extra = {}
        if with_fact:
            gen_ids = self.processor.tokenizer.encode(
                self._suffixes()[0], add_special_tokens=False
            )
            ends = _find_ends(rows[-1], gen_ids)
            if not ends:
                raise ValueError("fact row: generation prompt not found")
            # Target tokens are rows[-1][fact_start:]; scored from fact_start - 1.
            extra["fact_start"] = ends[-1] + 1
        pad = self.processor.tokenizer.pad_token_id or 0
        input_ids, attention_mask = right_pad(rows, pad)
        mm_types, _ = right_pad(types, 0)
        grid = enc["video_grid_thw"][:1]
        per_video = int(grid[0].prod())
        return dict(
            inputs={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "mm_token_type_ids": mm_types,
                "pixel_values_videos": enc["pixel_values_videos"][:per_video],
                "video_grid_thw": grid,
            },
            prefix_len=shared_prefix_len(rows),
            last_index=torch.tensor(
                [len(r) - 1 for r in rows[: len(group)]], dtype=torch.long
            ),
            labels=torch.tensor([ex.label for ex in group], dtype=torch.long),
            qa_ids=[ex.qa_id for ex in group],
            tasks=[ex.task for ex in group],
            perms=[ex.perm for ex in group],
            **self._counterfactual(group, grouped=True),
            **extra,
        )

    def _encode_shared_video(self, texts, frames, meta):
        """(enc, rows, types) for prompts that share one video.

        The clip goes through the processor once, with the common head up to
        the video's `<|vision_end|>`; each prompt's tail after it is tokenized
        as plain text. Special tokens are hard tokenizer boundaries, so the
        ids equal encoding every full prompt with its own copy of the video,
        without processing the frames once per row.
        """
        heads, _, tails = zip(*(t.rpartition(VISION_END) for t in texts))
        if len(set(heads)) != 1 or not heads[0]:
            raise ValueError("branched prompts must share the head up to the video")
        enc = self.processor(
            text=[heads[0] + VISION_END],
            videos=[frames],
            video_metadata=[meta],
            do_sample_frames=False,
            return_tensors="pt",
            **self.video_kwargs(),
        )
        head_ids = enc["input_ids"][0].tolist()
        head_types = enc["mm_token_type_ids"][0].tolist()
        tok = self.processor.tokenizer
        tail_ids = [tok.encode(t, add_special_tokens=False) for t in tails]
        rows = [head_ids + ids for ids in tail_ids]
        types = [head_types + [0] * len(ids) for ids in tail_ids]
        return enc, rows, types


def _find_ends(seq: List[int], pattern: List[int]) -> List[int]:
    """Index of the last token of every occurrence of `pattern` in `seq`."""
    n, out = len(pattern), []
    for i in range(len(seq) - n + 1):
        if seq[i : i + n] == pattern:
            out.append(i + n - 1)
    return out
