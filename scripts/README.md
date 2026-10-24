# scripts/

Entrypoints for getting the data, training and inference. Run them from the
repo root, e.g. `uv run python scripts/<category>/<name>.py ...`.

```
scripts/
├── data/        download, split, frame extraction, camera-motion cue
├── training/    train / evaluate / smoke test
└── submission/  Codabench packaging
```

## data/

| Script | Purpose |
|---|---|
| `data/fetch_data.py` | Download ReVA from the HF Hub (`--annotations_only`, `--folders`) |
| `data/make_splits.py` | Leak-free fit/dev split (seen video, unseen question) |
| `data/extract_frames.py` | Decode each clip once into a 64-frame JPEG cache + fps/indices (`--delete_videos`) |
| `data/compute_motion_cues.py` | Camera-motion text per clip → `data/processed/cues/motion.jsonl` |

## training/

| Script | Purpose |
|---|---|
| `training/train.py` | Thin wrapper over `src.pipelines.train` (the real CLI) |
| `training/evaluate.py` | Thin wrapper over `src.pipelines.predict`: score a split with a checkpoint |
| `training/smoke_test.py` | Offline end-to-end check on synthetic clips; `--model` adds a real VLM forward |

## submission/

| Script | Purpose |
|---|---|
| `submission/make_submission.py` | Validate all 4,000 test ids / letters, write `submission.zip` |
