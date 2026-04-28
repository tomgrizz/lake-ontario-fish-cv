# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project overview

Computer vision system for classifying salmonid fish species in Lake Ontario
tributary Riverwatcher camera footage, operated by the Ontario Ministry of
Natural Resources, Lake Ontario Management Unit.

Classifies five salmonid species: Chinook Salmon, Coho Salmon, Rainbow Trout,
Atlantic Salmon, Brown Trout.

> Note: confirm the exact species list against `model/mar10_2026_detr-resnet50/config.json`
> (`id2label` field) before relying on this for code generation.

Architecture: RT-DETR (`PekingU/rtdetr_v2_r50vd`) for detection and
classification, BoT-SORT for tracking.

## Current focus

Building inference pipeline v2 in `inference/`. The previous pipeline (in
`gradio_app/salmonid_app-main/`) outputs only top-1 species, direction, and
entrance side per video as JSON. The new pipeline must output full
per-detection records (bboxes, full softmax, track IDs) to support
human-in-the-loop labeling and retraining.

The new pipeline reuses the working detection / tracking / direction logic
from the previous code but rewrites output handling and adds robustness for
long batch runs (~244k videos across Ganaraska and Credit).

## Hard rules

- **`gradio_app/salmonid_app-main/` is read-only reference.** Do not modify it.
  Read it to understand existing logic, but the new pipeline lives in
  `inference/`.
- **Test sets are immutable once frozen.** Once a `test_set/test_set_vN/` is
  created and labeled, never modify it in place. Corrections create a new
  version (`test_set_vN+1`).
- **No model weights, outputs, or large binaries in git.** Outputs go to
  the path documented in `docs/local_paths.md`. Model checkpoints stay in
  `model/` (gitignored).
- **No credentials in git.** API keys, database passwords, etc. live in
  environment variables or gitignored config, never committed.

## Repository structure

```
inference/      # NEW pipeline being built (currently empty)
  config/       # tracker and inference configs
labeling/       # Label Studio integration (planned, not started)
test_set/       # frozen evaluation sets (none created yet)
training/       # retraining workflows (planned, not started)
docs/           # additional documentation
gradio_app/     # READ-ONLY reference code from previous project (gitignored)
model/          # model checkpoints (gitignored)
```

## Reference code structure (`gradio_app/salmonid_app-main/`)

Modules to study when building the new pipeline:

- `video.py` — frame-by-frame RT-DETR detection → BoT-SORT tracking →
  per-detection aggregation. The core inference logic to preserve.
- `config.py` — `AppConfig` dataclass with model paths, detection thresholds,
  tracker parameters.
- `gradio_app.py` / `terminal_app.py` — UI and CLI entry points. The new
  pipeline will replace the CLI for batch runs; the Gradio UI is not part
  of the new build.
- `recording_log.py` — parses Riverwatcher recording logs to attach
  wall-clock timestamps to detections.

## Paths

Machine-specific paths are documented in `docs/local_paths.md` (gitignored,
not committed). Refer to that file for the model checkpoint location, source
video locations, output storage location, and any external service endpoints.

## Conventions

- **Model naming**: `{MMMDD_YYYY}_{architecture}` (e.g. `mar10_2026_detr-resnet50`)
- **Test set naming**: `test_set_v1`, `test_set_v2`, etc.
- **Run output naming**: `run_{timestamp}/` under the outputs directory
- **Commit messages**: short imperative summary ("add parquet writer",
  "fix track ID flickering"), commit at coherent stopping points

## Environment

- Windows, GPU inference (CUDA)
- `numpy < 2` is pinned in requirements (compatibility constraint with
  current dependencies)
- BoT-SORT ReID weights downloaded separately
  (`osnet_x0_25_msmt17.pt`), not in git
- No automated tests or linting configured yet

## Reference commands

For running the existing reference pipeline (read-only — useful for
understanding behavior, but don't modify):

```bash
cd gradio_app/salmonid_app-main
python terminal_app.py INPUT_VIDEO [OPTIONS]    # CLI
python gradio_app.py                            # web UI on :7860
```

## Working with this repository

When starting a task:

1. Read this file and `docs/local_paths.md` (if local context is needed).
2. For inference work, read the relevant modules in
   `gradio_app/salmonid_app-main/` to understand existing behavior.
3. Propose a structure or approach before writing code; build incrementally
   with commits at each major piece.
4. Don't create files outside the established structure without flagging it.
5. Keep changes focused — separate commits for separate concerns.
