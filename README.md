# Lake Ontario Fish CV

Fish classification system for Lake Ontario tributary Riverwatcher camera systems.
Operated by the Ontario Ministry of Natural Resources, Lake Ontario Management Unit.

## Purpose

This project builds and maintains the fish classification system for Lake Ontario
tributary Riverwatcher systems (Ganaraska River, Credit River). It comprises:

- **Inference pipeline** (DETR + BoT-SORT) for fish detection, tracking, and species classification
- **Human-in-the-loop labeling infrastructure** (Label Studio) for tech review of model outputs
- **Frozen test set** for evaluating model versions across retraining cycles
- **Retraining workflows** for iterative model improvement

The system is designed to iteratively improve through tech-reviewed label feedback while
maintaining a defensible evaluation baseline across model versions.

## Target species

Six salmonid species classified by the model:

- Chinook Salmon
- Coho Salmon
- Rainbow Trout
- Atlantic Salmon
- Brown Trout

## Deployment context

Riverwatcher (Smith-Root) underwater camera systems deployed on Lake Ontario tributaries
record fish passage events year-round. Footage is processed through the inference
pipeline to produce per-detection outputs (bounding boxes, species probabilities, track
IDs), which feed both operational fish counts and the labeling workflow used to
continuously improve the model.

## Repository structure

```
inference/      # DETR + BoT-SORT inference pipeline
  config/       # BoT-SORT and inference settings
labeling/       # Label Studio config, queue prioritization, label export
test_set/       # Frozen test set construction and storage
training/       # Retraining workflows (planned)
docs/           # Additional documentation
```

Outputs (parquet detections, SQLite databases, model checkpoints) are stored outside the
repo. See `docs/local_paths.md` for environment-specific paths.

## Local setup

Paths on the primary development machine are documented in `docs/local_paths.md`
(gitignored). See that file for model checkpoint, reference code, and output locations
specific to this environment.

## Versioning conventions

- **Model checkpoints** are named `{date}_{architecture}` — e.g. `mar10_2026_detr-resnet50`
- **Inference runs** are tagged with model version, code commit hash, and run date
- **Test set** is versioned as `test_set_v1`, `test_set_v2`, etc. and is never modified
  in place

## Status

Active development. Currently building inference pipeline v2 with full per-detection
outputs (bounding boxes, full softmax, track IDs) to support the labeling workflow and
downstream retraining.
