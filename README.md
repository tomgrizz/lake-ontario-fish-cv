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

Five salmonid species classified by the model:

- Chinook Salmon
- Coho Salmon
- Rainbow Trout
- Atlantic Salmon
- Brown Trout

## Deployment context

Riverwatcher (VAKI) underwater camera systems deployed on Lake Ontario tributaries
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

## Labeling — reviewer quick-start

Fish tracks from the inference pipeline are reviewed in Label Studio. As a reviewer
you just need a browser — no installation required.

### Access

Open this URL in your browser (you must be connected to Tailscale):

```
http://100.68.58.2:8080
```

Log in with the credentials Tom provided. You will land on the
**Salmonid Fish Review** project.

### How to label

1. Click **Label All Tasks** to enter the labeling stream.
2. A short clip plays automatically showing one fish track with a bounding box overlay.
3. The model's best guess is already selected — if it's correct, just press **Submit**
   (or **Enter**). That's most tasks.
4. If the model is wrong, press the correct species key before submitting:

| Key | Action |
|-----|--------|
| **1** | Chinook Salmon |
| **2** | Coho Salmon |
| **3** | Atlantic Salmon |
| **4** | Rainbow Trout |
| **5** | Brown Trout |
| **n** | Not a fish (debris, shadow, artifact) |
| **m** | Multiple fish merged into one track |
| **u** | Unsure — flag for senior review |

5. Label Studio advances to the next task automatically after each submission.

### Species ID cues (quick reference)

| Species | Key features |
|---------|-------------|
| Chinook Salmon | Largest body, black gum line, black spots on both lobes of tail |
| Coho Salmon | Smaller spots on upper tail lobe only, white gum line |
| Atlantic Salmon | Spots above lateral line, forked tail, slender |
| Rainbow Trout | Pink/red lateral stripe, heavily spotted including lower body |
| Brown Trout | Orange/red spots with pale halos, squared tail |

**Edge cases:** Label juveniles by species if confident. For partial views, label if
confident — otherwise press **u**. If two fish appear in the same clip, press **m**.
Some tasks have known correct answers for quality monitoring — treat them like any
other task.

### Target pace

Confident tasks (Confirm) should take 2–3 seconds. Exception tasks (wrong species,
not a fish, unsure) take longer — that's expected.

## Status

Active development. Inference pipeline v2 running (~23% complete as of April 2026).
Labeling infrastructure operational. Retraining pipeline planned after first label batch.
