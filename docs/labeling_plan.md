# Labeling Infrastructure Plan (revised)

## Context

The 6-day inference run is producing ~500k fish tracks across 258k videos. This plan
covers the labeling infrastructure needed to turn those tracks into human-verified
ground truth labels that feed model retraining and a frozen test set.

Reviewer model: "sneakernet" — reviewers get local copies of clips + a Label Studio
task JSON; they run Label Studio on their own machines, annotate, and return a JSON
export. Labels are merged into a master labels.sqlite on the main machine.

---

## Label Studio capabilities (verified)

- **Video tag**: scrubbing and timeline supported. No native autoplay or loop.
- **Space-key behavior**: MUST be verified in step 3 (manual XML test). If Space
  submits the task rather than playing the video, the JS autoplay workaround is
  required for MVP — not deferred. Document the actual behavior and decide then.
- **Keyboard shortcuts**: `hotkey` attribute on `<Choice>` works natively.
- **Local file serving**: `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` +
  `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<path>`. URL pattern:
  `http://localhost:8080/data/local-files/?d=relative/path/clip.mp4`
- **Custom HTML**: `<HyperText name="x" value="$html_field" inline="true">` renders
  styled HTML including confidence bars — Community Edition.
- **Auto-advance**: built-in when using labeling stream ("Label All Tasks").
- **Pre-selected predictions**: `predictions` array in task JSON pre-populates the
  species choice. Confirm = just submit (Space/Enter once Space behavior is verified).
- **Metadata round-trip**: Only the `data` field round-trips reliably through export.
  Top-level `meta` is not reliably available in the export JSON. All metadata that
  export_labels.py needs must live in `data` with `_`-prefixed keys.
- **Underscore fields**: `data._field` keys not referenced in the XML config are
  invisible to the reviewer. Verify in step 3.
- **lead_time captured**: cumulative seconds on task (includes idle). Imperfect proxy
  for time-on-task; document this in quality.py output.
- **SQLite**: fine — one reviewer per machine, no concurrency.

---

## Pre-build step: shared scoring module

To avoid drift between the post-hoc computation (extract_clips.py) and the future
pipeline.py integration, factor the best_id_frame formula into a shared module:

**`inference/scoring.py`** — exports `compute_best_id_frame(detections_df, frame_width, frame_height)`:

```python
def compute_best_id_frame(detections_df, frame_width, frame_height):
    """
    detections_df: DataFrame with columns frame_number, bbox_x1, bbox_y1,
                   bbox_x2, bbox_y2, detection_confidence
    Returns: frame_number with highest best_id_score (tie-break: earliest frame)
    """
    w, h = frame_width, frame_height
    cx = (detections_df.bbox_x1 + detections_df.bbox_x2) / 2
    cy = (detections_df.bbox_y1 + detections_df.bbox_y2) / 2
    bw = detections_df.bbox_x2 - detections_df.bbox_x1
    bh = detections_df.bbox_y2 - detections_df.bbox_y1
    bbox_area_norm = (bw * bh) / (w * h)
    dist = ((cx - w/2)**2 + (cy - h/2)**2) ** 0.5
    dist_norm = dist / ((w/2)**2 + (h/2)**2) ** 0.5
    score = detections_df.detection_confidence * bbox_area_norm * (1 - dist_norm)
    return int(detections_df.loc[score.idxmax(), "frame_number"])
```

Unit test: synthetic 5-detection DataFrame with a known best frame → verify output.
Both `extract_clips.py` and the future `pipeline.py` update import from `scoring.py`.

---

## tracks.sqlite additions (post-hoc, via extract_clips.py)

```sql
ALTER TABLE tracks ADD COLUMN best_id_frame      INTEGER;
ALTER TABLE tracks ADD COLUMN clip_start_frame   INTEGER;
ALTER TABLE tracks ADD COLUMN clip_end_frame     INTEGER;
ALTER TABLE tracks ADD COLUMN clip_path          TEXT;
```

**Clip bounds**:
```
lead_in_frames   = round(1.5 * fps)   # configurable, default 37 at 25fps
lead_out_frames  = round(1.5 * fps)
clip_start_frame = max(track_start_frame, best_id_frame - lead_in_frames)
clip_end_frame   = min(track_end_frame, best_id_frame + lead_out_frames)
```
Full track used when shorter than lead_in + lead_out. No padding.

---

## labels.sqlite schema

```sql
CREATE TABLE labels (
    label_id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id                        TEXT    NOT NULL,
    track_id                        INTEGER NOT NULL,
    final_label                     TEXT    NOT NULL,
    -- Values: "Chinook" / "Coho" / "Atlantic" / "Rainbow Trout" / "Brown Trout"
    --         / "Not a fish" / "Multiple fish" / "Unsure"
    label_action                    TEXT    NOT NULL,
    -- Values: "confirm" / "specify" / "not_a_fish" / "multiple_fish" / "unsure"
    reviewer_id                     TEXT    NOT NULL,
    reviewed_at                     TEXT    NOT NULL,
    time_on_task                    REAL,
    original_predicted_class_6      TEXT    NOT NULL,
    original_predicted_species      TEXT    NOT NULL,
    original_confidence             REAL    NOT NULL,
    model_version                   TEXT    NOT NULL,
    calibration_task                INTEGER NOT NULL DEFAULT 0,
    multi_reviewed                  INTEGER NOT NULL DEFAULT 0,
    disagreement_resolution         TEXT,
    ls_annotation_id                INTEGER,
    batch_id                        TEXT,
    UNIQUE (video_id, track_id, reviewer_id)
);

CREATE TABLE skipped_tasks (
    video_id     TEXT    NOT NULL,
    track_id     INTEGER NOT NULL,
    reviewer_id  TEXT    NOT NULL,
    skipped_at   TEXT    NOT NULL,
    reason       TEXT,
    batch_id     TEXT,
    PRIMARY KEY (video_id, track_id, reviewer_id)
);

CREATE TABLE calibration_ground_truth (
    video_id            TEXT    NOT NULL,
    track_id            INTEGER NOT NULL,
    ground_truth_label  TEXT    NOT NULL,
    added_by            TEXT    NOT NULL,
    added_at            TEXT    NOT NULL,
    PRIMARY KEY (video_id, track_id)
);

CREATE TABLE reviewer_stats (
    reviewer_id             TEXT    NOT NULL,
    computed_at             TEXT    NOT NULL,
    n_labels                INTEGER,
    n_calibration_correct   INTEGER,
    n_calibration_total     INTEGER,
    calibration_accuracy    REAL,
    mean_time_on_task       REAL,
    n_confirmed             INTEGER,
    n_specified             INTEGER,
    n_not_a_fish            INTEGER,
    n_multiple_fish         INTEGER,
    n_unsure                INTEGER,
    PRIMARY KEY (reviewer_id)
);
```

**Training data semantics**:
- `final_label` in {Chinook, Coho, Atlantic, Rainbow Trout, Brown Trout} → training positive
- `final_label = "Not a fish"` → hard negative for detection training
- `final_label = "Multiple fish"` → EXCLUDED from training data
- `final_label = "Unsure"` → EXCLUDED from training data
- Multi-reviewed with `disagreement_resolution = "pending"` → EXCLUDED
- Multi-reviewed with `disagreement_resolution = "senior"` → included; overruled row excluded

---

## Queue strategy

### Phase 0 — Calibration
- Source: `calibration_ground_truth` table (you populate manually)
- n_tasks: 200–300, identical set across all reviewers

### Phase 1 — Stratified high-confidence
- Filters: `mean_detection_confidence >= 0.85`, background not dominant
- Species stratification: balanced per-species (n_tasks/5 per species)
- 3% calibration injection (random positions, not clustered)

### Phase 2 — Mixed queue
```
priority = w1*uncertainty + w2*temporal_anomaly + w3*confusion_pair + w4*jitter
```
Default weights: w1=0.50, w2=0.00 (disabled), w3=0.15, w4=0.05

- `uncertainty_score` = entropy of `mean_probs[:5]`
- `confusion_pair_score` = 1.0 if top-2 are a known pair (Chinook/Coho, Atlantic/Brown)
- `temporal_anomaly_score` = 0.0 until historical prior CSV provided

### Multi-review mechanics
- 7% of tasks sent to >1 reviewer (they don't know it's a duplicate)
- Disagreements flagged as "pending" → resolved by senior via resolve_disagreements.py

---

## File layout

```
labeling/
  schema.py                 ✓ done
  task_config.xml           ✓ done
  queue.py                  ← next (step 4)
  import_tasks.py           ← step 5
  export_labels.py          ← step 6
  merge_labels.py           ← step 7
  resolve_disagreements.py  ← step 8
  quality.py                ← step 9
  REVIEWER_GUIDE.md         ← step 10
  config/
    queue_config.yaml

inference/
  scoring.py                ✓ done
  extract_clips.py          ✓ done
```

---

## Reviewer package layout

```
reviewer_batch_001/
  clips/           <- copy to LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
  tasks.json       <- import into Label Studio project
  setup.md         <- LS install + env var + import instructions
  REVIEWER_GUIDE.md
```

---

## Deferred items

- Temporal anomaly score: disabled until historical species × day-of-year CSV provided
- pipeline.py best_id_frame: update after extract_clips.py validated on current run
- Test set v1 sampling: after Phase 0 calibration labels exist
- JS autoplay/loop: decision after Space-key verification in step 3
