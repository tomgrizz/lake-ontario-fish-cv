"""
labels.sqlite schema for the labeling pipeline.

Tables:
  labels                  — one row per (track, reviewer) annotation
  skipped_tasks           — tasks a reviewer skipped (tracked separately)
  calibration_ground_truth — known-answer tasks populated by senior reviewer
  reviewer_stats          — per-reviewer quality metrics (refreshed by quality.py)
  meta                    — key/value store for schema_version, run metadata
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCHEMA_VERSION: int = 1

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

LABELS_DDL: str = """
CREATE TABLE IF NOT EXISTS labels (
    label_id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id                        TEXT    NOT NULL,
    track_id                        INTEGER NOT NULL,
    final_label                     TEXT    NOT NULL,
    -- "Chinook" | "Coho" | "Atlantic" | "Rainbow Trout" | "Brown Trout"
    --   | "Not a fish" | "Multiple fish" | "Unsure"
    label_action                    TEXT    NOT NULL,
    -- "confirm" | "specify" | "not_a_fish" | "multiple_fish" | "unsure"
    reviewer_id                     TEXT    NOT NULL,
    reviewed_at                     TEXT    NOT NULL,
    time_on_task                    REAL,
    -- lead_time from Label Studio (cumulative seconds, includes idle time)
    original_predicted_class_6      TEXT    NOT NULL,
    -- top-1 across all 6 classes including Background
    original_predicted_species      TEXT    NOT NULL,
    -- top-1 across 5 fish classes only
    original_confidence             REAL    NOT NULL,
    -- mean_detection_confidence from the track
    model_version                   TEXT    NOT NULL,
    calibration_task                INTEGER NOT NULL DEFAULT 0,
    -- 1 = known-answer calibration task; interface is identical to normal tasks
    multi_reviewed                  INTEGER NOT NULL DEFAULT 0,
    -- 1 = this track was sent to more than one reviewer
    disagreement_resolution         TEXT,
    -- NULL        = not multi-reviewed (or reviewed once and agreed)
    -- "pending"   = multi-reviewed, labels differ, awaiting senior resolution
    -- "senior"    = senior confirmed this label as correct
    -- "senior_overruled" = senior rejected this label
    ls_annotation_id                INTEGER,
    -- Label Studio annotation ID for audit trail
    batch_id                        TEXT,
    UNIQUE (video_id, track_id, reviewer_id)
)
"""

SKIPPED_TASKS_DDL: str = """
CREATE TABLE IF NOT EXISTS skipped_tasks (
    video_id     TEXT    NOT NULL,
    track_id     INTEGER NOT NULL,
    reviewer_id  TEXT    NOT NULL,
    skipped_at   TEXT    NOT NULL,
    reason       TEXT,
    batch_id     TEXT,
    PRIMARY KEY (video_id, track_id, reviewer_id)
)
"""
# Skips never become label rows. The queue re-surfaces skipped tasks to other
# reviewers. tracking here prevents re-sending to the same reviewer.

CALIBRATION_DDL: str = """
CREATE TABLE IF NOT EXISTS calibration_ground_truth (
    video_id            TEXT    NOT NULL,
    track_id            INTEGER NOT NULL,
    ground_truth_label  TEXT    NOT NULL,
    added_by            TEXT    NOT NULL,
    added_at            TEXT    NOT NULL,
    PRIMARY KEY (video_id, track_id)
)
"""

REVIEWER_STATS_DDL: str = """
CREATE TABLE IF NOT EXISTS reviewer_stats (
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
)
"""

META_DDL: str = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""

_META_INSERT_VERSION: str = (
    "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)"
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LabelRecord:
    video_id: str
    track_id: int
    final_label: str
    label_action: str
    reviewer_id: str
    reviewed_at: str
    original_predicted_class_6: str
    original_predicted_species: str
    original_confidence: float
    model_version: str
    time_on_task: Optional[float] = None
    calibration_task: int = 0
    multi_reviewed: int = 0
    disagreement_resolution: Optional[str] = None
    ls_annotation_id: Optional[int] = None
    batch_id: Optional[str] = None


@dataclass
class SkippedRecord:
    video_id: str
    track_id: int
    reviewer_id: str
    skipped_at: str
    batch_id: Optional[str] = None
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Training data inclusion rules
# ---------------------------------------------------------------------------

#: Labels that contribute positives for species classification training.
TRAINING_POSITIVE_LABELS = frozenset([
    "Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"
])
#: Label that contributes hard negatives for detection training.
TRAINING_HARD_NEGATIVE_LABEL = "Not a fish"
#: Labels excluded from training data entirely.
TRAINING_EXCLUDED_LABELS = frozenset(["Multiple fish", "Unsure"])
#: disagreement_resolution values that exclude a row from training data.
TRAINING_EXCLUDED_RESOLUTIONS = frozenset(["pending", "senior_overruled"])


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

def init_labels_db(conn: sqlite3.Connection) -> None:
    """Create all tables and stamp schema_version. Idempotent."""
    conn.execute(LABELS_DDL)
    conn.execute(SKIPPED_TASKS_DDL)
    conn.execute(CALIBRATION_DDL)
    conn.execute(REVIEWER_STATS_DDL)
    conn.execute(META_DDL)
    conn.execute(_META_INSERT_VERSION, (str(SCHEMA_VERSION),))
    conn.commit()


def open_labels_db(db_path: Path) -> sqlite3.Connection:
    """Open labels.sqlite with WAL mode and init tables."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    init_labels_db(conn)
    return conn
