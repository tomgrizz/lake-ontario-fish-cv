"""
Output schemas for inference pipeline v2.

SCHEMA_VERSION, dataclasses, PyArrow schema for per-detection Parquet,
and SQLite DDL for tracks and processing-log databases.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pyarrow as pa

SCHEMA_VERSION: int = 1

# Class labels matching model/mar10_2026_detr-resnet50/config.json id2label.
# Index 5 is the DETR no-object class, stored as "Background" throughout.
FISH_CLASSES: dict[int, str] = {
    0: "Chinook",
    1: "Coho",
    2: "Atlantic",
    3: "Rainbow Trout",
    4: "Brown Trout",
}
BACKGROUND_CLASS_IDX: int = 5
ALL_CLASSES: dict[int, str] = {**FISH_CLASSES, BACKGROUND_CLASS_IDX: "Background"}
NUM_CLASSES: int = len(ALL_CLASSES)  # 6


@dataclass
class DetectionRecord:
    video_id: str
    frame_number: int
    timestamp_seconds: float
    track_id: int
    detection_id: int        # 0-based row index within this video's Parquet
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    detection_confidence: float
    prob_chinook: float
    prob_coho: float
    prob_atlantic: float
    prob_rainbow: float
    prob_brown: float
    prob_background: float
    predicted_class: int     # argmax of fish-only probs (0–4, excludes Background)
    predicted_class_6: int   # argmax across all 6 classes (0–5, includes Background)


@dataclass
class TrackSummary:
    video_id: str
    track_id: int
    start_frame: int
    end_frame: int
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    n_frames: int
    mean_prob_chinook: float
    mean_prob_coho: float
    mean_prob_atlantic: float
    mean_prob_rainbow: float
    mean_prob_brown: float
    mean_prob_background: float
    # mean_class_probs is an unweighted mean — each frame contributes equally
    # regardless of detection_confidence. Matches existing reference behavior.
    # Future enhancement: confidence-weighted mean if low-confidence boundary
    # frames are observed to drag down track-level predictions.
    predicted_class: int     # argmax of mean fish-only probs (0–4)
    predicted_class_6: int   # argmax of mean probs across all 6 classes
    mean_detection_confidence: float
    direction: str           # "Left" | "Right" | "Unknown"
    entrance_side: str       # first side fish appeared on: "Left" | "Right" | "None"
    exit_side: str           # last side fish appeared on: "Left" | "Right" | "None"
    # Earliest frame with the highest detection_confidence in the track.
    # Tie-break: when multiple frames share the max confidence (common with float32),
    # the smallest frame_number is chosen.
    representative_frame: int


# ---------------------------------------------------------------------------
# PyArrow schema for per-detection Parquet files.
# schema_version is stored as file-level metadata so every file is self-describing.
#
# y_coord_fix_applied is stamped here so files written by the post-fix
# pipeline are self-identifying as "born with correct Y coordinates."
# inference/fix_bbox_y.py uses the presence of this metadata key to decide
# whether to skip a Parquet file (already correct) or rewrite it (legacy
# buggy file from before pipeline.py target_sizes was patched).
# Value "baseline_post_fix" => correct at write time.
# Value "<ISO timestamp>"   => corrected post-hoc by fix_bbox_y.py.
# ---------------------------------------------------------------------------
DETECTION_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("frame_number", pa.int32()),
        pa.field("timestamp_seconds", pa.float32()),
        pa.field("track_id", pa.int32()),
        pa.field("detection_id", pa.int32()),
        pa.field("bbox_x1", pa.int16()),
        pa.field("bbox_y1", pa.int16()),
        pa.field("bbox_x2", pa.int16()),
        pa.field("bbox_y2", pa.int16()),
        pa.field("detection_confidence", pa.float32()),
        pa.field("prob_chinook", pa.float32()),
        pa.field("prob_coho", pa.float32()),
        pa.field("prob_atlantic", pa.float32()),
        pa.field("prob_rainbow", pa.float32()),
        pa.field("prob_brown", pa.float32()),
        pa.field("prob_background", pa.float32()),
        pa.field("predicted_class", pa.int8()),
        pa.field("predicted_class_6", pa.int8()),
    ],
    metadata={
        b"schema_version":       str(SCHEMA_VERSION).encode(),
        b"y_coord_fix_applied":  b"baseline_post_fix",
    },
)


# ---------------------------------------------------------------------------
# SQLite DDL
# ---------------------------------------------------------------------------
TRACKS_DDL: str = """
CREATE TABLE IF NOT EXISTS tracks (
    video_id                    TEXT    NOT NULL,
    track_id                    INTEGER NOT NULL,
    start_frame                 INTEGER NOT NULL,
    end_frame                   INTEGER NOT NULL,
    start_timestamp_seconds     REAL,
    end_timestamp_seconds       REAL,
    n_frames                    INTEGER NOT NULL,
    mean_prob_chinook           REAL,
    mean_prob_coho              REAL,
    mean_prob_atlantic          REAL,
    mean_prob_rainbow           REAL,
    mean_prob_brown             REAL,
    mean_prob_background        REAL,
    predicted_class             INTEGER,
    predicted_class_6           INTEGER,
    mean_detection_confidence   REAL,
    direction                   TEXT,
    entrance_side               TEXT,
    exit_side                   TEXT,
    representative_frame        INTEGER,
    PRIMARY KEY (video_id, track_id)
)
"""

PROCESSING_LOG_DDL: str = """
CREATE TABLE IF NOT EXISTS processing_log (
    video_id                        TEXT    PRIMARY KEY,
    status                          TEXT    NOT NULL,
    processed_at                    TEXT    NOT NULL,
    processing_duration_seconds     REAL,
    n_detections                    INTEGER,
    n_tracks                        INTEGER,
    error_message                   TEXT
)
"""

# Both tracks.sqlite and processing_log.sqlite include a meta table for
# schema_version and any future key/value metadata.
META_DDL: str = """
CREATE TABLE IF NOT EXISTS meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
)
"""

_META_INSERT_VERSION: str = (
    "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)"
)


def init_tracks_db(conn) -> None:
    """Create tracks and meta tables; stamp schema_version. Idempotent."""
    conn.execute(TRACKS_DDL)
    conn.execute(META_DDL)
    conn.execute(_META_INSERT_VERSION, (str(SCHEMA_VERSION),))
    conn.commit()


def init_processing_log_db(conn) -> None:
    """Create processing_log and meta tables; stamp schema_version. Idempotent."""
    conn.execute(PROCESSING_LOG_DDL)
    conn.execute(META_DDL)
    conn.execute(_META_INSERT_VERSION, (str(SCHEMA_VERSION),))
    conn.commit()


# ---------------------------------------------------------------------------
# Parquet filename derivation
# ---------------------------------------------------------------------------
def video_id_to_parquet_stem(video_id: str) -> str:
    """Return the Parquet filename stem for a given video_id.

    Uses the first 12 hex chars of SHA1(video_id). 12 chars gives ~4.8e14
    possible values; at 244k videos the birthday-collision probability is
    effectively zero (~1e-7). The user-facing example used 8 chars, but 8
    chars (~4.3e9 values) has near-certain collision at this scale.

    Example: "Ganaraska/Ganaraska 2020/1.mp4" -> "det_a0b1c2d3e4f5"
    """
    digest = hashlib.sha1(video_id.encode()).hexdigest()
    return f"det_{digest[:12]}"
