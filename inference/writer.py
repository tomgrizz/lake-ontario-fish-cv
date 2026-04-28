"""
Output writers for inference pipeline v2.

Three public functions:
  write_detections()   — atomic Parquet write for one video's detections
  flush_tracks()       — bulk-insert TrackSummary rows into tracks.sqlite
  log_video()          — upsert one row into processing_log.sqlite
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from schema import (
    DETECTION_SCHEMA,
    SCHEMA_VERSION,
    DetectionRecord,
    TrackSummary,
    init_processing_log_db,
    init_tracks_db,
    video_id_to_parquet_stem,
)


# ---------------------------------------------------------------------------
# Parquet — per-detection
# ---------------------------------------------------------------------------

def write_detections(
    video_id: str,
    records: List[DetectionRecord],
    detections_dir: Path,
) -> Path:
    """Write detection records for one video to a Parquet file atomically.

    Writes to a .tmp file first, then renames on success so a crash mid-write
    never leaves a partial file in the output directory.

    Args:
        video_id: Relative video path used as the key in the Parquet filename.
        records:  Per-detection records for this video (may be empty).
        detections_dir: Directory to write Parquet files into.

    Returns:
        Path to the written Parquet file.
    """
    stem = video_id_to_parquet_stem(video_id)
    final_path = detections_dir / f"{stem}.parquet"
    tmp_path = detections_dir / f"{stem}.parquet.tmp"

    if records:
        table = _records_to_table(records)
    else:
        table = pa.table({field.name: pa.array([], type=field.type)
                          for field in DETECTION_SCHEMA},
                         schema=DETECTION_SCHEMA)

    detections_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, final_path)
    return final_path


def _records_to_table(records: List[DetectionRecord]) -> pa.Table:
    """Convert a list of DetectionRecord dataclass instances to a PyArrow table."""
    cols: dict[str, list] = {field.name: [] for field in DETECTION_SCHEMA}
    for r in records:
        cols["video_id"].append(r.video_id)
        cols["frame_number"].append(r.frame_number)
        cols["timestamp_seconds"].append(r.timestamp_seconds)
        cols["track_id"].append(r.track_id)
        cols["detection_id"].append(r.detection_id)
        cols["bbox_x1"].append(r.bbox_x1)
        cols["bbox_y1"].append(r.bbox_y1)
        cols["bbox_x2"].append(r.bbox_x2)
        cols["bbox_y2"].append(r.bbox_y2)
        cols["detection_confidence"].append(r.detection_confidence)
        cols["prob_chinook"].append(r.prob_chinook)
        cols["prob_coho"].append(r.prob_coho)
        cols["prob_atlantic"].append(r.prob_atlantic)
        cols["prob_rainbow"].append(r.prob_rainbow)
        cols["prob_brown"].append(r.prob_brown)
        cols["prob_background"].append(r.prob_background)
        cols["predicted_class"].append(r.predicted_class)
        cols["predicted_class_6"].append(r.predicted_class_6)

    arrays = [
        pa.array(cols[field.name], type=field.type)
        for field in DETECTION_SCHEMA
    ]
    return pa.table(
        {field.name: arr for field, arr in zip(DETECTION_SCHEMA, arrays)},
        schema=DETECTION_SCHEMA,
    )


# ---------------------------------------------------------------------------
# SQLite — per-track
# ---------------------------------------------------------------------------

def flush_tracks(
    summaries: List[TrackSummary],
    db_path: Path,
) -> None:
    """Bulk-insert TrackSummary rows into tracks.sqlite.

    Creates and initialises the database on first call. Subsequent calls
    append; existing (video_id, track_id) pairs are left unchanged (INSERT OR
    IGNORE) so replaying a checkpoint is safe.

    Args:
        summaries: Track summaries to insert (may be empty — no-op).
        db_path:   Path to tracks.sqlite.
    """
    if not summaries:
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _open_wal(db_path)
    try:
        init_tracks_db(conn)
        conn.executemany(_TRACKS_INSERT_SQL, [_track_row(t) for t in summaries])
        conn.commit()
    finally:
        conn.close()


_TRACKS_INSERT_SQL: str = """
INSERT OR IGNORE INTO tracks (
    video_id, track_id,
    start_frame, end_frame,
    start_timestamp_seconds, end_timestamp_seconds,
    n_frames,
    mean_prob_chinook, mean_prob_coho, mean_prob_atlantic,
    mean_prob_rainbow, mean_prob_brown, mean_prob_background,
    predicted_class, predicted_class_6,
    mean_detection_confidence,
    direction, entrance_side, exit_side,
    representative_frame
) VALUES (
    ?, ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?, ?,
    ?, ?,
    ?,
    ?, ?, ?,
    ?
)
"""


def _track_row(t: TrackSummary) -> tuple:
    return (
        t.video_id, t.track_id,
        t.start_frame, t.end_frame,
        t.start_timestamp_seconds, t.end_timestamp_seconds,
        t.n_frames,
        t.mean_prob_chinook, t.mean_prob_coho, t.mean_prob_atlantic,
        t.mean_prob_rainbow, t.mean_prob_brown, t.mean_prob_background,
        t.predicted_class, t.predicted_class_6,
        t.mean_detection_confidence,
        t.direction, t.entrance_side, t.exit_side,
        t.representative_frame,
    )


# ---------------------------------------------------------------------------
# SQLite — processing log
# ---------------------------------------------------------------------------

def log_video(
    video_id: str,
    status: str,
    db_path: Path,
    processing_duration_seconds: Optional[float] = None,
    n_detections: Optional[int] = None,
    n_tracks: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """Upsert one row into processing_log.sqlite.

    status must be one of: 'success', 'error', 'skipped'.
    Uses INSERT OR REPLACE so re-running a video (e.g. with --retry-errors)
    overwrites the previous entry.

    Args:
        video_id:     Relative video path (primary key).
        status:       'success' | 'error' | 'skipped'
        db_path:      Path to processing_log.sqlite.
        processing_duration_seconds: Wall time to process the video.
        n_detections: Total detections written (None on error/skipped).
        n_tracks:     Total tracks written (None on error/skipped).
        error_message: Exception string (None on success/skipped).
    """
    assert status in ("success", "error", "skipped"), f"Invalid status: {status!r}"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _open_wal(db_path)
    try:
        init_processing_log_db(conn)
        conn.execute(
            _LOG_UPSERT_SQL,
            (
                video_id,
                status,
                datetime.now(timezone.utc).isoformat(),
                processing_duration_seconds,
                n_detections,
                n_tracks,
                error_message,
            ),
        )
        conn.commit()
    finally:
        conn.close()


_LOG_UPSERT_SQL: str = """
INSERT OR REPLACE INTO processing_log (
    video_id, status, processed_at,
    processing_duration_seconds, n_detections, n_tracks, error_message
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_wal(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
