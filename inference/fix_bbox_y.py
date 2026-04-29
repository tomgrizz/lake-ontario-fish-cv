"""
Post-hoc bbox Y-coordinate correction for Parquet detections produced
before the target_sizes letterbox bug was fixed in pipeline.py.

Bug: pipeline.py was calling DETR's post_process_object_detection with
target_sizes=(H, W). The DETR image processor letterboxes every frame
into an 800x800 padded square, so the model's normalized output is
relative to that padded canvas — target_sizes must equal
padded_size / resize_scale, which for landscape 1280x960 is (1280, 1280),
not (960, 1280). Y was therefore under-scaled by 0.75 (= 960 / 1280),
shifting every detection's Y up by 25 percent of fish height. X was
correct because the resize was width-limited.

Correction: multiply bbox_y1, bbox_y2 by max(W,H) / H (= 4/3 for
1280x960) and clamp to [0, H]. Same pattern handles portrait video
with (X, max(W,H)/W) — only the buggy axis differs.

Resumability: each Parquet's Arrow schema metadata gets a
`y_coord_fix_applied = <iso ts>` entry on rewrite. Re-runs skip
already-stamped files, so an interrupted run can be resumed simply
by re-invoking the same command.

This script does NOT touch:
  * Track-level summaries in tracks.sqlite (direction / sides / probs
    are independent of bbox coords).
  * Already-rendered clip MP4s (the bbox overlays are burned in pixels;
    use --null-clip-paths to clear clip_path so extract_clips.py
    re-renders, which is the proper way to repair them).

Usage:
  python inference/fix_bbox_y.py
    --run-dir       PATH/run_<timestamp>
    [--frame-width  1280]
    [--frame-height 960]
    [--pad-size     800]
    [--null-clip-paths]      # also clear clip_* columns so re-running
                             #   extract_clips.py regenerates every clip
    [--limit N]              # only process N parquet files (testing)
"""
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FILE_META_KEY = b"y_coord_fix_applied"
DB_META_KEY = "y_coord_fix_applied"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    det_dir = run_dir / "detections"
    tracks_db = run_dir / "tracks.sqlite"
    if not det_dir.is_dir():
        sys.exit(f"detections/ not found: {det_dir}")
    if not tracks_db.is_file():
        sys.exit(f"tracks.sqlite not found: {tracks_db}")

    factors = _compute_factors(args.frame_width, args.frame_height, args.pad_size)
    if factors["y_factor"] == 1.0 and factors["x_factor"] == 1.0:
        sys.exit(
            f"No correction needed: {args.frame_width}x{args.frame_height} "
            f"with pad {args.pad_size} produces y_factor=1.0 and x_factor=1.0."
        )
    print(_summary(args, factors))

    prev = _get_db_meta(tracks_db, DB_META_KEY)
    if prev:
        print(f"NOTE: tracks.sqlite already carries '{DB_META_KEY}={prev}'. "
              "Per-file markers still gate whether each Parquet gets rewritten; "
              "re-running this command will skip already-fixed files.")

    files = sorted(det_dir.glob("*.parquet"))
    if args.limit is not None:
        files = files[:args.limit]
    print(f"Found {len(files)} parquet files in {det_dir}")
    if not files:
        return

    n_fixed = n_skipped = n_errors = 0
    n_rows_corrected = 0
    t0 = dt.datetime.now()
    for i, path in enumerate(files):
        try:
            result, rows = _fix_one(
                path,
                factors=factors,
                frame_height=args.frame_height,
                frame_width=args.frame_width,
            )
            if result == "fixed":
                n_fixed += 1
                n_rows_corrected += rows
            else:
                n_skipped += 1
        except Exception as exc:
            n_errors += 1
            print(f"ERROR {path.name}: {exc}", file=sys.stderr)

        is_last = (i + 1) == len(files)
        if (i + 1) % 500 == 0 or is_last:
            elapsed = (dt.datetime.now() - t0).total_seconds()
            rate = (i + 1) / elapsed if elapsed else 0
            print(
                f"  [{i+1:>6}/{len(files)}] "
                f"fixed={n_fixed} skipped={n_skipped} err={n_errors}  "
                f"rows_corrected={n_rows_corrected:,}  ({rate:.1f} files/s)"
            )

    if n_fixed > 0:
        ts = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        _set_db_meta(tracks_db, DB_META_KEY, ts)
        print(f"Stamped tracks.sqlite.meta['{DB_META_KEY}'] = {ts}")

    if args.null_clip_paths:
        n = _null_clip_paths(tracks_db)
        print(f"Cleared clip_path/clip_*/best_id_frame for {n} tracks. "
              "Run extract_clips.py to regenerate clips with corrected overlays.")

    print(f"\nDone. fixed={n_fixed}  skipped={n_skipped}  errors={n_errors}  "
          f"rows_corrected={n_rows_corrected:,}")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _compute_factors(W: int, H: int, pad: int) -> dict:
    """Per-axis correction factor.

    Buggy code passed target_sizes=(H, W). Correct value is
    (round(pad/scale), round(pad/scale)) where scale = min(pad/W, pad/H).
    For inputs >= pad in at least one dim, that simplifies to max(W, H).
    """
    scale = min(pad / W, pad / H)
    target = round(pad / scale)
    return {
        "scale":    scale,
        "y_factor": target / H,
        "x_factor": target / W,
    }


def _summary(args, factors) -> str:
    return (
        f"Frame {args.frame_width}x{args.frame_height} pad {args.pad_size}\n"
        f"  resize_scale={factors['scale']:.4f}\n"
        f"  x_factor={factors['x_factor']:.6f}  y_factor={factors['y_factor']:.6f}"
    )


def _fix_one(
    path: Path,
    factors: dict,
    frame_height: int,
    frame_width: int,
) -> tuple[str, int]:
    """Rewrite one Parquet file in place. Returns (status, n_rows_corrected).

    status is 'fixed' or 'skipped'. Skipped means file already carries the
    y_coord_fix marker; no rewrite occurred.
    """
    # Use read_schema (one-shot) rather than ParquetFile so Windows
    # releases the handle before _atomic_write tries to replace the file.
    schema = pq.read_schema(path)
    md = dict(schema.metadata or {})
    if FILE_META_KEY in md:
        return ("skipped", 0)

    table = pq.read_table(path)
    n = len(table)

    if n > 0:
        if factors["y_factor"] != 1.0:
            new_y1 = _scale_clamp(
                table.column("bbox_y1").to_numpy(),
                factors["y_factor"], 0, frame_height,
            )
            new_y2 = _scale_clamp(
                table.column("bbox_y2").to_numpy(),
                factors["y_factor"], 0, frame_height,
            )
            table = (
                table
                .set_column(table.schema.get_field_index("bbox_y1"),
                            "bbox_y1", pa.array(new_y1, type=pa.int16()))
                .set_column(table.schema.get_field_index("bbox_y2"),
                            "bbox_y2", pa.array(new_y2, type=pa.int16()))
            )
        if factors["x_factor"] != 1.0:
            new_x1 = _scale_clamp(
                table.column("bbox_x1").to_numpy(),
                factors["x_factor"], 0, frame_width,
            )
            new_x2 = _scale_clamp(
                table.column("bbox_x2").to_numpy(),
                factors["x_factor"], 0, frame_width,
            )
            table = (
                table
                .set_column(table.schema.get_field_index("bbox_x1"),
                            "bbox_x1", pa.array(new_x1, type=pa.int16()))
                .set_column(table.schema.get_field_index("bbox_x2"),
                            "bbox_x2", pa.array(new_x2, type=pa.int16()))
            )

    md[FILE_META_KEY] = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ").encode()
    table = table.replace_schema_metadata(md)
    _atomic_write(table, path)
    return ("fixed", n)


def _scale_clamp(arr: np.ndarray, factor: float,
                 lo: int, hi: int) -> np.ndarray:
    scaled = arr.astype(np.float32) * factor
    return np.clip(np.round(scaled), lo, hi).astype(np.int16)


def _atomic_write(table: pa.Table, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(dst)


# ---------------------------------------------------------------------------
# tracks.sqlite helpers
# ---------------------------------------------------------------------------

def _get_db_meta(db: Path, key: str) -> Optional[str]:
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _set_db_meta(db: Path, key: str, value: str) -> None:
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def _null_clip_paths(db: Path) -> int:
    conn = sqlite3.connect(str(db))
    try:
        cur = conn.execute(
            "UPDATE tracks SET "
            "clip_path=NULL, best_id_frame=NULL, "
            "clip_start_frame=NULL, clip_end_frame=NULL "
            "WHERE clip_path IS NOT NULL"
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-hoc bbox Y correction for pre-fix Parquet output."
    )
    p.add_argument("--run-dir", required=True, metavar="PATH",
                   help="Run directory containing detections/ and tracks.sqlite.")
    p.add_argument("--frame-width",  type=int, default=1280, metavar="W",
                   help="Original frame width  (default 1280, RiverWatcher).")
    p.add_argument("--frame-height", type=int, default=960,  metavar="H",
                   help="Original frame height (default 960,  RiverWatcher).")
    p.add_argument("--pad-size",     type=int, default=800,  metavar="P",
                   help="DETR processor pad_size (default 800, "
                        "from preprocessor_config.json).")
    p.add_argument("--null-clip-paths", action="store_true",
                   help="Clear clip_path / clip_*_frame / best_id_frame in "
                        "tracks.sqlite so a follow-up extract_clips.py "
                        "regenerates every clip with corrected overlays.")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Process at most N Parquet files (for testing).")
    return p.parse_args()


if __name__ == "__main__":
    main()
