"""
Post-inference clip extraction.

Walks tracks.sqlite, computes best_id_frame from per-detection Parquet data,
extracts ~3-second annotated clips centred on that frame, and writes results
back to tracks.sqlite.

Usage:
  python extract_clips.py
    --tracks   G:/Projects/model_outputs/run_20260428/tracks.sqlite
    --parquet  G:/Projects/model_outputs/run_20260428/detections/
    --video-roots Ganaraska=G:/RiverWatcher/Ganaraska Credit=G:/RiverWatcher/Credit
    --clips-dir  G:/Projects/clips
    [--lead-in 1.5]   # seconds before best_id_frame (default 1.5)
    [--lead-out 1.5]  # seconds after best_id_frame (default 1.5)
    [--limit N]       # process only N tracks (for testing)
    [--workers N]     # parallel ffmpeg workers (default 4, CPU-bound)

Output layout under --clips-dir:
  {clip_stem}.mp4       where clip_stem = "clip_" + sha1(video_id+str(track_id))[:12]

tracks.sqlite gains four new columns (added if absent, idempotent):
  best_id_frame      INTEGER
  clip_start_frame   INTEGER
  clip_end_frame     INTEGER
  clip_path          TEXT     (absolute path to the clip MP4)
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from schema import video_id_to_parquet_stem
from scoring import compute_best_id_frame

FISH_CLASSES = {0: "Chinook", 1: "Coho", 2: "Atlantic", 3: "Rainbow Trout", 4: "Brown Trout"}
BACKGROUND_CLASS_IDX = 5

# Bbox overlay style
BBOX_COLOR_BGR = (0, 255, 255)   # cyan
BBOX_THICKNESS = 2
LABEL_SCALE = 0.45
LABEL_THICKNESS = 1

# Target clip resolution (height; width scales proportionally)
CLIP_HEIGHT = 360


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    tracks_db = Path(args.tracks)
    parquet_dir = Path(args.parquet)
    clips_dir = Path(args.clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    video_roots: Dict[str, str] = {}
    for spec in args.video_roots:
        name, _, path = spec.partition("=")
        video_roots[name.strip()] = path.strip()

    # Add new columns to tracks.sqlite if they don't exist yet (idempotent).
    _ensure_columns(tracks_db)

    # Load tracks that still need clips (clip_path IS NULL).
    tracks = _load_pending_tracks(tracks_db, limit=args.limit)
    print(f"Tracks needing clips: {len(tracks):,}")

    if not tracks:
        print("Nothing to do.")
        return

    # Disk space check before starting.
    _check_disk(clips_dir, len(tracks), args)

    # Process tracks (threaded ffmpeg extract, but clip rendering is per-thread).
    n_done = n_error = 0
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _process_track,
                row,
                parquet_dir,
                video_roots,
                clips_dir,
                args.lead_in,
                args.lead_out,
            ): row
            for row in tracks
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                if result is not None:
                    _update_track(tracks_db, result)
                    n_done += 1
                else:
                    n_error += 1
            except Exception as exc:
                print(f"ERROR {row['video_id']} track {row['track_id']}: {exc}")
                n_error += 1

            total = n_done + n_error
            if total % 100 == 0 or total == len(tracks):
                elapsed = time.monotonic() - t0
                fps = total / elapsed if elapsed > 0 else 0
                print(f"[{total}/{len(tracks)}] done={n_done} err={n_error} "
                      f"rate={fps:.2f} tracks/sec")

    elapsed = time.monotonic() - t0
    total_gb = sum(p.stat().st_size for p in clips_dir.glob("*.mp4")) / 1e9
    print(f"\nDone in {elapsed/60:.1f}m. "
          f"Success: {n_done:,}  Error: {n_error:,}  "
          f"Total clip storage: {total_gb:.2f} GB")


# ---------------------------------------------------------------------------
# Per-track processing
# ---------------------------------------------------------------------------

def _process_track(
    row: dict,
    parquet_dir: Path,
    video_roots: Dict[str, str],
    clips_dir: Path,
    lead_in: float,
    lead_out: float,
) -> Optional[dict]:
    """Compute best_id_frame, extract and annotate clip. Returns update dict or None."""
    video_id = row["video_id"]
    track_id = row["track_id"]

    # Locate source video.
    video_path = _resolve_video_path(video_id, video_roots)
    if video_path is None or not Path(video_path).is_file():
        print(f"  SKIP: source video not found for {video_id}")
        return None

    # Read per-track detections from Parquet.
    parquet_stem = video_id_to_parquet_stem(video_id)
    parquet_file = parquet_dir / f"{parquet_stem}.parquet"
    if not parquet_file.exists():
        print(f"  SKIP: parquet not found: {parquet_file.name}")
        return None

    tbl = pq.read_table(parquet_file,
                        filters=[("track_id", "=", track_id)])
    if len(tbl) == 0:
        print(f"  SKIP: no detections for track {track_id} in {parquet_file.name}")
        return None

    det_df = tbl.to_pandas()

    # Get frame dimensions from the source video.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    best_id_frame = compute_best_id_frame(det_df, fw, fh)

    lead_in_frames = round(lead_in * fps)
    lead_out_frames = round(lead_out * fps)
    clip_start = max(row["start_frame"], best_id_frame - lead_in_frames)
    clip_end = min(row["end_frame"], best_id_frame + lead_out_frames)

    clip_stem = _clip_stem(video_id, track_id)
    clip_path = clips_dir / f"{clip_stem}.mp4"

    # Extract and annotate clip.
    ok = _extract_annotated_clip(
        video_path=video_path,
        clip_path=clip_path,
        start_frame=clip_start,
        end_frame=clip_end,
        fps=fps,
        fw=fw,
        fh=fh,
        det_df=det_df,
    )
    if not ok:
        return None

    return {
        "video_id": video_id,
        "track_id": track_id,
        "best_id_frame": best_id_frame,
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "clip_path": str(clip_path),
    }


def _extract_annotated_clip(
    video_path: str,
    clip_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    fw: int,
    fh: int,
    det_df: pd.DataFrame,
) -> bool:
    """Read frames start_frame..end_frame, draw bbox overlay, write 360p MP4.

    Bboxes are linearly interpolated between detected frames so the overlay
    is smooth even with frame_skip > 1. Without interpolation the bbox
    flickers on/off every N frames.
    """
    det_by_frame = {
        int(r["frame_number"]): r for _, r in det_df.iterrows()
    }
    # Pre-compute interpolated bbox for every frame in the clip range.
    det_frames = sorted(det_by_frame.keys())
    interp_bbox: dict[int, tuple[int, int, int, int]] = {}
    if det_frames:
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx in det_by_frame:
                r = det_by_frame[frame_idx]
                interp_bbox[frame_idx] = (
                    int(r["bbox_x1"]), int(r["bbox_y1"]),
                    int(r["bbox_x2"]), int(r["bbox_y2"]),
                )
            else:
                # Find surrounding detected frames and lerp.
                prev = [f for f in det_frames if f <= frame_idx]
                nxt  = [f for f in det_frames if f >  frame_idx]
                if prev and nxt:
                    f0, f1 = prev[-1], nxt[0]
                    t = (frame_idx - f0) / (f1 - f0)
                    r0, r1 = det_by_frame[f0], det_by_frame[f1]
                    interp_bbox[frame_idx] = (
                        int(r0["bbox_x1"] + t * (r1["bbox_x1"] - r0["bbox_x1"])),
                        int(r0["bbox_y1"] + t * (r1["bbox_y1"] - r0["bbox_y1"])),
                        int(r0["bbox_x2"] + t * (r1["bbox_x2"] - r0["bbox_x2"])),
                        int(r0["bbox_y2"] + t * (r1["bbox_y2"] - r0["bbox_y2"])),
                    )
                elif prev:
                    r0 = det_by_frame[prev[-1]]
                    interp_bbox[frame_idx] = (
                        int(r0["bbox_x1"]), int(r0["bbox_y1"]),
                        int(r0["bbox_x2"]), int(r0["bbox_y2"]),
                    )
                elif nxt:
                    r1 = det_by_frame[nxt[0]]
                    interp_bbox[frame_idx] = (
                        int(r1["bbox_x1"]), int(r1["bbox_y1"]),
                        int(r1["bbox_x2"]), int(r1["bbox_y2"]),
                    )

    # Scale factor for 360p output.
    scale = CLIP_HEIGHT / fh
    out_w = int(fw * scale)
    out_h = CLIP_HEIGHT

    tmp_path = clip_path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        return False

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    try:
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break

            # Draw interpolated bbox overlay on every frame.
            if frame_idx in interp_bbox:
                x1, y1, x2, y2 = interp_bbox[frame_idx]
                cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR_BGR, BBOX_THICKNESS)

                # Species label only on frames with actual detections.
                if frame_idx in det_by_frame:
                    det = det_by_frame[frame_idx]
                    cls_idx = int(det["predicted_class"])
                    label = FISH_CLASSES.get(cls_idx, "?")
                    conf = float(det["detection_confidence"])
                    text = f"{label} {conf:.2f}"
                    cv2.putText(frame, text, (x1 + 3, y1 + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE,
                                BBOX_COLOR_BGR, LABEL_THICKNESS, cv2.LINE_AA)

            # Resize to 360p.
            resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(resized)
    finally:
        cap.release()
        writer.release()

    # Re-encode with ffmpeg for h264 + proper MP4 container.
    # Prefer system ffmpeg; fall back to imageio_ffmpeg bundle so this works
    # even without a system ffmpeg installation.
    ffmpeg_cmd = "ffmpeg"
    try:
        import imageio_ffmpeg
        ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    try:
        subprocess.run(
            [ffmpeg_cmd, "-y", "-i", str(tmp_path),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-an", str(clip_path)],
            check=True,
            capture_output=True,
        )
        tmp_path.unlink(missing_ok=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg failed — keep raw file but warn (browser may not play mp4v).
        print(f"  WARN: ffmpeg re-encode failed for {clip_path.name}, keeping raw mp4v")
        tmp_path.rename(clip_path)
        return True


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _ensure_columns(db_path: Path) -> None:
    """Add the four new columns to tracks if they don't exist. Idempotent."""
    conn = sqlite3.connect(str(db_path))
    existing = {row[1] for row in conn.execute("PRAGMA table_info(tracks)").fetchall()}
    new_cols = [
        ("best_id_frame",    "INTEGER"),
        ("clip_start_frame", "INTEGER"),
        ("clip_end_frame",   "INTEGER"),
        ("clip_path",        "TEXT"),
    ]
    for col, dtype in new_cols:
        if col not in existing:
            conn.execute(f"ALTER TABLE tracks ADD COLUMN {col} {dtype}")
    conn.commit()
    conn.close()


def _load_pending_tracks(db_path: Path, limit: Optional[int]) -> List[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    sql = "SELECT * FROM tracks WHERE clip_path IS NULL"
    if limit:
        sql += f" LIMIT {limit}"
    rows = [dict(r) for r in conn.execute(sql).fetchall()]
    conn.close()
    return rows


def _update_track(db_path: Path, result: dict) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """UPDATE tracks SET
             best_id_frame=?, clip_start_frame=?, clip_end_frame=?, clip_path=?
           WHERE video_id=? AND track_id=?""",
        (result["best_id_frame"], result["clip_start_frame"],
         result["clip_end_frame"], result["clip_path"],
         result["video_id"], result["track_id"]),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip_stem(video_id: str, track_id: int) -> str:
    key = video_id + str(track_id)
    return "clip_" + hashlib.sha1(key.encode()).hexdigest()[:12]


def _resolve_video_path(video_id: str, video_roots: Dict[str, str]) -> Optional[str]:
    site = video_id.split("/")[0]
    if site not in video_roots:
        return None
    tail = "/".join(video_id.split("/")[1:])
    return str(Path(video_roots[site]) / tail)


def _check_disk(clips_dir: Path, n_tracks: int, args) -> None:
    """Extrapolate from a small sample and warn if disk is tight."""
    sample_size = min(20, n_tracks)
    if sample_size == 0:
        return
    # Estimate ~750 KB per clip as baseline (360p h264 ~3s).
    estimate_bytes_per_clip = 750 * 1024
    extrapolated_gb = (estimate_bytes_per_clip * n_tracks) / 1e9
    free_gb = shutil.disk_usage(clips_dir).free / 1e9
    ratio = free_gb / extrapolated_gb if extrapolated_gb > 0 else float("inf")
    status = "SUFFICIENT" if ratio >= 1.5 else "WARNING"
    print(f"[Disk] Extrapolated clips: {extrapolated_gb:.1f} GB  "
          f"| Free: {free_gb:.1f} GB  -> {status}")
    if status == "WARNING":
        print("ERROR: Insufficient disk space for clips. Aborting.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract annotated clips for Label Studio review."
    )
    p.add_argument("--tracks", required=True, metavar="PATH",
                   help="Path to tracks.sqlite from inference run.")
    p.add_argument("--parquet", required=True, metavar="DIR",
                   help="Directory containing per-video detection Parquet files.")
    p.add_argument("--video-roots", nargs="+", metavar="NAME=PATH", default=[],
                   help="Site name and root path, e.g. Ganaraska=G:/RiverWatcher/Ganaraska.")
    p.add_argument("--clips-dir", required=True, metavar="DIR",
                   help="Output directory for extracted clip MP4s.")
    p.add_argument("--lead-in", type=float, default=1.5, metavar="SEC",
                   help="Seconds before best_id_frame (default 1.5).")
    p.add_argument("--lead-out", type=float, default=1.5, metavar="SEC",
                   help="Seconds after best_id_frame (default 1.5).")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Process only N tracks (for testing).")
    p.add_argument("--workers", type=int, default=4, metavar="N",
                   help="Parallel workers for clip extraction (default 4).")
    return p.parse_args()


if __name__ == "__main__":
    main()
