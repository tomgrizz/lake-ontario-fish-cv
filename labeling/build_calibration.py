"""
Build the calibration task pool from CVAT annotations.

For each annotated track in the CVAT export:
  1. Finds the best frame (largest bbox area, outside=0, keyframe=1)
  2. Extracts a ~3-second clip centred on that frame
  3. Inserts a row into calibration_ground_truth in labels.sqlite

The calibration clips are stored alongside regular clips and referenced
by queue.py for injection into reviewer batches (at ~3% of batch size).
Reviewers see calibration tasks as identical to normal tasks; their
answer is compared to the CVAT ground truth by quality.py.

Usage:
  python labeling/build_calibration.py
    --cvat       old_data
    --labels     G:/Projects/labels/labels.sqlite
    --clips-dir  G:/Projects/clips/calibration
    [--drive-map D:/Credit=G:/RiverWatcher/Credit D:/Ganaraska=G:/RiverWatcher/Ganaraska]
    [--lead-in 1.5] [--lead-out 1.5]
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import open_labels_db

SPECIES_MAP = {
    "Chinook": "Chinook", "Coho": "Coho", "Atlantic": "Atlantic",
    "Rainbow": "Rainbow Trout", "Brown": "Brown Trout",
    "Rainbow Trout": "Rainbow Trout", "Brown Trout": "Brown Trout",
}
FISH_CLASSES = {"Chinook": 0, "Coho": 1, "Atlantic": 2, "Rainbow Trout": 3, "Brown Trout": 4}

BBOX_COLOR = (0, 255, 255)   # cyan
CLIP_HEIGHT = 360


def main() -> None:
    args = _parse_args()

    drive_map = {}
    for spec in args.drive_map:
        src, _, dst = spec.partition("=")
        drive_map[src.replace("\\", "/")] = dst.replace("\\", "/")
    if not drive_map:
        drive_map = {
            "D:/Credit":    "G:/RiverWatcher/Credit",
            "D:/Ganaraska": "G:/RiverWatcher/Ganaraska",
        }

    clips_dir = Path(args.clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    labels_db = Path(args.labels)
    labels_db.parent.mkdir(parents=True, exist_ok=True)
    conn = open_labels_db(labels_db)

    # Parse CVAT XML
    text = Path(args.cvat).read_text(encoding="utf-8")
    xml_start = text.index("<?xml")
    xml_end   = text.rfind("</annotations>") + len("</annotations>")
    root = ET.fromstring(text[xml_start:xml_end])

    # Build task metadata with global frame offsets
    task_meta = {}
    global_offset = 0
    for task in root.findall(".//task"):
        tid  = task.find("id").text
        name = task.find("name").text.strip('"').replace("\\", "/")
        for src, dst in drive_map.items():
            if name.startswith(src):
                name = dst + name[len(src):]
                break
        size    = int(task.find("size").text)
        overlap = int(task.find("overlap").text) if task.find("overlap") is not None else 0
        task_meta[tid] = {
            "path": name,
            "width": int(task.find("original_size/width").text),
            "height": int(task.find("original_size/height").text),
            "offset": global_offset,
        }
        global_offset += size - overlap

    inserted = skipped_no_video = skipped_species = skipped_existing = 0

    for track in root.findall(".//track"):
        label = track.get("label", "")
        species = SPECIES_MAP.get(label)
        if species is None:
            skipped_species += 1
            continue

        tid  = track.get("task_id", "")
        meta = task_meta.get(tid, {})
        video_path = meta.get("path", "")
        if not video_path or not Path(video_path).exists():
            skipped_no_video += 1
            continue

        offset = meta.get("offset", 0)

        # Find best frame: largest bbox area among keyframes with outside=0
        best_box = None
        best_area = -1.0
        for box in track.findall("box"):
            if box.get("outside") == "1" or box.get("keyframe") != "1":
                continue
            w = float(box.get("xbr")) - float(box.get("xtl"))
            h = float(box.get("ybr")) - float(box.get("ytl"))
            area = w * h
            if area > best_area:
                best_area = area
                best_box  = box

        if best_box is None:
            skipped_no_video += 1
            continue

        global_frame = int(best_box.get("frame"))
        local_frame  = global_frame - offset
        xtl = float(best_box.get("xtl"))
        ytl = float(best_box.get("ytl"))
        xbr = float(best_box.get("xbr"))
        ybr = float(best_box.get("ybr"))

        # Clip filename derived from video path + track id (stable, unique)
        key = video_path + "_track_" + track.get("id", "0")
        stem = "cal_" + hashlib.sha1(key.encode()).hexdigest()[:12]
        clip_path = clips_dir / f"{stem}.mp4"

        if not clip_path.exists():
            ok = _extract_clip(
                video_path, local_frame, clip_path,
                meta["width"], meta["height"],
                xtl, ytl, xbr, ybr, args.lead_in, args.lead_out,
            )
            if not ok:
                skipped_no_video += 1
                continue

        # Build a stable video_id (relative path from site root)
        video_id = _make_video_id(video_path, drive_map)

        # Insert into calibration_ground_truth (skip if already exists)
        try:
            conn.execute(
                """INSERT INTO calibration_ground_truth
                   (video_id, track_id, ground_truth_label, added_by, added_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    video_id,
                    int(track.get("id", 0)),
                    species,
                    "cvat_annotation",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            # Also store clip path in a metadata table for easy lookup
            conn.execute(
                """INSERT OR REPLACE INTO calibration_clips
                   (video_id, track_id, clip_path, best_frame,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (video_id, int(track.get("id", 0)), str(clip_path),
                 local_frame, int(xtl), int(ytl), int(xbr), int(ybr)),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            skipped_existing += 1

    conn.commit()
    conn.close()

    from collections import Counter
    # Summary
    conn2 = sqlite3.connect(str(labels_db))
    dist = dict(conn2.execute(
        "SELECT ground_truth_label, COUNT(*) FROM calibration_ground_truth GROUP BY ground_truth_label"
    ).fetchall())
    conn2.close()

    print(f"\nCalibration pool built: {args.clips_dir}")
    print(f"  Inserted  : {inserted}")
    print(f"  Existing  : {skipped_existing}")
    print(f"  No video  : {skipped_no_video}")
    print(f"  Unknown sp: {skipped_species}")
    print(f"\nPool by species:")
    for sp, n in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {sp}: {n}")


def _extract_clip(
    video_path, local_frame, clip_path, fw, fh,
    xtl, ytl, xbr, ybr, lead_in_sec, lead_out_sec,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    lead_in  = round(lead_in_sec  * fps)
    lead_out = round(lead_out_sec * fps)
    start = max(0, local_frame - lead_in)
    end   = min(total - 1, local_frame + lead_out)

    scale  = CLIP_HEIGHT / fh
    out_w  = int(fw * scale)
    tmp    = clip_path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (out_w, CLIP_HEIGHT))
    if not writer.isOpened():
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for fi in range(start, end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        # Draw bbox on the best frame only
        if fi == local_frame:
            x1, y1 = int(xtl), int(ytl)
            x2, y2 = int(xbr), int(ybr)
            cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, 2)
        resized = cv2.resize(frame, (out_w, CLIP_HEIGHT), interpolation=cv2.INTER_AREA)
        writer.write(resized)
    cap.release()
    writer.release()

    # Re-encode to H.264
    ffmpeg = "ffmpeg"
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(tmp),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-an", str(clip_path)],
            check=True, capture_output=True,
        )
        tmp.unlink(missing_ok=True)
    except Exception:
        tmp.rename(clip_path)
    return True


def _make_video_id(video_path: str, drive_map: Dict[str, str]) -> str:
    p = video_path.replace("\\", "/")
    for src, dst in drive_map.items():
        # Convert back to relative: strip the G:/RiverWatcher/{Site} prefix
        site_root = dst  # e.g. G:/RiverWatcher/Credit
        if p.startswith(site_root):
            site = src.split("/")[-1]  # "Credit" or "Ganaraska"
            rel  = p[len(site_root):].lstrip("/")
            return site + "/" + rel
    return video_path


def _ensure_calibration_clips_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_clips (
            video_id     TEXT NOT NULL,
            track_id     INTEGER NOT NULL,
            clip_path    TEXT NOT NULL,
            best_frame   INTEGER,
            bbox_x1      INTEGER, bbox_y1 INTEGER,
            bbox_x2      INTEGER, bbox_y2 INTEGER,
            PRIMARY KEY (video_id, track_id)
        )
    """)
    conn.commit()


# Patch open_labels_db to also create calibration_clips
_orig_open = open_labels_db
def open_labels_db(path):  # noqa: F811
    conn = _orig_open(path)
    _ensure_calibration_clips_table(conn)
    return conn


def _parse_args():
    p = argparse.ArgumentParser(description="Build calibration task pool from CVAT annotations.")
    p.add_argument("--cvat",      required=True, metavar="PATH")
    p.add_argument("--labels",    required=True, metavar="PATH",
                   help="labels.sqlite (calibration_ground_truth table updated in place)")
    p.add_argument("--clips-dir", required=True, metavar="DIR")
    p.add_argument("--drive-map", nargs="+", default=[], metavar="SRC=DST")
    p.add_argument("--lead-in",   type=float, default=1.5)
    p.add_argument("--lead-out",  type=float, default=1.5)
    return p.parse_args()


if __name__ == "__main__":
    main()
