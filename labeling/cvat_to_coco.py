"""
Convert CVAT XML annotations (from old_data) to COCO-format JSON for
DETR fine-tuning.

Extracts keyframe bounding boxes from each annotated track, maps the
original D:\\Credit / D:\\Ganaraska paths to G:\\RiverWatcher equivalents,
extracts the corresponding video frames as JPEG images, and writes a
COCO annotations.json.

Usage:
  python labeling/cvat_to_coco.py
    --cvat    old_data
    --out     G:/Projects/cvat_dataset
    [--drive-map D:/Credit=G:/RiverWatcher/Credit D:/Ganaraska=G:/RiverWatcher/Ganaraska]
    [--skip-outside]      # skip boxes where outside=1 (default: True)
    [--keyframes-only]    # only use manually-drawn frames (default: True)

Output layout:
  G:/Projects/cvat_dataset/
    images/
      {task_id}_frame_{n}.jpg
    annotations.json     (COCO format)
    summary.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import cv2

# Species mapping — must match model/mar10_2026_detr-resnet50/config.json
SPECIES_MAP: Dict[str, int] = {
    "Chinook":      0,
    "Coho":         1,
    "Atlantic":     2,
    "Rainbow":      3,
    "Brown":        4,
    # Aliases
    "Rainbow Trout": 3,
    "Brown Trout":   4,
}
CATEGORIES = [
    {"id": 0, "name": "Chinook",      "supercategory": "fish"},
    {"id": 1, "name": "Coho",         "supercategory": "fish"},
    {"id": 2, "name": "Atlantic",     "supercategory": "fish"},
    {"id": 3, "name": "Rainbow Trout","supercategory": "fish"},
    {"id": 4, "name": "Brown Trout",  "supercategory": "fish"},
]


def main() -> None:
    args = _parse_args()

    # Drive mapping: original path prefix -> replacement prefix
    drive_map: Dict[str, str] = {}
    for spec in args.drive_map:
        src, _, dst = spec.partition("=")
        drive_map[src.replace("\\", "/")] = dst.replace("\\", "/")
    if not drive_map:
        drive_map = {
            "D:/Credit":    "G:/RiverWatcher/Credit",
            "D:/Ganaraska": "G:/RiverWatcher/Ganaraska",
        }

    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Parse CVAT XML (skip leading non-XML line if present)
    text = Path(args.cvat).read_text(encoding="utf-8")
    xml_start = text.index("<?xml")
    xml_end   = text.rfind("</annotations>") + len("</annotations>")
    root = ET.fromstring(text[xml_start:xml_end])

    # Build task_id -> (video_path, width, height, global_offset) index.
    # CVAT stores frame numbers as global indices across the whole project;
    # local_frame = global_frame - task_offset.
    # Offset accumulates as (size - overlap) per task in project order.
    task_meta: Dict[str, dict] = {}
    global_offset = 0
    for task in root.findall(".//task"):
        tid  = task.find("id").text
        name = task.find("name").text.strip('"').replace("\\", "/")
        for src, dst in drive_map.items():
            if name.startswith(src):
                name = dst + name[len(src):]
                break
        w       = int(task.find("original_size/width").text)
        h       = int(task.find("original_size/height").text)
        size    = int(task.find("size").text)
        overlap = int(task.find("overlap").text) if task.find("overlap") is not None else 0
        task_meta[tid] = {
            "path": name, "width": w, "height": h, "offset": global_offset
        }
        global_offset += size - overlap

    coco_images: List[dict]      = []
    coco_annotations: List[dict] = []
    image_id = ann_id = 0

    skipped_no_video = skipped_outside = skipped_species = 0
    extracted = 0

    for track in root.findall(".//track"):
        label = track.get("label", "")
        task_id = track.get("task_id", "")  # may not be present in all exports
        cat_id = SPECIES_MAP.get(label)
        if cat_id is None:
            skipped_species += 1
            continue  # Lamprey scar, Brook Trout, etc.

        # Find the task this track belongs to — track is nested under task in XML
        # If task_id attr isn't available, get it from parent
        task_elem = track.getparent() if hasattr(track, "getparent") else None
        if task_id == "":
            # Walk up to find task id via tree parent map
            pass

        boxes = track.findall("box")
        for box in boxes:
            if args.keyframes_only and box.get("keyframe") != "1":
                continue
            if args.skip_outside and box.get("outside") == "1":
                skipped_outside += 1
                continue

            global_frame = int(box.get("frame"))
            offset = (task_meta.get(_task_id_for_track(track, root), {})
                      .get("offset", 0))
            frame_n = global_frame - offset   # local frame index in video file
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            bw = xbr - xtl
            bh = ybr - ytl
            if bw <= 0 or bh <= 0:
                continue

            # Find video path — track task_id attribute not always present;
            # look at the task that contains this track's image range.
            video_path = _find_video_for_track(track, task_meta, root)
            if video_path is None or not Path(video_path).exists():
                skipped_no_video += 1
                continue

            img_fname = f"t{_task_id_for_track(track, root)}_f{frame_n:05d}.jpg"
            img_path  = img_dir / img_fname

            if not img_path.exists():
                ok = _extract_frame(video_path, frame_n, img_path)
                if not ok:
                    skipped_no_video += 1
                    continue
                extracted += 1

            meta = task_meta.get(_task_id_for_track(track, root), {})
            coco_images.append({
                "id": image_id,
                "file_name": str(img_path),
                "width": meta.get("width", 1280),
                "height": meta.get("height", 960),
                "video_path": video_path,
                "frame_number": frame_n,
                "global_frame": global_frame,
            })
            coco_annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [xtl, ytl, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            image_id += 1
            ann_id   += 1

    coco = {
        "info": {"description": "CVAT annotations converted from old_data",
                 "year": 2026},
        "categories": CATEGORIES,
        "images": coco_images,
        "annotations": coco_annotations,
    }
    ann_path = out_dir / "annotations.json"
    ann_path.write_text(json.dumps(coco, indent=2))

    # Summary
    from collections import Counter
    species_dist = Counter(
        CATEGORIES[a["category_id"]]["name"] for a in coco_annotations
    )
    summary = [
        f"CVAT -> COCO conversion summary",
        f"  Images extracted : {extracted:,}",
        f"  Total images     : {len(coco_images):,}",
        f"  Total annotations: {len(coco_annotations):,}",
        f"  Skipped (no video): {skipped_no_video:,}",
        f"  Skipped (outside) : {skipped_outside:,}",
        f"  Skipped (species) : {skipped_species:,}",
        f"",
        f"  Annotations by species:",
    ] + [f"    {sp}: {n}" for sp, n in sorted(species_dist.items(), key=lambda x: -x[1])]
    summary_text = "\n".join(summary)
    print(summary_text)
    (out_dir / "summary.txt").write_text(summary_text)
    print(f"\nWrote {ann_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_id_for_track(track, root) -> str:
    """Return the task id for this track — available directly as task_id attribute."""
    return track.get("task_id", "")


def _find_video_for_track(track, task_meta, root) -> str | None:
    tid = _task_id_for_track(track, root)
    meta = task_meta.get(tid)
    return meta["path"] if meta else None


def _extract_frame(video_path: str, frame_n: int, out_path: Path) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert CVAT XML to COCO JSON.")
    p.add_argument("--cvat", required=True, metavar="PATH",
                   help="Path to the CVAT export file (old_data).")
    p.add_argument("--out", required=True, metavar="DIR",
                   help="Output directory for images and annotations.json.")
    p.add_argument("--drive-map", nargs="+", default=[],
                   metavar="SRC=DST",
                   help="Path prefix remapping, e.g. D:/Credit=G:/RiverWatcher/Credit.")
    p.add_argument("--keyframes-only", action="store_true", default=True,
                   help="Only extract manually-drawn keyframes (default: True).")
    p.add_argument("--skip-outside", action="store_true", default=True,
                   help="Skip boxes where outside=1 (default: True).")
    return p.parse_args()


if __name__ == "__main__":
    main()
