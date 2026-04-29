"""
Build a Label Studio reviewer batch from a queue.json artifact.

Reads:
  * queue.json  produced by queue.py
  * tracks.sqlite  (must have clip_path populated by extract_clips.py)
  * each track's source clip mp4 referenced by clip_path

Writes:
  reviewer_batch_<batch_id>/
    clips/<basename of each clip>
    tasks.json    (Label Studio import format, per task_config.xml)
    setup.md      (env-var + start instructions for the reviewer)
    skipped.json  (audit log of items dropped for missing tracks/clips)

The output folder is the unit a reviewer carries to their machine
("sneakernet" model, per docs/labeling_plan.md).

LS-test-verified field handling (2026-04-29 against LS 1.23 Community):
  * data._-prefixed fields round-trip through JSON export.
  * predictions[0].model_version and predictions[0].score also round-trip,
    so _model_version and _original_confidence are NOT included in data.
  * The seven retained data._* fields are needed by export_labels.py.

Run:
  python labeling/import_tasks.py
    --queue          PATH/queue.json
    --tracks-db      PATH/tracks.sqlite
    --out            PATH/reviewer_batch_<id>
    --model-version  mar10_2026_detr-resnet50
    [--ls-base       http://localhost:8080/data/local-files/?d=]
    [--no-copy-clips]   # leave clips in place; useful for dry-run
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_LS_BASE = "http://localhost:8080/data/local-files/?d="

# Match inference/schema.py.
FISH_CLASSES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]
BACKGROUND_CLASS_IDX = 5
NOT_A_FISH_CHOICE = "Not a fish"   # task_config.xml hotkey 'n'

# Track row columns we read from tracks.sqlite (post extract_clips.py).
_TRACK_COLS = (
    "video_id", "track_id",
    "mean_prob_chinook", "mean_prob_coho", "mean_prob_atlantic",
    "mean_prob_rainbow", "mean_prob_brown", "mean_prob_background",
    "mean_detection_confidence",
    "n_frames", "direction", "predicted_class", "predicted_class_6",
    "clip_path",
)


@dataclass
class _BuildResult:
    tasks: list
    skipped: list


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    queue_path = Path(args.queue)
    tracks_db = Path(args.tracks_db)
    out_dir = Path(args.out)

    if not queue_path.is_file():
        sys.exit(f"queue.json not found: {queue_path}")
    if not tracks_db.is_file():
        sys.exit(f"tracks.sqlite not found: {tracks_db}")

    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    if "items" not in queue or "batch_id" not in queue:
        sys.exit(f"queue.json missing 'items' or 'batch_id': {queue_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    clips_out = out_dir / "clips"
    clips_out.mkdir(exist_ok=True)

    conn = sqlite3.connect(str(tracks_db))
    conn.row_factory = sqlite3.Row

    result = _build_tasks(
        queue=queue,
        conn=conn,
        clips_out=clips_out,
        ls_base=args.ls_base,
        model_version=args.model_version,
        copy_clips=not args.no_copy_clips,
    )

    (out_dir / "tasks.json").write_text(
        json.dumps(result.tasks, indent=2),
        encoding="utf-8",
    )
    (out_dir / "skipped.json").write_text(
        json.dumps(result.skipped, indent=2),
        encoding="utf-8",
    )
    _write_setup_md(out_dir, queue=queue)

    print(
        f"Batch '{queue['batch_id']}' -> {out_dir}\n"
        f"  reviewer_id: {queue.get('reviewer_id')}  "
        f"phase: {queue.get('phase')}\n"
        f"  tasks emitted: {len(result.tasks)}  "
        f"skipped: {len(result.skipped)}\n"
        f"  clips dir:   {clips_out}\n"
        f"  tasks.json:  {out_dir / 'tasks.json'}"
    )
    if result.skipped:
        print(f"  See {out_dir / 'skipped.json'} for skip reasons.")


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def _build_tasks(
    *,
    queue: dict,
    conn: sqlite3.Connection,
    clips_out: Path,
    ls_base: str,
    model_version: str,
    copy_clips: bool,
) -> _BuildResult:
    tasks: list = []
    skipped: list = []
    batch_id = queue["batch_id"]

    select_sql = (
        f"SELECT {', '.join(_TRACK_COLS)} FROM tracks "
        f"WHERE video_id=? AND track_id=?"
    )

    for item in queue["items"]:
        video_id = item["video_id"]
        track_id = int(item["track_id"])
        row = conn.execute(select_sql, (video_id, track_id)).fetchone()
        if row is None:
            skipped.append({**item, "_skip_reason": "track_not_in_tracks_db"})
            continue
        if row["clip_path"] is None:
            skipped.append({**item, "_skip_reason": "clip_path_null"})
            continue

        clip_src = Path(row["clip_path"])
        if not clip_src.is_file():
            skipped.append({
                **item,
                "_skip_reason": "clip_file_missing",
                "_clip_path": str(clip_src),
            })
            continue

        clip_basename = clip_src.name
        clip_dest = clips_out / clip_basename
        if copy_clips and not clip_dest.exists():
            shutil.copy2(clip_src, clip_dest)

        task = _build_one_task(
            item=item,
            row=row,
            ls_base=ls_base,
            clip_basename=clip_basename,
            model_version=model_version,
            batch_id=batch_id,
        )
        tasks.append(task)

    return _BuildResult(tasks=tasks, skipped=skipped)


def _build_one_task(
    *,
    item: dict,
    row: sqlite3.Row,
    ls_base: str,
    clip_basename: str,
    model_version: str,
    batch_id: str,
) -> dict:
    """Produce one Label Studio task dict matching task_config.xml."""
    fish_probs = [
        float(row["mean_prob_chinook"]),
        float(row["mean_prob_coho"]),
        float(row["mean_prob_atlantic"]),
        float(row["mean_prob_rainbow"]),
        float(row["mean_prob_brown"]),
    ]
    bg_prob = float(row["mean_prob_background"])
    pred_6 = int(row["predicted_class_6"])
    mean_conf = float(row["mean_detection_confidence"])

    indexed = sorted(enumerate(fish_probs), key=lambda x: -x[1])
    top1_fish_name = FISH_CLASSES[indexed[0][0]]
    top2 = [(FISH_CLASSES[i], p) for i, p in indexed[:2]]

    # Pre-select rule: if the model's overall top-1 is Background, pre-select
    # "Not a fish" so phase-2 background-dominant items are honest about the
    # model's call. Otherwise pre-select the top fish species.
    if pred_6 == BACKGROUND_CLASS_IDX:
        pre_selected = NOT_A_FISH_CHOICE
        original_predicted_class_6 = "Background"
    else:
        pre_selected = FISH_CLASSES[pred_6]
        original_predicted_class_6 = FISH_CLASSES[pred_6]

    site, approx_dt = _parse_site_and_time(item["video_id"])

    task_html = _build_task_html(
        site=site,
        approx_dt=approx_dt,
        n_frames=int(row["n_frames"]),
        direction=str(row["direction"]),
        top2=top2,
        bg_prob=bg_prob,
        mean_conf=mean_conf,
    )

    clip_url = ls_base + "clips/" + clip_basename

    return {
        "data": {
            "clip_url":  clip_url,
            "task_html": task_html,
            # _model_version and _original_confidence intentionally omitted;
            # predictions[0].model_version / .score round-trip and supersede.
            "_video_id":                  item["video_id"],
            "_track_id":                  int(item["track_id"]),
            "_batch_id":                  batch_id,
            "_calibration":               bool(item.get("calibration_task", 0)),
            "_multi_reviewed":            bool(item.get("multi_reviewed", 0)),
            "_original_predicted_class_6": original_predicted_class_6,
            "_original_predicted_species": top1_fish_name,
        },
        "predictions": [
            {
                "model_version": model_version,
                "score":         mean_conf,
                "result": [
                    {
                        "from_name": "label",
                        "to_name":   "video",
                        "type":      "choices",
                        "value":     {"choices": [pre_selected]},
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FILENAME_TS_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}\.\d{2}\.\d{2})"
)


def _parse_site_and_time(video_id: str) -> tuple[str, str]:
    """Best-effort site / approx-timestamp from video_id.

    Production format example: "Ganaraska/Ganaraska 2020/1.mp4" -> site
    "Ganaraska". Filename timestamp is parsed only when it matches
    YYYY-MM-DD_HH.MM.SS at the start of the basename (RiverWatcher
    Trovis pattern). Returns ("unknown", "unknown") on a miss.
    """
    parts = video_id.replace("\\", "/").split("/")
    site = parts[0] if len(parts) >= 2 else "unknown"
    stem = parts[-1]

    m = _FILENAME_TS_RE.search(stem)
    if not m:
        return site, "unknown"
    date_s, time_s = m.group("date"), m.group("time").replace(".", ":")
    try:
        dt = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M:%S")
        return site, dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return site, "unknown"


def _build_task_html(
    *,
    site: str,
    approx_dt: str,
    n_frames: int,
    direction: str,
    top2: list[tuple[str, float]],
    bg_prob: float,
    mean_conf: float,
) -> str:
    """Pre-rendered metadata panel HTML.

    Inline CSS so it works regardless of LS theme. Kept narrow so it sits
    comfortably under the video tag without forcing horizontal scroll.
    """
    bars = ""
    for name, prob in top2:
        pct = max(0.0, min(1.0, prob)) * 100.0
        bars += (
            '<div style="margin:2px 0;display:flex;align-items:center;'
            'font-family:system-ui,sans-serif;font-size:13px;">'
            f'<div style="width:120px;">{html.escape(name)}</div>'
            '<div style="flex:1;background:#eef;border-radius:3px;'
            'overflow:hidden;height:14px;">'
            f'<div style="width:{pct:.1f}%;height:100%;background:#5b9bd5;"></div>'
            '</div>'
            '<div style="width:60px;text-align:right;'
            f'font-variant-numeric:tabular-nums;">{pct:.1f}%</div>'
            '</div>'
        )

    bg_pct = max(0.0, min(1.0, bg_prob)) * 100.0
    return (
        '<div style="font-family:system-ui,sans-serif;font-size:13px;'
        'line-height:1.45;padding:6px 4px;">'
        f'<div><b>Site:</b> {html.escape(site)} '
        f'&nbsp;<b>Approx time:</b> {html.escape(approx_dt)}</div>'
        f'<div><b>Direction:</b> {html.escape(direction)} '
        f'&nbsp;<b>Frames:</b> {n_frames} '
        f'&nbsp;<b>Mean det conf:</b> {mean_conf:.3f}</div>'
        '<div style="margin-top:6px;"><b>Top-2 species (mean track probs):</b></div>'
        f'{bars}'
        '<div style="margin-top:4px;color:#666;">'
        f'Background probability: {bg_pct:.1f}% '
        '<span style="color:#999;">(diagnostic; high values suggest spurious detection)</span>'
        '</div></div>'
    )


def _write_setup_md(out_dir: Path, queue: dict) -> None:
    batch_id = queue.get("batch_id", "?")
    reviewer_id = queue.get("reviewer_id", "?")
    phase = queue.get("phase", "?")
    abs_out = str(out_dir.resolve())
    md = f"""# Reviewer batch: {batch_id}

Reviewer: {reviewer_id}  | Phase: {phase}

## Setup (PowerShell)

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LOCAL_FILES_SERVING_ENABLED              = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT   = "{abs_out}"
$env:LOCAL_FILES_DOCUMENT_ROOT                = "{abs_out}"
label-studio start
```

## Project setup (one-time, in the LS UI)

1. Create a new project, name it after this batch.
2. Settings -> Labeling Interface -> Code -> paste contents of
   `labeling/task_config.xml` from the repo -> Save.
3. Settings -> Cloud Storage -> Add Source Storage -> Local files.
   Absolute path: `{abs_out}\\clips`
   File filter regex: `.*\\.mp4$`
   Save (do NOT click Sync).

## Import the batch

In the project view, click Import and select `tasks.json` from this folder.

## Label

Click "Label All Tasks". Hotkeys 1-5 (species), n (Not a fish),
m (Multiple fish), u (Unsure). Submit advances.

## When done

Export -> JSON -> send the file back to the project lead.
"""
    (out_dir / "setup.md").write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a Label Studio reviewer batch from queue.json."
    )
    p.add_argument("--queue",         required=True, metavar="PATH",
                   help="queue.json produced by queue.py")
    p.add_argument("--tracks-db",     required=True, metavar="PATH",
                   help="tracks.sqlite (post extract_clips.py)")
    p.add_argument("--out",           required=True, metavar="DIR",
                   help="output reviewer_batch_<id> directory")
    p.add_argument("--model-version", required=True, metavar="VERSION",
                   help="model checkpoint name, e.g. mar10_2026_detr-resnet50. "
                        "Long-term this should be read from tracks.sqlite.meta.")
    p.add_argument("--ls-base",       default=DEFAULT_LS_BASE, metavar="URL",
                   help=f"LS local-files URL prefix (default {DEFAULT_LS_BASE})")
    p.add_argument("--no-copy-clips", action="store_true",
                   help="Do not copy clip files; useful for dry-run.")
    return p.parse_args()


if __name__ == "__main__":
    main()
