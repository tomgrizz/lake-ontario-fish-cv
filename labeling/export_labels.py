"""
Parse a Label Studio JSON export and write labels to labels.sqlite.

Usage:
  python export_labels.py
    --export   G:/Projects/label_exports/pilot_export.json
    --labels   G:/Projects/labels/labels.sqlite
    --reviewer-id tom
    --model-version mar10_2026_detr-resnet50
    [--batch-id batch_20260429_143853_tom]

label_action is inferred by comparing the annotation choice against the
model's original prediction stored in task data (_original_predicted_species):
  confirm      — annotation matches model's top fish-class prediction
  specify      — annotation is a different fish species
  not_a_fish   — annotation is "Not a fish"
  multiple_fish— annotation is "Multiple fish"
  unsure       — annotation is "Unsure"

Skipped tasks (no annotation) are ignored — they stay in the queue.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import (
    SCHEMA_VERSION,
    LabelRecord,
    SkippedRecord,
    TRAINING_POSITIVE_LABELS,
    open_labels_db,
)

FISH_LABELS = {"Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"}


def main() -> None:
    args = _parse_args()

    export_path = Path(args.export)
    if not export_path.exists():
        print(f"ERROR: export file not found: {export_path}")
        sys.exit(1)

    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(tasks)} tasks from {export_path.name}")

    labels_db = Path(args.labels)
    labels_db.parent.mkdir(parents=True, exist_ok=True)
    conn = open_labels_db(labels_db)

    n_written = n_skipped = n_duplicate = 0

    for task in tasks:
        data = task.get("data", {})
        annotations = task.get("annotations", [])

        if not annotations:
            continue

        ann = annotations[0]  # take the first annotation
        result = ann.get("result", [])
        if not result:
            continue

        choice = result[0]["value"]["choices"][0]
        video_id = data.get("_video_id", "")
        track_id = int(data.get("_track_id", 0))
        original_species = data.get("_original_predicted_species", "")
        original_class6 = data.get("_original_predicted_class_6", "")

        label_action = _infer_action(choice, original_species)

        record = LabelRecord(
            video_id=video_id,
            track_id=track_id,
            final_label=choice,
            label_action=label_action,
            reviewer_id=args.reviewer_id,
            reviewed_at=ann.get("updated_at", datetime.now(timezone.utc).isoformat()),
            time_on_task=ann.get("lead_time"),
            original_predicted_class_6=original_class6,
            original_predicted_species=original_species,
            original_confidence=float(data.get("_original_confidence", 0.0)) if "_original_confidence" in data else 0.0,
            model_version=args.model_version,
            calibration_task=int(data.get("_calibration", False)),
            multi_reviewed=int(data.get("_multi_reviewed", False)),
            ls_annotation_id=ann.get("id"),
            batch_id=args.batch_id or data.get("_batch_id"),
        )

        try:
            conn.execute(
                """INSERT INTO labels (
                    video_id, track_id, final_label, label_action,
                    reviewer_id, reviewed_at, time_on_task,
                    original_predicted_class_6, original_predicted_species,
                    original_confidence, model_version,
                    calibration_task, multi_reviewed,
                    ls_annotation_id, batch_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.video_id, record.track_id, record.final_label,
                    record.label_action, record.reviewer_id, record.reviewed_at,
                    record.time_on_task, record.original_predicted_class_6,
                    record.original_predicted_species, record.original_confidence,
                    record.model_version, record.calibration_task,
                    record.multi_reviewed, record.ls_annotation_id, record.batch_id,
                ),
            )
            n_written += 1
        except sqlite3.IntegrityError:
            n_duplicate += 1

    conn.commit()
    conn.close()

    print(f"\nWritten : {n_written}")
    print(f"Duplicate (already in DB): {n_duplicate}")

    # Summary by action
    conn2 = sqlite3.connect(str(labels_db))
    rows = conn2.execute(
        "SELECT label_action, COUNT(*) FROM labels GROUP BY label_action ORDER BY COUNT(*) DESC"
    ).fetchall()
    print("\nBy action:")
    for action, n in rows:
        print(f"  {action:15s}: {n}")

    rows2 = conn2.execute(
        "SELECT final_label, COUNT(*) FROM labels GROUP BY final_label ORDER BY COUNT(*) DESC"
    ).fetchall()
    print("\nBy label:")
    for label, n in rows2:
        print(f"  {label:20s}: {n}")

    mean_time = conn2.execute(
        "SELECT AVG(time_on_task) FROM labels WHERE reviewer_id=?",
        (args.reviewer_id,)
    ).fetchone()[0]
    print(f"\nMean time on task ({args.reviewer_id}): {mean_time:.1f}s" if mean_time else "")
    conn2.close()


def _infer_action(choice: str, original_species: str) -> str:
    if choice == "Not a fish":
        return "not_a_fish"
    if choice == "Multiple fish":
        return "multiple_fish"
    if choice == "Unsure":
        return "unsure"
    if choice in FISH_LABELS:
        return "confirm" if choice == original_species else "specify"
    return "specify"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Label Studio annotations to labels.sqlite.")
    p.add_argument("--export", required=True, metavar="PATH",
                   help="Label Studio JSON export file.")
    p.add_argument("--labels", required=True, metavar="PATH",
                   help="Path to labels.sqlite (created if absent).")
    p.add_argument("--reviewer-id", required=True, metavar="ID")
    p.add_argument("--model-version", required=True, metavar="VERSION")
    p.add_argument("--batch-id", default=None, metavar="ID")
    return p.parse_args()


if __name__ == "__main__":
    main()
