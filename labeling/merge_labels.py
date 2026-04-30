"""
Merge reviewer labels.sqlite files into a master labels.sqlite.

Detects multi-reviewer overlaps (same video_id + track_id, different reviewer_id),
marks them multi_reviewed=1, and flags disagreements as disagreement_resolution='pending'.

Usage:
  python merge_labels.py
    --master  G:/Projects/labels/labels.sqlite
    --add     G:/Projects/labels/mikayla_export.sqlite
    [--add    G:/Projects/labels/reviewer3_export.sqlite ...]

The master is updated in place. Run repeatedly as new exports arrive.
After merging, run resolve_disagreements.py to surface pending items for
senior review.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import open_labels_db


def main() -> None:
    args = _parse_args()
    master_db = Path(args.master)
    master_conn = open_labels_db(master_db)

    for src_path in args.add:
        src = Path(src_path)
        if not src.exists():
            print(f"ERROR: not found: {src}")
            continue
        _merge_one(master_conn, src)

    _flag_multi_reviewed(master_conn)
    master_conn.commit()
    master_conn.close()

    # Print summary
    conn = sqlite3.connect(str(master_db))
    total  = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    multi  = conn.execute("SELECT COUNT(*) FROM labels WHERE multi_reviewed=1").fetchone()[0]
    pending = conn.execute(
        "SELECT COUNT(*) FROM labels WHERE disagreement_resolution='pending'"
    ).fetchone()[0]
    reviewers = [r[0] for r in conn.execute(
        "SELECT DISTINCT reviewer_id FROM labels ORDER BY reviewer_id"
    ).fetchall()]
    conn.close()

    print(f"\nMaster labels.sqlite: {master_db}")
    print(f"  Total labels    : {total:,}")
    print(f"  Reviewers       : {', '.join(reviewers)}")
    print(f"  Multi-reviewed  : {multi:,}")
    print(f"  Pending review  : {pending:,}")
    if pending:
        print("  -> Run resolve_disagreements.py to action pending items.")


def _merge_one(master: sqlite3.Connection, src_path: Path) -> None:
    src = sqlite3.connect(str(src_path))
    src.row_factory = sqlite3.Row
    rows = src.execute("SELECT * FROM labels").fetchall()
    src.close()

    inserted = skipped = 0
    for row in rows:
        try:
            master.execute(
                """INSERT INTO labels (
                    video_id, track_id, final_label, label_action,
                    reviewer_id, reviewed_at, time_on_task,
                    original_predicted_class_6, original_predicted_species,
                    original_confidence, model_version,
                    calibration_task, multi_reviewed, disagreement_resolution,
                    ls_annotation_id, batch_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row["video_id"], row["track_id"], row["final_label"],
                    row["label_action"], row["reviewer_id"], row["reviewed_at"],
                    row["time_on_task"], row["original_predicted_class_6"],
                    row["original_predicted_species"], row["original_confidence"],
                    row["model_version"], row["calibration_task"],
                    row["multi_reviewed"], row["disagreement_resolution"],
                    row["ls_annotation_id"], row["batch_id"],
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            skipped += 1

    print(f"  {src_path.name}: inserted {inserted}, skipped {skipped} duplicates")


def _flag_multi_reviewed(conn: sqlite3.Connection) -> None:
    """Mark rows where multiple reviewers labeled the same track, flag disagreements."""
    # Find (video_id, track_id) pairs with more than one reviewer.
    pairs = conn.execute("""
        SELECT video_id, track_id
        FROM labels
        GROUP BY video_id, track_id
        HAVING COUNT(DISTINCT reviewer_id) > 1
    """).fetchall()

    for video_id, track_id in pairs:
        rows = conn.execute(
            "SELECT reviewer_id, final_label, disagreement_resolution FROM labels "
            "WHERE video_id=? AND track_id=?",
            (video_id, track_id)
        ).fetchall()

        # Mark all as multi_reviewed.
        conn.execute(
            "UPDATE labels SET multi_reviewed=1 WHERE video_id=? AND track_id=?",
            (video_id, track_id)
        )

        # Check for disagreement (compare final_label values, not reviewer IDs).
        labels = {r[1] for r in rows}  # unique final_labels
        if len(labels) > 1:
            # Disagreement — flag as pending if not already resolved.
            conn.execute(
                """UPDATE labels SET disagreement_resolution='pending'
                   WHERE video_id=? AND track_id=?
                   AND (disagreement_resolution IS NULL OR disagreement_resolution='pending')""",
                (video_id, track_id)
            )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge reviewer label exports into master labels.sqlite.")
    p.add_argument("--master", required=True, metavar="PATH",
                   help="Master labels.sqlite (updated in place, created if absent).")
    p.add_argument("--add", action="append", default=[], metavar="PATH",
                   help="Reviewer labels.sqlite to merge in. Repeat for multiple reviewers.")
    return p.parse_args()


if __name__ == "__main__":
    main()
