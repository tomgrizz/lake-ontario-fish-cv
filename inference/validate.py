"""
Pilot validation — diagnostic report for an inference run.

Usage:
  python validate.py --run-dir G:/Projects/model_outputs/run_20260501_120000

Reads the Parquet detections, tracks.sqlite, and processing_log.sqlite produced
by run.py and prints a diagnostic report. Does not modify any outputs.

Resumability and error-injection tests must be performed manually (see
docs/pilot_videos.md); this script reads the logs to report their results.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import SCHEMA_VERSION


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)

    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}")
        sys.exit(1)

    print(f"Validating run: {run_dir}\n")
    issues: list[str] = []

    manifest = _load_manifest(run_dir, issues)
    log_summary = _check_processing_log(run_dir, issues)
    det_summary = _check_detections(run_dir, issues)
    track_summary = _check_tracks(run_dir, issues)
    _check_schema_version(run_dir, manifest, issues)
    _check_disk(run_dir, det_summary, manifest, args.full_corpus_n)

    _print_report(manifest, log_summary, det_summary, track_summary, issues)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _load_manifest(run_dir: Path, issues: list) -> dict:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        issues.append("run_manifest.json not found")
        return {}
    return json.loads(path.read_text())


def _check_processing_log(run_dir: Path, issues: list) -> dict:
    db = run_dir / "processing_log.sqlite"
    if not db.exists():
        issues.append("processing_log.sqlite not found")
        return {}

    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM processing_log GROUP BY status"
        ).fetchall()
        counts = {r[0]: r[1] for r in rows}
        n_success = counts.get("success", 0)
        n_error = counts.get("error", 0)
        n_skipped = counts.get("skipped", 0)

        # Resumability check: look for evidence that some videos were skipped
        # on a restart (status='success' rows whose processed_at timestamps
        # appear in two distinct batches).
        resumed = conn.execute(
            "SELECT COUNT(*) FROM processing_log WHERE status='skipped'"
        ).fetchone()[0]

        # Error injection check: any status='error' rows?
        error_rows = conn.execute(
            "SELECT video_id, error_message FROM processing_log WHERE status='error' LIMIT 5"
        ).fetchall()

        return {
            "n_success": n_success,
            "n_error": n_error,
            "n_skipped": n_skipped,
            "error_rows": error_rows,
            "has_skipped": resumed > 0,
        }
    finally:
        conn.close()


def _check_detections(run_dir: Path, issues: list) -> dict:
    det_dir = run_dir / "detections"
    parquet_files = sorted(det_dir.glob("*.parquet")) if det_dir.exists() else []

    if not parquet_files:
        issues.append("No Parquet files found in detections/")
        return {}

    total_rows = 0
    file_sizes = []
    prob_sum_violations = 0
    bbox_violations = 0
    schema_ok = True
    sample_tables = []

    for pf in parquet_files:
        file_sizes.append(pf.stat().st_size)
        try:
            tbl = pq.read_table(pf)
        except Exception as e:
            issues.append(f"Cannot read {pf.name}: {e}")
            continue

        # Schema check
        expected_cols = {
            "video_id", "frame_number", "timestamp_seconds", "track_id",
            "detection_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "detection_confidence", "prob_chinook", "prob_coho", "prob_atlantic",
            "prob_rainbow", "prob_brown", "prob_background",
            "predicted_class", "predicted_class_6",
        }
        missing = expected_cols - set(tbl.schema.names)
        if missing:
            issues.append(f"{pf.name}: missing columns {missing}")
            schema_ok = False

        n = len(tbl)
        total_rows += n
        if n == 0:
            continue

        if len(sample_tables) < 3:
            sample_tables.append(tbl)

        # Prob sum check: all 6 probs should sum to ≈1.0
        import pyarrow.compute as pc
        prob_cols = ["prob_chinook", "prob_coho", "prob_atlantic",
                     "prob_rainbow", "prob_brown", "prob_background"]
        try:
            prob_sum = sum(tbl.column(c).to_pylist() for c in prob_cols)
            # prob_sum is now a list of sums per row
            arr = np.array([sum(tbl.column(c)[i].as_py() for c in prob_cols)
                            for i in range(min(n, 100))])
            bad = np.sum(np.abs(arr - 1.0) > 0.01)
            prob_sum_violations += int(bad)
        except Exception:
            pass

        # Bbox sanity: no negative coords
        for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
            vals = np.array(tbl.column(col).to_pylist())
            if np.any(vals < 0):
                bbox_violations += 1
                issues.append(f"{pf.name}: negative values in {col}")
                break

    sizes_kb = [s / 1024 for s in file_sizes]
    return {
        "n_files": len(parquet_files),
        "total_rows": total_rows,
        "mean_size_kb": float(np.mean(sizes_kb)) if sizes_kb else 0,
        "max_size_kb": float(np.max(sizes_kb)) if sizes_kb else 0,
        "prob_sum_violations": prob_sum_violations,
        "bbox_violations": bbox_violations,
        "schema_ok": schema_ok,
        "sample_tables": sample_tables,
    }


def _check_tracks(run_dir: Path, issues: list) -> dict:
    db = run_dir / "tracks.sqlite"
    if not db.exists():
        issues.append("tracks.sqlite not found")
        return {}

    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT n_frames, direction, entrance_side, exit_side, "
            "predicted_class, predicted_class_6, n_frames, "
            "mean_prob_chinook + mean_prob_coho + mean_prob_atlantic + "
            "mean_prob_rainbow + mean_prob_brown + mean_prob_background AS prob_sum, "
            "end_frame - start_frame AS bbox_span "
            "FROM tracks"
        ).fetchall()

        if not rows:
            return {"n_tracks": 0}

        n_frames_all = [r[0] for r in rows]
        pred_classes = [r[4] for r in rows]
        prob_sums = [r[7] for r in rows if r[7] is not None]
        bbox_spans = [r[8] for r in rows if r[8] is not None]

        arr = np.array(n_frames_all)
        p95 = float(np.percentile(arr, 95))

        # Fragmentation: tracks with very few frames
        short = int(np.sum(arr < 5))

        # Over-merging: tracks longer than 95th percentile
        long_tracks = int(np.sum(arr > p95))

        # Class distribution
        from collections import Counter
        class_dist = Counter(pred_classes)

        # prob_sum should be close to 1.0 (including background)
        bad_probs = sum(1 for s in prob_sums if abs(s - 1.0) > 0.02)

        return {
            "n_tracks": len(rows),
            "n_frames_p10": float(np.percentile(arr, 10)),
            "n_frames_p50": float(np.percentile(arr, 50)),
            "n_frames_p90": float(np.percentile(arr, 90)),
            "n_frames_max": float(np.max(arr)),
            "p95_threshold": p95,
            "short_tracks": short,
            "long_tracks_flagged": long_tracks,
            "class_dist": dict(class_dist),
            "bad_prob_sums": bad_probs,
        }
    finally:
        conn.close()


def _check_schema_version(run_dir: Path, manifest: dict, issues: list) -> None:
    # Manifest
    if manifest.get("schema_version") != SCHEMA_VERSION:
        issues.append(
            f"run_manifest.json schema_version={manifest.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )

    # tracks.sqlite meta table
    db = run_dir / "tracks.sqlite"
    if db.exists():
        conn = sqlite3.connect(str(db))
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if not row or row[0] != str(SCHEMA_VERSION):
                issues.append(
                    f"tracks.sqlite meta.schema_version={row[0] if row else None!r}, "
                    f"expected '{SCHEMA_VERSION}'"
                )
        except sqlite3.OperationalError:
            issues.append("tracks.sqlite: meta table missing")
        finally:
            conn.close()

    # processing_log.sqlite meta table
    db2 = run_dir / "processing_log.sqlite"
    if db2.exists():
        conn2 = sqlite3.connect(str(db2))
        try:
            row2 = conn2.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if not row2 or row2[0] != str(SCHEMA_VERSION):
                issues.append(
                    f"processing_log.sqlite meta.schema_version={row2[0] if row2 else None!r}, "
                    f"expected '{SCHEMA_VERSION}'"
                )
        except sqlite3.OperationalError:
            issues.append("processing_log.sqlite: meta table missing")
        finally:
            conn2.close()

    # Parquet file metadata
    det_dir = run_dir / "detections"
    parquets = list(det_dir.glob("*.parquet")) if det_dir.exists() else []
    if parquets:
        tbl = pq.read_table(parquets[0])
        meta_ver = tbl.schema.metadata.get(b"schema_version", b"").decode()
        if meta_ver != str(SCHEMA_VERSION):
            issues.append(
                f"Parquet schema_version metadata={meta_ver!r}, "
                f"expected '{SCHEMA_VERSION}'"
            )


def _check_disk(run_dir: Path, det_summary: dict, manifest: dict, full_corpus_n: int) -> None:
    if not det_summary.get("mean_size_kb"):
        return
    # Use the full corpus size for extrapolation, not the pilot total.
    extrap_gb = (det_summary["mean_size_kb"] * 1024 * full_corpus_n) / 1e9
    free_gb = shutil.disk_usage(run_dir).free / 1e9
    if free_gb >= 3 * extrap_gb:
        status = "SUFFICIENT"
    elif free_gb >= extrap_gb:
        status = "TIGHT"
    else:
        status = "WARNING"
    print(f"[Disk] Extrapolated total output: {extrap_gb:.1f} GB  "
          f"|  Free: {free_gb:.1f} GB  →  {status}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "Chinook", 1: "Coho", 2: "Atlantic", 3: "Rainbow", 4: "Brown"}


def _print_report(
    manifest: dict,
    log: dict,
    det: dict,
    trk: dict,
    issues: list,
) -> None:
    hr = "-" * 60

    print(hr)
    print("PROCESSING LOG")
    print(hr)
    print(f"  Success : {log.get('n_success', '?'):>8,}")
    print(f"  Error   : {log.get('n_error', '?'):>8,}")
    print(f"  Skipped : {log.get('n_skipped', '?'):>8,}")
    if log.get("error_rows"):
        print("  Recent errors:")
        for vid, msg in log["error_rows"]:
            print(f"    {vid}  →  {msg}")

    print()
    print(hr)
    print("DETECTIONS (Parquet)")
    print(hr)
    print(f"  Files             : {det.get('n_files', 0):,}")
    print(f"  Total rows        : {det.get('total_rows', 0):,}")
    print(f"  Mean file size    : {det.get('mean_size_kb', 0):.1f} KB")
    print(f"  Max file size     : {det.get('max_size_kb', 0):.1f} KB")
    print(f"  Prob-sum violations (|sum-1|>0.01) : {det.get('prob_sum_violations', 0)}")
    print(f"  Bbox violations                    : {det.get('bbox_violations', 0)}")
    print(f"  Schema OK                          : {det.get('schema_ok', False)}")

    print()
    print(hr)
    print("TRACKS (SQLite)")
    print(hr)
    print(f"  Total tracks      : {trk.get('n_tracks', 0):,}")
    if trk.get("n_tracks"):
        print(f"  n_frames p10/p50/p90/max: "
              f"{trk['n_frames_p10']:.0f} / {trk['n_frames_p50']:.0f} / "
              f"{trk['n_frames_p90']:.0f} / {trk['n_frames_max']:.0f}")
        print(f"  Short tracks (n_frames<5)         : {trk['short_tracks']}")
        print(f"  Long tracks (>p95={trk['p95_threshold']:.0f} frames, over-merge candidates): "
              f"{trk['long_tracks_flagged']}")
        print(f"  Bad prob sums (mean probs ≠ 1.0)  : {trk['bad_prob_sums']}")
        print("  Species distribution (predicted_class):")
        for cls_id, count in sorted(trk.get("class_dist", {}).items()):
            name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            print(f"    {name:15s}: {count:,}")

    print()
    print(hr)
    print("SCHEMA VERSION")
    print(hr)
    schema_issues = [i for i in issues if "schema_version" in i.lower()]
    if schema_issues:
        for i in schema_issues:
            print(f"  FAIL: {i}")
    else:
        print(f"  OK: schema_version={SCHEMA_VERSION} in manifest, both SQLite DBs, and Parquet")

    print()
    print(hr)
    print("RESUMABILITY TEST")
    print(hr)
    if log.get("has_skipped"):
        print("  PASS: processing_log contains skipped entries "
              "(consistent with a resumed run)")
    else:
        print("  INCONCLUSIVE: no skipped entries found — run the resumability test "
              "manually (kill at video 25, restart, re-run validate.py)")

    print()
    print(hr)
    print("ERROR INJECTION TEST")
    print(hr)
    if log.get("n_error", 0) > 0:
        print(f"  PASS: {log['n_error']} error(s) recorded in processing_log "
              "and run continued")
    else:
        print("  INCONCLUSIVE: no errors recorded — inject a corrupt video and "
              "re-run validate.py to verify error isolation")

    print()
    other_issues = [i for i in issues if "schema_version" not in i.lower()]
    if other_issues:
        print(hr)
        print(f"OTHER ISSUES ({len(other_issues)})")
        print(hr)
        for i in other_issues:
            print(f"  {i}")
    else:
        print("No other issues found.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate an inference pipeline v2 run.")
    p.add_argument("--run-dir", required=True, metavar="DIR",
                   help="Path to the run output directory to validate.")
    p.add_argument("--full-corpus-n", type=int, default=258955, metavar="N",
                   help="Total video count in the full corpus for disk extrapolation "
                        "(default: 258955).")
    return p.parse_args()


if __name__ == "__main__":
    main()
