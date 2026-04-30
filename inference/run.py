"""
Inference pipeline v2 — batch orchestrator.

Usage:
  python run.py \\
    --model-dir  C:/path/to/model/checkpoint \\
    --output-dir G:/Projects/model_outputs/run_20260501_120000 \\
    --site Ganaraska=G:/RiverWatcher/Ganaraska \\
    --site Credit=G:/RiverWatcher/Credit

Options:
  --model-dir DIR       Local model checkpoint directory (required)
  --output-dir DIR      Run output directory (required, created if absent)
  --site NAME=PATH      Site name and root path; repeat for each site (required)
  --tracker NAME        bytetrack (default) or botsort
  --device DEVICE       cuda:0 (default) or cpu
  --box-score-thresh F  Detection confidence threshold (default 0.6)
  --checkpoint-every N  Flush tracks to SQLite every N videos (default 50)
  --retry-errors        Re-process videos previously logged as 'error'

Output layout (under --output-dir):
  detections/           One Parquet file per video
  tracks.sqlite         Per-track summaries (flushed every N videos)
  processing_log.sqlite Per-video status log (written after every video)
  run_manifest.json     Run metadata (written at start, updated at end)
"""
from __future__ import annotations

import argparse
import random
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import InferenceConfig
from manifest import capture_env, write_manifest
from pipeline import VideoResult, process_video
from writer import flush_tracks, log_video, write_detections

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    cfg = _build_config(args)

    run_dir = Path(cfg.output_dir)
    det_dir = run_dir / "detections"
    tracks_db = run_dir / "tracks.sqlite"
    log_db = run_dir / "processing_log.sqlite"

    run_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {run_dir}")
    print(f"Tracker: {cfg.tracker}  |  threshold: {cfg.box_score_thresh}")

    # Write initial manifest so the run is self-describing even if killed early.
    env = capture_env(cfg)
    write_manifest(env, run_dir)

    # Load model once for the entire run.
    device = _resolve_device(cfg.cuda_device)
    print(f"Loading model from {cfg.model_checkpoint} on {device} …")
    model = (
        AutoModelForObjectDetection.from_pretrained(
            cfg.model_checkpoint, local_files_only=True
        )
        .to(device)
        .eval()
    )
    image_processor = AutoImageProcessor.from_pretrained(
        cfg.model_checkpoint, local_files_only=True
    )
    if cfg.fp16:
        model = model.half()
        print("FP16 enabled.")
    if cfg.compile_model:
        print("Compiling model (first video will be slow — compilation warm-up) …")
        model = torch.compile(model)
        print("Model compiled.")
    print("Model loaded.")

    # Enumerate all videos across all sites.
    all_videos = _enumerate_videos(cfg.video_roots)
    total_on_disk = len(all_videos)
    print(f"Found {total_on_disk:,} videos across {len(cfg.video_roots)} site(s).")

    # Determine which videos to skip based on processing_log.
    done_ids = _load_done_ids(log_db, retry_errors=args.retry_errors)
    todo = [(vid, path) for vid, path in all_videos if vid not in done_ids]

    # --limit: shuffle randomly then cap (for pilot runs).
    if args.limit is not None:
        random.shuffle(todo)
        todo = todo[: args.limit]
        print(f"--limit {args.limit}: randomly selected {len(todo):,} videos.")

    total = len(todo) + len(done_ids)
    skipped_upfront = len(done_ids)
    print(
        f"Resuming: {len(todo):,} to process, "
        f"{skipped_upfront:,} already done (skipped)."
    )

    # Main processing loop.
    pending_tracks = []
    n_success = skipped_upfront  # count previously-done as success for manifest
    n_error = 0
    n_skipped = 0
    run_start_time = time.monotonic()
    disk_checked = False

    for i, (video_id, video_path) in enumerate(todo):
        t0 = time.monotonic()
        try:
            result: VideoResult = process_video(
                video_path=video_path,
                video_id=video_id,
                model=model,
                image_processor=image_processor,
                cfg=cfg,
                device=device,
            )

            write_detections(video_id, result.detections, det_dir)
            pending_tracks.extend(result.tracks)

            log_video(
                video_id, "success", log_db,
                processing_duration_seconds=time.monotonic() - t0,
                n_detections=len(result.detections),
                n_tracks=len(result.tracks),
            )
            n_success += 1

        except FileNotFoundError:
            log_video(video_id, "skipped", log_db,
                      processing_duration_seconds=time.monotonic() - t0,
                      error_message="file not found")
            n_skipped += 1

        except Exception as exc:
            log_video(
                video_id, "error", log_db,
                processing_duration_seconds=time.monotonic() - t0,
                error_message=f"{type(exc).__name__}: {exc}",
            )
            n_error += 1

        # Flush tracks to SQLite periodically.
        if len(pending_tracks) >= cfg.checkpoint_every_n * 5 or (
            (i + 1) % cfg.checkpoint_every_n == 0 and pending_tracks
        ):
            flush_tracks(pending_tracks, tracks_db)
            pending_tracks.clear()

        # Disk space check: after processing the first 20 videos.
        if not disk_checked and (i + 1) >= 20:
            _check_disk_space(det_dir, i + 1, total)
            disk_checked = True

        # Progress report every 100 videos.
        if (i + 1) % 100 == 0 or (i + 1) == len(todo):
            _print_progress(i + 1, len(todo), total, run_start_time, n_error)

    # Final flush and manifest.
    if pending_tracks:
        flush_tracks(pending_tracks, tracks_db)

    run_end = datetime.now(timezone.utc).isoformat()
    write_manifest(
        env, run_dir,
        run_end=run_end,
        n_total=total,
        n_success=n_success,
        n_error=n_error,
        n_skipped=n_skipped,
    )

    elapsed = time.monotonic() - run_start_time
    print(
        f"\nDone in {elapsed/3600:.1f}h. "
        f"Success: {n_success:,}  Error: {n_error:,}  Skipped: {n_skipped:,}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enumerate_videos(
    video_roots: Dict[str, str],
) -> List[Tuple[str, str]]:
    """Return (video_id, absolute_path) pairs for every video under all site roots."""
    result = []
    for site_name, root_str in video_roots.items():
        root = Path(root_str)
        if not root.is_dir():
            print(f"WARNING: site root not found, skipping: {root}")
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file():
                # Skip _m.mp4 files — these are pre-annotated versions with
                # burned-in overlays from a previous processing pass. Running
                # inference on them would give the model corrupted input.
                if p.stem.endswith("_m"):
                    continue
                rel = p.relative_to(root)
                video_id = site_name + "/" + "/".join(rel.parts)
                result.append((video_id, str(p)))
    return result


def _load_done_ids(log_db: Path, retry_errors: bool) -> Set[str]:
    """Return video_ids that should be skipped on this run."""
    if not log_db.exists():
        return set()
    conn = sqlite3.connect(str(log_db))
    try:
        statuses = ("'success'", "'skipped'")
        if not retry_errors:
            statuses = ("'success'", "'skipped'", "'error'")
        rows = conn.execute(
            f"SELECT video_id FROM processing_log WHERE status IN ({','.join(statuses)})"
        ).fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


def _check_disk_space(det_dir: Path, n_processed: int, n_total: int) -> None:
    """Estimate total output size from pilot videos and warn if disk is tight."""
    parquet_files = list(det_dir.glob("*.parquet"))
    if not parquet_files:
        return

    total_bytes = sum(p.stat().st_size for p in parquet_files)
    mean_bytes = total_bytes / len(parquet_files)
    extrapolated_gb = (mean_bytes * n_total) / 1e9

    free_gb = shutil.disk_usage(det_dir).free / 1e9

    if free_gb >= 3 * extrapolated_gb:
        status = "SUFFICIENT"
    elif free_gb >= extrapolated_gb:
        status = "TIGHT"
    else:
        status = "WARNING"

    print(
        f"\n[Disk check] Extrapolated total output: {extrapolated_gb:.1f} GB  |  "
        f"Free: {free_gb:.1f} GB  ->  {status}"
    )
    if status == "WARNING":
        print("ERROR: Insufficient disk space. Aborting.")
        sys.exit(1)
    if status == "TIGHT":
        print("Notice: Disk space is tight. Monitor during run.")


def _print_progress(
    n_done: int, n_todo: int, n_total: int, run_start: float, n_error: int
) -> None:
    elapsed = time.monotonic() - run_start
    fps = n_done / elapsed if elapsed > 0 else 0.0
    remaining = n_todo - n_done
    eta_s = remaining / fps if fps > 0 else 0
    eta_h, eta_m = divmod(int(eta_s / 60), 60)
    pct = 100.0 * (n_total - n_todo + n_done) / n_total
    print(
        f"[{n_total - n_todo + n_done:,}/{n_total:,}] "
        f"{pct:.1f}% | ETA {eta_h}h {eta_m}m | "
        f"fps {fps:.2f} | errors {n_error:,}"
    )


def _resolve_device(cuda_device: str) -> str:
    if torch.cuda.is_available():
        return f"cuda:{cuda_device}"
    return "cpu"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference pipeline v2: DETR + ByteTrack batch processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-dir", required=True, metavar="DIR",
                   help="Local model checkpoint directory.")
    p.add_argument("--output-dir", required=True, metavar="DIR",
                   help="Run output directory (created if absent).")
    p.add_argument("--site", action="append", default=[], metavar="NAME=PATH",
                   help="Site name and root path, e.g. Ganaraska=G:/RiverWatcher/Ganaraska. "
                        "Repeat for each site.")
    p.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"],
                   help="Tracker algorithm (default: bytetrack).")
    p.add_argument("--device", default=None, metavar="DEVICE",
                   help="PyTorch device override, e.g. cpu or cuda:1.")
    p.add_argument("--box-score-thresh", type=float, default=0.6, metavar="F",
                   help="Detection confidence threshold (default: 0.6).")
    p.add_argument("--checkpoint-every", type=int, default=50, metavar="N",
                   help="Flush tracks to SQLite every N videos (default: 50).")
    p.add_argument("--retry-errors", action="store_true",
                   help="Re-attempt videos previously logged as 'error'.")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Process only N randomly selected videos (for pilot runs).")
    p.add_argument("--fp16", action="store_true",
                   help="Half-precision inference (~1.5-2x throughput on RTX GPUs).")
    p.add_argument("--compile", action="store_true", dest="compile_model",
                   help="torch.compile the model after loading (~20-40%% extra throughput).")
    p.add_argument("--frame-skip", type=int, default=1, metavar="N",
                   help="Process every Nth frame (default: 1 = all frames).")
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> InferenceConfig:
    video_roots: Dict[str, str] = {}
    for spec in args.site:
        if "=" not in spec:
            print(f"ERROR: --site must be NAME=PATH, got: {spec!r}")
            sys.exit(1)
        name, _, path = spec.partition("=")
        video_roots[name.strip()] = path.strip()

    if not video_roots:
        print("ERROR: at least one --site NAME=PATH is required.")
        sys.exit(1)

    cfg = InferenceConfig(
        model_checkpoint=args.model_dir,
        output_dir=args.output_dir,
        video_roots=video_roots,
        tracker=args.tracker,
        box_score_thresh=args.box_score_thresh,
        checkpoint_every_n=args.checkpoint_every,
        fp16=args.fp16,
        compile_model=args.compile_model,
        frame_skip=args.frame_skip,
    )
    if args.device is not None:
        # Manual device override bypasses cuda_device field.
        cfg.cuda_device = args.device.replace("cuda:", "")

    return cfg


if __name__ == "__main__":
    main()
