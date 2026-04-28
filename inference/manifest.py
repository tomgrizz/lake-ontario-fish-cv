"""
Run manifest for inference pipeline v2.

capture_env()    — snapshot of model, code, and hardware state at run start
write_manifest() — write/update run_manifest.json in the run output directory
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from schema import SCHEMA_VERSION
from config import InferenceConfig


def capture_env(cfg: InferenceConfig) -> Dict[str, Any]:
    """Return a dict capturing the model, code, and hardware state.

    Called once at run start. run_start is set here; run_end, n_total,
    n_success, n_error, and n_skipped are filled in by write_manifest()
    at run completion.
    """
    import torch

    model_path = Path(cfg.model_checkpoint)
    try:
        model_mtime = datetime.fromtimestamp(
            model_path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
    except OSError:
        model_mtime = None

    gpu_name: Optional[str] = None
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        cudnn_version = str(torch.backends.cudnn.version())

    return {
        "schema_version": SCHEMA_VERSION,
        "model_checkpoint": str(cfg.model_checkpoint),
        "model_last_modified": model_mtime,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "tracker": cfg.tracker,
        "tracker_config": _tracker_config(cfg),
        "video_roots": dict(cfg.video_roots),
        "pytorch_version": torch.__version__,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "gpu_name": gpu_name,
        "run_start": datetime.now(timezone.utc).isoformat(),
        "run_end": None,
        "n_total": None,
        "n_success": None,
        "n_error": None,
        "n_skipped": None,
    }


def write_manifest(
    env: Dict[str, Any],
    run_dir: Path,
    *,
    run_end: Optional[str] = None,
    n_total: Optional[int] = None,
    n_success: Optional[int] = None,
    n_error: Optional[int] = None,
    n_skipped: Optional[int] = None,
) -> Path:
    """Write (or overwrite) run_manifest.json in run_dir.

    Call once at run start (with only env), then again at completion with
    the final counters. Overwrites in place — the file is small and JSON
    writes are atomic enough for a single-process run.
    """
    manifest = dict(env)
    if run_end is not None:
        manifest["run_end"] = run_end
    if n_total is not None:
        manifest["n_total"] = n_total
    if n_success is not None:
        manifest["n_success"] = n_success
    if n_error is not None:
        manifest["n_error"] = n_error
    if n_skipped is not None:
        manifest["n_skipped"] = n_skipped

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _git_dirty() -> Optional[bool]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


def _tracker_config(cfg: InferenceConfig) -> Dict[str, Any]:
    if cfg.tracker == "bytetrack":
        return {
            "min_conf": cfg.bytetrack_min_conf,
            "track_thresh": cfg.bytetrack_track_thresh,
            "match_thresh": cfg.bytetrack_match_thresh,
            "track_buffer": cfg.bytetrack_track_buffer,
        }
    if cfg.tracker == "botsort":
        return {
            "weights": cfg.botsort_weights,
            "track_high_thresh": cfg.botsort_track_high_thresh,
            "track_low_thresh": cfg.botsort_track_low_thresh,
            "new_track_thresh": cfg.botsort_new_track_thresh,
            "track_buffer": cfg.botsort_track_buffer,
            "match_thresh": cfg.botsort_match_thresh,
        }
    return {}
