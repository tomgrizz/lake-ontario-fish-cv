"""
InferenceConfig: all runtime settings for inference pipeline v2.

model_checkpoint, output_dir, and video_roots must be set before use;
all other fields have working defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class InferenceConfig:

    # ---- Model ----
    # Path to local model checkpoint directory (required).
    model_checkpoint: str = ""
    # CUDA device index. Ignored when CUDA is unavailable.
    cuda_device: str = "0"

    # ---- Detection ----
    box_score_thresh: float = 0.6

    # ---- Tracker ----
    # Only "bytetrack" is recommended. "botsort" is available but uses
    # pedestrian-trained ReID weights, which are unreliable for fish.
    tracker: str = "bytetrack"

    # ByteTrack parameters (see config/bytetrack.yaml for annotated reference).
    # match_thresh is a cost threshold (cost = 1 - IoU); 0.99 accepts any IoU > 0.01.
    bytetrack_min_conf: float = 0.6
    bytetrack_track_thresh: float = 0.12
    bytetrack_match_thresh: float = 0.99
    bytetrack_track_buffer: int = 30

    # BotSort parameters (used only when tracker="botsort").
    botsort_weights: str = ""
    botsort_track_high_thresh: float = 0.8
    botsort_track_low_thresh: float = 0.3
    botsort_new_track_thresh: float = 0.9
    botsort_track_buffer: int = 25
    botsort_match_thresh: float = 0.9

    # ---- Track filtering ----
    # Tracks shorter than this are dropped from output.
    min_frames_for_track: int = 5
    # Minimum horizontal displacement (as fraction of frame width) to assign direction.
    min_displacement_frac: float = 0.05
    # Minimum horizontal displacement in pixels (max of frac and px is used).
    min_displacement_px: float = 20.0

    # ---- Video sources ----
    # Maps site name -> absolute root path for that site's videos.
    # Site name becomes the first component of each video_id.
    # Example: {"Ganaraska": "G:\\RiverWatcher\\Ganaraska",
    #            "Credit":    "G:\\RiverWatcher\\Credit"}
    video_roots: Dict[str, str] = field(default_factory=dict)

    # ---- Output ----
    # Run output directory (required). Typically run_YYYYMMDD_HHMMSS under
    # the outputs root documented in docs/local_paths.md.
    output_dir: str = ""
    # Flush per-track summaries to SQLite every N successfully processed videos.
    checkpoint_every_n: int = 50
