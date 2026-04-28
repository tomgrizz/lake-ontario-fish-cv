"""
Track-level scoring functions shared between pipeline.py (live inference)
and extract_clips.py (post-hoc computation for the current run).

Keeping this in one place prevents the two implementations from drifting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_best_id_frame(
    detections_df: pd.DataFrame,
    frame_width: int,
    frame_height: int,
) -> int:
    """Return the frame_number with the highest best_id_score for a track.

    best_id_score = detection_confidence
                  × bbox_area_normalized
                  × (1 − distance_from_center_normalized)

    Prioritizes the moment where the fish is large, centered, and confidently
    detected — not just the highest-confidence detection.

    Tie-break: earliest frame_number when multiple frames share the max score.

    Args:
        detections_df: DataFrame with columns:
            frame_number (int), bbox_x1, bbox_y1, bbox_x2, bbox_y2 (int/float),
            detection_confidence (float).
        frame_width:  Width of the source video frame in pixels.
        frame_height: Height of the source video frame in pixels.

    Returns:
        frame_number (int) of the best-ID frame.
    """
    if detections_df.empty:
        raise ValueError("detections_df must not be empty")

    fw, fh = float(frame_width), float(frame_height)

    cx = (detections_df["bbox_x1"] + detections_df["bbox_x2"]) / 2.0
    cy = (detections_df["bbox_y1"] + detections_df["bbox_y2"]) / 2.0
    bw = detections_df["bbox_x2"] - detections_df["bbox_x1"]
    bh = detections_df["bbox_y2"] - detections_df["bbox_y1"]

    bbox_area_norm = (bw * bh) / (fw * fh)

    # Distance from frame centre, normalised by the half-diagonal.
    dist = np.sqrt((cx - fw / 2) ** 2 + (cy - fh / 2) ** 2)
    half_diag = np.sqrt((fw / 2) ** 2 + (fh / 2) ** 2)
    dist_norm = dist / half_diag if half_diag > 0 else 0.0

    score = (
        detections_df["detection_confidence"].values
        * bbox_area_norm.values
        * (1.0 - dist_norm.values)
    )

    # idxmax returns the first occurrence of the maximum (earliest frame),
    # satisfying the tie-break requirement.
    best_idx = int(np.argmax(score))
    return int(detections_df.iloc[best_idx]["frame_number"])
