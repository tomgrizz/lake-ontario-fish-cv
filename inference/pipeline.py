"""
Per-video inference: DETR detection + tracker → structured records.

process_video() is the single entry point. No I/O is performed here;
writing is handled by writer.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from decord import VideoReader as _DecordVR
    from decord import cpu as _decord_cpu
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False

from config import InferenceConfig
from schema import (
    BACKGROUND_CLASS_IDX,
    NUM_CLASSES,
    DetectionRecord,
    TrackSummary,
)


@dataclass
class VideoResult:
    detections: List[DetectionRecord]
    tracks: List[TrackSummary]


def process_video(
    video_path: str,
    video_id: str,
    model,
    image_processor,
    cfg: InferenceConfig,
    device: str,
) -> VideoResult:
    """Run DETR detection and tracking on a single video.

    Args:
        video_path:      Absolute path to the video file.
        video_id:        Relative path identifier stored in all output records.
        model:           Loaded DETR model (on device, eval mode).
        image_processor: HuggingFace AutoImageProcessor for the model.
        cfg:             InferenceConfig with detection and tracker settings.
        device:          PyTorch device string, e.g. "cuda:0" or "cpu".

    Returns:
        VideoResult containing all DetectionRecords and TrackSummaries.

    Raises:
        IOError: if the video cannot be opened.
    """
    fps, total_frames, height, width = _video_meta(video_path)
    tracker = _build_tracker(cfg, fps, device)

    # track_id -> {boxes, confs, frames, class_probs}
    # boxes/confs/frames accumulate for every tracker output (including Kalman-only
    # frames with no matched detection), matching reference behaviour for direction
    # and side calculations. class_probs accumulates only for real detections
    # (det_ind valid); Kalman-only frames contribute zeros to avoid diluting means.
    track_state: Dict[int, dict] = {}
    all_detections: List[DetectionRecord] = []
    detection_id = 0
    last_frame_idx = 0

    try:
        for frame_idx, frame_rgb in _iter_frames(video_path, cfg.frame_skip):
            last_frame_idx = frame_idx
            pil_image = Image.fromarray(frame_rgb)

            inputs = image_processor(pil_image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            if cfg.fp16:
                pixel_values = pixel_values.half()

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)

            # Full 6-class softmax over all logits (5 fish + Background at index 5).
            # Single computation used for both the keep mask and class_probs, removing
            # the dual-computation alignment risk present in the reference code.
            probs_full = torch.softmax(outputs.logits[0], dim=-1)  # (Q, 6)
            fish_max = probs_full[:, :BACKGROUND_CLASS_IDX].max(dim=-1).values
            keep = fish_max > cfg.box_score_thresh
            class_probs_np = probs_full[keep].cpu().numpy()  # (N_kept, 6)

            # The DETR image processor letterboxes every frame into a fixed square
            # (pad_size, e.g. 800x800 for this checkpoint): resize keeping aspect
            # ratio so longest side fits, then zero-pad bottom/right to a square.
            # The model's outputs are normalized to that padded canvas, so
            # post_process target_sizes must equal padded_size / resize_scale,
            # not the original frame size. For 1280x960 with pad 800x800, that's
            # (1280, 1280) — passing (960, 1280) under-scales Y by 0.75 and shifts
            # boxes up. pixel_values.shape[-2:] gives the padded canvas; deriving
            # the scale from there is robust to different processor configs.
            padded_h, padded_w = pixel_values.shape[-2:]
            resize_scale = min(padded_h / height, padded_w / width)
            target_h = round(padded_h / resize_scale)
            target_w = round(padded_w / resize_scale)
            detections = image_processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([[target_h, target_w]]),
                threshold=cfg.box_score_thresh,
            )[0]

            boxes = detections["boxes"].cpu().numpy()
            scores = detections["scores"].cpu().numpy()
            labels = detections["labels"].cpu().numpy()

            if len(boxes) == 0:
                dets = np.empty((0, 6))
            else:
                dets = np.column_stack([boxes, scores, labels])

            # BYTETracker ignores the frame argument (used only by appearance-based
            # trackers). Pass the RGB array; it's unused either way.
            res = tracker.update(dets, frame_rgb)

            for track in res:
                x1, y1, x2, y2, track_id, conf, cls, det_ind = track
                # Clamp to frame bounds. DETR can predict boxes slightly outside
                # [0, width] × [0, height]; the tracker Kalman state can too.
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(width, int(x2))
                y2 = min(height, int(y2))
                track_id, cls, det_ind = int(track_id), int(cls), int(det_ind)

                is_real_detection = 0 <= det_ind < len(class_probs_np)
                prob_vec = (
                    class_probs_np[det_ind]
                    if is_real_detection
                    else np.zeros(NUM_CLASSES, dtype=np.float32)
                )

                if is_real_detection:
                    pred_fish = int(np.argmax(prob_vec[:BACKGROUND_CLASS_IDX]))
                    pred_6 = int(np.argmax(prob_vec))
                    all_detections.append(DetectionRecord(
                        video_id=video_id,
                        frame_number=frame_idx,
                        timestamp_seconds=float(frame_idx / fps),
                        track_id=track_id,
                        detection_id=detection_id,
                        bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                        detection_confidence=float(conf),
                        prob_chinook=float(prob_vec[0]),
                        prob_coho=float(prob_vec[1]),
                        prob_atlantic=float(prob_vec[2]),
                        prob_rainbow=float(prob_vec[3]),
                        prob_brown=float(prob_vec[4]),
                        prob_background=float(prob_vec[5]),
                        predicted_class=pred_fish,
                        predicted_class_6=pred_6,
                    ))
                    detection_id += 1

                if track_id not in track_state:
                    track_state[track_id] = {
                        "boxes": [], "confs": [], "frames": [], "class_probs": [],
                    }
                track_state[track_id]["boxes"].append([x1, y1, x2, y2])
                track_state[track_id]["confs"].append(float(conf))
                track_state[track_id]["frames"].append(frame_idx)
                track_state[track_id]["class_probs"].append(prob_vec)

    finally:
        pass  # reader cleanup handled inside _iter_frames

    tracks = _build_tracks(track_state, video_id, fps, width, cfg)
    return VideoResult(detections=all_detections, tracks=tracks)


# ---------------------------------------------------------------------------
# Video reading helpers (decord with OpenCV fallback)
# ---------------------------------------------------------------------------

def _video_meta(video_path: str) -> tuple[float, int, int, int]:
    """Return (fps, total_frames, height, width) using whichever reader is available."""
    if _DECORD_AVAILABLE:
        try:
            vr = _DecordVR(video_path, ctx=_decord_cpu(0))
            fps = float(vr.get_avg_fps()) or 25.0
            total_frames = len(vr)
            h, w = vr[0].shape[:2]
            return fps, total_frames, h, w
        except Exception:
            pass  # fall through to OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return fps, total_frames, height, width


def _iter_frames(video_path: str, frame_skip: int):
    """Yield (frame_idx, frame_rgb_uint8) for every sampled frame.

    Uses decord when available (2x faster CPU decode than OpenCV, returns RGB
    directly). Falls back to OpenCV on ImportError or decord open failure.
    """
    if _DECORD_AVAILABLE:
        try:
            vr = _DecordVR(video_path, ctx=_decord_cpu(0))
            indices = range(0, len(vr), frame_skip)
            for frame_idx in indices:
                yield frame_idx, vr[frame_idx].asnumpy()  # already RGB
            return
        except Exception:
            pass  # fall through to OpenCV

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                yield frame_idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Track summarisation
# ---------------------------------------------------------------------------

def _build_tracks(
    track_state: Dict[int, dict],
    video_id: str,
    fps: float,
    frame_width: int,
    cfg: InferenceConfig,
) -> List[TrackSummary]:
    summaries = []
    min_disp = max(frame_width * cfg.min_displacement_frac, cfg.min_displacement_px)

    for track_id, state in track_state.items():
        frames = state["frames"]
        if len(frames) < cfg.min_frames_for_track:
            continue

        boxes = state["boxes"]
        confs = state["confs"]
        class_probs = state["class_probs"]

        # Direction: 3-frame average center_x at start vs end (preserves reference logic).
        centers_x = [(b[0] + b[2]) / 2.0 for b in boxes]
        window = min(3, len(centers_x))
        delta_x = float(np.mean(centers_x[-window:])) - float(np.mean(centers_x[:window]))
        if abs(delta_x) >= min_disp:
            direction = "Right" if delta_x > 0 else "Left"
        else:
            direction = "Unknown"

        entrance_side = _horizontal_side(boxes[0], frame_width)
        exit_side = _horizontal_side(boxes[-1], frame_width)

        # Mean class probabilities — unweighted (each frame contributes equally).
        # This matches existing reference behaviour. Consider confidence-weighted
        # mean as a future enhancement if low-confidence start/end frames are
        # observed to drag down track-level species predictions.
        probs_arr = np.stack(class_probs, axis=0)   # (n_frames, 6)
        mean_probs = probs_arr.mean(axis=0)          # (6,)

        predicted_class = int(np.argmax(mean_probs[:BACKGROUND_CLASS_IDX]))
        predicted_class_6 = int(np.argmax(mean_probs))

        # Representative frame: earliest frame with maximum detection_confidence.
        # Tie-break = smallest frame_number (confs.index() returns first occurrence).
        max_conf = max(confs)
        rep_frame = frames[confs.index(max_conf)]

        start_frame, end_frame = frames[0], frames[-1]
        summaries.append(TrackSummary(
            video_id=video_id,
            track_id=track_id,
            start_frame=start_frame,
            end_frame=end_frame,
            start_timestamp_seconds=float(start_frame / fps),
            end_timestamp_seconds=float(end_frame / fps),
            n_frames=len(frames),
            mean_prob_chinook=float(mean_probs[0]),
            mean_prob_coho=float(mean_probs[1]),
            mean_prob_atlantic=float(mean_probs[2]),
            mean_prob_rainbow=float(mean_probs[3]),
            mean_prob_brown=float(mean_probs[4]),
            mean_prob_background=float(mean_probs[5]),
            predicted_class=predicted_class,
            predicted_class_6=predicted_class_6,
            mean_detection_confidence=float(np.mean(confs)),
            direction=direction,
            entrance_side=entrance_side,
            exit_side=exit_side,
            representative_frame=rep_frame,
        ))

    return summaries


def _horizontal_side(box: list, frame_width: int) -> str:
    center_x = (box[0] + box[2]) / 2.0
    if center_x < frame_width * 0.5:
        return "Left"
    if center_x > frame_width * 0.5:
        return "Right"
    return "None"


# ---------------------------------------------------------------------------
# Tracker construction
# ---------------------------------------------------------------------------

def _build_tracker(cfg: InferenceConfig, fps: float, device: str):
    """Build a fresh tracker instance for one video."""
    frame_rate = max(1, round(fps))

    if cfg.tracker == "bytetrack":
        # boxmot >=10: BYTETracker (no min_conf parameter; pre-filtering to
        # box_score_thresh upstream makes it redundant anyway).
        from boxmot import BYTETracker
        return BYTETracker(
            track_thresh=cfg.bytetrack_track_thresh,
            match_thresh=cfg.bytetrack_match_thresh,
            track_buffer=cfg.bytetrack_track_buffer,
            frame_rate=frame_rate,
        )

    if cfg.tracker == "botsort":
        # boxmot >=10: BoTSORT (fp16 replaces half; model_weights replaces reid_weights).
        from boxmot import BoTSORT
        return BoTSORT(
            model_weights=Path(cfg.botsort_weights),
            device=torch.device(device),
            fp16=False,
            track_high_thresh=cfg.botsort_track_high_thresh,
            track_low_thresh=cfg.botsort_track_low_thresh,
            new_track_thresh=cfg.botsort_new_track_thresh,
            track_buffer=cfg.botsort_track_buffer,
            match_thresh=cfg.botsort_match_thresh,
            frame_rate=frame_rate,
        )

    raise ValueError(f"Unknown tracker: {cfg.tracker!r}. Use 'bytetrack' or 'botsort'.")
