"""
Benchmark video decoding backends: OpenCV vs decord CPU vs decord GPU.

Also benchmarks full inference pipeline with each backend + GPU preprocessing
(bypasses HuggingFace image_processor for the decord-GPU path, which keeps
frames on GPU end-to-end).

Usage:
  python bench_decode.py --site Ganaraska=G:/RiverWatcher/Ganaraska --n-videos 10
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# DETR ImageNet normalisation constants
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Decoding backends
# ---------------------------------------------------------------------------

def decode_opencv(path: str) -> tuple[int, float]:
    """Return (n_frames_decoded, seconds)."""
    cap = cv2.VideoCapture(path)
    n = 0
    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        n += 1
    cap.release()
    return n, time.perf_counter() - t0


def decode_decord_cpu(path: str) -> tuple[int, float]:
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0))
    n = len(vr)
    t0 = time.perf_counter()
    for i in range(n):
        _ = vr[i].asnumpy()  # RGB uint8 HxWxC
    return n, time.perf_counter() - t0


def decode_decord_gpu(path: str, device_id: int = 0) -> tuple[int, float]:
    from decord import VideoReader, gpu
    vr = VideoReader(path, ctx=gpu(device_id))
    n = len(vr)
    t0 = time.perf_counter()
    for i in range(n):
        _ = vr[i]  # stays on GPU as decord NDArray
    torch.cuda.synchronize()
    return n, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Full inference pipeline variants
# ---------------------------------------------------------------------------

def _preprocess_hf(frame_rgb_np, image_processor, device, fp16):
    """Standard HuggingFace image_processor path (CPU preprocessing)."""
    from PIL import Image
    pil = Image.fromarray(frame_rgb_np)
    inputs = image_processor(pil, return_tensors="pt")
    pv = inputs["pixel_values"].to(device)
    return pv.half() if fp16 else pv


def _preprocess_gpu(frame_rgb_np, target_h, target_w, device, fp16):
    """GPU preprocessing path: normalize + resize entirely on CUDA."""
    import torchvision.transforms.functional as TF
    # (H,W,3) uint8 -> (3,H,W) float32 on GPU
    t = torch.from_numpy(frame_rgb_np).to(device).permute(2, 0, 1).float().div_(255.0)
    mean = _MEAN.to(device).view(3, 1, 1)
    std  = _STD.to(device).view(3, 1, 1)
    t = (t - mean) / std
    t = TF.resize(t.unsqueeze(0), [target_h, target_w]).squeeze(0)
    return (t.unsqueeze(0).half() if fp16 else t.unsqueeze(0))


def _get_resize_dims(h, w, min_size=800, max_size=1333):
    """DETR resize: shortest side -> min_size, longest capped at max_size."""
    scale = min_size / min(h, w)
    nh, nw = int(h * scale), int(w * scale)
    if max(nh, nw) > max_size:
        scale = max_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
    return nh, nw


def bench_pipeline(videos: list[str], model, image_processor, device: str,
                   fp16: bool, frame_skip: int, n_frames_limit: int,
                   label: str) -> float:
    """Run model inference on videos; return mean videos/sec."""
    total_time = 0.0
    n_done = 0

    for path in videos:
        cap = cv2.VideoCapture(path)
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 25.0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        th, tw = _get_resize_dims(h, w)
        cap.release()

        # -- OpenCV + HF image_processor (current pipeline)
        if label.startswith("opencv"):
            cap = cv2.VideoCapture(path)
            n_frames = 0
            t0 = time.perf_counter()
            fi = 0
            while n_frames < n_frames_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                fi += 1
                if fi % frame_skip != 0:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pv = _preprocess_hf(frame_rgb, image_processor, device, fp16)
                with torch.no_grad():
                    _ = model(pixel_values=pv)
                n_frames += 1
            cap.release()
            total_time += time.perf_counter() - t0

        # -- decord CPU + HF image_processor
        elif label.startswith("decord_cpu"):
            from decord import VideoReader, cpu as dcpu
            vr = VideoReader(path, ctx=dcpu(0))
            n_total = len(vr)
            idxs = list(range(0, n_total, frame_skip))[:n_frames_limit]
            t0 = time.perf_counter()
            for i in idxs:
                frame_rgb = vr[i].asnumpy()
                pv = _preprocess_hf(frame_rgb, image_processor, device, fp16)
                with torch.no_grad():
                    _ = model(pixel_values=pv)
            total_time += time.perf_counter() - t0

        # -- decord CPU + GPU preprocessing (CPU decode, GPU preprocess+infer)
        elif label == "decord_cpu+gpu_preprocess":
            from decord import VideoReader, cpu as dcpu
            vr = VideoReader(path, ctx=dcpu(0))
            n_total = len(vr)
            idxs = list(range(0, n_total, frame_skip))[:n_frames_limit]
            t0 = time.perf_counter()
            for i in idxs:
                frame_np = vr[i].asnumpy()  # RGB uint8, CPU
                pv = _preprocess_gpu(frame_np, th, tw, device, fp16)
                with torch.no_grad():
                    _ = model(pixel_values=pv)
            torch.cuda.synchronize()
            total_time += time.perf_counter() - t0

        # -- decord GPU + GPU preprocessing (fully GPU pipeline)
        elif label.startswith("decord_gpu"):
            from decord import VideoReader, gpu as dgpu
            vr = VideoReader(path, ctx=dgpu(0))
            n_total = len(vr)
            idxs = list(range(0, n_total, frame_skip))[:n_frames_limit]
            t0 = time.perf_counter()
            for i in idxs:
                frame_np = vr[i].asnumpy()  # to numpy first (decord 0.6 limitation)
                pv = _preprocess_gpu(frame_np, th, tw, device, fp16)
                with torch.no_grad():
                    _ = model(pixel_values=pv)
            torch.cuda.synchronize()
            total_time += time.perf_counter() - t0

        n_done += 1

    return n_done / total_time if total_time > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--site", action="append", default=[])
    ap.add_argument("--n-videos", type=int, default=10)
    ap.add_argument("--n-frames", type=int, default=200,
                    help="Max frames per video for inference benchmarks.")
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    # Collect videos
    videos = []
    for spec in args.site:
        name, _, root = spec.partition("=")
        for p in Path(root).rglob("*"):
            if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file():
                videos.append(str(p))
    if not videos:
        print("No videos found."); sys.exit(1)

    random.shuffle(videos)
    bench_vids = videos[: args.n_videos]
    print(f"Benchmarking on {len(bench_vids)} videos (frame_skip={args.frame_skip}, "
          f"n_frames_limit={args.n_frames}, fp16={args.fp16})\n")

    # --- 1. Raw decode speed (no inference) ---
    print("=== Raw decode speed (frames/sec) ===")
    for label, fn in [("opencv    ", decode_opencv),
                       ("decord_cpu", decode_decord_cpu),
                       ("decord_gpu", decode_decord_gpu)]:
        total_frames, total_time = 0, 0.0
        for v in bench_vids:
            try:
                nf, t = fn(v)
                total_frames += nf
                total_time += t
            except Exception as e:
                print(f"  {label} error on {Path(v).name}: {e}")
        fps = total_frames / total_time if total_time > 0 else 0
        print(f"  {label}: {fps:,.0f} frames/sec  "
              f"({total_frames:,} frames in {total_time:.1f}s)")

    # --- 2. Full inference pipeline (videos/sec) ---
    print("\n=== Full inference pipeline (videos/sec) ===")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    print("Loading model …")
    model = AutoModelForObjectDetection.from_pretrained(
        args.model_dir, local_files_only=True
    ).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_dir, local_files_only=True
    )
    if args.fp16:
        model = model.half()

    for label in ["opencv+hf", "decord_cpu+hf", "decord_cpu+gpu_preprocess", "decord_gpu+gpu_preprocess"]:
        try:
            vps = bench_pipeline(bench_vids, model, image_processor, device,
                                 args.fp16, args.frame_skip, args.n_frames, label)
            days = (258955 / vps / 3600 / 24) if vps > 0 else float("inf")
            print(f"  {label:30s}: {vps:.3f} videos/sec  "
                  f"-> {days:.1f} days for 258k videos")
        except Exception as e:
            print(f"  {label:30s}: ERROR - {e}")


if __name__ == "__main__":
    main()
