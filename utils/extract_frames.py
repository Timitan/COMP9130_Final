"""
extract_frames.py
─────────────────
Extracts frames from a video at original resolution.
YOLO handles resizing internally during inference and returns bounding box
coordinates in the original image's coordinate space, so no preprocessing
is needed here.

Usage:
    python extract_frames.py --video input.mp4 --output ./frames
    python extract_frames.py --video input.mp4 --output ./frames --fps 15

Output structure:
    ./frames/
        raw/
            000000.png
            000001.png
            ...
        meta.json   ← original resolution, fps, frame count, frame map
"""

import argparse
import json
import os
import sys

import cv2


def extract(video_path: str, output_dir: str, max_fps: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Frame-skip ratio so we don't exceed max_fps (0 = keep every frame)
    keep_every    = max(1, int(round(src_fps / max_fps))) if max_fps else 1
    effective_fps = src_fps / keep_every

    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Video        : {video_path}")
    print(f"Resolution   : {src_w}x{src_h}")
    print(f"Source FPS   : {src_fps:.2f}  |  Total frames: {total}")
    print(f"Keep 1 in {keep_every} -> effective {effective_fps:.2f} fps")
    print(f"Output       : {raw_dir}")

    src_idx   = 0   # frame counter from video stream
    out_idx   = 0   # output frame counter (0-based)
    frame_map = {}  # out_idx -> src_idx

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if src_idx % keep_every == 0:
            name = f"{out_idx:06d}"
            cv2.imwrite(os.path.join(raw_dir, f"{name}.png"), frame)
            frame_map[out_idx] = src_idx
            out_idx += 1

        src_idx += 1

    cap.release()

    meta = {
        "video_path":       os.path.abspath(video_path),
        "src_fps":          src_fps,
        "effective_fps":    effective_fps,
        "src_width":        src_w,
        "src_height":       src_h,
        "total_src_frames": src_idx,
        "total_out_frames": out_idx,
        "keep_every":       keep_every,
        "frame_map":        frame_map,
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nExtracted {out_idx} frames -> {raw_dir}")
    print(f"Meta         -> {meta_path}")
    return meta


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract video frames at original resolution for YOLO inference"
    )
    p.add_argument("--video",  required=True, help="Path to input video file")
    p.add_argument("--output", required=True, help="Output directory for frames")
    p.add_argument("--fps",    type=float, default=0,
                   help="Max output FPS -- 0 keeps every frame (default: 0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract(
        video_path=args.video,
        output_dir=args.output,
        max_fps=args.fps,
    )
