"""
compose_video.py
----------------
Composites raw video frames with transparent PNG overlays produced by
run_yolo.py into an annotated MP4.

Usage:
    python compose_video.py --frames ./frames --overlays ./overlays --output annotated.mp4
    python compose_video.py --frames ./frames --overlays ./overlays --output annotated.mp4 \
        --fps 30 --crf 18 --preset fast

How it works:
    1. Reads raw frames from  ./frames/raw/
    2. Reads RGBA overlays from ./overlays/all/
    3. Alpha-composites overlay onto raw frame (preserves original resolution)
    4. Writes composited frames to MP4 via OpenCV VideoWriter
    5. Re-encodes with libx264 via ffmpeg if available (better quality/size)

Output:
    annotated.mp4  at original video resolution and fps
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_meta(frames_dir: str) -> dict:
    meta_path = os.path.join(frames_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def composite(raw_bgr: np.ndarray, overlay_rgba: Image.Image) -> np.ndarray:
    """
    Alpha-composite an RGBA PIL overlay onto a BGR numpy frame.
    Returns a BGR numpy array at the same resolution.
    """
    base = Image.fromarray(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # Guard against size mismatch (should never happen if scripts ran correctly)
    if overlay_rgba.size != base.size:
        overlay_rgba = overlay_rgba.resize(base.size, Image.NEAREST)

    composited = Image.alpha_composite(base, overlay_rgba)
    return cv2.cvtColor(np.array(composited.convert("RGB")), cv2.COLOR_RGB2BGR)


def compose(frames_dir, overlays_dir, output_path, fps_override, crf, preset):

    raw_dir     = os.path.join(frames_dir, "raw")
    overlay_dir = os.path.join(overlays_dir, "all")

    if not os.path.isdir(raw_dir):
        sys.exit(f"[ERROR] Raw frames directory not found: {raw_dir}")
    if not os.path.isdir(overlay_dir):
        sys.exit(f"[ERROR] Overlay directory not found: {overlay_dir}")

    # Collect and sort raw frames
    raw_files = sorted(
        f for f in os.listdir(raw_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not raw_files:
        sys.exit(f"[ERROR] No frames found in {raw_dir}")

    # Resolve FPS: CLI override -> meta.json -> fallback 30
    meta = load_meta(frames_dir)
    fps  = fps_override or meta.get("effective_fps") or meta.get("src_fps") or 30.0

    # Get frame dimensions from first raw frame
    first = cv2.imread(os.path.join(raw_dir, raw_files[0]))
    if first is None:
        sys.exit(f"[ERROR] Cannot read first frame: {raw_files[0]}")
    h, w = first.shape[:2]

    # Set up VideoWriter (mp4v; re-encoded to libx264 below if ffmpeg available)
    os.makedirs(str(Path(output_path).parent), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        sys.exit(f"[ERROR] Could not open VideoWriter for: {output_path}")

    print(f"Frames       : {len(raw_files)}  ({w}x{h}  {fps:.2f} fps)")
    print(f"Raw dir      : {raw_dir}")
    print(f"Overlay dir  : {overlay_dir}")
    print(f"Output       : {output_path}")
    print()

    missing_overlays = 0

    for i, fname in enumerate(raw_files):
        stem     = Path(fname).stem
        raw_path = os.path.join(raw_dir, fname)
        ovl_path = os.path.join(overlay_dir, f"{stem}.png")

        raw_bgr = cv2.imread(raw_path)
        if raw_bgr is None:
            print(f"  [WARN] Cannot read {fname}, writing black frame")
            raw_bgr = np.zeros((h, w, 3), dtype=np.uint8)

        if os.path.isfile(ovl_path):
            overlay_rgba = Image.open(ovl_path).convert("RGBA")
            out_frame = composite(raw_bgr, overlay_rgba)
        else:
            # No overlay for this frame — write raw frame unchanged
            missing_overlays += 1
            out_frame = raw_bgr

        writer.write(out_frame)

        if (i + 1) % 100 == 0 or (i + 1) == len(raw_files):
            print(f"  Composited {i + 1}/{len(raw_files)} frames")

    writer.release()

    if missing_overlays:
        print(f"\n  [NOTE] {missing_overlays} frames had no overlay (written as raw)")

    print(f"\nWritten (mp4v) -> {output_path}")

    # Re-encode with libx264 via ffmpeg for better quality and compatibility.
    # mp4v from OpenCV is a valid fallback if ffmpeg is unavailable.
    tmp_path = output_path + ".tmp.mp4"
    try:
        import subprocess
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", output_path,
                "-c:v", "libx264",
                "-crf", str(crf),
                "-preset", preset,
                "-pix_fmt", "yuv420p",   # broadest player compatibility
                tmp_path,
            ],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            os.replace(tmp_path, output_path)
            print(f"Re-encoded with libx264  crf={crf}  preset={preset}")
        else:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print("ffmpeg re-encode failed — keeping mp4v output")
            print(result.stderr[-500:] if result.stderr else "")
    except FileNotFoundError:
        print("ffmpeg not found — keeping mp4v output (still playable)")

    print(f"\nDone -> {output_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Composite raw frames + YOLO overlays into an annotated MP4"
    )
    p.add_argument("--frames",   required=True,
                   help="Frame directory from extract_frames.py (contains raw/)")
    p.add_argument("--overlays", required=True,
                   help="Overlay directory from run_yolo.py (contains all/)")
    p.add_argument("--output",   required=True,
                   help="Output MP4 path  e.g. ./annotated.mp4")
    p.add_argument("--fps",      type=float, default=0,
                   help="Output FPS -- 0 reads from meta.json or defaults to 30 (default: 0)")
    p.add_argument("--crf",      type=int, default=18,
                   help="libx264 CRF quality 0=lossless 51=worst, used if ffmpeg available (default: 18)")
    p.add_argument("--preset",   default="fast",
                   choices=["ultrafast","superfast","veryfast","faster",
                            "fast","medium","slow","slower","veryslow"],
                   help="libx264 preset -- slower=smaller file (default: fast)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compose(
        frames_dir=args.frames,
        overlays_dir=args.overlays,
        output_path=args.output,
        fps_override=args.fps,
        crf=args.crf,
        preset=args.preset,
    )
