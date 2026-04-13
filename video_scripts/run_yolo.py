"""
run_yolo.py
───────────
Runs YOLOv12 inference on raw original-resolution frames and writes
transparent PNG bounding-box overlays at the same resolution.

YOLO resizes frames internally for inference and returns bounding box
coordinates mapped back to the original image space, so overlays composite
correctly over the source video without any coordinate remapping.

Usage:
    python run_yolo.py --weights best.pt --frames ./frames --output ./overlays
    python run_yolo.py --weights best.pt --frames ./frames --output ./overlays \\
        --conf 0.25 --iou 0.45 --size 640 --device 0

Output structure:
    ./overlays/
        all/
            000000.png      <- all classes, transparent background, original resolution
            000001.png
            ...
        car/
            000000.png      <- only 'car' detections (fully transparent if none present)
            ...
        bus/  truck/  motorcycle/  bicycle/  pedestrian/  cyclist/
        summary.json        <- per-frame detection counts + colour map

Each PNG is RGBA:
    - Alpha = 0    -> fully transparent (background)
    - Alpha = 255  -> opaque (box outline + label text)
    - Label fill uses LABEL_ALPHA (semi-transparent)

Compositing with ffmpeg:
    ffmpeg -framerate 30 -i raw/%06d.png \\
           -framerate 30 -i overlays/all/%06d.png \\
           -filter_complex "[0][1]overlay" -c:v libx264 output.mp4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# Class → bounding-box colour mapping  (RGB)
# Edit to match your dataset's CLASS_NAMES order.
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "car",          # 0
    "bus",          # 1
    "truck",        # 2
    "motorcycle",   # 3
    "bicycle",      # 4
    "pedestrian",   # 5
    "cyclist",      # 6
]

CLASS_COLOURS = {
    "car":         (255,  82,  82),   # red
    "bus":         ( 68, 138, 255),   # blue
    "truck":       (105, 240, 174),   # green
    "motorcycle":  (255, 215,  64),   # amber
    "bicycle":     (234, 128, 252),   # purple
    "pedestrian":  (255, 109,   0),   # orange
    "cyclist":     ( 24, 255, 255),   # cyan
}

# Fallback colour for any class not in the map above
DEFAULT_COLOUR = (200, 200, 200)

# Visual settings
BOX_THICKNESS   = 2      # px outline width
FONT_SIZE       = 14     # pt
LABEL_PAD       = 3      # px padding inside label background
LABEL_ALPHA     = 200    # 0-255 opacity of label background fill


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_font(size: int):
    """Load a truetype font if available, else fall back to PIL default."""
    try:
        # Works on most Linux/Colab environments
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", size)
        except OSError:
            return ImageFont.load_default()


def _colour(class_name: str):
    return CLASS_COLOURS.get(class_name, DEFAULT_COLOUR)


def blank_rgba(width: int, height: int) -> Image.Image:
    """Fully transparent RGBA canvas."""
    return Image.new("RGBA", (width, height), (0, 0, 0, 0))


def draw_box(draw: ImageDraw.Draw, x1, y1, x2, y2,
             label: str, colour: tuple, font, img_w: int, img_h: int):
    """
    Draw an RGBA bounding box + label onto `draw`.
    Box outline is fully opaque; label background uses LABEL_ALPHA.
    """
    r, g, b = colour
    outline  = (r, g, b, 255)

    # Clamp coordinates
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img_w - 1, int(x2)), min(img_h - 1, int(y2))

    # Box outline — drawn as four filled rectangles (no fill so centre stays transparent)
    t = BOX_THICKNESS
    draw.rectangle([x1, y1, x2, y1 + t], fill=outline)        # top
    draw.rectangle([x1, y2 - t, x2, y2], fill=outline)        # bottom
    draw.rectangle([x1, y1, x1 + t, y2], fill=outline)        # left
    draw.rectangle([x2 - t, y1, x2, y2], fill=outline)        # right

    # Label background + text
    bbox_text = draw.textbbox((0, 0), label, font=font)
    tw = bbox_text[2] - bbox_text[0]
    th = bbox_text[3] - bbox_text[1]

    lx1 = x1
    ly1 = max(0, y1 - th - LABEL_PAD * 2)
    lx2 = lx1 + tw + LABEL_PAD * 2
    ly2 = ly1 + th + LABEL_PAD * 2

    draw.rectangle([lx1, ly1, lx2, ly2], fill=(r, g, b, LABEL_ALPHA))
    draw.text((lx1 + LABEL_PAD, ly1 + LABEL_PAD), label,
              fill=(255, 255, 255, 255), font=font)


# ─────────────────────────────────────────────────────────────────────────────
# Core inference + render
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(requested: str) -> str:
    """
    Return a valid device string, falling back to 'cpu' if the requested
    CUDA device is unavailable so the script never crashes on CPU-only machines.
    'auto' picks GPU 0 if available, otherwise cpu.
    """
    import torch
    if requested == "cpu":
        return "cpu"
    cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if requested == "auto":
        device = "0" if cuda_ok else "cpu"
        print(f"[INFO] Device auto-selected: {device}")
        return device
    # Explicit GPU request (e.g. "0", "0,1")
    if cuda_ok:
        return requested
    print(f"[INFO] CUDA not available (requested '{requested}') -- falling back to cpu")
    return "cpu"


def run_inference(
    weights: str,
    frames_dir: str,
    output_dir: str,
    conf: float,
    iou: float,
    img_size: int,
    device: str,
    class_names: list,
):
    # ── Resolve device (auto-fallback to cpu if CUDA unavailable) ───────────
    device = resolve_device(device)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading model: {weights}")
    model = YOLO(weights)

    # ── Always use raw/ frames (original resolution) ─────────────────────────
    # YOLO resizes internally and returns boxes in the input image's coordinate
    # space, so overlays will match the source video without any remapping.
    raw_dir = os.path.join(frames_dir, "raw")
    if os.path.isdir(raw_dir):
        src_dir = raw_dir
    else:
        # Fallback: treat frames_dir itself as the source
        src_dir = frames_dir

    print(f"Frame source : {src_dir}")

    frame_files = sorted(
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not frame_files:
        sys.exit(f"[ERROR] No image files found in {src_dir}")

    # ── Build output directories ─────────────────────────────────────────────
    all_dir = os.path.join(output_dir, "all")
    os.makedirs(all_dir, exist_ok=True)

    class_dirs = {}
    for name in class_names:
        d = os.path.join(output_dir, name.replace(" ", "_"))
        os.makedirs(d, exist_ok=True)
        class_dirs[name] = d

    font = _get_font(FONT_SIZE)

    print(f"\nRunning inference on {len(frame_files)} frames...")
    print(f"conf={conf}  iou={iou}  imgsz={img_size}  device={device}")
    print(f"Overlay resolution: original frame size (YOLO boxes remapped automatically)\n")

    detection_counts = {}   # frame_stem → {class: count}

    for i, fname in enumerate(frame_files):
        stem     = Path(fname).stem          # e.g. "000042"
        img_path = os.path.join(src_dir, fname)

        # Read to get dimensions
        img_cv   = cv2.imread(img_path)
        if img_cv is None:
            print(f"  [WARN] Cannot read {img_path}, skipping")
            continue
        h, w = img_cv.shape[:2]

        # ── Inference ────────────────────────────────────────────────────────
        results = model(
            img_path,
            conf=conf,
            iou=iou,
            imgsz=img_size,
            device=device,
            verbose=False,
        )[0]

        # ── Parse detections ─────────────────────────────────────────────────
        detections = []   # list of (x1, y1, x2, y2, class_name, conf_score)
        if results.boxes and len(results.boxes) > 0:
            xyxy   = results.boxes.xyxy.cpu().numpy()
            confs  = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, cls_ids):
                cname = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                detections.append((x1, y1, x2, y2, cname, float(score)))

        detection_counts[stem] = {}
        for *_, cname, _ in detections:
            detection_counts[stem][cname] = detection_counts[stem].get(cname, 0) + 1

        # ── Render: all-classes overlay ───────────────────────────────────────
        all_img = blank_rgba(w, h)
        draw_all = ImageDraw.Draw(all_img)
        for x1, y1, x2, y2, cname, score in detections:
            colour = _colour(cname)
            label  = f"{cname} {score:.2f}"
            draw_box(draw_all, x1, y1, x2, y2, label, colour, font, w, h)
        all_img.save(os.path.join(all_dir, f"{stem}.png"))

        # ── Render: per-class overlays ────────────────────────────────────────
        # Group detections by class
        by_class = {name: [] for name in class_names}
        for det in detections:
            cname = det[4]
            if cname in by_class:
                by_class[cname].append(det)

        for cname, dets in by_class.items():
            cls_img  = blank_rgba(w, h)
            draw_cls = ImageDraw.Draw(cls_img)
            for x1, y1, x2, y2, cn, score in dets:
                colour = _colour(cn)
                label  = f"{cn} {score:.2f}"
                draw_box(draw_cls, x1, y1, x2, y2, label, colour, font, w, h)
            cls_img.save(os.path.join(class_dirs[cname], f"{stem}.png"))

        if (i + 1) % 50 == 0 or (i + 1) == len(frame_files):
            print(f"  Processed {i + 1}/{len(frame_files)} frames")

    # ── Write detection summary ───────────────────────────────────────────────
    summary = {
        "weights":          os.path.abspath(weights),
        "frames_dir":       os.path.abspath(frames_dir),
        "output_dir":       os.path.abspath(output_dir),
        "conf_threshold":   conf,
        "iou_threshold":    iou,
        "img_size":         img_size,
        "class_names":      class_names,
        "class_colours":    {k: list(v) for k, v in CLASS_COLOURS.items()},
        "total_frames":     len(frame_files),
        "detection_counts": detection_counts,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Overlays written to: {output_dir}")
    print(f"Summary → {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="YOLOv12 video inference → transparent PNG overlays")
    p.add_argument("--weights", required=True,
                   help="Path to YOLO weights file (e.g. best.pt)")
    p.add_argument("--frames",  required=True,
                   help="Directory produced by extract_frames.py (must contain raw/ sub-folder)")
    p.add_argument("--output",  required=True,
                   help="Output directory for transparent PNG overlays")
    p.add_argument("--conf",    type=float, default=0.25,
                   help="Confidence threshold (default: 0.25)")
    p.add_argument("--iou",     type=float, default=0.45,
                   help="NMS IoU threshold (default: 0.45)")
    p.add_argument("--size",    type=int, default=640,
                   help="Inference image size (default: 640)")
    p.add_argument("--device",  default="auto",
                   help="Device: 'auto' detects GPU or falls back to cpu, '0' for GPU 0, 'cpu' to force cpu (default: auto)")
    p.add_argument("--classes", nargs="+", default=CLASS_NAMES,
                   help="Class names in index order (default: DAWN classes)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        weights=args.weights,
        frames_dir=args.frames,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        img_size=args.size,
        device=args.device,
        class_names=args.classes,
    )
