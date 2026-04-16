import kagglehub
import os
import shutil
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict
from typing import Optional, Union


# ─────────────────────────────────────────────
# 1. DATASET ACQUISITION & FILTERING
# ─────────────────────────────────────────────

WEATHER_CONDITIONS = ["fog", "rain", "snow", "sand"]

# Class names in 0-based index order, matching remapped YOLO label files.
# Labels have been remapped from original DAWN indices [1,2,3,4,6,7,8] → [0,1,2,3,4,5].
# CLASS_NAMES = ["car", "bus", "truck", "motorcycle", "bicycle", "pedestrian", "cyclist"]
CLASS_NAMES = ["pedestrian", "bicycle", "car", "motorcycle", "bus", "truck"]
LABEL_LIST  = CLASS_NAMES   # alias used by confusion matrix / visualisation helpers
NUM_CLASSES = len(CLASS_NAMES)  # 6


def get_dataset():
    """Download the DAWN dataset from Kaggle and sort images by weather condition."""
    os.makedirs("./data", exist_ok=True)
    path = kagglehub.dataset_download("shuvoalok/dawn-dataset", output_dir="./data")
    print("data/", path)

    output_dir = "./data/filtered"
    for cond in WEATHER_CONDITIONS:
        os.makedirs(os.path.join(output_dir, cond), exist_ok=True)

    def _filter(input_dir, out):
        for filename in os.listdir(input_dir):
            if not filename.endswith(".jpg"):
                continue
            name = filename.lower()
            if "snow" in name:
                dest = os.path.join(out, "snow", filename)
            elif "rain" in name or "mist" in name:
                dest = os.path.join(out, "rain", filename)
            elif "fog" in name or "haze" in name:
                dest = os.path.join(out, "fog", filename)
            elif "sand" in name or "dust" in name:
                dest = os.path.join(out, "sand", filename)
            else:
                continue
            shutil.copy(os.path.join(input_dir, filename), dest)

    _filter("./data/images", output_dir)


# ─────────────────────────────────────────────
# 2. DATASET READING
# ─────────────────────────────────────────────

def read_dataset(
    data_root: str = "./data/filtered",
    label_root: str = "./data/labels",
    conditions: Optional[list] = None,
) -> dict:
    """
    Read dataset where:
      - images are in ./data/filtered/{condition}
      - labels are flat in ./data/labels

    Returns
    -------
    dict: {condition: [{"image": path, "label": path_or_none}, ...]}
    """
    conditions = conditions or WEATHER_CONDITIONS
    dataset = {}

    for cond in conditions:
        img_dir = os.path.join(data_root, cond)

        if not os.path.isdir(img_dir):
            dataset[cond] = []
            continue

        entries = []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(img_dir, fname)
            stem = Path(fname).stem

            lbl_path = os.path.join(label_root, stem + ".txt")

            entries.append({
                "image": img_path,
                "label": lbl_path if os.path.isfile(lbl_path) else None
            })

        dataset[cond] = entries

    return dataset


def dataset_summary(dataset: dict) -> None:
    """Print a summary table of image counts per weather condition."""
    print(f"{'Condition':<12} {'Images':>8}")
    print("-" * 22)
    total = 0
    for cond, entries in dataset.items():
        print(f"{cond:<12} {len(entries):>8}")
        total += len(entries)
    print("-" * 22)
    print(f"{'TOTAL':<12} {total:>8}")


# ─────────────────────────────────────────────
# 3. DATASET SPLITTING (25 / 50 / 75 / 100%)
# ─────────────────────────────────────────────

def split_dataset(
    image_dir: str,
    label_dir: str,
    output_root: str = "./data/splits",
    fractions: tuple = (0.25, 0.50, 0.75, 1.0),
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> dict:
    """
    Split and balance the DAWN dataset into multiple fractions for convergence experiments.

    Parameters
    ----------
    image_dir : str   Path to flat image directory (all conditions mixed, YOLO layout).
    label_dir : str   Matching YOLO label directory.
    output_root : str Where to write split sub-datasets.
    fractions : tuple Fractions of total data to experiment with.
    val_ratio  : float Fraction of each split reserved for validation.
    test_ratio : float Fraction of each split reserved for testing.
    seed : int        Random seed for reproducibility.

    Returns
    -------
    splits : dict   {fraction: {"train": [...], "val": [...], "test": [...]}}
    """
    random.seed(seed)
    np.random.seed(seed)

    # Collect all (image, label) pairs
    all_pairs = []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(image_dir, fname)
        stem = Path(fname).stem
        lbl_path = os.path.join(label_dir, stem + ".txt")
        if os.path.isfile(lbl_path):
            all_pairs.append((img_path, lbl_path))

    random.shuffle(all_pairs)
    n_total = len(all_pairs)

    # Balance by class: read labels, bucket images per class
    class_buckets = defaultdict(list)
    for img_path, lbl_path in all_pairs:
        with open(lbl_path) as f:
            classes_in_image = {int(line.split()[0]) for line in f if line.strip()}
        for cls_id in classes_in_image:
            class_buckets[cls_id].append((img_path, lbl_path))

    splits = {}
    for frac in fractions:
        n_frac = max(1, int(n_total * frac))
        # Sample proportionally from each class bucket then de-duplicate
        sampled = set()
        per_class = max(1, n_frac // max(len(class_buckets), 1))
        for cls_id, pairs in class_buckets.items():
            for pair in random.sample(pairs, min(per_class, len(pairs))):
                sampled.add(pair)
            if len(sampled) >= n_frac:
                break
        # Fill remainder randomly if needed
        remaining = [p for p in all_pairs if p not in sampled]
        while len(sampled) < n_frac and remaining:
            sampled.add(remaining.pop())

        sampled = list(sampled)
        random.shuffle(sampled)

        n_val  = max(1, int(len(sampled) * val_ratio))
        n_test = max(1, int(len(sampled) * test_ratio))
        val_set   = sampled[:n_val]
        test_set  = sampled[n_val:n_val + n_test]
        train_set = sampled[n_val + n_test:]

        frac_key = f"{int(frac * 100)}pct"
        splits[frac_key] = {"train": train_set, "val": val_set, "test": test_set}

        # Write symlinked YOLO directory structure
        for split_name, pairs in splits[frac_key].items():
            for sub in ("images", "labels"):
                os.makedirs(os.path.join(output_root, frac_key, split_name, sub), exist_ok=True)
            for img_path, lbl_path in pairs:
                fname = Path(img_path).name
                dst_img = os.path.join(output_root, frac_key, split_name, "images", fname)
                dst_lbl = os.path.join(output_root, frac_key, split_name, "labels", Path(lbl_path).name)
                if not os.path.exists(dst_img):
                    shutil.copy(img_path, dst_img)
                if not os.path.exists(dst_lbl):
                    shutil.copy(lbl_path, dst_lbl)

        print(f"[{frac_key}]  train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    return splits


def write_yolo_yaml(
    split_root: str,
    frac_key: str,
    num_classes: int = NUM_CLASSES,
    output_path: Optional[str] = None,
) -> str:
    """Write a YOLO data.yaml for a given fraction split."""
    import yaml as _yaml
    base = os.path.join(split_root, frac_key)
    data = {
        "path":  os.path.abspath(base),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    num_classes,
        "names": CLASS_NAMES,
    }
    output_path = output_path or os.path.join(base, "data.yaml")
    with open(output_path, "w") as f:
        _yaml.dump(data, f, default_flow_style=False, sort_keys=True)
    print(f"YAML written to {output_path}")
    return output_path


# ─────────────────────────────────────────────
# 4. IMAGE PREPROCESSING & AUGMENTATION
# ─────────────────────────────────────────────

def preprocess_and_augment(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    target_size: tuple = (640, 640),
    augment: bool = True,
    seed: int = 42,
) -> None:
    """
    Works with:
        image_dir = ./data/filtered/{cond}
        label_dir = ./data/labels  (flat)

    Output:
        ./data/augmented/{cond}/images
        ./data/augmented/{cond}/labels
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("pip install opencv-python")

    random.seed(seed)
    np.random.seed(seed)

    out_img_dir = os.path.join(output_dir, "images")
    out_lbl_dir = os.path.join(output_dir, "labels")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    seen_hashes = set()
    processed = 0
    skipped = 0
    missing_labels = 0

    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(image_dir, fname)
        img = cv2.imread(img_path)

        if img is None:
            skipped += 1
            continue

        # Deduplicate
        img_hash = hash(img.tobytes())
        if img_hash in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(img_hash)

        # Resize
        img = cv2.resize(img, target_size)

        stem = Path(fname).stem

        lbl_src = os.path.join(label_dir, stem + ".txt")
        has_label = os.path.isfile(lbl_src)

        if not has_label:
            missing_labels += 1  # debug visibility

        # ───────── CLEAN ─────────
        clean_img = f"{stem}_clean.jpg"
        clean_lbl = f"{stem}_clean.txt"

        cv2.imwrite(os.path.join(out_img_dir, clean_img), img)

        if has_label:
            shutil.copy(lbl_src, os.path.join(out_lbl_dir, clean_lbl))

        if not augment:
            processed += 1
            continue

        # ───────── GAUSSIAN NOISE ─────────
        sigma = random.uniform(20, 30)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        aug = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        name = f"{stem}_gauss"
        cv2.imwrite(os.path.join(out_img_dir, name + ".jpg"), aug)
        if has_label:
            shutil.copy(lbl_src, os.path.join(out_lbl_dir, name + ".txt"))

        # ───────── SALT & PEPPER ─────────
        aug = img.copy()
        sp_ratio = random.uniform(0.05, 0.08)
        n_px = int(sp_ratio * img.shape[0] * img.shape[1])

        coords = [np.random.randint(0, d, n_px) for d in img.shape[:2]]
        aug[coords[0], coords[1]] = 255

        coords = [np.random.randint(0, d, n_px) for d in img.shape[:2]]
        aug[coords[0], coords[1]] = 0

        name = f"{stem}_saltpepper"
        cv2.imwrite(os.path.join(out_img_dir, name + ".jpg"), aug)
        if has_label:
            shutil.copy(lbl_src, os.path.join(out_lbl_dir, name + ".txt"))

        # ───────── BLUR ─────────
        for k in [5, 7]:
            aug = cv2.GaussianBlur(img, (k, k), 0)
            name = f"{stem}_blur{k}"
            cv2.imwrite(os.path.join(out_img_dir, name + ".jpg"), aug)
            if has_label:
                shutil.copy(lbl_src, os.path.join(out_lbl_dir, name + ".txt"))

        # ───────── HAZE ─────────
        opacity = random.uniform(0.3, 0.4)
        haze = np.full_like(img, 255)
        aug = cv2.addWeighted(img, 1 - opacity, haze, opacity, 0)

        name = f"{stem}_haze"
        cv2.imwrite(os.path.join(out_img_dir, name + ".jpg"), aug)
        if has_label:
            shutil.copy(lbl_src, os.path.join(out_lbl_dir, name + ".txt"))

        processed += 1

    print(
        f"[{Path(image_dir).name}] processed={processed}, "
        f"skipped={skipped}, missing_labels={missing_labels}"
    )


# ─────────────────────────────────────────────
# 5. TRAINING CURVE VISUALISATION
# ─────────────────────────────────────────────

def plot_training_curves(
    results: Union[dict, str],
    model_name: str = "YOLOv12n",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training / validation loss and metric curves.

    Parameters
    ----------
    results : dict or str
        If dict: keys like "train_loss", "val_loss", "map50", "map50_95",
                 "precision", "recall" → lists of per-epoch values.
        If str:  path to a CSV produced by Ultralytics (results.csv).
    model_name : str  Title label.
    save_path : str   Optional path to save the figure.

    Returns
    -------
    fig : plt.Figure
    """
    if isinstance(results, str):
        import pandas as pd
        df = pd.read_csv(results)
        df.columns = df.columns.str.strip()
        results = {
            "train_box_loss": df.get("train/box_loss", pd.Series(dtype=float)).tolist(),
            "val_box_loss":   df.get("val/box_loss",   pd.Series(dtype=float)).tolist(),
            "map50":          df.get("metrics/mAP50(B)",    pd.Series(dtype=float)).tolist(),
            "map50_95":       df.get("metrics/mAP50-95(B)", pd.Series(dtype=float)).tolist(),
            "precision":      df.get("metrics/precision(B)", pd.Series(dtype=float)).tolist(),
            "recall":         df.get("metrics/recall(B)",    pd.Series(dtype=float)).tolist(),
        }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=16, fontweight="bold")
    palette = {"train": "#2196F3", "val": "#FF5722", "metric": "#4CAF50"}

    panels = [
        ("train_box_loss", "Train Box Loss",     palette["train"]),
        ("val_box_loss",   "Val Box Loss",       palette["val"]),
        ("map50",          "mAP@0.5",            palette["metric"]),
        ("map50_95",       "mAP@0.5:0.95",       palette["metric"]),
        ("precision",      "Precision",          "#9C27B0"),
        ("recall",         "Recall",             "#FF9800"),
    ]

    for ax, (key, title, color) in zip(axes.flat, panels):
        values = results.get(key, [])
        if values:
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, color=color, linewidth=2, marker="o", markersize=3)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{title} (no data)")
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")
    return fig


def plot_convergence_comparison(
    results_by_fraction: dict,
    metric: str = "map50",
    model_name: str = "YOLOv12n",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay convergence curves for 25%, 50%, 75% data fractions on a single axis.

    Parameters
    ----------
    results_by_fraction : dict
        {"25pct": {"map50": [...], ...}, "50pct": {...}, "75pct": {...}}
    metric : str  Key to plot (e.g. "map50", "val_box_loss").
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"25pct": "#F44336", "50pct": "#FF9800", "75pct": "#4CAF50"}
    for frac, res in results_by_fraction.items():
        values = res.get(metric, [])
        if values:
            ax.plot(range(1, len(values) + 1), values,
                    label=f"{frac} data", color=colors.get(frac, None), linewidth=2)

    ax.set_title(f"Convergence Comparison ({metric.upper()}) — {model_name}", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# 6. CONFUSION MATRIX
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: Optional[list] = None,
    model_name: str = "Model",
    normalize: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render a styled confusion matrix from flat classification lists.

    Parameters
    ----------
    y_true : list[int]  Ground-truth class indices (0-based after label remap).
    y_pred : list[int]  Predicted class indices (0-based).
    class_names : list  Human-readable names. Defaults to CLASS_NAMES.
    normalize : bool    Show row-normalised percentages when True.
    """
    class_names = class_names or CLASS_NAMES
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            cm[t, p] += 1

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.where(row_sums > 0, cm / row_sums, 0)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm_display.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = f"{cm_display[i, j]:{fmt}}"
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    return fig


# ─────────────────────────────────────────────
# 7. MISCLASSIFIED IMAGE VISUALISATION
# ─────────────────────────────────────────────

def visualize_misclassifications(
    image_paths: list,
    true_labels: list,
    pred_labels: list,
    class_names: Optional[list] = None,
    n_samples: int = 12,
    cols: int = 4,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Display a grid of misclassified images with their true and predicted labels.

    Parameters
    ----------
    image_paths : list[str]  Paths to all evaluated images.
    true_labels : list[int]  True class indices (one per image).
    pred_labels : list[int]  Predicted class indices (one per image).
    n_samples : int          How many misclassified examples to show.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise ImportError("Install Pillow: pip install Pillow")

    class_names = class_names or CLASS_NAMES

    def _safe_name(idx):
        return class_names[idx] if 0 <= idx < len(class_names) else str(idx)

    errors = [(p, t, pr) for p, t, pr in zip(image_paths, true_labels, pred_labels) if t != pr]
    if not errors:
        print("No misclassifications found.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No misclassifications!", ha="center", va="center")
        return fig

    sample = random.sample(errors, min(n_samples, len(errors)))
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle(f"Misclassified Images — {model_name}", fontsize=14, fontweight="bold")
    axes = np.array(axes).flatten()

    for ax, (img_path, true_cls, pred_cls) in zip(axes, sample):
        try:
            img = PILImage.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.set_facecolor("#EEEEEE")
        ax.set_title(
            f"True: {_safe_name(true_cls)}\nPred: {_safe_name(pred_cls)}",
            fontsize=9,
            color="red" if true_cls != pred_cls else "green",
        )
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Misclassification grid saved to {save_path}")
    return fig


# ─────────────────────────────────────────────
# 8. mAP@0.5 COMPARISON ACROSS WEATHER CONDITIONS
# ─────────────────────────────────────────────

# Default baseline data extracted from previous experiments and the paper's reported results.
BASELINE_MAP50 = {
    "YOLOv5s":      {"fog": 0.670, "rain": 0.695, "snow": 0.648},
    "YOLOv8m":      {"fog": 0.693, "rain": 0.855, "snow": 0.723},
    "YOLOv10n":     {"fog": 0.836, "rain": 0.800, "snow": 0.826},
    "Faster R-CNN": {"fog": 0.811, "rain": 0.783, "snow": 0.762},
}


def plot_map50_weather_comparison(
    results: Optional[dict] = None,
    models: Optional[list] = None,
    conditions: Optional[list] = None,
    include_baselines: bool = True,
    title: str = "mAP@0.5 Comparison Across Weather Conditions",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing mAP@0.5 across weather conditions for one or more models.

    Parameters
    ----------
    results : dict or None
        New experimental results in the format:
        {"ModelName": {"fog": 0.82, "rain": 0.79, "snow": 0.85, ...}}
        When None only baselines are shown (if include_baselines is True).
    models : list or None
        Subset of model names to display.  Shows all if None.
    conditions : list or None
        Subset of weather conditions.  Shows all available if None.
    include_baselines : bool
        Whether to add the four baseline models from the paper.

    Returns
    -------
    fig : plt.Figure
    """
    # Merge baseline + new results
    all_results = {}
    if include_baselines:
        all_results.update(BASELINE_MAP50)
    if results:
        all_results.update(results)

    if models:
        all_results = {m: v for m, v in all_results.items() if m in models}

    all_conditions = conditions or sorted({c for v in all_results.values() for c in v})
    model_names    = list(all_results.keys())

    n_models     = len(model_names)
    n_conditions = len(all_conditions)
    bar_width    = 0.8 / n_models
    x            = np.arange(n_conditions)

    # Colour palette (cycles if >8 models)
    palette = [
        "#FF6B9D", "#4FC3F7", "#80DEEA", "#FFB74D",
        "#AED581", "#CE93D8", "#F48FB1", "#80CBC4",
    ]

    fig, ax = plt.subplots(figsize=(max(10, n_conditions * 3), 6))

    for i, model in enumerate(model_names):
        offsets = x + (i - (n_models - 1) / 2) * bar_width
        values  = [all_results[model].get(cond, 0.0) for cond in all_conditions]
        color   = palette[i % len(palette)]
        bars    = ax.bar(offsets, values, width=bar_width * 0.9,
                         label=model, color=color, edgecolor="white", linewidth=0.5)
        # Value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in all_conditions], fontsize=12)
    ax.set_xlabel("Weather Condition", fontsize=12)
    ax.set_ylabel("mAP@0.5", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(
        loc="lower right",
        ncol=min(n_models, 4),
        framealpha=0.9,
        markerscale=1.2,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"mAP@0.5 chart saved to {save_path}")
    return fig


# ─────────────────────────────────────────────
# 9. BOUNDING-BOX ANNOTATION VISUALISER (helper)
# ─────────────────────────────────────────────

def visualize_annotations(
    image_path: str,
    label_path: Optional[str] = None,
    class_names: Optional[list] = None,
    pred_boxes: Optional[list] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Draw ground-truth (green) and/or predicted (red) bounding boxes on an image.

    Parameters
    ----------
    image_path : str
    label_path : str or None   YOLO .txt label file.
    pred_boxes : list or None  List of [cls_id, x_c, y_c, w, h, conf] (normalised).
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise ImportError("Install Pillow: pip install Pillow")

    class_names = class_names or CLASS_NAMES

    def _name(cls_id):
        return class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)

    img = PILImage.open(image_path).convert("RGB")
    W, H = img.size

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)

    def _yolo_to_abs(box, W, H):
        cls_id, xc, yc, w, h = int(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
        x1 = (xc - w / 2) * W
        y1 = (yc - h / 2) * H
        return cls_id, x1, y1, w * W, h * H

    if label_path and os.path.isfile(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id, x1, y1, bw, bh = _yolo_to_abs(parts, W, H)
                rect = patches.Rectangle((x1, y1), bw, bh, linewidth=2,
                                         edgecolor="#4CAF50", facecolor="none")
                ax.add_patch(rect)
                name = _name(cls_id)
                ax.text(x1, y1 - 3, name, color="white", fontsize=8,
                        bbox=dict(facecolor="#4CAF50", alpha=0.8, pad=1))

    if pred_boxes:
        for box in pred_boxes:
            cls_id, x1, y1, bw, bh = _yolo_to_abs(box, W, H)
            conf = float(box[5]) if len(box) > 5 else None
            rect = patches.Rectangle((x1, y1), bw, bh, linewidth=2,
                                     edgecolor="#F44336", facecolor="none", linestyle="--")
            ax.add_patch(rect)
            name = _name(cls_id)
            label = f"{name} {conf:.2f}" if conf else name
            ax.text(x1, y1 + bh + 12, label, color="white", fontsize=8,
                    bbox=dict(facecolor="#F44336", alpha=0.8, pad=1))

    ax.set_title(Path(image_path).name)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig