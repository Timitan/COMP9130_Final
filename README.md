# COMP9130 Final Project

# Convergence and Robustness of Object Detection Models in Adverse Weather Conditions

A comparative study of CNN-based, transformer-based, and one-pass object detectors evaluated on the [DAWN (Detection in Adverse Weather Nature)](https://www.kaggle.com/datasets/shuvoalok/dawn-dataset) benchmark dataset under fog, rain, sand, and snow conditions.

---

## Repository Structure

```
├── notebooks/
│   ├── cascade_rcnn/         # Cascade R-CNN training, evaluation, and data fraction experiments
│   ├── rtdetr/               # RT-DETR / RT-DETRv2 fine-tuning and evaluation
│   └── yolov12s_v5s/         # YOLOv12s and YOLOv5s training and comparison
├── results/
│   ├── cascade_rcnn/         # Output metrics, plots, and confusion matrices for Cascade R-CNN
│   ├── rtdetr_comparison/    # Weather-wise F1 charts and model comparison tables for RT-DETR
│   └── yolov12s_v5s/         # mAP plots, confusion matrices, and per-class results for YOLO models
├── utils/
│   ├── compose_video.py      # Combine annotated frames into an .mp4
│   ├── extract_frames.py     # Extract frames from a video file
│   ├── run_yolo.py           # Run YOLO inference on extracted video frames
│   └── utils.py              # Shared utility functions for dataset preparation and augmentation
├── videos/
│   ├── fog1.mp4                        # Raw example video clip (foggy road scene)
│   ├── fog1_rtdetrv2_annotated.mp4     # RT-DETRv2 inference output
│   ├── fog1_yolov12s_100pct.mp4        # YOLOv12s (100%) inference output
│   └── rcnn_output_annotated.mp4       # Cascade R-CNN inference output
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Results Summary

All models were fine-tuned on the DAWN dataset (≈1,000 images) across four adverse weather conditions. The table below reports performance on the held-out test split at IoU = 0.50.

| Model                  | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------------------------|-----------|--------|---------|--------------|
| YOLOv5s (Baseline)     | 0.909     | **0.835**  | 0.893   | 0.651        |
| YOLOv12s (100%)        | 0.901     | 0.739  | 0.843   | 0.600        |
| RT-DETRv2 (100%)       | **0.983** | 0.778  | **0.912** | **0.708**  |
| Cascade R-CNN (100%)   | 0.367     | —      | 0.640   | 0.367        |

---

## Dataset

The [DAWN dataset](https://www.kaggle.com/datasets/shuvoalok/dawn-dataset) is automatically downloaded via `kagglehub` inside each notebook — no manual download is required.

```python
import kagglehub
path = kagglehub.dataset_download("shuvoalok/dawn-dataset")
```

The dataset contains approximately 1,000 road images annotated in YOLO format across six object classes (car, person, truck, bus, motorcycle, bicycle) under four weather conditions: **fog, rain, sand/dust, and snow**.

---

## Setup & Usage

### Option 1 — Google Colab (Recommended)

Each notebook in `notebooks/` is self-contained and designed to run on Google Colab with a free or Pro GPU.

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (A100 recommended for Cascade R-CNN)
3. Move any utility files into the runtime or drive storage 
4. Run all cells — the dataset is fetched automatically via `kagglehub`
5. Results and plots are saved to the `results/` subdirectory

> **Note:** Cascade R-CNN requires Detectron2, which is installed via the first cell of the notebook. RT-DETRv2 requires `transformers >= 4.45.0`.

### Option 2 — Run Locally

**Prerequisites:** Python 3.10+, CUDA-compatible GPU recommended

```bash
# Clone the repository
git clone https://github.com/Timitan/COMP9130_Final

# Install dependencies
pip install -r requirements.txt

# Launch a notebook
jupyter notebook notebooks/rtdetr/rtdetr_training.ipynb
```

For Detectron2 (Cascade R-CNN), follow the [official installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) as it must be built from source for your specific CUDA version.

---

## Contributions

| Author | Contributions |
|--------|--------------|
| **Michael Persson** | Dataset cleaning and label adjustment, RT-DETR and RT-DETRv2 training pipeline, RT-DETR robustness evaluation, RT-DETR weatherwise breakdown, video inference for RT-DETRv2 |
| **Brendan Zapf** | Dataset download and data folder management, Cascade R-CNN training pipeline, Cascade R-CNN evaluation, Cascade R-CNN video inference |
| **Timothy Tan** | Dataset annotation remapping, YOLOv12s and YOLOv5s training pipeline, confusion matrix analysis, per-class and per-weather breakdown, video inference utilities for YOLO models(`run_yolo.py`, `compose_video.py`) |

All authors contributed to the paper writing, experimental design, and results interpretation.