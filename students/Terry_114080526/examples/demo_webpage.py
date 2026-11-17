#!/usr/bin/env python3
"""Streamlit demo webpage for Kather (2016) colorectal histology classification and slide scanning.

Usage (from students/Terry_114080526):
    uv run streamlit run examples/demo_webpage.py

Headless verification (from students/Terry_114080526):
    uv run python examples/demo_webpage.py --check
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Ensure local src/mini_ml is importable when invoked from any working directory
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mini_ml.decomposition import (
    PCA,
    project,
    reconstruct,
    reconstruction_error,
)
from mini_ml.information import (
    cross_entropy,
    kl_divergence,
    shannon_entropy,
)
from mini_ml.linalg import euclidean_distance
from mini_ml.preprocessing import Standardizer
from mini_ml.probability import class_proportions
from mini_ml.statistics import mean, std, variance

try:
    from mini_ml.models.linear_models import LogisticClassifier, softmax as torch_softmax
    HAS_TORCH = True
except ModuleNotFoundError:
    LogisticClassifier = None
    torch_softmax = None
    HAS_TORCH = False


# Paths
DATA_DIR = PACKAGE_ROOT / "data"
WEEK08_DIR = DATA_DIR / "week-08"
TILES_DIR = DATA_DIR / "images" / "Kather_texture_2016_image_tiles_5000"
LARGER_IMAGES_DIR = DATA_DIR / "images" / "Kather_texture_2016_larger_images_10"
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "test_code" / "week_09_best_model.npz"
WEEK09_RESULTS_CSV = PACKAGE_ROOT / "test_code" / "week_09_results.csv"
WEEK09_HISTORY_CSV = PACKAGE_ROOT / "test_code" / "week_09_history.csv"

CLASS_NAMES = (
    "01_TUMOR",
    "02_STROMA",
    "03_COMPLEX",
    "04_LYMPHO",
    "05_DEBRIS",
    "06_MUCOSA",
    "07_ADIPOSE",
    "08_EMPTY",
)

# Matches matplotlib tab10 palette used in data/answers/slide-scan.png
CLASS_COLORS_RGB = np.array(
    [
        [0.12156863, 0.46666667, 0.70588235],  # 01_TUMOR (blue)
        [1.00000000, 0.49803922, 0.05490196],  # 02_STROMA (orange)
        [0.17254902, 0.62745098, 0.17254902],  # 03_COMPLEX (green)
        [0.83921569, 0.15294118, 0.15686275],  # 04_LYMPHO (red)
        [0.58039216, 0.40392157, 0.74117647],  # 05_DEBRIS (purple)
        [0.54901961, 0.33725490, 0.29411765],  # 06_MUCOSA (brown)
        [0.89019608, 0.46666667, 0.76078431],  # 07_ADIPOSE (pink)
        [0.49803922, 0.49803922, 0.49803922],  # 08_EMPTY (gray)
    ],
    dtype=np.float64,
)
LOW_PROB_COLOR_RGB = np.array([0.84, 0.86, 0.88], dtype=np.float64)
PAPER = "#fffdf7"
TEAL = "#008782"
RUST = "#bd5d35"


def numpy_softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax matching mini_ml.models.linear_models.softmax."""
    arr = np.asarray(scores, dtype=np.float64)
    shifted = arr - np.max(arr, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def preprocess_patch(
    patch_rgb: np.ndarray | Image.Image,
    target_size: tuple[int, int] = (28, 28),
    intensity_divisor: float = 255.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-process a 150x150 RGB tile/window following the Week 8/9 provenance recipe.

    Recipe:
        1. Convert to RGB uint8.
        2. Resize from 150x150 to 28x28 with Pillow BILINEAR resampling.
        3. Flatten in C-order to (2352,) and divide by 255.0 as float64.

    Returns:
        resized_uint8: (28, 28, 3) uint8 image array.
        features_float64: (2352,) float64 feature vector in [0, 1].
    """
    if isinstance(patch_rgb, Image.Image):
        pil_img = patch_rgb.convert("RGB")
    else:
        pil_img = Image.fromarray(np.asarray(patch_rgb, dtype=np.uint8), mode="RGB")

    resized_pil = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    resized_uint8 = np.asarray(resized_pil, dtype=np.uint8)
    features_float64 = resized_uint8.reshape(-1).astype(np.float64) / float(intensity_divisor)
    return resized_uint8, features_float64


def image_features(images: np.ndarray, intensity_divisor: float = 255.0) -> np.ndarray:
    """Convert a batch of (N, 28, 28, 3) uint8 images to (N, 2352) float64 features."""
    return images.reshape(len(images), -1).astype(np.float64) / float(intensity_divisor)


def slice_and_preprocess_large_image(
    slide_rgb: np.ndarray,
    window_size: int = 150,
    stride: int = 75,
    target_size: tuple[int, int] = (28, 28),
    intensity_divisor: float = 255.0,
) -> dict[str, np.ndarray | list[int] | int]:
    """Slice a large histology image into sliding windows and pre-process each into 28x28 RGB features.

    Args:
        slide_rgb: (H, W, 3) uint8 RGB array (e.g. 5000x5000x3).
        window_size: Side length of square patch in pixels (default 150).
        stride: Step size in pixels between adjacent windows (default 75).
        target_size: Downsampled resolution (default (28, 28)).
        intensity_divisor: Normalization divisor (default 255.0).

    Returns:
        Dictionary with y_positions, x_positions, resized_patches, and feature matrix X.
    """
    height, width = slide_rgb.shape[:2]
    y_positions = list(range(0, height - window_size + 1, stride))
    x_positions = list(range(0, width - window_size + 1, stride))

    resized_list = []
    for y in y_positions:
        row_slice = slide_rgb[y : y + window_size]
        for x in x_positions:
            patch = row_slice[:, x : x + window_size]
            resized_patch = np.asarray(
                Image.fromarray(patch).resize(target_size, Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
            resized_list.append(resized_patch)

    resized_patches = np.stack(resized_list, axis=0)
    X = image_features(resized_patches, intensity_divisor=intensity_divisor)
    return {
        "y_positions": y_positions,
        "x_positions": x_positions,
        "window_size": window_size,
        "stride": stride,
        "resized_patches": resized_patches,
        "X": X,
    }


def aggregate_overlapping_probabilities(
    window_probs: np.ndarray,
    y_positions: list[int],
    x_positions: list[int],
    window_size: int = 150,
    stride: int = 75,
) -> dict[str, np.ndarray | int]:
    """Average class probability vectors across overlapping windows on a regular sub-tile grid.

    When window_size is a multiple of stride (e.g. 150px windows with 75px stride), each window
    covers (window_size // stride) x (window_size // stride) sub-cells of size stride x stride.
    Averaging probabilities on this sub-cell grid is exact for every pixel in the covered area.
    """
    n_y = len(y_positions)
    n_x = len(x_positions)
    n_classes = window_probs.shape[1]
    probs_grid = window_probs.reshape(n_y, n_x, n_classes)

    cell_size = int(np.gcd(window_size, stride))
    span = window_size // cell_size
    step = stride // cell_size
    grid_h = (n_y - 1) * step + span
    grid_w = (n_x - 1) * step + span

    prob_sum = np.zeros((grid_h, grid_w, n_classes), dtype=np.float64)
    counts = np.zeros((grid_h, grid_w, 1), dtype=np.float64)

    for iy in range(n_y):
        cy = iy * step
        for ix in range(n_x):
            cx = ix * step
            prob_sum[cy : cy + span, cx : cx + span] += probs_grid[iy, ix]
            counts[cy : cy + span, cx : cx + span] += 1.0

    avg_probs = prob_sum / np.maximum(counts, 1.0)
    max_prob = np.max(avg_probs, axis=-1)
    pred_class = np.argmax(avg_probs, axis=-1)

    # Compute Shannon entropy (in bits) per grid cell using mini_ml.information.shannon_entropy
    flat_probs = avg_probs.reshape(-1, n_classes)
    entropies = np.array(
        [shannon_entropy(row.tolist()) for row in flat_probs],
        dtype=np.float64,
    ).reshape(grid_h, grid_w)

    return {
        "avg_probs": avg_probs,
        "max_prob": max_prob,
        "pred_class": pred_class,
        "entropy_bits": entropies,
        "cell_size": cell_size,
        "covered_height": grid_h * cell_size,
        "covered_width": grid_w * cell_size,
    }


def select_representative_patches(
    window_probs: np.ndarray,
    y_positions: list[int],
    x_positions: list[int],
    window_size: int = 150,
    n_patches: int = 8,
) -> list[dict[str, int | float | str]]:
    """Select high-confidence non-overlapping patches (e.g. Tumor and Stroma/Complex) to number on the slide."""
    n_x = len(x_positions)
    pred_classes = np.argmax(window_probs, axis=1)
    max_probs = np.max(window_probs, axis=1)

    selected: list[dict[str, int | float | str]] = []
    used_boxes: list[tuple[int, int]] = []

    def is_non_overlapping(y_coord: int, x_coord: int) -> bool:
        for uy, ux in used_boxes:
            if abs(y_coord - uy) < window_size and abs(x_coord - ux) < window_size:
                return False
        return True

    # Pick up to half from class 0 (01_TUMOR) and the rest from other confident classes
    half = n_patches // 2
    tumor_candidates = np.argsort(-window_probs[:, 0])
    for idx in tumor_candidates:
        if len(selected) >= half:
            break
        iy, ix = divmod(int(idx), n_x)
        y_coord, x_coord = y_positions[iy], x_positions[ix]
        if is_non_overlapping(y_coord, x_coord):
            used_boxes.append((y_coord, x_coord))
            c_idx = int(pred_classes[idx])
            selected.append(
                {
                    "patch_number": len(selected) + 1,
                    "flat_index": int(idx),
                    "iy": iy,
                    "ix": ix,
                    "y": y_coord,
                    "x": x_coord,
                    "pred_class": c_idx,
                    "pred_name": CLASS_NAMES[c_idx],
                    "confidence": float(max_probs[idx]),
                    "tumor_prob": float(window_probs[idx, 0]),
                }
            )

    non_tumor_mask = pred_classes != 0
    other_indices = np.flatnonzero(non_tumor_mask)
    other_order = other_indices[np.argsort(-max_probs[other_indices])]
    for idx in other_order:
        if len(selected) >= n_patches:
            break
        iy, ix = divmod(int(idx), n_x)
        y_coord, x_coord = y_positions[iy], x_positions[ix]
        if is_non_overlapping(y_coord, x_coord):
            used_boxes.append((y_coord, x_coord))
            c_idx = int(pred_classes[idx])
            selected.append(
                {
                    "patch_number": len(selected) + 1,
                    "flat_index": int(idx),
                    "iy": iy,
                    "ix": ix,
                    "y": y_coord,
                    "x": x_coord,
                    "pred_class": c_idx,
                    "pred_name": CLASS_NAMES[c_idx],
                    "confidence": float(max_probs[idx]),
                    "tumor_prob": float(window_probs[idx, 0]),
                }
            )

    # Fallback if fewer than n_patches were found
    for idx in np.argsort(-max_probs):
        if len(selected) >= n_patches:
            break
        iy, ix = divmod(int(idx), n_x)
        y_coord, x_coord = y_positions[iy], x_positions[ix]
        if is_non_overlapping(y_coord, x_coord):
            used_boxes.append((y_coord, x_coord))
            c_idx = int(pred_classes[idx])
            selected.append(
                {
                    "patch_number": len(selected) + 1,
                    "flat_index": int(idx),
                    "iy": iy,
                    "ix": ix,
                    "y": y_coord,
                    "x": x_coord,
                    "pred_class": c_idx,
                    "pred_name": CLASS_NAMES[c_idx],
                    "confidence": float(max_probs[idx]),
                    "tumor_prob": float(window_probs[idx, 0]),
                }
            )

    return selected


def fit_mini_ml_logistic_sgd(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    learning_rate: float = 0.1,
    epochs: int = 100,
    batch_size: int = 128,
    random_state: int = 42,
) -> dict[str, np.ndarray | list[float]]:
    """Train the Week 9 mini_ml LogisticClassifier (uses PyTorch if installed, or exact analytical SGD fallback)."""
    if HAS_TORCH and LogisticClassifier is not None:
        clf = LogisticClassifier(
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
        )
        val_arg = (Z_val, y_val) if (Z_val is not None and y_val is not None) else None
        clf.fit(Z_train, y_train, validation_data=val_arg)
        return {
            "coef": clf.coef_,
            "intercept": clf.intercept_,
            "classes": clf.classes_,
            "loss_history": list(clf.loss_history_),
            "validation_loss_history": list(clf.validation_loss_history_),
        }

    # Exact NumPy analytical equivalent of mini_ml.models.linear_models.LogisticClassifier
    classes, encoded = np.unique(y_train, return_inverse=True)
    n_samples, n_features = Z_train.shape
    n_classes = len(classes)
    W = np.zeros((n_classes, n_features), dtype=np.float64)
    b = np.zeros(n_classes, dtype=np.float64)
    rng = np.random.default_rng(random_state)

    def cross_entropy_nats(X_mat: np.ndarray, y_idx: np.ndarray) -> float:
        scores = X_mat @ W.T + b
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))
        return float(np.mean(log_sum_exp - shifted[np.arange(len(y_idx)), y_idx]))

    loss_history = [cross_entropy_nats(Z_train, encoded)]
    val_encoded = np.searchsorted(classes, y_val) if y_val is not None else None
    val_loss_history = (
        [cross_entropy_nats(Z_val, val_encoded)]
        if (Z_val is not None and val_encoded is not None)
        else []
    )

    for _ in range(epochs):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = order[start : start + batch_size]
            X_batch = Z_train[idx]
            y_batch = encoded[idx]
            probs = numpy_softmax(X_batch @ W.T + b)
            grad_scores = probs
            grad_scores[np.arange(len(y_batch)), y_batch] -= 1.0
            grad_scores /= float(len(y_batch))
            W -= learning_rate * (grad_scores.T @ X_batch)
            b -= learning_rate * np.sum(grad_scores, axis=0)

        loss_history.append(cross_entropy_nats(Z_train, encoded))
        if Z_val is not None and val_encoded is not None:
            val_loss_history.append(cross_entropy_nats(Z_val, val_encoded))

    return {
        "coef": W,
        "intercept": b,
        "classes": classes,
        "loss_history": loss_history,
        "validation_loss_history": val_loss_history,
    }


def make_slide_scan_figure(
    slide_rgb: np.ndarray,
    agg: dict[str, np.ndarray | int],
    selected_patches: list[dict[str, int | float | str]],
    slide_name: str,
    model_label: str,
    n_pcs: int,
    window_size: int,
    stride: int,
    low_prob_threshold: float = 0.50,
    target_class_idx: int = 0,
    overlay_alpha: float = 0.58,
) -> plt.Figure:
    """Build the 4-panel whole-slide scan figure modeled on data/answers/slide-scan.png."""
    covered_h = int(agg["covered_height"])
    covered_w = int(agg["covered_width"])
    avg_probs = np.asarray(agg["avg_probs"])
    max_prob = np.asarray(agg["max_prob"])
    pred_class = np.asarray(agg["pred_class"])

    # Downsample the 5000x5000 display image to 990x990 for crisp, fast matplotlib rendering
    cropped_rgb = slide_rgb[:covered_h, :covered_w]
    display_size = 990
    base_display = (
        np.asarray(
            Image.fromarray(cropped_rgb).resize(
                (display_size, display_size), Image.Resampling.BILINEAR
            ),
            dtype=np.float64,
        )
        / 255.0
    )
    scale_factor = display_size / float(covered_w)

    # Build RGB class map on grid and upsample with NEAREST to match display_size
    low_mask = max_prob < low_prob_threshold
    low_fraction = float(np.mean(low_mask))
    grid_colors = CLASS_COLORS_RGB[pred_class]
    grid_colors_uint8 = np.clip(np.round(grid_colors * 255.0), 0, 255).astype(np.uint8)
    upsampled_colors = (
        np.asarray(
            Image.fromarray(grid_colors_uint8).resize(
                (display_size, display_size), Image.Resampling.NEAREST
            ),
            dtype=np.float64,
        )
        / 255.0
    )
    upsampled_low_mask = (
        np.asarray(
            Image.fromarray((low_mask.astype(np.uint8) * 255)).resize(
                (display_size, display_size), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        > 127
    )

    blended = (1.0 - overlay_alpha) * base_display + overlay_alpha * upsampled_colors
    blended[upsampled_low_mask] = LOW_PROB_COLOR_RGB

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 13.0), facecolor="white")
    fig.suptitle(
        f"Whole-slide scan | {slide_name} | {model_label}, {n_pcs} PCs\n"
        f"{window_size}px windows, stride {stride} | Model predictions; no region-level ground truth",
        fontsize=15,
        y=0.97,
    )

    # Panel 1: Original with numbered patches
    ax0 = axes[0, 0]
    ax0.imshow(base_display)
    for item in selected_patches:
        cy = (float(item["y"]) + window_size / 2.0) * scale_factor
        cx = (float(item["x"]) + window_size / 2.0) * scale_factor
        is_tumor = int(item["pred_class"]) == 0
        badge_color = "#b8252a" if is_tumor else "#ebba52"
        text_color = "white" if is_tumor else "#222222"
        ax0.scatter(
            [cx],
            [cy],
            s=360,
            c=badge_color,
            edgecolors="none",
            alpha=0.92,
            zorder=4,
        )
        ax0.text(
            cx,
            cy,
            str(item["patch_number"]),
            color=text_color,
            fontsize=9.5,
            weight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
    ax0.set_title("Original | numbered patches shown separately", fontsize=12)
    ax0.set_axis_off()

    # Panel 2: Eight-class model prediction overlay
    ax1 = axes[0, 1]
    ax1.imshow(np.clip(blended, 0.0, 1.0))
    ax1.set_title(
        f"Eight-class model prediction\n"
        f"Gray: max avg probability < {low_prob_threshold:.2f} ({low_fraction:.1%})",
        fontsize=12,
    )
    ax1.set_axis_off()

    # Panel 3: Selected class score heatmap
    target_name = CLASS_NAMES[target_class_idx]
    short_label = target_name.split("_", 1)[1].capitalize()
    ax2 = axes[1, 0]
    im2 = ax2.imshow(
        avg_probs[:, :, target_class_idx],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax2.set_title(f"{short_label} score | {target_name}", fontsize=12)
    ax2.set_axis_off()
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.04, shrink=0.72)
    cbar2.set_label("Predicted probability", fontsize=10)

    # Panel 4: Maximum averaged class probability
    ax3 = axes[1, 1]
    im3 = ax3.imshow(
        max_prob,
        cmap="viridis",
        vmin=max(0.125, float(np.min(max_prob))),
        vmax=1.0,
        interpolation="nearest",
    )
    ax3.set_title("Maximum averaged class probability", fontsize=12)
    ax3.set_axis_off()
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.035, pad=0.04, shrink=0.72)
    cbar3.set_label("Maximum probability", fontsize=10)

    # Shared bottom legend
    legend_handles = [
        mpatches.Patch(color=CLASS_COLORS_RGB[i], label=CLASS_NAMES[i])
        for i in range(len(CLASS_NAMES))
    ]
    legend_handles.append(
        mpatches.Patch(
            color=LOW_PROB_COLOR_RGB,
            label=f"Low max probability (< {low_prob_threshold:.2f})",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=9.5,
        frameon=True,
    )
    fig.subplots_adjust(left=0.03, right=0.95, top=0.89, bottom=0.10, wspace=0.16, hspace=0.18)
    return fig


def load_week08_bundles() -> dict[str, dict[str, np.ndarray]]:
    """Load the verified Week 8 train, validation, and example .npz bundles."""
    bundles = {}
    for key, filename in (
        ("train", "kather_train.npz"),
        ("validation", "kather_validation.npz"),
        ("examples", "kather_examples.npz"),
    ):
        with np.load(WEEK08_DIR / filename, allow_pickle=False) as data:
            bundles[key] = {name: data[name] for name in data.files}
    return bundles


def load_week09_checkpoint(path: Path = DEFAULT_MODEL_PATH) -> dict[str, np.ndarray]:
    """Load the saved Week 9 best model checkpoint (.npz)."""
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def build_classifier_pipeline(
    model_choice: str,
    n_pcs: int = 30,
    standardize_pcs: bool = False,
    learning_rate: float = 0.1,
    epochs: int = 100,
    batch_size: int = 128,
    svm_c: float = 1.0,
) -> dict[str, object]:
    """Build or load the requested PCA + classifier pipeline using mini_ml components."""
    bundles = load_week08_bundles()
    X_train = image_features(bundles["train"]["images"])
    y_train = bundles["train"]["labels"]
    X_val = image_features(bundles["validation"]["images"])
    y_val = bundles["validation"]["labels"]

    if model_choice == "week09_checkpoint" and DEFAULT_MODEL_PATH.is_file() and n_pcs == 30 and not standardize_pcs:
        ckpt = load_week09_checkpoint(DEFAULT_MODEL_PATH)
        pca_mean = ckpt["pca_mean"]
        pca_components = ckpt["pca_components"]
        coef = ckpt["coef"]
        intercept = ckpt["intercept"]
        classes = ckpt["classes"]

        Z_train = project(X_train, pca_mean, pca_components)
        Z_val = project(X_val, pca_mean, pca_components)

        def predict_proba_fn(X_input: np.ndarray) -> np.ndarray:
            Z = project(X_input, pca_mean, pca_components)
            return numpy_softmax(Z @ coef.T + intercept)

        val_probs = predict_proba_fn(X_val)
        val_preds = classes[np.argmax(val_probs, axis=1)]
        val_acc = float(np.mean(val_preds == y_val))
        return {
            "label": "mini_ml LogisticClassifier (Week 9 checkpoint)",
            "short_label": "mini_ml_logistic",
            "n_pcs": 30,
            "pca_mean": pca_mean,
            "pca_components": pca_components,
            "Z_train": Z_train,
            "y_train": y_train,
            "Z_val": Z_val,
            "y_val": y_val,
            "val_probs": val_probs,
            "val_preds": val_preds,
            "val_accuracy": val_acc,
            "predict_proba": predict_proba_fn,
            "loss_history": None,
            "validation_loss_history": None,
        }

    # Fit mini_ml PCA on the 3,000 training images
    pca = PCA(n_components=n_pcs, solver="auto", random_state=42).fit(X_train)
    pca_mean = pca.mean_
    pca_components = pca.components_
    Z_train_raw = pca.transform(X_train)
    Z_val_raw = pca.transform(X_val)

    scaler = Standardizer().fit(Z_train_raw) if standardize_pcs else None
    Z_train = scaler.transform(Z_train_raw) if scaler is not None else Z_train_raw
    Z_val = scaler.transform(Z_val_raw) if scaler is not None else Z_val_raw

    if model_choice in ("mini_ml_logistic", "week09_checkpoint"):
        trained = fit_mini_ml_logistic_sgd(
            Z_train,
            y_train,
            Z_val=Z_val,
            y_val=y_val,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_state=42,
        )
        coef = np.asarray(trained["coef"])
        intercept = np.asarray(trained["intercept"])
        classes = np.asarray(trained["classes"])

        def predict_proba_fn(X_input: np.ndarray) -> np.ndarray:
            Z = project(X_input, pca_mean, pca_components)
            if scaler is not None:
                Z = scaler.transform(Z)
            return numpy_softmax(Z @ coef.T + intercept)

        val_probs = predict_proba_fn(X_val)
        val_preds = classes[np.argmax(val_probs, axis=1)]
        val_acc = float(np.mean(val_preds == y_val))
        return {
            "label": f"mini_ml LogisticClassifier (lr={learning_rate:g}, epochs={epochs})",
            "short_label": "mini_ml_logistic",
            "n_pcs": n_pcs,
            "pca_mean": pca_mean,
            "pca_components": pca_components,
            "pca_explained_variance_ratio": pca.explained_variance_ratio_,
            "Z_train": Z_train_raw,
            "y_train": y_train,
            "Z_val": Z_val_raw,
            "y_val": y_val,
            "val_probs": val_probs,
            "val_preds": val_preds,
            "val_accuracy": val_acc,
            "predict_proba": predict_proba_fn,
            "loss_history": trained["loss_history"],
            "validation_loss_history": trained["validation_loss_history"],
        }

    # Reference RBF SVM on mini_ml PCA scores (matches instructor prototype slide-scan.png)
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC

    base_svm = SVC(kernel="rbf", C=svm_c, random_state=42)
    clf = CalibratedClassifierCV(base_svm, cv=5).fit(Z_train, y_train)
    classes = np.unique(y_train)

    def predict_proba_fn(X_input: np.ndarray) -> np.ndarray:
        Z = project(X_input, pca_mean, pca_components)
        if scaler is not None:
            Z = scaler.transform(Z)
        return clf.predict_proba(Z)

    val_probs = predict_proba_fn(X_val)
    val_preds = classes[np.argmax(val_probs, axis=1)]
    val_acc = float(np.mean(val_preds == y_val))
    return {
        "label": f"rbf_svm on mini_ml PCA (C={svm_c:g})",
        "short_label": "rbf_svm",
        "n_pcs": n_pcs,
        "pca_mean": pca_mean,
        "pca_components": pca_components,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "Z_train": Z_train_raw,
        "y_train": y_train,
        "Z_val": Z_val_raw,
        "y_val": y_val,
        "val_probs": val_probs,
        "val_preds": val_preds,
        "val_accuracy": val_acc,
        "predict_proba": predict_proba_fn,
        "loss_history": None,
        "validation_loss_history": None,
    }


def render_streamlit_app() -> None:
    """Render the interactive Streamlit application."""
    import streamlit as st

    st.set_page_config(
        page_title="Kather 2016 Histology ML Demo — mini_ml",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Colorectal Histology Whole-Slide & Tile Explorer (`mini_ml`)")
    st.markdown(
        """
        Interactive machine learning webpage powered by **`src/mini_ml`** (`PCA`, `project`, `reconstruct`,
        `reconstruction_error`, `Standardizer`, `LogisticClassifier`, `shannon_entropy`, `cross_entropy`,
        `kl_divergence`, `class_proportions`, `euclidean_distance`, `mean`, `std`, `variance`).
        """
    )

    @st.cache_data(show_spinner=False)
    def cached_week08_bundles():
        return load_week08_bundles()

    @st.cache_data(show_spinner=False)
    def cached_large_image_paths() -> list[Path]:
        if not LARGER_IMAGES_DIR.is_dir():
            return []
        return sorted(LARGER_IMAGES_DIR.glob("*.tif"))

    @st.cache_data(show_spinner=False)
    def cached_tile_paths_by_class() -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for cls_name in CLASS_NAMES:
            cls_dir = TILES_DIR / cls_name
            if cls_dir.is_dir():
                mapping[cls_name] = [str(p) for p in sorted(cls_dir.glob("*.tif"))]
            else:
                mapping[cls_name] = []
        return mapping

    @st.cache_data(show_spinner="Loading 5000×5000 histology slide...")
    def cached_load_slide_rgb(path_str: str) -> np.ndarray:
        with Image.open(path_str) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    @st.cache_data(show_spinner="Slicing slide & downsampling 150×150 windows to 28×28 RGB...")
    def cached_slice_slide(path_str: str, window_size: int, stride: int):
        slide_rgb = cached_load_slide_rgb(path_str)
        return slice_and_preprocess_large_image(
            slide_rgb, window_size=window_size, stride=stride
        )

    @st.cache_resource(show_spinner="Preparing mini_ml PCA + Classifier pipeline...")
    def cached_pipeline(
        model_choice: str,
        n_pcs: int,
        standardize_pcs: bool,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        svm_c: float,
    ):
        return build_classifier_pipeline(
            model_choice=model_choice,
            n_pcs=n_pcs,
            standardize_pcs=standardize_pcs,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            svm_c=svm_c,
        )

    @st.cache_resource(show_spinner="Fitting 300-component mini_ml PCA for multi-k reconstruction...")
    def cached_pca_300():
        bundles = cached_week08_bundles()
        X_train = image_features(bundles["train"]["images"])
        return PCA(n_components=300, solver="auto", random_state=42).fit(X_train)

    # Sidebar configuration
    st.sidebar.header("1. Large Slide Input")
    large_paths = cached_large_image_paths()
    large_names = [p.name for p in large_paths]
    selected_slide_name = st.sidebar.selectbox(
        "Select 5000×5000 pathology slide (`Kather_texture_2016_larger_images_10`):",
        options=large_names,
        index=0,
    )
    selected_slide_path = LARGER_IMAGES_DIR / selected_slide_name if large_names else None

    st.sidebar.header("2. Window Slicing & Overlay")
    window_size = st.sidebar.selectbox(
        "Window size (px)",
        options=[150],
        index=0,
        help="Matches the 150×150 pixel tiles in Kather_texture_2016_image_tiles_5000.",
    )
    stride = st.sidebar.select_slider(
        "Sliding window stride (px)",
        options=[50, 75, 100, 150],
        value=75,
        help="Stride 75 overlaps 150×150 windows by 50% and averages class probabilities across overlapping windows.",
    )
    low_prob_threshold = st.sidebar.slider(
        "Low-confidence gray mask threshold",
        min_value=0.15,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Regions where maximum averaged class probability falls below this threshold are masked in gray.",
    )
    target_class_name = st.sidebar.selectbox(
        "Class score heatmap panel",
        options=list(CLASS_NAMES),
        index=0,
    )
    target_class_idx = CLASS_NAMES.index(target_class_name)

    st.sidebar.header("3. `mini_ml` Model Configuration")
    model_label_map = {
        "week09_checkpoint": "Week 9 Saved Checkpoint (mini_ml Logistic + 30 PCs)",
        "mini_ml_logistic": "Live mini_ml PCA + LogisticClassifier (SGD)",
        "rbf_svm": "Reference RBF SVM on mini_ml PCA (Instructor Prototype)",
    }
    model_choice = st.sidebar.selectbox(
        "Classifier pipeline",
        options=list(model_label_map.keys()),
        format_func=lambda k: model_label_map[k],
        index=0,
    )

    if model_choice == "week09_checkpoint":
        n_pcs = 30
        standardize_pcs = False
        learning_rate = 0.1
        epochs = 100
        batch_size = 128
        svm_c = 1.0
        st.sidebar.caption(
            "Using saved `test_code/week_09_best_model.npz` (30 PCs, lr=0.1, epochs=100, batch_size=128)."
        )
    elif model_choice == "mini_ml_logistic":
        n_pcs = st.sidebar.select_slider("PCA components (k)", options=[3, 10, 30, 50, 100], value=30)
        standardize_pcs = st.sidebar.checkbox("Apply `mini_ml.preprocessing.Standardizer` to PCs", value=False)
        learning_rate = st.sidebar.select_slider(
            "SGD learning rate", options=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1], value=1e-1
        )
        epochs = st.sidebar.slider("Training epochs", min_value=10, max_value=200, value=100, step=10)
        batch_size = st.sidebar.select_slider("Mini-batch size", options=[32, 64, 128, 256, 512], value=128)
        svm_c = 1.0
    else:
        n_pcs = st.sidebar.select_slider("PCA components (k)", options=[10, 30, 50, 100], value=30)
        standardize_pcs = st.sidebar.checkbox("Apply `mini_ml.preprocessing.Standardizer` to PCs", value=False)
        svm_c = st.sidebar.select_slider("RBF SVM regularization C", options=[0.5, 1.0, 2.0, 5.0], value=1.0)
        learning_rate = 0.1
        epochs = 100
        batch_size = 128

    pipeline = cached_pipeline(
        model_choice=model_choice,
        n_pcs=n_pcs,
        standardize_pcs=standardize_pcs,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        svm_c=svm_c,
    )

    tab_scan, tab_tiles, tab_diagnostics = st.tabs(
        [
            "🗺️ 1. Whole-Slide Scan (`Kather_texture_2016_larger_images_10`)",
            "🧩 2. Tile Slicing & Pre-processing (`Kather_texture_2016_image_tiles_5000`)",
            "📊 3. `mini_ml` Model & PCA Diagnostics",
        ]
    )

    # =========================================================================
    # TAB 1: Whole-Slide Scan on Kather_texture_2016_larger_images_10
    # =========================================================================
    with tab_scan:
        if selected_slide_path is None or not selected_slide_path.is_file():
            st.error(f"Could not find large images in `{LARGER_IMAGES_DIR}`.")
        else:
            t_start = time.perf_counter()
            slide_rgb = cached_load_slide_rgb(str(selected_slide_path))
            sliced = cached_slice_slide(str(selected_slide_path), window_size, stride)
            window_probs = pipeline["predict_proba"](sliced["X"])
            agg = aggregate_overlapping_probabilities(
                window_probs,
                sliced["y_positions"],
                sliced["x_positions"],
                window_size=window_size,
                stride=stride,
            )
            selected_patches = select_representative_patches(
                window_probs,
                sliced["y_positions"],
                sliced["x_positions"],
                window_size=window_size,
                n_patches=8,
            )
            elapsed = time.perf_counter() - t_start

            n_windows = len(sliced["X"])
            max_prob_flat = agg["max_prob"].ravel().tolist()
            entropy_flat = agg["entropy_bits"].ravel().tolist()
            low_conf_pct = float(np.mean(agg["max_prob"] < low_prob_threshold)) * 100.0

            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Slide Dimensions", f"{slide_rgb.shape[1]}×{slide_rgb.shape[0]} RGB")
            col_m2.metric(
                "Sliding Windows",
                f"{n_windows:,} ({len(sliced['y_positions'])}×{len(sliced['x_positions'])})",
            )
            col_m3.metric(
                "Validation Accuracy (1,000 tiles)",
                f"{pipeline['val_accuracy']:.1%}",
            )
            col_m4.metric(
                "Mean Max Probability (`mini_ml`)",
                f"{mean(max_prob_flat):.3f} ± {std(max_prob_flat):.3f}",
            )
            col_m5.metric(
                f"Low Confidence (< {low_prob_threshold:.2f})",
                f"{low_conf_pct:.1f}%",
            )
            st.caption(
                f"Scanned `{selected_slide_name}` in {elapsed:.2f}s — each `{window_size}×{window_size}` window "
                f"is resized to `28×28` (`BILINEAR`), flattened to `2,352` float64 features (`/ 255.0`), projected onto "
                f"`{pipeline['n_pcs']}` training-only PCs via `mini_ml.decomposition.project`, and averaged across overlapping stride-{stride} cells."
            )

            # Main 4-panel Whole-Slide Scan Figure
            scan_fig = make_slide_scan_figure(
                slide_rgb=slide_rgb,
                agg=agg,
                selected_patches=selected_patches,
                slide_name=selected_slide_name.replace("_APPLICATION.tif", "").replace("-APPLICATION.tif", ""),
                model_label=str(pipeline["short_label"]),
                n_pcs=int(pipeline["n_pcs"]),
                window_size=window_size,
                stride=stride,
                low_prob_threshold=low_prob_threshold,
                target_class_idx=target_class_idx,
            )
            st.pyplot(scan_fig, width="stretch")
            plt.close(scan_fig)

            # Numbered patches inspector & PCA reconstruction
            st.subheader("🔎 Numbered Patch Inspector (150×150 Window → 28×28 Input → PCA Reconstruction)")
            st.markdown(
                "The 8 numbered markers on the original slide above correspond to the `150×150` patches shown below, "
                "along with their `28×28` downsampled model input, their `mini_ml.decomposition.reconstruct` image, "
                "and their predicted class probability & Shannon entropy (`mini_ml.information.shannon_entropy`)."
            )

            patch_fig, patch_axes = plt.subplots(2, 8, figsize=(16, 4.6), facecolor=PAPER)
            pca_mean = np.asarray(pipeline["pca_mean"])
            pca_components = np.asarray(pipeline["pca_components"])
            for col_idx, item in enumerate(selected_patches):
                y0, x0 = int(item["y"]), int(item["x"])
                flat_idx = int(item["flat_index"])
                raw_patch = slide_rgb[y0 : y0 + window_size, x0 : x0 + window_size]
                downsampled_patch = sliced["resized_patches"][flat_idx]
                feat_vec = sliced["X"][flat_idx : flat_idx + 1]
                z_vec = project(feat_vec, pca_mean, pca_components)
                rebuilt_vec = reconstruct(z_vec, pca_mean, pca_components)
                mse_val = reconstruction_error(feat_vec, rebuilt_vec)
                ent_val = shannon_entropy(window_probs[flat_idx].tolist())

                ax_top = patch_axes[0, col_idx]
                ax_top.imshow(raw_patch)
                ax_top.set_title(
                    f"#{item['patch_number']} ({x0},{y0})\n150×150 RGB",
                    fontsize=8.5,
                )
                ax_top.set_axis_off()

                ax_bot = patch_axes[1, col_idx]
                ax_bot.imshow(downsampled_patch, interpolation="nearest")
                ax_bot.set_title(
                    f"{item['pred_name']}\np={item['confidence']:.2f} | H={ent_val:.2f}b\nMSE={mse_val:.4f}",
                    fontsize=8,
                )
                ax_bot.set_axis_off()

            patch_fig.tight_layout()
            st.pyplot(patch_fig, width="stretch")
            plt.close(patch_fig)

            # Slide-wide class proportions & Shannon entropy map
            col_prop, col_ent = st.columns([1.15, 1.0])
            with col_prop:
                st.subheader("📈 Slide Predicted Class Proportions (`mini_ml.probability.class_proportions`)")
                pred_labels_all = [CLASS_NAMES[int(c)] for c in agg["pred_class"].ravel()]
                confident_mask = (agg["max_prob"] >= low_prob_threshold).ravel()
                confident_labels = [
                    CLASS_NAMES[int(c)]
                    for c, keep in zip(agg["pred_class"].ravel(), confident_mask)
                    if keep
                ]
                props_all = class_proportions(pred_labels_all)
                props_conf = class_proportions(confident_labels) if confident_labels else {}

                prop_rows = []
                for idx_c, cname in enumerate(CLASS_NAMES):
                    mean_prob_c = mean(agg["avg_probs"][:, :, idx_c].ravel().tolist())
                    prop_rows.append(
                        {
                            "Class": cname,
                            "All Regions (%)": round(props_all.get(cname, 0.0) * 100.0, 2),
                            f"Confident ≥ {low_prob_threshold:.2f} (%)": round(
                                props_conf.get(cname, 0.0) * 100.0, 2
                            ),
                            "Mean Probability": round(mean_prob_c, 4),
                        }
                    )
                st.dataframe(pd.DataFrame(prop_rows), width="stretch", hide_index=True)

            with col_ent:
                st.subheader("🌡️ Prediction Uncertainty (`mini_ml.information.shannon_entropy`)")
                ent_fig, ent_ax = plt.subplots(figsize=(5.6, 4.5), facecolor="white")
                im_ent = ent_ax.imshow(
                    agg["entropy_bits"],
                    cmap="inferno",
                    vmin=0.0,
                    vmax=3.0,
                    interpolation="nearest",
                )
                ent_ax.set_title(
                    f"Shannon Entropy H(p) in bits (max = log₂(8) = 3.0 bits)\n"
                    f"Slide mean = {mean(entropy_flat):.2f} bits, variance = {variance(entropy_flat):.3f}",
                    fontsize=10,
                )
                ent_ax.set_axis_off()
                cbar_ent = ent_fig.colorbar(im_ent, ax=ent_ax, fraction=0.042, pad=0.04)
                cbar_ent.set_label("Shannon entropy (bits)", fontsize=9)
                ent_fig.tight_layout()
                st.pyplot(ent_fig, width="stretch")
                plt.close(ent_fig)

    # =========================================================================
    # TAB 2: Tile Slicing & Pre-processing Explorer (Kather_texture_2016_image_tiles_5000)
    # =========================================================================
    with tab_tiles:
        st.subheader("🧩 Pre-processing Pipeline for `Kather_texture_2016_image_tiles_5000` & Image Slicing")
        st.markdown(
            """
            ### How Inputs Are Sliced & Pre-processing Works
            1. **Source Tiles (`data/images/Kather_texture_2016_image_tiles_5000/`)**:
               Contains **5,000** histological `.tif` tiles (`625` per class across 8 tissue types), each of size **`150×150×3` (`uint8` RGB)**.
            2. **Sliding-Window Slicing on Large Slides (`5000×5000×3` RGB)**:
               A `150×150` window steps across the `5000×5000` image with stride `s` (e.g. `s = 75` yields `65×65 = 4,225` windows). Each window has the exact same `150×150×3` dimensions as a training tile, and overlapping predictions are averaged on the `75×75` sub-tile grid.
            3. **Downsampling & Vectorization (`preprocess_patch` / `image_features`)**:
               - Resize from `150×150` to **`28×28`** using **`PIL.Image.Resampling.BILINEAR`** (`uint8`).
               - Flatten in **C-order** (`reshape(-1)`) into a **`2,352`-dimensional vector** (`28 × 28 × 3 = 2,352`) and scale by **`/ 255.0`** into `float64` in `[0.0, 1.0]`.
            4. **Subspace Projection & Reconstruction (`mini_ml.decomposition`)**:
               - Center using the training mean $\\mu \\in \\mathbb{R}^{2352}$ and project onto orthonormal rows $V_k \\in \\mathbb{R}^{k \\times 2352}$ via `project(X, mean, components)`:
                 $$Z = (X - \\mu) V_k^\\top$$
               - Reconstruct via `reconstruct(Z, mean, components)` ($\\hat{X} = Z V_k + \\mu$) and evaluate entry-wise MSE before display clipping via `reconstruction_error(X, X_hat)`.
            """
        )

        tile_map = cached_tile_paths_by_class()
        col_t1, col_t2, col_t3 = st.columns([1.0, 1.8, 1.0])
        with col_t1:
            chosen_tile_class = st.selectbox(
                "Tissue folder in `Kather_texture_2016_image_tiles_5000/`:",
                options=list(CLASS_NAMES),
                index=0,
            )
        class_files = tile_map.get(chosen_tile_class, [])
        with col_t2:
            chosen_tile_idx = st.slider(
                f"Select tile index in `{chosen_tile_class}` (0 to {max(0, len(class_files) - 1)}):",
                min_value=0,
                max_value=max(0, len(class_files) - 1),
                value=0,
            )
        with col_t3:
            inspect_k = st.select_slider(
                "PCA components `k` for reconstruction:",
                options=[3, 10, 30, 100, 300],
                value=30,
            )

        if class_files:
            tile_path = Path(class_files[chosen_tile_idx])
            with Image.open(tile_path) as raw_pil:
                raw_tile_rgb = np.asarray(raw_pil.convert("RGB"), dtype=np.uint8)

            resized_28, x_vec = preprocess_patch(raw_tile_rgb)
            X_single = x_vec[None, :]

            pca_300 = cached_pca_300()
            ks_list = (3, 10, 30, 100, 300)
            reconstructions = {}
            mse_by_k = {}
            for k_val in ks_list:
                basis_k = pca_300.components_[:k_val]
                z_k = project(X_single, pca_300.mean_, basis_k)
                x_hat_k = reconstruct(z_k, pca_300.mean_, basis_k)
                reconstructions[k_val] = np.clip(x_hat_k.reshape(28, 28, 3), 0.0, 1.0)
                mse_by_k[k_val] = reconstruction_error(X_single, x_hat_k)

            # Visual comparison across pre-processing and multi-k PCA reconstruction
            tile_fig, tile_axes = plt.subplots(1, 7, figsize=(15.5, 2.8), facecolor=PAPER)
            tile_axes[0].imshow(raw_tile_rgb)
            tile_axes[0].set_title("1. Raw Tile\n150×150×3 uint8", fontsize=9.5)
            tile_axes[0].set_axis_off()

            tile_axes[1].imshow(resized_28, interpolation="nearest")
            tile_axes[1].set_title("2. BILINEAR Input\n28×28×3 (2,352d)", fontsize=9.5)
            tile_axes[1].set_axis_off()

            for idx_k, k_val in enumerate(ks_list):
                ax_k = tile_axes[idx_k + 2]
                ax_k.imshow(reconstructions[k_val], interpolation="nearest")
                highlight = " ★" if k_val == inspect_k else ""
                ax_k.set_title(
                    f"Recon ({k_val} PCs){highlight}\nMSE={mse_by_k[k_val]:.5f}",
                    fontsize=9.5,
                    weight="bold" if k_val == inspect_k else "normal",
                )
                ax_k.set_axis_off()

            tile_fig.suptitle(
                f"File: {chosen_tile_class}/{tile_path.name} — Pre-processing & nested PCA reconstructions",
                fontsize=11,
                y=1.02,
            )
            tile_fig.tight_layout()
            st.pyplot(tile_fig, width="stretch")
            plt.close(tile_fig)

            # Classification & Information-Theoretic Metrics for this Tile
            tile_probs = pipeline["predict_proba"](X_single)[0]
            true_class_idx = CLASS_NAMES.index(chosen_tile_class)
            true_dist = [1.0 if i == true_class_idx else 0.0 for i in range(len(CLASS_NAMES))]
            uniform_dist = [1.0 / len(CLASS_NAMES)] * len(CLASS_NAMES)
            prob_list = tile_probs.tolist()

            tile_entropy = shannon_entropy(prob_list)
            tile_ce_bits = cross_entropy(true_dist, prob_list)
            tile_kl_uniform = kl_divergence(prob_list, uniform_dist)

            # Nearest training neighbor in PCA space using mini_ml.linalg.euclidean_distance
            z_query = project(X_single, np.asarray(pipeline["pca_mean"]), np.asarray(pipeline["pca_components"]))[0]
            Z_train_mat = np.asarray(pipeline["Z_train"])
            approx_dists = np.linalg.norm(Z_train_mat - z_query, axis=1)
            top_candidates = np.argsort(approx_dists)[:25]
            exact_dists = [
                (int(i), float(euclidean_distance(z_query, Z_train_mat[int(i)])))
                for i in top_candidates
            ]
            exact_dists.sort(key=lambda item: item[1])
            nn_idx, nn_dist = exact_dists[0]
            bundles = cached_week08_bundles()
            nn_label_idx = int(bundles["train"]["labels"][nn_idx])
            nn_class_name = CLASS_NAMES[nn_label_idx]

            c_info1, c_info2 = st.columns([1.2, 1.0])
            with c_info1:
                prob_fig, prob_ax = plt.subplots(figsize=(6.8, 3.4), facecolor=PAPER)
                bars = prob_ax.barh(
                    np.arange(len(CLASS_NAMES)),
                    tile_probs,
                    color=CLASS_COLORS_RGB,
                    height=0.68,
                )
                prob_ax.set_yticks(np.arange(len(CLASS_NAMES)))
                prob_ax.set_yticklabels(CLASS_NAMES, fontsize=9)
                prob_ax.invert_yaxis()
                prob_ax.set_xlim(0.0, 1.0)
                prob_ax.set_xlabel("Predicted Probability")
                prob_ax.set_title(
                    f"Predicted Class Probabilities (True: {chosen_tile_class})",
                    fontsize=10.5,
                )
                prob_ax.grid(axis="x", alpha=0.2)
                for bar, p_val in zip(bars, tile_probs):
                    prob_ax.text(
                        min(p_val + 0.015, 0.93),
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{p_val:.1%}",
                        va="center",
                        fontsize=8.5,
                    )
                prob_fig.tight_layout()
                st.pyplot(prob_fig, width="stretch")
                plt.close(prob_fig)

            with c_info2:
                st.markdown("#### `mini_ml` Information & Distance Readout")
                st.markdown(
                    f"""
                    - **Predicted Class**: `{CLASS_NAMES[int(np.argmax(tile_probs))]}` (`{np.max(tile_probs):.1%}`)
                    - **True Class Probability**: `{tile_probs[true_class_idx]:.1%}` (`{chosen_tile_class}`)
                    - **Shannon Entropy `shannon_entropy(q)`**: `{tile_entropy:.4f} bits`
                    - **Cross-Entropy `cross_entropy(p_true, q)`**: `{tile_ce_bits:.4f} bits`
                    - **KL vs. Uniform `kl_divergence(q || u)`**: `{tile_kl_uniform:.4f} bits`
                    - **Reconstruction Error (`k={inspect_k}`)**: `{mse_by_k[inspect_k]:.6f}`
                    - **Nearest Training Tile (`euclidean_distance` in {pipeline['n_pcs']}-PC space)**:
                      Training row `#{nn_idx}` (`{nn_class_name}`, distance = `{nn_dist:.4f}`)
                    """
                )

    # =========================================================================
    # TAB 3: mini_ml Model & PCA Diagnostics
    # =========================================================================
    with tab_diagnostics:
        st.subheader("📊 Validation Split Diagnostics & PCA Basis")
        y_val = np.asarray(pipeline["y_val"])
        val_preds = np.asarray(pipeline["val_preds"])

        # Confusion matrix (8x8)
        conf_mat = np.zeros((8, 8), dtype=int)
        for yt, yp in zip(y_val, val_preds):
            conf_mat[int(yt), int(yp)] += 1

        col_d1, col_d2 = st.columns([1.05, 1.0])
        with col_d1:
            cm_fig, cm_ax = plt.subplots(figsize=(6.2, 5.2), facecolor=PAPER)
            im_cm = cm_ax.imshow(conf_mat, cmap="Blues")
            cm_ax.set_xticks(np.arange(8))
            cm_ax.set_yticks(np.arange(8))
            cm_ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=8.5)
            cm_ax.set_yticklabels(CLASS_NAMES, fontsize=8.5)
            cm_ax.set_xlabel("Predicted class")
            cm_ax.set_ylabel("True class")
            cm_ax.set_title(
                f"Validation Confusion Matrix (1,000 tiles) | Acc = {pipeline['val_accuracy']:.1%}",
                fontsize=10.5,
            )
            for r in range(8):
                for c in range(8):
                    val_c = conf_mat[r, c]
                    cm_ax.text(
                        c,
                        r,
                        str(val_c),
                        ha="center",
                        va="center",
                        color="white" if val_c > conf_mat.max() * 0.55 else "#1f2933",
                        fontsize=8,
                    )
            cm_fig.colorbar(im_cm, ax=cm_ax, fraction=0.04, pad=0.04)
            cm_fig.tight_layout()
            st.pyplot(cm_fig, width="stretch")
            plt.close(cm_fig)

        with col_d2:
            if pipeline.get("loss_history") and pipeline.get("validation_loss_history"):
                loss_fig, loss_ax = plt.subplots(figsize=(6.0, 4.8), facecolor=PAPER)
                epochs_axis = np.arange(len(pipeline["loss_history"]))
                loss_ax.plot(epochs_axis, pipeline["loss_history"], color=TEAL, linewidth=2, label="Training CE")
                loss_ax.plot(
                    epochs_axis,
                    pipeline["validation_loss_history"],
                    color=RUST,
                    linewidth=2,
                    label="Validation CE",
                )
                loss_ax.set_xlabel("Epoch")
                loss_ax.set_ylabel("Cross-entropy (nats)")
                loss_ax.set_title("Live `mini_ml` LogisticClassifier SGD Loss Curve", fontsize=10.5)
                loss_ax.grid(alpha=0.25)
                loss_ax.legend()
                loss_fig.tight_layout()
                st.pyplot(loss_fig, width="stretch")
                plt.close(loss_fig)
            elif WEEK09_HISTORY_CSV.is_file():
                hist_df = pd.read_csv(WEEK09_HISTORY_CSV)
                loss_fig, loss_ax = plt.subplots(figsize=(6.0, 4.8), facecolor=PAPER)
                for lr_val, group in hist_df.groupby("learning_rate"):
                    loss_ax.plot(
                        group["epoch"],
                        group["validation_loss"],
                        linewidth=1.8,
                        label=f"Val CE (lr={lr_val:g})",
                    )
                loss_ax.set_xlabel("Epoch")
                loss_ax.set_ylabel("Validation cross-entropy (nats)")
                loss_ax.set_title("Week 9 Learning-Rate Sweep (`week_09_history.csv`)", fontsize=10.5)
                loss_ax.grid(alpha=0.25)
                loss_ax.legend(fontsize=8.5)
                loss_fig.tight_layout()
                st.pyplot(loss_fig, width="stretch")
                plt.close(loss_fig)

        if WEEK09_RESULTS_CSV.is_file():
            st.markdown("#### Week 9 Hyperparameter Sweep Summary (`week_09_results.csv`)")
            st.dataframe(pd.read_csv(WEEK09_RESULTS_CSV), width="stretch", hide_index=True)


def run_cli_check(output_fig_path: Path | None = None) -> int:
    """Run a headless end-to-end verification of tile preprocessing and whole-slide scanning."""
    import matplotlib
    matplotlib.use("Agg")

    print("1. Verifying tile pre-processing against Week 8 validation bundle...")
    bundles = load_week08_bundles()
    val = bundles["validation"]
    sample_rel_path = str(val["filenames"][0])
    sample_tile_path = DATA_DIR / "images" / sample_rel_path
    with Image.open(sample_tile_path) as im:
        resized_28, feat_vec = preprocess_patch(im)
    assert np.array_equal(resized_28, val["images"][0]), "28x28 BILINEAR output mismatch"
    assert feat_vec.shape == (2352,) and feat_vec.dtype == np.float64
    print(f"   Verified tile {sample_rel_path} -> exact match with kather_validation.npz.")

    print("2. Building mini_ml classifier pipeline and checking validation accuracy...")
    pipeline = build_classifier_pipeline("week09_checkpoint")
    print(f"   Pipeline: {pipeline['label']}, Validation accuracy: {pipeline['val_accuracy']:.3f}")

    slide_path = LARGER_IMAGES_DIR / "CRC-Prim-HE-01_APPLICATION.tif"
    print(f"3. Slicing and scanning large slide {slide_path.name} (150px windows, stride 75)...")
    with Image.open(slide_path) as slide_im:
        slide_rgb = np.asarray(slide_im.convert("RGB"), dtype=np.uint8)
    sliced = slice_and_preprocess_large_image(slide_rgb, window_size=150, stride=75)
    window_probs = pipeline["predict_proba"](sliced["X"])
    agg = aggregate_overlapping_probabilities(
        window_probs, sliced["y_positions"], sliced["x_positions"], window_size=150, stride=75
    )
    selected_patches = select_representative_patches(
        window_probs, sliced["y_positions"], sliced["x_positions"], window_size=150, n_patches=8
    )
    props = class_proportions([CLASS_NAMES[int(c)] for c in agg["pred_class"].ravel()])
    print(f"   Windows: {len(sliced['X'])}, Aggregated grid: {agg['avg_probs'].shape}")
    print(f"   Predicted slide proportions: { {k: round(v, 3) for k, v in props.items()} }")

    if output_fig_path is not None:
        fig = make_slide_scan_figure(
            slide_rgb=slide_rgb,
            agg=agg,
            selected_patches=selected_patches,
            slide_name="CRC-Prim-HE-01",
            model_label=str(pipeline["short_label"]),
            n_pcs=int(pipeline["n_pcs"]),
            window_size=150,
            stride=75,
        )
        fig.savefig(output_fig_path, dpi=140)
        plt.close(fig)
        print(f"   Saved preview figure to {output_fig_path}")

    print("All checks passed successfully.")
    return 0


def _in_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _in_streamlit_runtime():
        render_streamlit_app()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--check",
            action="store_true",
            help="Run headless verification of tile pre-processing and 5000x5000 slide scan",
        )
        parser.add_argument(
            "--save-preview",
            type=Path,
            help="Optional path to save a 4-panel whole-slide scan PNG during --check",
        )
        args = parser.parse_args()
        if args.check or args.save_preview is not None:
            raise SystemExit(run_cli_check(args.save_preview))
        render_streamlit_app()
