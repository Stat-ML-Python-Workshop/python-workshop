"""Week 7: student functions drive projection, reconstruction and every MSE.

Run from package root: uv run python test_code/week_07.py
Add --dataset wdbc for the two-feature bridge, or --headless for both checks.
Source: instructor-designed interface; Kather 2016 and UCI WDBC (data provenance).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from mini_ml.decomposition import project, reconstruct, reconstruction_error
from mini_ml.preprocessing import Standardizer

KS = (3, 10, 30, 100, 300)
TEAL, BLUE, RUST, PAPER = "#008782", "#38649b", "#bd5d35", "#fffdf7"


def default_data_dir():
    """Support the source template and the course-synced student package."""
    adjacent = Path(__file__).resolve().parent / "data"
    return adjacent if adjacent.is_dir() else Path(__file__).resolve().parents[1] / "data/week-07"


def read_npz(path):
    with np.load(path, allow_pickle=False) as supplied:
        return {name: supplied[name] for name in supplied.files}


def calculate(X, mean, components, counts):
    """Only the student functions produce the data that the interface plots."""
    results = {}
    for k in counts:
        basis = components[:k]
        scores = project(X, mean, basis)
        rebuilt = reconstruct(scores, mean, basis)
        mse = reconstruction_error(X, rebuilt)
        row_mse = np.array([reconstruction_error(X[i:i + 1], rebuilt[i:i + 1])
                            for i in range(len(X))])
        results[k] = dict(scores=scores, reconstructed=rebuilt, mse=mse, row_mse=row_mse)
    return results


def prepare(data_dir):
    """Fit the W5 scaler once on the exact training rows; reuse teacher bases."""
    tissue = read_npz(data_dir / "pca_basis.npz")
    if tissue["images"].shape != (8, 28, 28, 3) or tissue["components"].shape != (300, 2352):
        raise ValueError("Use the supplied eight-image, 300-component Week 7 bundle")
    tissue["X"] = tissue["images"].reshape(8, -1).astype(np.float64) / 255.0
    tissue["results"] = calculate(tissue["X"], tissue["mean"], tissue["components"], KS)
    wdbc = read_npz(data_dir / "wdbc_basis.npz")
    frame = pd.read_csv(data_dir / "train.csv")
    raw = frame[["mean_radius", "mean_texture"]].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(frame["malignant"], wdbc["labels"])
    np.testing.assert_allclose(raw, wdbc["X"], rtol=0, atol=1e-12)
    wdbc["raw"] = raw
    wdbc["scaler"] = Standardizer().fit(raw)
    wdbc["X"] = wdbc["scaler"].transform(raw)
    wdbc["results"] = calculate(wdbc["X"], wdbc["mean"], wdbc["components"], (1, 2))
    return {"tissue": tissue, "wdbc": wdbc}


def validate(cache, data_dir):
    """Independent NumPy references detect mistakes; they never replace output."""
    for dataset, entry in cache.items():
        if dataset == "tissue":
            reference_x = entry["images"].reshape(8, -1).astype(np.float64) / 255.0
        else:
            raw = entry["raw"]
            reference_x = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)
            np.testing.assert_allclose(entry["scaler"].mean_, entry["standardizer_mean"], atol=1e-12)
            np.testing.assert_allclose(entry["scaler"].scale_, entry["standardizer_scale"], atol=1e-12)
        np.testing.assert_allclose(entry["X"], reference_x, rtol=1e-10, atol=1e-12)
        previous = np.inf
        for k, result in entry["results"].items():
            basis = entry["components"][:k]
            expected_scores = (reference_x - entry["mean"]) @ basis.T
            expected_rebuilt = expected_scores @ basis + entry["mean"]
            expected_row_mse = np.mean((reference_x - expected_rebuilt) ** 2, axis=1)
            np.testing.assert_allclose(result["scores"], expected_scores, rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(result["reconstructed"], expected_rebuilt, rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(result["row_mse"], expected_row_mse, rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(result["mse"], expected_row_mse.mean(), rtol=1e-9, atol=1e-12)
            if result["mse"] > previous + 1e-12:
                raise AssertionError("Adding the supplied orthogonal components increased MSE")
            previous = result["mse"]
            print(f"{dataset}: k={k:3d}; scores={result['scores'].shape}; MSE={result['mse']:.12f}")
    np.testing.assert_allclose(cache["wdbc"]["results"][2]["reconstructed"], cache["wdbc"]["X"], atol=1e-10)
    meta = json.loads((data_dir / "provenance.json").read_text())
    for k, result in cache["tissue"]["results"].items():
        np.testing.assert_allclose(result["mse"], meta["tissue"]["selected_eight_mse"][str(k)], rtol=1e-9, atol=1e-12)
    print("NumPy reference checks passed for tissue and WDBC. MSE is measured before clipping.")


class TissueExplorer:
    """Eight held-out samples, one fixed mean/basis, and five choices of k."""

    def __init__(self, entry):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons

        self.entry, self.sample, self.k = entry, 0, 30
        self.fig = plt.figure(figsize=(13.5, 9), facecolor=PAPER)
        self.fig.canvas.manager.set_window_title("Week 7 — Image reconstruction")
        self.fig.text(.045, .95, "How much can 30 scores preserve?", fontsize=23, weight="bold", color="#80512f")
        self.heading = self.fig.text(.045, .908, "", fontsize=11, color="#556663")
        self.image_axes = [self.fig.add_axes([x, .49, .27, .36]) for x in (.045, .36, .675)]
        self.class_labels = entry["classes"][entry["labels"]].tolist()
        self.sample_buttons = RadioButtons(self.fig.add_axes([.045, .08, .20, .32]), self.class_labels)
        self.k_buttons = RadioButtons(self.fig.add_axes([.28, .13, .11, .25]), [str(k) for k in KS], active=2)
        self.fig.text(.285, .405, "Keep k PCs", fontsize=11, weight="bold")
        self.curve_ax = self.fig.add_axes([.48, .17, .46, .24])
        self.readout = self.fig.text(.48, .065, "", fontsize=10, family="monospace", linespacing=1.6)
        self.fig.text(.045, .025, "Images clipped only for display. MSE compares the un-clipped reconstruction with the 28×28 input.", fontsize=10, color="#556663")
        for buttons in (self.sample_buttons, self.k_buttons):
            buttons.ax.set_facecolor("#f1f0e8")
            for label in buttons.labels:
                label.set_fontsize(9)
        self.sample_buttons.on_clicked(self.select_sample)
        self.k_buttons.on_clicked(self.select_k)
        self.update()

    def select_sample(self, label):
        self.sample = self.class_labels.index(label)
        self.update()

    def select_k(self, label):
        self.k = int(label)
        self.update()

    def update(self):
        e, i, k = self.entry, self.sample, self.k
        result = e["results"][k]
        images = (e["original_images"][i], e["images"][i],
                  np.clip(result["reconstructed"][i].reshape(28, 28, 3), 0, 1))
        for ax, image, title in zip(self.image_axes, images,
                                    ("Source · 150×150 RGB", "Model input · 28×28 RGB", f"Reconstruction · {k} PCs")):
            ax.clear()
            ax.imshow(image, interpolation="nearest")
            ax.set_title(title, fontsize=12)
            ax.set_axis_off()
        self.heading.set_text(f"{self.class_labels[i]} · validation image {e['image_ids'][i]} · all images share the training mean and directions")
        self.fig.texts[0].set_text(f"How much can {k} scores preserve?")
        ax = self.curve_ax
        ax.clear()
        selected = [e["results"][count]["row_mse"][i] for count in KS]
        mean_eight = [e["results"][count]["mse"] for count in KS]
        ax.plot(KS, selected, "o-", color=RUST, label="Selected image")
        ax.plot(KS, mean_eight, "o-", color=TEAL, label="Mean of supplied 8")
        ax.plot([k], [result["row_mse"][i]], "o", color=RUST, markersize=10)
        ax.set(xscale="log", xticks=KS, xticklabels=[str(count) for count in KS],
               xlabel="Number of components, k (log scale)", ylabel="MSE before clipping")
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
        scores = ", ".join(f"{score:.3f}" for score in result["scores"][i, :3])
        self.readout.set_text(f"First 3 scores: [{scores}]\nSelected MSE={result['row_mse'][i]:.6f}   Mean of 8={result['mse']:.6f}")
        self.fig.canvas.draw_idle()


class WDBCExplorer:
    """The same two standardized training features as W6, with one or two PCs."""

    def __init__(self, entry):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.widgets import RadioButtons

        self.entry, self.k = entry, 1
        self.fig = plt.figure(figsize=(11.5, 7.8), facecolor=PAPER)
        self.fig.canvas.manager.set_window_title("Week 7 — WDBC reconstruction")
        self.fig.text(.07, .94, "Keep one direction or keep both", fontsize=23, weight="bold", color="#80512f")
        self.fig.text(.07, .89, "341 training patients · mean radius / mean texture · W5 Standardizer · fixed teacher directions", fontsize=10)
        self.ax = self.fig.add_axes([.075, .18, .50, .64])
        self.readout = self.fig.text(.64, .53, "", fontsize=11, family="monospace", linespacing=1.8)
        self.fig.text(.64, .38, "Filled = standardized observation\nHollow = reconstruction\nDashed = discarded residual\nLabels only color observations.", fontsize=10, linespacing=1.8)
        self.buttons = RadioButtons(self.fig.add_axes([.65, .18, .18, .13]), ["1 PC", "2 PCs"])
        self.buttons.on_clicked(self.select_k)
        self.LineCollection = LineCollection
        self.update()

    def select_k(self, label):
        self.k = 1 if label == "1 PC" else 2
        self.update()

    def update(self):
        e, ax, result = self.entry, self.ax, self.entry["results"][self.k]
        x, rebuilt = e["X"], result["reconstructed"]
        bound = max(float(np.max(np.linalg.norm(x, axis=1))) * 1.1, 1.0)
        ax.clear()
        for label, name, color in ((0, "Benign", BLUE), (1, "Malignant", RUST)):
            subset = x[e["labels"] == label]
            ax.scatter(*subset.T, s=14, alpha=.6, label=name, color=color, zorder=3)
        ax.scatter(*rebuilt.T, s=18, facecolors="none", edgecolors=TEAL, linewidths=.6, zorder=4)
        ax.add_collection(self.LineCollection(np.stack([x, rebuilt], axis=1), colors="#8b938c",
                                             linewidths=.5, linestyles="dashed", alpha=.45))
        for index, direction in enumerate(e["components"][:self.k]):
            endpoints = np.array([-bound, bound])[:, None] * direction + e["mean"]
            ax.plot(*endpoints.T, color=TEAL, linewidth=1.2, label=f"PC{index + 1} direction")
        ax.set(xlim=(-bound, bound), ylim=(-bound, bound), xlabel="Mean radius (standardized)",
               ylabel="Mean texture (standardized)", title=f"{self.k} PC{'s' if self.k > 1 else ''}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", fontsize=9)
        self.readout.set_text(f"X shape: {x.shape}\nZ shape: {result['scores'].shape}\nX_hat:   {rebuilt.shape}\nMSE:     {result['mse']:.8f}")
        self.fig.canvas.draw_idle()


def save_reference_figures(cache, output_dir):
    """Static panels and tables; every numeric result comes from calculate()."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    e = cache["tissue"]
    fig, axes = plt.subplots(8, 7, figsize=(14, 16), facecolor=PAPER)
    fig.subplots_adjust(top=.94, bottom=.045, left=.12, right=.99, wspace=.06, hspace=.18)
    for row in range(8):
        samples = [e["original_images"][row], e["images"][row]] + [
            np.clip(e["results"][k]["reconstructed"][row].reshape(28, 28, 3), 0, 1) for k in KS]
        for ax, sample in zip(axes[row], samples):
            ax.imshow(sample, interpolation="nearest")
            ax.set(xticks=[], yticks=[])
        axes[row, 0].set_ylabel(str(e["classes"][e["labels"][row]]), rotation=0, ha="right", va="center", fontsize=10)
    for ax, title in zip(axes[0], ["Source 150×150", "Input 28×28"] + [f"{k} PCs" for k in KS]):
        ax.set_title(title, fontsize=11)
    fig.suptitle("Eight fixed validation images · student projection and reconstruction", fontsize=17)
    fig.text(.5, .015, "Display clipping only. MSE uses the 28×28 input and un-clipped reconstruction.", ha="center", fontsize=10)
    fig.savefig(output_dir / "week_07_tissue_reconstruction.png", dpi=130, facecolor=PAPER)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.7), layout="constrained", facecolor=PAPER)
    for i, name in enumerate(e["classes"]):
        ax.plot(KS, [e["results"][k]["row_mse"][i] for k in KS], "o-", alpha=.5, linewidth=1, label=str(name))
    ax.plot(KS, [e["results"][k]["mse"] for k in KS], "o-", color="black", linewidth=2.5, label="Mean of supplied 8")
    ax.set(xscale="log", xticks=KS, xticklabels=[str(k) for k in KS], xlabel="Components, k (log scale)",
           ylabel="Mean squared entry error, before clipping", title="Reconstruction loss on the eight supplied validation images")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output_dir / "week_07_tissue_mse.png", dpi=140, facecolor=PAPER)
    plt.close(fig)
    tissue_explorer = TissueExplorer(e)
    tissue_explorer.fig.savefig(output_dir / "week_07_tissue_explorer.png", dpi=120, facecolor=PAPER)
    # Exercise both controls headlessly, in addition to numerical reference checks.
    tissue_explorer.sample_buttons.set_active(7)
    tissue_explorer.k_buttons.set_active(4)
    assert tissue_explorer.sample == 7 and tissue_explorer.k == 300
    plt.close(tissue_explorer.fig)
    wdbc_explorer = WDBCExplorer(cache["wdbc"])
    wdbc_explorer.fig.savefig(output_dir / "week_07_wdbc.png", dpi=130, facecolor=PAPER)
    wdbc_explorer.buttons.set_active(1)
    assert wdbc_explorer.k == 2
    wdbc_explorer.fig.savefig(output_dir / "week_07_wdbc_full.png", dpi=130, facecolor=PAPER)
    plt.close(wdbc_explorer.fig)
    with (output_dir / "week_07_tissue_scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "class_name"] + [f"PC{i + 1}" for i in range(300)])
        for i, scores in enumerate(e["results"][300]["scores"]):
            writer.writerow([int(e["image_ids"][i]), str(e["classes"][e["labels"][i]])] + scores.tolist())
    with (output_dir / "week_07_tissue_mse.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["k", "mean_of_eight_mse"] + e["classes"].tolist())
        for k in KS:
            result = e["results"][k]
            writer.writerow([k, result["mse"]] + result["row_mse"].tolist())
    print(f"Saved five reference PNGs, scores CSV and MSE CSV to {output_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("tissue", "wdbc"), default="tissue")
    parser.add_argument("--headless", action="store_true", help="validate both datasets and save reference PNGs/CSVs")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache = prepare(args.data_dir)
    validate(cache, args.data_dir)
    if args.headless:
        save_reference_figures(cache, args.output_dir)
    else:
        explorer = TissueExplorer(cache["tissue"]) if args.dataset == "tissue" else WDBCExplorer(cache["wdbc"])
        if explorer.fig.canvas.required_interactive_framework is None:
            plt.close(explorer.fig)
            parser.error("No native GUI backend. Run in a local desktop terminal with GUI-enabled Python/Matplotlib; "
                         "unset MPLBACKEND if it forces Agg. Use --headless for checks and PNGs.")
        plt.show()


if __name__ == "__main__":
    main()
