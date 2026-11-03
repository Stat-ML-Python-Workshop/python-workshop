"""Week 8: fit student PCA once, then explore scores, components and reconstruction.

From package root: uv run python test_code/week_08.py
Use --dataset wdbc for the 30-feature bridge; --headless checks both and saves figures.
Data and plotting are instructor supplied. mini_ml performs the fitted mathematics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
import numpy as np
from mini_ml.decomposition import PCA, reconstruct, reconstruction_error
from mini_ml.preprocessing import Standardizer

KS = (3, 10, 30, 100, 300)
TEAL, BLUE, RUST, PAPER = "#008782", "#38649b", "#bd5d35", "#fffdf7"
COLORS = ("#a84735", "#486994", "#8b548a", "#23756d", "#a27923", "#d06e44", "#65a06b", "#74818e")


def default_data_dir():
    adjacent = Path(__file__).resolve().parent / "data"
    return adjacent if adjacent.is_dir() else Path(__file__).resolve().parents[1] / "data/week-08"


def read_npz(path):
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files}


def load_data(data_dir=None):
    """Load and verify the pickle-free offline pack; never load test images/state."""
    data_dir = Path(data_dir or default_data_dir())
    provenance = json.loads((data_dir / "provenance.json").read_text())
    expected_files = {"wdbc.npz", "kather_train.npz", "kather_validation.npz", "kather_examples.npz"}
    if set(provenance["files"]) != expected_files:
        raise ValueError("Use the complete supplied Week 8 data pack")
    for filename, record in provenance["files"].items():
        content = (data_dir / filename).read_bytes()
        if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"Week 8 data checksum mismatch: {filename}")
    data = {"wdbc": read_npz(data_dir / "wdbc.npz"),
            "tissue_train": read_npz(data_dir / "kather_train.npz"),
            "tissue_validation": read_npz(data_dir / "kather_validation.npz"),
            "examples": read_npz(data_dir / "kather_examples.npz"), "provenance": provenance}
    train, validation, examples, wdbc = (data[key] for key in ("tissue_train", "tissue_validation", "examples", "wdbc"))
    if train["images"].shape != (3000, 28, 28, 3) or validation["images"].shape != (1000, 28, 28, 3):
        raise ValueError("Expected the 3000/1000 RGB training/validation split")
    if train["images"].dtype != np.uint8 or validation["images"].dtype != np.uint8:
        raise ValueError("Supplied images must be uint8")
    if set(train["image_ids"]) & set(validation["image_ids"]):
        raise ValueError("Training and validation IDs overlap")
    if not set(examples["image_ids"]) <= set(validation["image_ids"]):
        raise ValueError("All display examples must belong to validation")
    if wdbc["X_train"].shape != (341, 30) or wdbc["X_validation"].shape != (114, 30):
        raise ValueError("Expected the fixed 30-feature WDBC split")
    if set(wdbc["train_ids"]) & set(wdbc["validation_ids"]):
        raise ValueError("WDBC training and validation IDs overlap")
    return data


def image_features(images):
    return images.reshape(len(images), -1).astype(np.float64) / 255.0


def fit_models(data):
    """Two training-only fits; UI controls never call this function again."""
    scaler = Standardizer().fit(data["wdbc"]["X_train"])
    started = time.perf_counter()
    wdbc = PCA(n_components=30, solver="auto", random_state=42).fit(scaler.transform(data["wdbc"]["X_train"]))
    wdbc_seconds = time.perf_counter() - started
    started = time.perf_counter()
    tissue = PCA(n_components=300, solver="auto", random_state=42).fit(image_features(data["tissue_train"]["images"]))
    tissue_seconds = time.perf_counter() - started
    print(f"Training-only fit: WDBC 341x30 {wdbc_seconds:.3f}s; tissue 3000x2352 {tissue_seconds:.3f}s. Reusing these fits.")
    return {"wdbc": wdbc, "tissue": tissue, "scaler": scaler,
            "fit_seconds": {"wdbc": wdbc_seconds, "tissue": tissue_seconds}}


def _state(model):
    return {"components": model.components_, "mean": model.mean_,
            "explained_variance": model.explained_variance_,
            "explained_variance_ratio": model.explained_variance_ratio_}


def build_results(data, models):
    """Compute once using student methods; nested-k views use one ordered basis."""
    wdbc, tissue, scaler = (models[key] for key in ("wdbc", "tissue", "scaler"))
    wtrain = scaler.transform(data["wdbc"]["X_train"])
    wvalidation = scaler.transform(data["wdbc"]["X_validation"])
    wscores = wdbc.transform(wvalidation)
    wresult = {**_state(wdbc), "X_train": wtrain, "X_validation": wvalidation,
               "train_scores": wdbc.transform(wtrain), "validation_scores": wscores,
               "validation_reconstructed": wdbc.inverse_transform(wscores)}
    training = image_features(data["tissue_train"]["images"])
    validation = image_features(data["tissue_validation"]["images"])
    train_scores, validation_scores = tissue.transform(training), tissue.transform(validation)
    lookup = {int(image_id): position for position, image_id in enumerate(data["tissue_validation"]["image_ids"])}
    positions = np.array([lookup[int(image_id)] for image_id in data["examples"]["image_ids"]])
    example_x = validation[positions]
    example_results, validation_mse = {}, {}
    for k in KS:
        # These are nested views of one fitted 300-component PCA, not five refits.
        rebuilt = (tissue.inverse_transform(validation_scores) if k == 300 else
                   reconstruct(validation_scores[:, :k], tissue.mean_, tissue.components_[:k]))
        validation_mse[k] = reconstruction_error(validation, rebuilt)
        selected = rebuilt[positions]
        row_mse = np.array([reconstruction_error(x[None, :], y[None, :]) for x, y in zip(example_x, selected)])
        example_results[k] = {"scores": validation_scores[positions, :k], "reconstructed": selected,
                              "row_mse": row_mse, "mse": reconstruction_error(example_x, selected)}
    tresult = {**_state(tissue), "train_scores": train_scores, "validation_scores": validation_scores,
               "example_positions": positions, "example_X": example_x, "example_results": example_results,
               "validation_mse": validation_mse}
    return {"wdbc": wresult, "tissue": tresult}


def validate(data, models, results):
    """NumPy references detect errors; they never substitute plotted student output."""
    raw = data["wdbc"]["X_train"]
    reference = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)
    np.testing.assert_allclose(results["wdbc"]["X_train"], reference, atol=1e-12)
    reference_validation = (data["wdbc"]["X_validation"] - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)
    np.testing.assert_allclose(results["wdbc"]["X_validation"], reference_validation, atol=1e-12)
    for dataset, training in (("wdbc", reference), ("tissue", image_features(data["tissue_train"]["images"]))):
        model, entry = models[dataset], results[dataset]
        np.testing.assert_allclose(model.mean_, training.mean(axis=0), atol=1e-12)
        if model.n_samples_ != len(training) or model.n_features_in_ != training.shape[1]:
            raise AssertionError("PCA must save the training sample/feature counts")
        np.testing.assert_allclose(model.components_ @ model.components_.T,
                                   np.eye(len(model.components_)), rtol=0, atol=1e-8)
        np.testing.assert_allclose(entry["train_scores"], (training - model.mean_) @ model.components_.T,
                                   rtol=1e-9, atol=1e-10)
        score_variance = np.var(entry["train_scores"], axis=0, ddof=1)
        # The image helper estimates singular values in a randomized sketch.
        # Its tail estimates need not equal the exact projected variances;
        # 1% bounds this numerical approximation, not the student's formulas.
        variance_tolerance = 1e-8 if dataset == "wdbc" else 1e-2
        np.testing.assert_allclose(model.explained_variance_, score_variance, rtol=variance_tolerance, atol=1e-10)
        total = float(np.sum(np.var(training, axis=0, ddof=1)))
        np.testing.assert_allclose(model.explained_variance_ratio_, model.explained_variance_ / total,
                                   rtol=1e-10, atol=1e-12)
        if np.any(np.diff(model.explained_variance_) > 1e-10):
            raise AssertionError("Explained variances must be descending")
    covariance = np.cov(reference, rowvar=False, ddof=1)
    expected_variance = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0)
    np.testing.assert_allclose(models["wdbc"].explained_variance_, expected_variance, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(results["wdbc"]["validation_reconstructed"], reference_validation, atol=1e-9)
    wreference = reference_validation @ models["wdbc"].components_.T - models["wdbc"].mean_ @ models["wdbc"].components_.T
    np.testing.assert_allclose(results["wdbc"]["validation_scores"], wreference, atol=1e-10)
    t, model = results["tissue"], models["tissue"]
    validation = image_features(data["tissue_validation"]["images"])
    np.testing.assert_allclose(t["validation_scores"], (validation - model.mean_) @ model.components_.T, atol=1e-10)
    previous_examples, previous_all = np.inf, np.inf
    for k in KS:
        result = t["example_results"][k]
        expected = ((validation - model.mean_) @ model.components_[:k].T) @ model.components_[:k] + model.mean_
        expected_rows = np.mean((validation - expected) ** 2, axis=1)
        np.testing.assert_allclose(t["validation_mse"][k], expected_rows.mean(), atol=1e-12)
        np.testing.assert_allclose(result["reconstructed"], expected[t["example_positions"]], atol=1e-10)
        np.testing.assert_allclose(result["row_mse"], expected_rows[t["example_positions"]], atol=1e-12)
        np.testing.assert_allclose(result["mse"], result["row_mse"].mean(), atol=1e-12)
        if result["mse"] > previous_examples + 1e-12 or t["validation_mse"][k] > previous_all + 1e-12:
            raise AssertionError("MSE must not increase along a nested orthonormal basis")
        previous_examples, previous_all = result["mse"], t["validation_mse"][k]
        print(f"tissue k={k:3d}: mean of 8={result['mse']:.10f}; all 1000 validation={t['validation_mse'][k]:.10f}")
    print("NumPy reference checks passed: training-only state, WDBC eigenvalues, projection/reconstruction and both MSE populations.")


def _score_axes(ax, scores, labels, names, ratios, title, colors=None):
    palette = colors or COLORS
    for label, name in enumerate(names):
        selected = scores[labels == label, :3]
        ax.scatter(*selected.T, s=12, alpha=.55, color=palette[label], label=str(name))
    ax.set(xlabel=f"PC1 · {ratios[0]:.1%}", ylabel=f"PC2 · {ratios[1]:.1%}",
           zlabel=f"PC3 · {ratios[2]:.1%}", title=title)
    ax.view_init(elev=23, azim=-58)


def score_figure(data, results, dataset):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.5, 6.2), facecolor=PAPER)
    entry = results[dataset]
    if dataset == "wdbc":
        labels = (data["wdbc"]["y_train"], data["wdbc"]["y_validation"])
        names, colors = ("Benign", "Malignant"), (BLUE, RUST)
        counts, heading = (341, 114), "WDBC · 30 features → first three student PCA scores"
    else:
        labels = (data["tissue_train"]["labels"], data["tissue_validation"]["labels"])
        names, colors = data["tissue_train"]["classes"], COLORS
        counts, heading = (3000, 1000), "Tissue · 2352 RGB features → first three student PCA scores"
    pooled = np.concatenate([entry["train_scores"][:, :3], entry["validation_scores"][:, :3]])
    lower, upper = pooled.min(axis=0), pooled.max(axis=0)
    margin = np.maximum((upper - lower) * .06, .1)
    for index, (split, label, count) in enumerate(zip(("train", "validation"), labels, counts)):
        ax = fig.add_subplot(1, 2, index + 1, projection="3d")
        _score_axes(ax, entry[f"{split}_scores"], label, names, entry["explained_variance_ratio"],
                    f"{'Training' if index == 0 else 'Validation'} · {count} observations", colors)
        ax.set(xlim=(lower[0] - margin[0], upper[0] + margin[0]),
               ylim=(lower[1] - margin[1], upper[1] + margin[1]),
               zlim=(lower[2] - margin[2], upper[2] + margin[2]))
        if index == 1:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, .98), fontsize=8, frameon=False)
    fig.suptitle(heading, fontsize=17, y=.97)
    fig.text(.045, .025, "Shared coordinate system and axis limits. Labels color points only. Rotate with the mouse in a desktop window.", fontsize=10)
    fig.subplots_adjust(left=.01, right=.85, top=.86, bottom=.14, wspace=.12)
    return fig


def loadings_figure(data, results, components=(0, 1, 2)):
    import matplotlib.pyplot as plt

    names, basis = data["wdbc"]["feature_names"], results["wdbc"]["components"]
    fig, axes = plt.subplots(1, len(components), figsize=(14, 9.8), squeeze=False, sharey=True, facecolor=PAPER)
    bound = float(np.max(np.abs(basis[list(components)]))) * 1.12
    for ax, component in zip(axes[0], components):
        values = basis[component]
        ax.barh(np.arange(30), values, color=np.where(values >= 0, TEAL, RUST), height=.72)
        ax.axvline(0, color="#66736d", linewidth=.7)
        ax.set(yticks=np.arange(30), yticklabels=names, xlim=(-bound, bound),
               xlabel="Signed coefficient", title=f"PC{component + 1} · {results['wdbc']['explained_variance_ratio'][component]:.1%}")
        ax.grid(axis="x", alpha=.15)
    axes[0, 0].invert_yaxis()
    fig.suptitle("All 30 feature coefficients · a component combines correlated measurements", fontsize=16, y=.97)
    fig.text(.28, .025, "Coefficient sign may flip together with scores. These are weights, not supervised feature importance.", fontsize=10)
    fig.subplots_adjust(left=.22, right=.98, top=.91, bottom=.09, wspace=.12)
    return fig


def variance_figure(results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), facecolor=PAPER, layout="constrained")
    for ax, dataset, title in zip(axes, ("wdbc", "tissue"), ("WDBC · all 30 components", "Tissue · randomized approximation, 300 of 2352 directions")):
        ratio = results[dataset]["explained_variance_ratio"]
        ax.plot(np.arange(1, len(ratio) + 1), np.cumsum(ratio), color=TEAL, linewidth=2.2)
        ax.axhline(1, color="#9b9d95", linestyle="--", linewidth=1)
        ax.set(xlabel="Number of retained components", ylabel="Cumulative explained variance ratio", ylim=(0, 1.04), title=title)
        ax.grid(alpha=.2)
        ax.text(.97, .08, f"Retained total: {ratio.sum():.2%}", transform=ax.transAxes, ha="right", fontsize=11)
    fig.suptitle("The denominator is total training variance across every feature", fontsize=16)
    return fig


def component_figure(results):
    import matplotlib.pyplot as plt

    entry = results["tissue"]
    basis = entry["components"][:3].reshape(3, 28, 28, 3)
    mean = entry["mean"].reshape(28, 28, 3)
    bound = float(np.max(np.abs(basis)))
    fig, axes = plt.subplots(3, 6, figsize=(14.2, 8.5), facecolor=PAPER)
    for row in range(3):
        for channel in range(3):
            axes[row, channel].imshow(basis[row, :, :, channel], cmap="RdBu_r", vmin=-bound, vmax=bound)
        amplitude = 2 * np.sqrt(entry["explained_variance"][row])
        for col, sign in ((3, -1), (4, 0), (5, 1)):
            axes[row, col].imshow(np.clip(mean + sign * amplitude * basis[row], 0, 1))
        axes[row, 0].set_ylabel(f"PC{row + 1}\n{entry['explained_variance_ratio'][row]:.1%}", fontsize=11)
    for ax, title in zip(axes[0], ("R coefficient", "G coefficient", "B coefficient", "Mean − 2σ × PC", "Training mean", "Mean + 2σ × PC")):
        ax.set_title(title, fontsize=10)
    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])
    fig.suptitle("A shared image direction · signed RGB coefficients and isolated variation", fontsize=16, y=.97)
    fig.text(.06, .045, f"All signed panels share [−{bound:.4f}, +{bound:.4f}] (blue negative, red positive). σ = √component variance.", fontsize=10)
    fig.text(.06, .018, "Mean± images are RGB, clipped only for display. A component is not a tissue identity; its overall sign is arbitrary.", fontsize=10)
    fig.subplots_adjust(left=.07, right=.99, top=.88, bottom=.10, wspace=.08, hspace=.17)
    return fig


def reconstruction_figure(data, results):
    import matplotlib.pyplot as plt

    t = results["tissue"]
    inputs = t["example_X"].reshape(8, 28, 28, 3)
    fig, axes = plt.subplots(8, 7, figsize=(14, 16), facecolor=PAPER)
    for row in range(8):
        images = [data["examples"]["original_images"][row], inputs[row]] + [
            np.clip(t["example_results"][k]["reconstructed"][row].reshape(28, 28, 3), 0, 1) for k in KS]
        for ax, value in zip(axes[row], images):
            ax.imshow(value, interpolation="nearest")
            ax.set(xticks=[], yticks=[])
        axes[row, 0].set_ylabel(str(data["examples"]["classes"][data["examples"]["labels"][row]]), rotation=0, ha="right", va="center", fontsize=9)
    for ax, title in zip(axes[0], ("Source 150×150", "Input 28×28", *[f"{k} PCs" for k in KS])):
        ax.set_title(title, fontsize=10)
    fig.suptitle("Eight fixed validation images · directions learned by student PCA", fontsize=17)
    fig.text(.5, .018, "One fit, nested directions. MSE compares with the 28×28 input before RGB clipping; resizing loss cannot return.", ha="center", fontsize=10)
    fig.subplots_adjust(left=.12, right=.99, top=.94, bottom=.05, wspace=.06, hspace=.18)
    return fig


def mse_figure(data, results):
    import matplotlib.pyplot as plt

    t = results["tissue"]
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PAPER, layout="constrained")
    for row, name in enumerate(data["examples"]["classes"]):
        ax.plot(KS, [t["example_results"][k]["row_mse"][row] for k in KS], "o-", linewidth=.8, alpha=.45, color=COLORS[row], label=str(name))
    ax.plot(KS, [t["example_results"][k]["mse"] for k in KS], "o-", color=TEAL, linewidth=2.5, label="Mean of fixed 8 examples")
    ax.plot(KS, [t["validation_mse"][k] for k in KS], "s--", color="#303e39", linewidth=2, label="Mean of all 1000 validation")
    ax.set(xscale="log", xticks=KS, xticklabels=list(map(str, KS)), xlabel="Retained components, k (log scale)",
           ylabel="MSE against 28×28 input, before clipping", title="Two averages, two explicitly named validation populations")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    return fig


class TissueExplorer:
    """Controls index cached output; changing k does not refit the model."""

    def __init__(self, data, results):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons

        self.data, self.results, self.sample, self.k = data, results, 0, 30
        self.fig = plt.figure(figsize=(13.5, 9), facecolor=PAPER)
        self.fig.canvas.manager.set_window_title("Week 8 — Student-fitted PCA")
        self.heading = self.fig.text(.045, .95, "", fontsize=23, weight="bold", color="#80512f")
        self.fig.text(.045, .908, "3000 training images → one fitted mean/basis. Controls reuse cached validation scores; no refitting.", fontsize=10)
        self.image_axes = [self.fig.add_axes([x, .49, .27, .36]) for x in (.045, .36, .675)]
        self.class_labels = data["examples"]["classes"][data["examples"]["labels"]].tolist()
        self.sample_buttons = RadioButtons(self.fig.add_axes([.045, .08, .20, .32]), self.class_labels)
        self.k_buttons = RadioButtons(self.fig.add_axes([.28, .13, .11, .25]), list(map(str, KS)), active=2)
        self.fig.text(.285, .405, "Keep k PCs", fontsize=11, weight="bold")
        self.curve_ax = self.fig.add_axes([.48, .18, .46, .24])
        self.readout = self.fig.text(.48, .065, "", fontsize=10, family="monospace", linespacing=1.5)
        self.fig.text(.045, .025, "A lower reconstruction error does not establish better classification. Labels are never supplied to PCA.", fontsize=10)
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
        t, row, k = self.results["tissue"], self.sample, self.k
        result = t["example_results"][k]
        images = (self.data["examples"]["original_images"][row], t["example_X"][row].reshape(28, 28, 3),
                  np.clip(result["reconstructed"][row].reshape(28, 28, 3), 0, 1))
        for ax, value, title in zip(self.image_axes, images, ("Source · 150×150", "Model input · 28×28", f"Reconstruction · {k} PCs")):
            ax.clear(); ax.imshow(value, interpolation="nearest"); ax.set_title(title, fontsize=12); ax.set_axis_off()
        self.heading.set_text(f"{self.class_labels[row]} · keep {k} learned directions")
        ax = self.curve_ax
        ax.clear()
        ax.plot(KS, [t["example_results"][count]["row_mse"][row] for count in KS], "o-", color=RUST, label="Selected image")
        ax.plot(KS, [t["example_results"][count]["mse"] for count in KS], "o-", color=TEAL, label="Mean of fixed 8")
        ax.plot(KS, [t["validation_mse"][count] for count in KS], "s--", color="#303e39", label="All 1000 validation")
        ax.plot([k], [result["row_mse"][row]], "o", color=RUST, markersize=10)
        ax.set(xscale="log", xticks=KS, xticklabels=list(map(str, KS)), xlabel="Components, k", ylabel="MSE before clipping")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
        self.readout.set_text(f"Image ID {int(self.data['examples']['image_ids'][row])}; selected MSE={result['row_mse'][row]:.6f}\n"
                              f"Mean of 8={result['mse']:.6f}; all 1000={t['validation_mse'][k]:.6f}")
        self.fig.canvas.draw_idle()


def state_summary(data, models, results):
    summary = {"week": 8, "random_state": 42, "image_fit_count": 1, "test_used": False,
               "data_sha256": {name: record["sha256"] for name, record in data["provenance"]["files"].items()},
               "mse_populations": data["provenance"]["tissue"]["mse_populations"], "models": {}}
    for name in ("wdbc", "tissue"):
        model = models[name]
        summary["models"][name] = {
            "n_samples": int(model.n_samples_), "n_features": int(model.n_features_in_),
            "mean_shape": list(model.mean_.shape), "components_shape": list(model.components_.shape),
            "fit_seconds": models["fit_seconds"][name],
            "explained_variance": model.explained_variance_.tolist(),
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            "retained_variance_ratio": float(model.explained_variance_ratio_.sum()),
            "solver": str(getattr(model, "solver_", getattr(model, "solver", "auto"))),
        }
    summary["tissue_mse"] = {str(k): {"fixed_eight": results["tissue"]["example_results"][k]["mse"],
                                     "all_1000_validation": results["tissue"]["validation_mse"][k]} for k in KS}
    return summary


def save_outputs(data, models, results, output_dir):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    factories = {
        "week_08_wdbc_scores.png": lambda: score_figure(data, results, "wdbc"),
        "week_08_wdbc_loadings.png": lambda: loadings_figure(data, results),
        "week_08_variance.png": lambda: variance_figure(results),
        "week_08_tissue_scores.png": lambda: score_figure(data, results, "tissue"),
        "week_08_tissue_components.png": lambda: component_figure(results),
        "week_08_tissue_reconstruction.png": lambda: reconstruction_figure(data, results),
        "week_08_tissue_mse.png": lambda: mse_figure(data, results),
    }
    for name, factory in factories.items():
        fig = factory(); fig.savefig(output_dir / name, dpi=130, facecolor=PAPER); plt.close(fig)
    explorer = TissueExplorer(data, results)
    explorer.fig.savefig(output_dir / "week_08_tissue_explorer.png", dpi=120, facecolor=PAPER)
    for sample, k_index in ((7, 4), (2, 0), (0, 2)):
        explorer.sample_buttons.set_active(sample); explorer.k_buttons.set_active(k_index)
        if explorer.sample != sample or explorer.k != KS[k_index]:
            raise AssertionError("Explorer controls did not select the cached result")
    plt.close(explorer.fig)
    with (output_dir / "week_08_tissue_mse.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["k", "mean_of_fixed_eight", "mean_of_all_1000_validation", *data["examples"]["classes"].tolist()])
        for k in KS:
            entry = results["tissue"]["example_results"][k]
            writer.writerow([k, entry["mse"], results["tissue"]["validation_mse"][k], *entry["row_mse"].tolist()])
    with (output_dir / "week_08_scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["dataset", "split", "observation_id", "class", "PC1", "PC2", "PC3"])
        for dataset in ("wdbc", "tissue"):
            for split in ("train", "validation"):
                if dataset == "wdbc":
                    ids, labels = data["wdbc"][f"{split}_ids"], data["wdbc"][f"y_{split}"]
                    names = np.array(["Benign", "Malignant"])
                else:
                    source = data[f"tissue_{split}"]
                    ids, labels, names = source["image_ids"], source["labels"], source["classes"]
                for image_id, label, scores in zip(ids, labels, results[dataset][f"{split}_scores"]):
                    writer.writerow([dataset, split, int(image_id), str(names[label]), *scores[:3].tolist()])
    with (output_dir / "week_08_wdbc_coefficients.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["feature", *[f"PC{i + 1}" for i in range(30)]])
        for feature, coefficients in zip(data["wdbc"]["feature_names"], results["wdbc"]["components"].T):
            writer.writerow([str(feature), *coefficients.tolist()])
    (output_dir / "week_08_state.json").write_text(json.dumps(state_summary(data, models, results), indent=2) + "\n")
    print(f"Saved eight PNGs, three CSV tables and week_08_state.json to {output_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("tissue", "wdbc"), default="tissue")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_data(args.data_dir)
    models = fit_models(data)
    results = build_results(data, models)
    validate(data, models, results)
    if args.headless:
        save_outputs(data, models, results, args.output_dir)
    else:
        figures = [score_figure(data, results, args.dataset), variance_figure(results)]
        if args.dataset == "wdbc":
            figures.append(loadings_figure(data, results))
        else:
            explorer = TissueExplorer(data, results)
            figures += [explorer.fig, component_figure(results)]
        if figures[0].canvas.required_interactive_framework is None:
            plt.close("all")
            parser.error("No native GUI backend. Use desktop Python/Matplotlib and unset MPLBACKEND=Agg; --headless saves checked figures.")
        plt.show()


if __name__ == "__main__":
    main()
