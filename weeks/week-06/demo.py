"""Teacher-supplied interface; mini_ml supplies every covariance/variance value.

Source: instructor-designed example; Matplotlib Slider/RadioButtons examples.
Run from package root: uv run python test_code/week_06.py [--headless]
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from mini_ml.decomposition import center, covariance_matrix, variance_along_direction
from mini_ml.preprocessing import Standardizer

SYNTHETIC = np.array([[2, 1], [3, 2], [4, 2], [5, 4],
                      [6, 3], [7, 6], [8, 5], [9, 7]], dtype=float)
ANGLES = np.arange(181)
MODES = ("Center only", "Standardize")
TEAL, BLUE, RUST = "#008782", "#38649b", "#bd5d35"


def unit_direction(angle):
    """angle: degrees -> ndarray (2,) of unit length."""
    radians = np.deg2rad(angle)
    return np.array([np.cos(radians), np.sin(radians)])


def load_datasets(data_path=None):
    if data_path is None:
        data_path = Path(__file__).resolve().parents[1] / "data/week-06/train.csv"
    frame = pd.read_csv(data_path)
    if len(frame) != 341:
        raise ValueError("Expected the supplied 341 WDBC training patients")
    return {
        "Synthetic": (SYNTHETIC.copy(), None, ("Feature 1", "Feature 2")),
        "WDBC": (frame[["mean_radius", "mean_texture"]].to_numpy(dtype=float),
                 frame["malignant"].to_numpy(), ("Mean radius", "Mean texture")),
    }


def prepare(datasets):
    """Fit each dataset/mode once; labels never enter numerical functions."""
    cache = {}
    for dataset, (raw, labels, names) in datasets.items():
        for mode in MODES:
            if mode == "Center only":
                Xc, _ = center(raw)
            else:
                scaler = Standardizer().fit(raw)
                Xc = scaler.transform(raw)  # Already centered; no second subtraction.
            C = covariance_matrix(Xc)
            curve = np.array([variance_along_direction(Xc, unit_direction(a))
                              for a in ANGLES])
            cache[dataset, mode] = dict(Xc=Xc, C=C, curve=curve, labels=labels,
                                       names=names, raw=raw)
    return cache


def validate(cache):
    """Independent references check results; they never supply plotted values."""
    for (dataset, mode), entry in cache.items():
        raw, Xc = entry["raw"], entry["Xc"]
        reference = raw - np.mean(raw, axis=0)
        if mode == "Standardize":
            scales = np.std(raw, axis=0, ddof=0)
            reference = reference / np.where(scales == 0, 1, scales)
        np.testing.assert_allclose(Xc, reference, atol=1e-10)
        np.testing.assert_allclose(entry["C"], np.cov(reference, rowvar=False, ddof=1), atol=1e-10)
        expected = [np.var(reference @ unit_direction(a), ddof=1) for a in ANGLES]
        np.testing.assert_allclose(entry["curve"], expected, atol=1e-10)
        best = int(np.argmax(entry["curve"]))
        current = variance_along_direction(Xc, unit_direction(41))
        np.testing.assert_allclose(current, expected[41], atol=1e-10)
        print(f"{dataset} / {mode}: shape={Xc.shape}; variance(41°)={current:.6f}; "
              f"best 1° grid angle={best}°")
    print("NumPy reference checks passed for all 4 dataset/mode combinations.")


class VarianceExplorer:
    """Supplied plotting/controller code; students edit decomposition.py only."""

    def __init__(self, cache):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.widgets import RadioButtons, Slider

        self.cache = cache
        self.dataset, self.mode, self.angle = "Synthetic", "Center only", 41
        self.fig = plt.figure(figsize=(12.8, 8), facecolor="#fffdf7", layout=None)
        self.fig.canvas.manager.set_window_title("Week 6 — Variance directions")
        self.fig.text(.06, .947, "Which direction keeps the variation?",
                      fontsize=23, weight="bold", color="#80512f")
        self.subtitle = self.fig.text(.06, .904, "", fontsize=11, color="#556663")
        self.left = self.fig.add_axes([.075, .37, .40, .47])
        self.right = self.fig.add_axes([.57, .51, .38, .32])
        self.readout = self.fig.text(.57, .315, "", fontsize=11, family="monospace", linespacing=1.6)
        self.fig.text(.075, .263, "Filled = observation   Hollow = projection\n"
                      "Dashed = residual (not the variance)", fontsize=10, color="#556663")
        self.fig.text(.57, .263, "Curve = variance of 1D scores, ddof=1\n"
                      "Maximum shown is sampled on a 1° grid.", fontsize=10, color="#556663")
        self.slider = Slider(self.fig.add_axes([.17, .19, .73, .035]), "Angle (°)",
                             0, 180, valinit=41, valstep=1, valfmt="%d°", color=TEAL)
        self.dataset_buttons = RadioButtons(self.fig.add_axes([.075, .04, .20, .105]),
                                            list(dict.fromkeys(k[0] for k in cache)), active=0)
        self.mode_buttons = RadioButtons(self.fig.add_axes([.31, .04, .22, .105]), MODES, active=0)
        self.fig.text(.58, .09, "Student functions → covariance + variance curve\n"
                      "PCA selects directions of large training variance.", fontsize=11)
        for controls in (self.dataset_buttons.ax, self.mode_buttons.ax):
            controls.set_facecolor("#f1f0e8")
            for spine in controls.spines.values():
                spine.set_color("#cbd1c9")
        self._line_collection = LineCollection
        self.slider.on_changed(self.set_angle)
        self.dataset_buttons.on_clicked(self.select_dataset)
        self.mode_buttons.on_clicked(self.select_mode)
        self.refresh()

    @property
    def entry(self):
        return self.cache[self.dataset, self.mode]

    def select_dataset(self, label):
        self.dataset = label
        self.refresh()

    def select_mode(self, label):
        self.mode = label
        self.refresh()

    def refresh(self):
        """Reset axes only when data/mode changes; dragging keeps limits fixed."""
        entry = self.entry
        Xc, labels = entry["Xc"], entry["labels"]
        self.left.clear()
        self.right.clear()
        if labels is None:
            self.left.scatter(*Xc.T, s=55, color=TEAL, zorder=4)
        else:
            for label, name, color in [(0, "Benign", BLUE), (1, "Malignant", RUST)]:
                selected = Xc[labels == label]
                self.left.scatter(*selected.T, s=14, alpha=.65, label=name, color=color, zorder=4)
            self.left.legend(loc="upper left", fontsize=9)
        self.bound = max(float(np.max(np.linalg.norm(Xc, axis=1))) * 1.12, 1.0)
        units = "centered original units" if self.mode == "Center only" else "standardized units"
        self.left.set(xlim=(-self.bound, self.bound), ylim=(-self.bound, self.bound),
                      xlabel=f'{entry["names"][0]} ({units})',
                      ylabel=f'{entry["names"][1]} ({units})')
        self.left.set_aspect("equal", adjustable="box")
        self.left.axhline(0, color="#9aa49c", linewidth=.7)
        self.left.axvline(0, color="#9aa49c", linewidth=.7)
        self.direction_line, = self.left.plot([], [], color=BLUE, linewidth=1.5)
        self.projections = self.left.scatter([], [], s=28, facecolors="none", edgecolors=BLUE, zorder=3)
        self.residuals = self._line_collection([], colors="#7f8880", linewidths=.7,
                                             linestyles="dashed", alpha=.4)
        self.left.add_collection(self.residuals)
        self.right.plot(ANGLES, entry["curve"], color=TEAL, linewidth=2)
        top = max(float(entry["curve"].max()) * 1.13, .1)
        self.right.set(xlim=(0, 180), ylim=(0, top), xticks=[0, 45, 90, 135, 180],
                       xlabel="Direction angle (°)", ylabel="Sample variance of scores")
        self.marker, = self.right.plot([], [], "o", color=RUST, markersize=9)
        for ax in (self.left, self.right):
            ax.set_facecolor("#fffdf7")
            ax.grid(alpha=.25)
            ax.set_axisbelow(True)
        label_note = "; diagnosis labels only color points" if labels is not None else ""
        self.subtitle.set_text(f"{self.dataset} / {self.mode}   ·   {len(Xc)} training samples{label_note}")
        self.set_angle(self.angle)

    def set_angle(self, angle):
        self.angle = int(angle)
        entry = self.entry
        Xc, C = entry["Xc"], entry["C"]
        v = unit_direction(self.angle)
        self.current_variance = variance_along_direction(Xc, v)  # Student function, every update.
        scores = Xc @ v
        projected = scores[:, None] * v  # Supplied projection drawing, not a W6 exercise.
        self.projections.set_offsets(projected)
        self.residuals.set_segments(np.stack([Xc, projected], axis=1))
        endpoints = np.array([-self.bound, self.bound])[:, None] * v
        self.direction_line.set_data(*endpoints.T)
        self.marker.set_data([self.angle], [self.current_variance])
        best = int(np.argmax(entry["curve"]))
        self.readout.set_text(
            f"v = [{v[0]: .3f}, {v[1]: .3f}]    angle = {self.angle}°\n"
            f"C = [[{C[0,0]:7.3f}, {C[0,1]:7.3f}],\n"
            f"     [{C[1,0]:7.3f}, {C[1,1]:7.3f}]]\n"
            f"Variance = {self.current_variance:.6f}\n"
            f"1° grid max: {best}° → {entry['curve'][best]:.6f}")
        self.fig.canvas.draw_idle()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="validate and save four static figures")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache = prepare(load_datasets())
    validate(cache)
    explorer = VarianceExplorer(cache)
    if args.headless:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for dataset_index, dataset in enumerate(("Synthetic", "WDBC")):
            explorer.dataset_buttons.set_active(dataset_index)
            for mode_index, mode in enumerate(MODES):
                explorer.mode_buttons.set_active(mode_index)
                explorer.slider.set_val(41)
                suffix = "center" if mode == "Center only" else "standardize"
                path = args.output_dir / f"week_06_{dataset.lower()}_{suffix}.png"
                explorer.fig.savefig(path, dpi=130, facecolor=explorer.fig.get_facecolor())
                print(f"Saved plot: {path.name}")
        plt.close(explorer.fig)
    else:
        if explorer.fig.canvas.required_interactive_framework is None:
            plt.close(explorer.fig)
            parser.error("No native GUI backend. Run in a local desktop terminal with GUI-enabled "
                         "Python/Matplotlib (unset MPLBACKEND if it forces Agg). "
                         "Use --headless for checks/PNGs; static images do not provide interaction.")
        plt.show()


if __name__ == "__main__":
    main()
