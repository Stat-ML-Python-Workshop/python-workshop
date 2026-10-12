"""Instructor-designed scale experiments and fixed WDBC training plot."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mini_ml.linalg import euclidean_distance
from mini_ml.preprocessing import Standardizer

# Hand calculation: training parameters are reused for new samples.
x_train = [[1, 10], [3, 10]]
scaler = Standardizer().fit(x_train)
train_scaled = scaler.transform(x_train)
new_scaled = scaler.transform([[5, 10], [5, 12]])
np.testing.assert_allclose(scaler.mean_, [2, 10])
np.testing.assert_allclose(scaler.scale_, [1, 1])
np.testing.assert_allclose(train_scaled, [[-1, 0], [1, 0]])
np.testing.assert_allclose(new_scaled, [[3, 0], [3, 2]])
assert euclidean_distance([0, 0], [3, 4]) == 5.0
print(f"Hand-example means: {scaler.mean_.tolist()}; scales: {scaler.scale_.tolist()}")
print(f"Training transformed: {train_scaled.tolist()}")
print(f"New samples transformed: {new_scaled.tolist()}")

# Scale experiment: fit once, transform query and candidates together.
scale_training = [[0, 0], [2, 200]]
query = [1, 100]
candidates = [[3, 100], [1, 200]]
scale_model = Standardizer().fit(scale_training)
query_scaled = scale_model.transform([query])[0]
candidates_scaled = scale_model.transform(candidates)
raw_distances = [euclidean_distance(query, point) for point in candidates]
scaled_distances = [euclidean_distance(query_scaled, point) for point in candidates_scaled]
assert raw_distances == [2, 100]
assert scaled_distances == [2, 1]
print(f"Candidate distances (A, B): raw={raw_distances}; standardized={scaled_distances}")
print("Nearest candidate: raw=A; standardized=B")

# Same fixed 341 training patients as Week 4; no validation/test data.
data_dir = Path(__file__).resolve().parents[1] / "data" / "week-05"
training = pd.read_csv(data_dir / "train.csv")
feature_names = ["mean_radius", "mean_texture"]
bio_x = training[feature_names].to_numpy(dtype=float)
bio_y = training["malignant"].to_numpy()
bio_scaler = Standardizer().fit(bio_x)
bio_z = bio_scaler.transform(bio_x)
reference = StandardScaler().fit(bio_x)
np.testing.assert_allclose(bio_scaler.mean_, reference.mean_)
np.testing.assert_allclose(bio_scaler.scale_, reference.scale_)
np.testing.assert_allclose(bio_z, reference.transform(bio_x))
np.testing.assert_allclose(bio_scaler.transform(bio_x[:3]), reference.transform(bio_x[:3]))
np.testing.assert_allclose(bio_z.mean(axis=0), [0, 0], atol=1e-12)
np.testing.assert_allclose(bio_z.std(axis=0, ddof=0), [1, 1])
raw_distance = euclidean_distance(bio_x[0], bio_x[1])
scaled_distance = euclidean_distance(bio_z[0], bio_z[1])
assert np.isclose(raw_distance, np.linalg.norm(bio_x[0] - bio_x[1]))
assert np.isclose(scaled_distance, np.linalg.norm(bio_z[0] - bio_z[1]))
preview = training[feature_names].head(3).copy()
preview["z_radius"], preview["z_texture"] = bio_z[:3].T
print(f"Features (in order): {feature_names}")
print(f"Training shape: {bio_x.shape}; transformed shape: {bio_z.shape}")
print(f"Training means: {np.round(bio_scaler.mean_, 4).tolist()}")
print(f"Training scales (ddof=0): {np.round(bio_scaler.scale_, 4).tolist()}")
print("First 3 patients:")
print(preview.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print(f"First-pair distance: raw={raw_distance:.4f}; standardized={scaled_distance:.4f}")
print("StandardScaler reference checks passed.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, values, title, names in zip(axes, [bio_x, bio_z],
        ["Original training data", "Using training parameters"],
        [["Mean radius", "Mean texture"], ["Standardized radius", "Standardized texture"]]):
    for label, name, color in [(0, "Benign", "#2878b5"), (1, "Malignant", "#d54b40")]:
        mask = bio_y == label
        ax.scatter(values[mask, 0], values[mask, 1], label=name, color=color, s=15, alpha=0.7)
    ax.set(xlabel=names[0], ylabel=names[1], title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
fig.tight_layout()
figure_path = Path(__file__).resolve().with_name("week_05_scaling.png")
fig.savefig(figure_path, dpi=150)
print(f"Saved plot: {figure_path.name}")
plt.show()
plt.close(fig)
