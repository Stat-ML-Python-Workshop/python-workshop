"""Instructor demo: read patient features and calculate arbitrary linear scores."""
from pathlib import Path

import numpy as np
import pandas as pd

from mini_ml.linalg import matmul, shape

# A small, independently calculated product checks your implementation first.
m1 = [[1, 2, 3], [4, 5, 6]]
m2 = [[7, 8], [9, 10], [11, 12]]
assert matmul(m1, m2) == [[58, 64], [139, 154]]
assert np.allclose(matmul(m1, m2), np.asarray(m1) @ np.asarray(m2))

# This file runs from test_code/week_04.py inside your student package.
data_dir = Path(__file__).resolve().parents[1] / "data" / "week-04"
training = pd.read_csv(data_dir / "train.csv")
feature_names = ["mean_radius", "mean_texture"]
M = training[feature_names].values.tolist()

# Arbitrary teaching values, chosen by the instructor, not fitted to the data.
# The order matches feature_names: radius gets 0.4; texture gets 0.1.
T = [[0.4], [0.1]]
bias = -7.0
weighted_sums = matmul(M, T)
scores = [[row[0] + bias] for row in weighted_sums]

# NumPy is used only to check the result of your pure-Python matmul.
assert np.allclose(weighted_sums, np.asarray(M) @ np.asarray(T))
assert np.allclose([row[0] for row in weighted_sums[:3]], [8.265, 4.9624, 6.474])
assert np.allclose([row[0] for row in scores[:3]], [1.265, -2.0376, -0.526])

preview = training[feature_names].head(3).copy()
preview["weighted_sum"] = [row[0] for row in weighted_sums[:3]]
preview["linear_score"] = [row[0] for row in scores[:3]]
print(f"Features (in order): {feature_names}")
print(f"Arbitrary weights: {T}; bias: {bias}")
print(f"M shape: {shape(M)}; T shape: {shape(T)}; result shape: {shape(weighted_sums)}")
print("First 3 patients:")
print(preview.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print("Matrix multiplication checks passed.")
