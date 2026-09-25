from mini_ml.information import cross_entropy, kl_divergence, shannon_entropy
from mini_ml.statistics import std, variance
import numpy as np


values = [2, 4, 6]
p = [0.75, 0.25]
q = [0.5, 0.5]

assert np.isclose(variance(values, ddof=0), np.var(values, ddof=0))
assert np.isclose(variance(values), np.var(values, ddof=1))
assert np.isclose(std(values), np.std(values, ddof=1))

reference_cross_entropy = -np.sum(np.asarray(p) * np.log2(q))
reference_kl = np.sum(np.asarray(p) * np.log2(np.asarray(p) / np.asarray(q)))

assert np.isclose(cross_entropy(p, q), reference_cross_entropy)
assert np.isclose(kl_divergence(p, q), reference_kl)
assert np.isclose(cross_entropy(p, p), shannon_entropy(p))
assert np.isclose(kl_divergence(p, p), 0.0)

print("NumPy reference checks passed.")
