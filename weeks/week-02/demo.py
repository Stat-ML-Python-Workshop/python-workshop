"""Reference checks only: implement the two core functions with Python loops."""
import numpy as np
from mini_ml.probability import class_proportions
from mini_ml.information import shannon_entropy

samples = {
    "skewed": ["T cell", "T cell", "T cell", "B cell"],
    "balanced": ["T cell", "T cell", "B cell", "B cell"],
}
entropies = {}
for name, labels in samples.items():
    proportions = class_proportions(labels)
    classes, counts = np.unique(labels, return_counts=True)
    reference_p = counts / len(labels)
    np.testing.assert_allclose([proportions[label] for label in classes], reference_p)
    positive = reference_p[reference_p > 0]
    reference_h = -np.sum(positive * np.log2(positive))
    entropy = shannon_entropy(list(proportions.values()))
    np.testing.assert_allclose(entropy, reference_h)
    entropies[name] = entropy
    print(f"{name}: {proportions}; entropy = {entropy:.6f} bits")
assert entropies["balanced"] > entropies["skewed"]
print("NumPy reference checks passed.")
