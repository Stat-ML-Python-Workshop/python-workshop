import numpy as np

def euclidean_distance(x, y):
    """x, y: (d,), d >= 1, matching feature order/scale -> float.

    Example: [0, 0], [3, 4] -> 5.0.
    """
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 1 or y_array.ndim != 1 or x_array.size == 0 or x_array.shape != y_array.shape:
        raise ValueError("nonempty equal-length vectors required")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("finite vector values required")
    # BEGIN STUDENT: Euclidean distance
    return float(np.sqrt(np.sum((x_array - y_array) ** 2)))
    # END STUDENT
