"""Week 6: centered data, sample covariance, and directional variance."""
import numpy as np

def _matrix(X, centered=False):
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape or not np.all(np.isfinite(array)):
        raise ValueError("X must be a nonempty finite (n,d) matrix")
    if centered and (len(array) < 2 or not np.allclose(array.mean(axis=0), 0, atol=1e-10, rtol=0)):
        raise ValueError("require n >= 2 and centered columns (atol=1e-10)")
    return array

def _vector(values, d):
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (d,) or not np.all(np.isfinite(vector)):
        raise ValueError("require a finite (d,) vector matching the features")
    return vector.copy()

def center(X, mean=None):
    """X: array_like (n,d); mean: array_like (d,) | None.
    Returns (X_centered, mean): float64 ndarrays, (n,d) and (d,).
    """
    array = _matrix(X)
    if mean is None:
        mean = np.mean(array, axis=0)  # STUDENT: column means
    else:
        mean = _vector(mean, array.shape[1])
    centered = array - mean  # STUDENT: broadcast subtraction
    return centered, mean

def covariance_matrix(X_centered):
    """X_centered: centered array_like (n,d), n >= 2.
    Returns sample covariance: float64 ndarray (d,d), ddof=1.
    """
    array = _matrix(X_centered, centered=True)
    n = array.shape[0]
    return array.T @ array / (n - 1)  # STUDENT: sample covariance

def variance_along_direction(X_centered, direction):
    """X_centered: centered array_like (n,d), n >= 2; direction: nonzero (d,) array_like.
    Returns variance along the unit direction: Python float (ddof=1).
    """
    C = covariance_matrix(X_centered)
    v = _vector(direction, C.shape[0])
    magnitude = np.max(np.abs(v))
    if magnitude == 0:
        raise ValueError("direction must be nonzero")
    v = v / magnitude  # Rescale first to avoid overflow/underflow in the norm.
    v = v / np.linalg.norm(v)
    return float(v @ C @ v)  # STUDENT: directional variance
