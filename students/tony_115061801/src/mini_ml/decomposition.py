"""Week 6: centered data, sample covariance, and directional variance."""
import numpy as np


def _week06_todo(hint):
    raise NotImplementedError("TODO: " + hint)

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
    return (array.T @ array)/(n-1) # STUDENT: sample covariance

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
    return float(v.T @ C @ v)  # STUDENT: directional variance


# Week 7: append below your completed Week 6 functions.


def _week07_todo(hint):
    raise NotImplementedError("TODO: " + hint)

def _orthonormal_components(components):
    basis = _matrix(components)
    k, d = basis.shape
    if k > d:
        raise ValueError("components require 1 <= k <= d")
    if not np.allclose(basis @ basis.T, np.eye(k), atol=1e-8, rtol=0):
        raise ValueError("component rows must be orthonormal (atol=1e-8)")
    return basis

def project(X, mean, components):
    """X: array_like (n,d); mean: (d,); components: orthonormal (k,d).
    Returns scores: float64 ndarray (n,k), using the supplied training mean.
    """
    array = _matrix(X)
    mean = _vector(mean, array.shape[1])
    components = _orthonormal_components(components)
    if components.shape[1] != array.shape[1]:
        raise ValueError("X and components must have matching feature widths")
    return (array - mean) @ components.T

def reconstruct(Z, mean, components):
    """Z: array_like (n,k); mean: (d,); components: orthonormal (k,d).
    Returns reconstruction: float64 ndarray (n,d), with the training mean restored.
    """
    scores = _matrix(Z)
    components = _orthonormal_components(components)
    mean = _vector(mean, components.shape[1])
    if scores.shape[1] != components.shape[0]:
        raise ValueError("Z width must match the number of component rows")
    return (scores @ components) + mean

def reconstruction_error(X, X_hat):
    """X, X_hat: finite nonempty array_like matrices with the same (n,d) shape.
    Returns mean squared entry error: Python float, averaged over all n*d entries.
    """
    array = _matrix(X)
    reconstructed = _matrix(X_hat)
    if array.shape != reconstructed.shape:
        raise ValueError("X and X_hat must have the same shape")
    return float(np.mean((array-reconstructed)**2))

