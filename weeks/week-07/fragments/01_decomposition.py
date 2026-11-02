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
    return _week07_todo("projection scores")  # STUDENT: projection scores

def reconstruct(Z, mean, components):
    """Z: array_like (n,k); mean: (d,); components: orthonormal (k,d).
    Returns reconstruction: float64 ndarray (n,d), with the training mean restored.
    """
    scores = _matrix(Z)
    components = _orthonormal_components(components)
    mean = _vector(mean, components.shape[1])
    if scores.shape[1] != components.shape[0]:
        raise ValueError("Z width must match the number of component rows")
    return _week07_todo("reconstruction")  # STUDENT: reconstruction

def reconstruction_error(X, X_hat):
    """X, X_hat: finite nonempty array_like matrices with the same (n,d) shape.
    Returns mean squared entry error: Python float, averaged over all n*d entries.
    """
    array = _matrix(X)
    reconstructed = _matrix(X_hat)
    if array.shape != reconstructed.shape:
        raise ValueError("X and X_hat must have the same shape")
    return _week07_todo("mean squared error")  # STUDENT: mean squared error
