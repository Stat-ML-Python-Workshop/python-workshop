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


def _week08_todo(hint):
    raise NotImplementedError("TODO: " + hint)

# Week 8: append after the completed Week 6 and Week 7 functions.

def _pca_randomized(X_centered, k, *, random_state):
    """Teacher SVD helper: candidate sample variances (r,), axes (d,r).

    r is the sketch width, with r >= k. PCA.fit applies the common ordering
    and sign convention. Covariance and eigh remain visible student work.
    Numerical modules use NumPy only; sklearn belongs in reference checks.
    """
    n, d = X_centered.shape
    if not np.any(X_centered):
        return np.zeros(k, dtype=np.float64), np.eye(d, k, dtype=np.float64)
    width = min(k + 10, n, d)
    if width == min(n, d):
        _, singular_values, right_rows = np.linalg.svd(X_centered, full_matrices=False)
    else:
        rng = np.random.default_rng(random_state)
        sketch = rng.standard_normal((d, width))
        Q, _ = np.linalg.qr(X_centered @ sketch, mode="reduced")
        # Normalize both products: seven unnormalized powers lose precision.
        for _ in range(7):
            Q_right, _ = np.linalg.qr(X_centered.T @ Q, mode="reduced")
            Q, _ = np.linalg.qr(X_centered @ Q_right, mode="reduced")
        _, singular_values, right_rows = np.linalg.svd(Q.T @ X_centered, full_matrices=False)
    return singular_values ** 2 / (n - 1), right_rows.T.copy()


class PCA:
    """Training-only PCA state, without whitening.

    n_components: integer k >= 1, with k <= min(n - 1, d) checked by fit.
    solver: "auto", "eigh", or "randomized". Auto uses covariance eigh for
    d <= 30 and a seeded NumPy SVD helper otherwise. random_state is a
    nonnegative integer; no global random state is changed.
    """

    def __init__(self, n_components, solver="auto", random_state=42):
        if isinstance(n_components, (bool, np.bool_)) or not isinstance(n_components, (int, np.integer)) or n_components < 1:
            raise ValueError("n_components must be a positive integer, not bool")
        if not isinstance(solver, str) or solver not in ("auto", "eigh", "randomized"):
            raise ValueError("solver must be auto, eigh, or randomized")
        if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)) or random_state < 0:
            raise ValueError("random_state must be a nonnegative integer, not bool")
        self.n_components = int(n_components)
        self.solver = solver
        self.random_state = int(random_state)
        self._fitted = False

    def fit(self, X):
        """Fit finite nonempty X (n,d), n >= 2; return self, preserving X.

        Compute new state locally so a failed refit preserves the previous fit.
        """
        array = _matrix(X)
        n, d = array.shape
        k = self.n_components
        if n < 2 or k > min(n - 1, d):
            raise ValueError("require n >= 2 and 1 <= n_components <= min(n - 1, d)")
        Xc, mean = center(array)  # STUDENT 1: center training data
        chosen = ("eigh" if d <= 30 else "randomized") if self.solver == "auto" else self.solver
        if chosen == "eigh":
            C = covariance_matrix(Xc)  # STUDENT 2: sample covariance of centered data
            candidate_variance, axes = np.linalg.eigh(C)  # STUDENT 3: eigenvalues and eigenvector columns
        else:
            candidate_variance, axes = _pca_randomized(Xc, k, random_state=self.random_state)
        # Supplied display convention; eigenvector signs are not mathematically unique.
        largest = np.argmax(np.abs(axes), axis=0)
        signs = np.where(axes[largest, np.arange(axes.shape[1])] < 0, -1.0, 1.0)
        axes = axes * signs
        order = np.argsort(candidate_variance)[::-1]  # STUDENT 4: descending variance order
        components = axes[:, order[:k]].T  # STUDENT 5: selected component rows
        variance = np.maximum(candidate_variance[order[:k]], 0)  # STUDENT 6: selected nonnegative variance
        total = np.sum(Xc ** 2) / (n - 1)  # STUDENT 7: total training variance
        ratio = variance / total if total > 0 else np.zeros_like(variance)  # STUDENT 8: explained variance ratio
        self.mean_ = mean.copy()
        self.components_ = components.copy()
        self.explained_variance_ = variance.copy()
        self.explained_variance_ratio_ = ratio.copy()
        self.n_features_in_ = d
        self.n_samples_ = n
        self._fitted = True
        return self

    def transform(self, X):
        """Finite nonempty X (m,d) -> float64 scores (m,k), using saved state."""
        if not self._fitted:
            raise ValueError("PCA must be fitted before transform")
        return project(X, self.mean_, self.components_)  # STUDENT 9: transform using saved state

    def inverse_transform(self, Z):
        """Finite nonempty Z (m,k) -> float64 reconstruction (m,d)."""
        if not self._fitted:
            raise ValueError("PCA must be fitted before inverse_transform")
        return reconstruct(Z, self.mean_, self.components_)  # STUDENT 10: inverse using saved state


