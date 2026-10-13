import numpy as np


def _week05_todo(hint):
    """Replace the call with a NumPy expression."""
    raise NotImplementedError("TODO: " + hint)

def _matrix(X):
    array = np.asarray(X, dtype=float)
    if array.ndim != 2 or 0 in array.shape or not np.all(np.isfinite(array)):
        raise ValueError("X must be a nonempty finite (n, d) matrix")
    return array

class Standardizer:
    """fit: (n, d) -> self; transform: (m, d) -> ndarray (m, d).

    mean_, scale_: (d,); population std (ddof=0), zero scale -> 1.
    """

    def fit(self, X):
        """Learn per-feature parameters from training rows only."""
        array = _matrix(X)
        # BEGIN STUDENT: training mean and population scale
        self.mean_ = np.mean(array, axis=0)
        self.scale_ = np.std(array, axis=0, ddof=0)
        self.scale_ = np.where(self.scale_ > 0, self.scale_, 1)
        # END STUDENT
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X):
        """Apply saved parameters; preserve X and learned state."""
        if not hasattr(self, "n_features_in_"):
            raise ValueError("fit must be called before transform")
        array = _matrix(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("feature dimensions differ")
        # BEGIN STUDENT: broadcast saved training parameters
        return (X - self.mean_) / self.scale_
        # END STUDENT
