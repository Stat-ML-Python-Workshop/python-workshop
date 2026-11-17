"""Teacher-supplied checks and checkpoint recording for Weeks 9 and 10.

No student exercises here. Computation uses CPU float64 tensors. NumPy is used
for label encoding and the public data/model interfaces, outside autograd.
"""
import numpy as np
import torch
from torch.nn import functional as F


class TrainingDivergedError(ValueError):
    """Nonfinite training arithmetic; retains completed epoch histories."""


def _matrix(values, name):
    try:
        array = torch.as_tensor(values, dtype=torch.float64, device='cpu')
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f'{name} must be a numeric matrix') from error
    if array.ndim != 2 or min(array.shape) == 0 or not torch.isfinite(array).all():
        raise ValueError(f'{name} must be a finite nonempty matrix')
    return array


def _labels(values, n):
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) != n or labels.dtype.kind not in 'iuUf':
        raise ValueError('Labels must be a vector of numbers or strings matching X')
    if labels.dtype.kind in 'iuf' and not np.isfinite(labels).all():
        raise ValueError('Labels must be finite')
    return labels


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def check_parameters(model):
    rate = model.learning_rate
    if (isinstance(rate, (bool, np.bool_))
            or not isinstance(rate, (int, float, np.integer, np.floating))
            or not np.isfinite(rate) or rate <= 0):
        raise ValueError('learning_rate must be positive and finite')
    _positive_integer(model.epochs, 'epochs')
    _positive_integer(model.batch_size, 'batch_size')
    seed = model.random_state
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError('random_state must be a nonnegative integer')


def check_scores(scores):
    # as_tensor preserves a tensor's computation graph; never detach here.
    scores = _matrix(scores, 'scores')
    if scores.shape[1] < 2:
        raise ValueError('Require at least two classes')
    return scores


def check_training_numbers(*arrays):
    if any(array is None or not torch.isfinite(array).all() for array in arrays):
        raise TrainingDivergedError('Nonfinite training values; try a smaller learning rate')


def prepare_data(X, y, validation_data):
    X = _matrix(X, 'X').detach()
    labels = _labels(y, len(X))
    classes, encoded = np.unique(labels, return_inverse=True)
    if len(classes) < 2:
        raise ValueError('Require at least two training classes')
    validation = None
    if validation_data is not None:
        if not isinstance(validation_data, (list, tuple)) or len(validation_data) != 2:
            raise ValueError('validation_data must be (X_validation, y_validation)')
        Xv = _matrix(validation_data[0], 'X_validation').detach()
        yv = _labels(validation_data[1], len(Xv))
        if Xv.shape[1] != X.shape[1] or not np.isin(yv, classes).all():
            raise ValueError('Validation requires matching features and known labels')
        validation = Xv, torch.as_tensor(np.searchsorted(classes, yv), dtype=torch.long, device='cpu')
    return X, torch.as_tensor(encoded, dtype=torch.long, device='cpu'), classes, validation


def initialize_model(model, X, classes):
    # New Parameter objects leave the previous fitted state intact on failure.
    previous = model.__dict__.copy()
    model.W = torch.nn.Parameter(torch.zeros((len(classes), X.shape[1]), dtype=torch.float64, device='cpu'))
    model.b = torch.nn.Parameter(torch.zeros(len(classes), dtype=torch.float64, device='cpu'))
    model.classes_ = classes.copy()
    model.n_features_in_ = X.shape[1]
    model.n_samples_ = len(X)
    model.loss_history_ = []
    model.validation_loss_history_ = []
    model.n_iter_ = 0
    return previous


def record_losses(model, X, encoded, validation):
    # Evaluate complete splits without a graph or any parameter update.
    with torch.no_grad():
        scores, _ = model.forward(X)
        train_loss = F.cross_entropy(scores, encoded)
        check_training_numbers(train_loss)
        validation_loss = None
        if validation is not None:
            Xv, yv = validation
            scores, _ = model.forward(Xv)
            validation_loss = F.cross_entropy(scores, yv)
            check_training_numbers(validation_loss)
    # Append both only when the entire checkpoint is finite.
    model.loss_history_.append(float(train_loss))
    if validation_loss is not None:
        model.validation_loss_history_.append(float(validation_loss))


def restore_model(model, previous, error):
    if isinstance(error, TrainingDivergedError):
        error.loss_history = model.loss_history_.copy()
        error.validation_loss_history = model.validation_loss_history_.copy()
    model.__dict__.clear()
    model.__dict__.update(previous)


def check_prediction_input(model, X):
    if not model._fitted:
        raise ValueError('Fit the classifier before prediction')
    X = _matrix(X, 'X')
    if X.shape[1] != model.n_features_in_:
        raise ValueError('Feature count differs from training')
    return X
