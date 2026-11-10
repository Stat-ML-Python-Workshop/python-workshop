def _week09_todo(number):
    raise NotImplementedError("TODO: Week 9 expression " + number)

"""Week 9: fill five forward expressions; PyTorch supplies backprop and SGD.

Read in order: softmax -> forward -> fit -> predict_proba.
Checks and loss recording are supplied in _training_utils.py.
"""
import numpy as np
import torch
from torch.nn import functional as F
from ._training_utils import (
    TrainingDivergedError,
    check_parameters,
    check_prediction_input,
    check_scores,
    check_training_numbers,
    initialize_model,
    prepare_data,
    record_losses,
    restore_model,
)


def softmax(scores):
    """Class scores (B, c) -> a differentiable probability tensor (B, c)."""
    scores = check_scores(scores)  # Supplied check; keeps the computation graph.
    # Subtract each sample's largest score. dim=1 means the class direction.
    # keepdim=True preserves (B, 1), so subtraction broadcasts across classes.
    shifted = shifted = scores - scores.max(dim=1, keepdim=True).values  # STUDENT 1: shift each row
    # Apply exp to every shifted score; shape stays (B, c).
    exponentials = torch.exp(shifted)  # STUDENT 2: exponentiate the scores
    # Divide each row by its own sum (B, 1); every probability row sums to 1.
    return exponentials / exponentials.sum(dim=1, keepdim=True)  # STUDENT 3: normalize each row


class LogisticClassifier:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=128, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        check_parameters(self)
        self._fitted = False

    @property
    def coef_(self):
        """A NumPy copy of W, for existing plotting and model-export code."""
        if not hasattr(self, 'W'):
            raise AttributeError('Weights are initialized by fit')
        return self.W.detach().numpy().copy()

    @property
    def intercept_(self):
        """A NumPy copy of b; editing it does not update the model."""
        if not hasattr(self, 'b'):
            raise AttributeError('Bias is initialized by fit')
        return self.b.detach().numpy().copy()

    def forward(self, X_batch):
        """Features (B, k) -> scores and probabilities, both (B, c)."""
        # X_batch: (B, k), W: (c, k), b: (c,).
        # Transpose W, multiply, then broadcast the same bias across samples.
        scores = X_batch @ self.W.T + self.b
        check_training_numbers(scores)  # Supplied divergence check.
        # Apply your softmax. Keep tensors here: NumPy/detach would break backprop.
        probabilities = softmax(scores)  # STUDENT 5: convert scores to probabilities
        check_training_numbers(probabilities)
        return scores, probabilities

    def _make_optimizer(self):
        """Teacher-supplied plain SGD; no momentum or weight decay."""
        return torch.optim.SGD([self.W, self.b], lr=self.learning_rate,
                               momentum=0, weight_decay=0)

    def _update_batch(self, X_batch, labels_batch, optimizer):
        """Teacher supplies gradients and the parameter update this week."""
        optimizer.zero_grad()  # Clear gradients from the previous mini-batch.
        scores, probabilities = self.forward(X_batch)
        # Cross-entropy takes raw scores and class indices, not probabilities.
        loss = F.cross_entropy(scores, labels_batch)
        check_training_numbers(loss)
        loss.backward()        # Autograd calculates W.grad and b.grad.
        check_training_numbers(self.W.grad, self.b.grad)
        optimizer.step()       # SGD uses those gradients to update W and b.

    def fit(self, X, y, validation_data=None):
        """Reset to zero parameters; train for the requested number of epochs."""
        check_parameters(self)
        X, encoded_labels, classes, validation = prepare_data(X, y, validation_data)
        previous_state = initialize_model(self, X, classes)
        rng = np.random.default_rng(self.random_state)

        try:
            optimizer = self._make_optimizer()
            record_losses(self, X, encoded_labels, validation)  # Epoch 0.
            for epoch in range(self.epochs):
                order = rng.permutation(len(X))  # Shuffle sample indices without repetition.
                for start in range(0, len(X), self.batch_size):
                    indices = order[start:start + self.batch_size]
                    X_batch = X[indices]                      # Features: (B, k).
                    labels_batch = encoded_labels[indices]    # Class indices: (B,).
                    self._update_batch(X_batch, labels_batch, optimizer)
                    check_training_numbers(self.W, self.b)

                # Evaluate both splits with the same completed-epoch parameters.
                record_losses(self, X, encoded_labels, validation)
                self.n_iter_ = epoch + 1
        except Exception as error:
            restore_model(self, previous_state, error)
            raise

        self._fitted = True
        return self

    def predict_proba(self, X):
        """Supplied: new features (m, k) -> NumPy probabilities (m, c)."""
        X = check_prediction_input(self, X)
        with torch.no_grad():
            scores, probabilities = self.forward(X)
        return probabilities.numpy().copy()

    def predict(self, X):
        """Return original labels using the probability columns in classes_ order."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
