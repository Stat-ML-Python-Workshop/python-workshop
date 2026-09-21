"""Information measures in bits for finite probability distributions."""
from math import isclose, log2

def _validate_distribution(probabilities):
    if not probabilities or any(value < 0 for value in probabilities):
        raise ValueError("probabilities must be nonempty and nonnegative")
    if not isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one")

def shannon_entropy(probabilities):
    """Return H(p) in bits, using the convention 0 log2(0) = 0."""
    _validate_distribution(probabilities)
    # BEGIN STUDENT: expected self-information
    entropy = 0.0
    for value in probabilities:
        # TODO: test whether this probability is positive.
        # Inside that condition, accumulate its entropy contribution.
        raise NotImplementedError("TODO: positive check and entropy update")
    return entropy
    # END STUDENT
