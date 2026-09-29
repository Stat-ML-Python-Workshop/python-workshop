"""Information measures in bits for finite probability distributions.

Sample inputs and outputs:
    [0.5, 0.5] -> 1.0
    [0.75, 0.25] -> 0.8112781244591328
    [1.0, 0.0] -> 0.0
"""
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
    for p in probabilities:
        if p>0:
            entropy -= p*log2(p)
    return entropy
    # END STUDENT
