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
    for value in probabilities:
        if value > 0:
            entropy -= value * log2(value)
    return entropy
    # END STUDENT

def cross_entropy(p, q):
    """Return H(p, q) in bits; impossible q events give infinity."""
    _validate_distribution(p)
    _validate_distribution(q)
    if len(p) != len(q):
        raise ValueError("distributions must have equal length")
    # BEGIN STUDENT: compare true distribution p with model q
    total = 0.0
    for actual, model in zip(p, q):
        if actual == 0:
            continue
        if model == 0:
            return float("inf")
        total -= actual * log2(model)
    return total
    # END STUDENT

def kl_divergence(p, q):
    """Return D_KL(p || q) in bits; direction is intentional."""
    _validate_distribution(p)
    _validate_distribution(q)
    if len(p) != len(q):
        raise ValueError("distributions must have equal length")
    # BEGIN STUDENT: relative entropy from p to q
    total = 0.0
    for actual, model in zip(p, q):
        if actual == 0:
            continue
        if model == 0:
            return float("inf")
        total += actual * log2(actual / model)
    return total
    # END STUDENT
