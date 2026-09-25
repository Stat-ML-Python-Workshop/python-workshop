def cross_entropy(p, q):
    """Return H(p, q) in bits; impossible q events give infinity."""
    _validate_distribution(p)
    _validate_distribution(q)
    if len(p) != len(q):
        raise ValueError("distributions must have equal length")
    # BEGIN STUDENT: compare true distribution p with model q
    total = 0.0
    for actual, model in zip(p, q):
        raise NotImplementedError("TODO: handle zero probabilities and add cross-entropy")
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
        raise NotImplementedError("TODO: handle zero probabilities and add KL divergence")
    return total
    # END STUDENT
