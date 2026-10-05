from math import exp, isfinite

def stable_softmax(logits):
    """Convert finite scores to probabilities without numerical overflow."""
    if not logits or any(not isfinite(value) for value in logits):
        raise ValueError("logits must be nonempty and finite")
    # BEGIN STUDENT: shift, exponentiate, and normalize
    raise NotImplementedError("Complete this week’s core exercise")
    # END STUDENT
