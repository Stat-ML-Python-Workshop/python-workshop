"""Build probability distributions from observations.

Sample inputs and outputs:
    ["T", "T", "B", "M"] -> {"T": 0.5, "B": 0.25, "M": 0.25}
    ["B", "B"] -> {"B": 1.0}
    [0, 0, 1] -> {0: 0.6666666666666666, 1: 0.3333333333333333}
"""
def class_proportions(labels):
    if not labels:
        raise ValueError("labels cannot be empty")
    counts = {}
    # BEGIN STUDENT: count observations and normalize the counts
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    proportions = {}
    for label, count in counts.items():
        proportions[label] = count/sum(counts.values())
    return proportions
    # END STUDENT
