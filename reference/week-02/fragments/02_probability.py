"""Build probability distributions from observations."""
def class_proportions(labels):
    if not labels:
        raise ValueError("labels cannot be empty")
    counts = {}
    # BEGIN STUDENT: count observations and normalize the counts
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    proportions = {}
    for label, count in counts.items():
        proportions[label] = count / len(labels)
    return proportions
    # END STUDENT
