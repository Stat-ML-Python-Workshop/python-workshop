"""Build probability distributions from observations."""
def class_proportions(labels):
    if not labels:
        raise ValueError("labels cannot be empty")
    counts = {}
    # BEGIN STUDENT: count observations and normalize the counts
    for label in labels:
        
    proportions = {}
    for label, count in counts.items():
        raise NotImplementedError("TODO: store this label's proportion")
    return proportions
    # END STUDENT
