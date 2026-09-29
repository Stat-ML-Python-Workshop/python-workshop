"""Week 2: explicit iteration. Do not call built-in sum."""
def sum_values(values):
    total = 0.0
    for value in values:
        total += value
    return total

def mean(values):
    if not values:
        raise ValueError("mean requires at least one value")
    return sum_values(values) / len(values)
