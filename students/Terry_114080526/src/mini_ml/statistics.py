"""Week 2: explicit iteration. Do not call built-in sum.

Sample inputs and outputs:
    sum_values([1, 2, 3]) -> 6.0
    sum_values([]) -> 0.0
    mean([2, 4, 6]) -> 4.0
"""
def sum_values(values):
    total = 0.0
    for value in values:
        total += value
    return total

def mean(values):
    if not values:
        raise ValueError("mean requires at least one value")
    return sum_values(values) / len(values)
