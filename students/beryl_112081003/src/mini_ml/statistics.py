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


"""Week 3: variance and standard deviation with explicit iteration."""
from math import sqrt

def variance(values, ddof=1):
    if ddof < 0 or len(values) <= ddof:
        raise ValueError("require 0 <= ddof < n")
    center = mean(values)
    total = 0.0
    # BEGIN STUDENT: accumulate squared deviations and normalize
    for value in values:
        total += (value - center) ** 2
    return total / (len(values) - ddof)
    # END STUDENT

def std(values, ddof=1):
    # BEGIN STUDENT: standard deviation is the square root of variance
    return sqrt(variance(values, ddof))
    # END STUDENT
