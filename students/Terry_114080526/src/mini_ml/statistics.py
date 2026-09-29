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
    n = len(values)
    for value in values:
        total += (value - center) ** 2
    if total == 0:
        return 0
    else:
        return total / (n - ddof)
    # END STUDENT

def std(values, ddof=1):
    # BEGIN STUDENT: standard deviation is the square root of variance
    if ddof < 0 or len(values) <= ddof:
        raise ValueError("Require 0 <= ddof <= n")
    center = mean(values)
    n = len(values)
    total = 0.0
    for value in values:
        total += (value - center) ** 2
    var = total / (n - ddof)
    return sqrt(var)
    # END STUDENT
