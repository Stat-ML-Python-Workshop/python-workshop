from math import sqrt

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
