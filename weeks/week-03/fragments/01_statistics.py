"""Week 3: variance and standard deviation with explicit iteration."""
from math import sqrt

def variance(values, ddof=1):
    if ddof < 0 or len(values) <= ddof:
        raise ValueError("require 0 <= ddof < n")
    center = mean(values)
    total = 0.0
    # BEGIN STUDENT: accumulate squared deviations and normalize
    for value in values:
        raise NotImplementedError("TODO: add this squared deviation")
    raise NotImplementedError("TODO: normalize the accumulated deviations")
    # END STUDENT

def std(values, ddof=1):
    # BEGIN STUDENT: standard deviation is the square root of variance
    raise NotImplementedError("TODO: return the standard deviation")
    # END STUDENT
