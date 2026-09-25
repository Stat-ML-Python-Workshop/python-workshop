import pytest
from mini_ml.statistics import std, variance


@pytest.mark.parametrize("values, ddof, expected", [
    ([2, 4, 6], 1, 4.0),
    ([2, 4, 6], 0, 8 / 3),
    ([7, 7, 7], 1, 0.0),
    ([42], 0, 0.0),
])
def test_variance(values, ddof, expected):
    assert variance(values, ddof=ddof) == pytest.approx(expected)


def test_standard_deviation():
    assert std([2, 4, 6]) == pytest.approx(2.0)


@pytest.mark.parametrize("values, ddof", [([], 0), ([2], 1), ([2, 4], -1)])
def test_invalid_variance(values, ddof):
    with pytest.raises(ValueError):
        variance(values, ddof=ddof)
