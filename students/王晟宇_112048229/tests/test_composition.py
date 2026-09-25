import pytest
from mini_ml.probability import class_proportions
from mini_ml.information import shannon_entropy


@pytest.mark.parametrize("labels, expected", [
    (["T", "T", "B", "M"], {"T": 0.5, "B": 0.25, "M": 0.25}),
    (["B", "B"], {"B": 1.0}),
    ([0, 0, 1], {0: 2 / 3, 1: 1 / 3}),
    # STUDENT TODO: add one more (labels, expected) example here.
    (["A", "A", "A", "B"], {"A": 0.75, "B": 0.25}),
])
def test_proportions(labels, expected):
    result = class_proportions(labels)
    assert result == pytest.approx(expected)
    assert sum(result.values()) == pytest.approx(1.0)


def test_empty_labels():
    with pytest.raises(ValueError):
        class_proportions([])


@pytest.mark.parametrize("probabilities, expected", [
    ([0.5, 0.5], 1.0),
    ([1.0], 0.0),
    ([1.0, 0.0], 0.0),
    ([0.5, 0.25, 0.25], 1.5),
    # STUDENT TODO: add one more (probabilities, expected) example here.
    ([0.35, 0.25, 0.25, 0.15], 1.9406),
])
def test_entropy(probabilities, expected):
    assert shannon_entropy(probabilities) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("probabilities", [[], [-0.1, 1.1], [0.2, 0.2]])
def test_invalid_distribution(probabilities):
    with pytest.raises(ValueError):
        shannon_entropy(probabilities)