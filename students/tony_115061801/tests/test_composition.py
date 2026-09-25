import pytest
from mini_ml.probability import class_proportions
from mini_ml.information import cross_entropy, kl_divergence, shannon_entropy

@pytest.mark.parametrize("labels, expected", [
    (["T", "T", "B", "M"], {"T": 0.5, "B": 0.25, "M": 0.25}),
    (["B", "B"], {"B": 1.0}),
    ([0, 0, 1], {0: 2 / 3, 1: 1 / 3}),
    # STUDENT TODO: add one more (labels, expected) example here.
    (["A", "A", "B", "A"], {"A" : 3/4, "B" : 1/4}),
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
    ([0.2, 0.8], 0.72192),
    # STUDENT TODO: add one more (probabilities, expected) example here.
])
def test_entropy(probabilities, expected):
    assert shannon_entropy(probabilities) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("probabilities", [[], [-0.1, 1.1], [0.2, 0.2]])
def test_invalid_distribution(probabilities):
    with pytest.raises(ValueError):
        shannon_entropy(probabilities)

@pytest.mark.parametrize("p, q, expected", [
    ([0.75, 0.25], [0.25, 0.75], 1.603759374819711),
    ([0.5, 0.5], [0.25, 0.75], 1.207518749639422),
    ([1.0, 0.0], [1.0, 0.0], 0.0),
    ([0.8, 0.2], [0.2, 0.8], 1.9219280948873623),
])
def test_cross_entropy(p, q, expected):
    assert cross_entropy(p, q) == pytest.approx(expected)


@pytest.mark.parametrize("p, q, expected", [
    ([0.75, 0.25], [0.25, 0.75], 0.792481250360578),
    ([0.2, 0.8], [0.2, 0.8], 0.0),
    ([1.0, 0.0], [0.25, 0.75], 2.0),
    ([0.25, 0.75], [0.5, 0.5], 0.18872187554086717),
])
def test_kl_divergence(p, q, expected):
    assert kl_divergence(p, q) == pytest.approx(expected, abs=1e-12)


def test_information_boundaries():
    assert cross_entropy([0, 1], [0, 1]) == pytest.approx(0.0)
    assert cross_entropy([1, 0], [0, 1]) == float("inf")
    assert kl_divergence([1, 0], [0, 1]) == float("inf")
    with pytest.raises(ValueError):
        kl_divergence([0.5, 0.5], [1.0])
