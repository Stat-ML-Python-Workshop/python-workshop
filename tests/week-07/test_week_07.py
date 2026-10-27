"""Independent W7 acceptance through the isolated student worker."""
import numpy as np
import pytest


def test_projection_known_values(call):
    assert call("decomposition", "project", [[2, 3], [-2, -3]], [0, 0], [[1, 0]]) == [[2], [-2]]
    assert call("decomposition", "project", [[11, 22, 35], [9, 16, 31]], [10, 20, 30], [[1, 0, 0], [0, 0, 1]]) == [[1, 5], [-1, 1]]
    assert call("decomposition", "project", [["5"]], ["2"], [["-1"]]) == [[-3]]


def test_reconstruction_known_values(call):
    assert call("decomposition", "reconstruct", [[2], [-2]], [0, 0], [[1, 0]]) == [[2, 0], [-2, 0]]
    assert call("decomposition", "reconstruct", [[1, 5], [-1, 1]], [10, 20, 30], [[1, 0, 0], [0, 0, 1]]) == [[11, 20, 35], [9, 20, 31]]


def test_error_averages_entries_without_clipping(call):
    assert call("decomposition", "reconstruction_error", [[2, 3], [-2, -3]], [[2, 0], [-2, 0]]) == pytest.approx(4.5)
    assert call("decomposition", "reconstruction_error", [[1, 2, 3], [4, 5, 6]], [[0, 0, 0], [0, 0, 0]]) == pytest.approx(91 / 6)
    assert call("decomposition", "reconstruction_error", [[0, 1]], [[-0.2, 1.1]]) == pytest.approx(0.025)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotated_basis_independent_reference(call, k):
    X = np.array([[3., 4., -2.], [1., 7., 5.], [-4., 2., 0.], [8., -1., 3.]])
    mean = np.array([2., -3., 1.])
    basis, _ = np.linalg.qr(np.array([[1., 2., 3.], [3., -1., 2.], [2., 3., -1.]]))
    components = basis[:k]
    expected_scores = np.array([[np.dot(row - mean, direction) for direction in components] for row in X])
    scores = call("decomposition", "project", X.tolist(), mean.tolist(), components.tolist())
    np.testing.assert_allclose(scores, expected_scores, atol=1e-12)
    expected = np.array([mean + sum((z * v for z, v in zip(row, components)), start=np.zeros(3)) for row in expected_scores])
    actual = call("decomposition", "reconstruct", expected_scores.tolist(), mean.tolist(), components.tolist())
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    expected_mse = sum((x - y) ** 2 for x, y in zip(X.flat, expected.flat)) / X.size
    assert call("decomposition", "reconstruction_error", X.tolist(), expected.tolist()) == pytest.approx(expected_mse, abs=1e-12)
    if k == 3:
        np.testing.assert_allclose(actual, X, atol=1e-12)


@pytest.mark.parametrize("function,args", [
    ("project", [[], [0], [[1]]]), ("project", [[1, 2], [0, 0], [[1, 0]]]),
    ("project", [[[1, 2], [3]], [0, 0], [[1, 0]]]),
    ("project", [[[float('nan')]], [0], [[1]]]),
    ("project", [[[float('inf')]], [0], [[1]]]),
    ("project", [[[1, 2]], [0], [[1, 0]]]),
    ("project", [[[1, 2]], [[0, 0]], [[1, 0]]]),
    ("project", [[[1, 2]], [0, float('inf')], [[1, 0]]]),
    ("project", [[[1, 2]], [0, 0], [[1]]]),
    ("project", [[[1, 2]], [0, 0], []]),
    ("project", [[[1, 2]], [0, 0], [[1, 0], [0, 1], [1, 0]]]),
    ("project", [[[1, 2]], [0, 0], [[2, 0]]]),
    ("project", [[[1, 2]], [0, 0], [[1, 0], [1, 0]]]),
    ("project", [[[1, 2]], [0, 0], [[float('nan'), 0]]]),
    ("project", [[[1, 2]], [0, 0], [[1 + 2e-8, 0]]]),
    ("reconstruct", [[], [0, 0], [[1, 0]]]),
    ("reconstruct", [[[1, 2]], [0, 0], [[1, 0]]]),
    ("reconstruct", [[[1]], [0], [[1, 0]]]),
    ("reconstruct", [[[1]], [float('inf'), 0], [[1, 0]]]),
    ("reconstruct", [[[1]], [0, 0], [[0, 0]]]),
    ("reconstruct", [[[1]], [0, 0], [[float('inf'), 0]]]),
    ("reconstruct", [[[1]], [0, 0], []]),
    ("reconstruction_error", [[], []]),
    ("reconstruction_error", [[1, 2], [1, 2]]),
    ("reconstruction_error", [[[1, 2]], [[1], [2]]]),
    ("reconstruction_error", [[[1, 2]], [[1, 2], [1, 2]]]),
    ("reconstruction_error", [[[float('nan')]], [[0]]]),
    ("reconstruction_error", [[[0]], [[float('inf')]]]),
])
def test_invalid_inputs(invoke, function, args):
    assert invoke("decomposition", function, args)["error"] == "ValueError"


def test_basis_tolerance_accepts_small_roundoff(call):
    np.testing.assert_allclose(call("decomposition", "project", [[1, 2]], [0, 0], [[1 + 2e-9, 0]]), [[1]], atol=3e-9)
