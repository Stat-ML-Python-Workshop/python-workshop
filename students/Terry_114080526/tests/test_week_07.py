"""Teacher-provided W7 tests. Run unchanged; no extra cases required."""
import numpy as np
import pytest
from mini_ml import decomposition as dec


def test_projection_hand_values_and_shape():
    scores = dec.project([[2, 3], [-2, -3]], [0, 0], [[1, 0]])
    assert scores.dtype == np.float64 and scores.shape == (2, 1)
    np.testing.assert_allclose(scores, [[2], [-2]])


def test_projection_uses_saved_mean_and_rectangular_basis():
    X = np.array([[11., 22., 35.], [9., 16., 31.]])
    mean = np.array([10., 20., 30.])
    components = np.array([[1., 0., 0.], [0., 0., 1.]])
    saved = [value.copy() for value in (X, mean, components)]
    expected = [[1, 5], [-1, 1]]
    np.testing.assert_allclose(dec.project(X, mean, components), expected)
    np.testing.assert_allclose(dec.project(X[:1], mean, components), expected[:1])
    for value, original in zip((X, mean, components), saved):
        np.testing.assert_array_equal(value, original)


def test_projection_numeric_strings_and_single_feature():
    scores = dec.project([["5"]], ["2"], [["-1"]])
    assert scores.dtype == np.float64
    np.testing.assert_array_equal(scores, [[-3]])


@pytest.mark.parametrize("X,mean,components", [
    ([], [0], [[1]]), ([[]], [0], [[1]]), ([1, 2], [0, 0], [[1, 0]]),
    ([[1, 2], [3]], [0, 0], [[1, 0]]), ([[np.nan]], [0], [[1]]),
    ([[np.inf]], [0], [[1]]), ([[1, 2]], [0], [[1, 0]]),
    ([[1, 2]], [[0, 0]], [[1, 0]]), ([[1, 2]], [0, np.inf], [[1, 0]]),
    ([[1, 2]], [0, 0], [[1]]), ([[1, 2]], [0, 0], []),
    ([[1, 2]], [0, 0], [[1, 0], [0, 1], [1, 0]]),
    ([[1, 2]], [0, 0], [[2, 0]]), ([[1, 2]], [0, 0], [[0, 0]]),
    ([[1, 2]], [0, 0], [[1, 0], [1, 0]]),
    ([[1, 2]], [0, 0], [[np.nan, 0]]),
])
def test_projection_invalid_inputs(X, mean, components):
    with pytest.raises(ValueError):
        dec.project(X, mean, components)


def test_projection_basis_tolerance():
    dec.project([[1, 2]], [0, 0], [[1 + 2e-9, 0]])
    with pytest.raises(ValueError, match="orthonormal"):
        dec.project([[1, 2]], [0, 0], [[1 + 2e-8, 0]])


def test_reconstruction_hand_values_and_shape():
    result = dec.reconstruct([[2], [-2]], [0, 0], [[1, 0]])
    assert result.dtype == np.float64 and result.shape == (2, 2)
    np.testing.assert_allclose(result, [[2, 0], [-2, 0]])


def test_reconstruction_restores_mean_and_preserves_inputs():
    Z = np.array([[1., 5.], [-1., 1.]])
    mean = np.array([10., 20., 30.])
    components = np.array([[1., 0., 0.], [0., 0., 1.]])
    saved = [value.copy() for value in (Z, mean, components)]
    result = dec.reconstruct(Z, mean, components)
    np.testing.assert_allclose(result, [[11, 20, 35], [9, 20, 31]])
    result[:] = 0
    for value, original in zip((Z, mean, components), saved):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("Z,mean,components", [
    ([], [0, 0], [[1, 0]]), ([1], [0, 0], [[1, 0]]),
    ([[np.inf]], [0, 0], [[1, 0]]), ([[1, 2]], [0, 0], [[1, 0]]),
    ([[1]], [0], [[1, 0]]), ([[1]], [np.nan, 0], [[1, 0]]),
    ([[1]], [0, 0], [[0, 0]]), ([[1]], [0, 0], [[np.inf, 0]]),
    ([[1]], [0, 0], []), ([[1]], [0, 0], [1, 0]),
])
def test_reconstruction_invalid_inputs(Z, mean, components):
    with pytest.raises(ValueError):
        dec.reconstruct(Z, mean, components)


def test_mse_hand_values_and_all_entries():
    assert type(dec.reconstruction_error([[2, 3], [-2, -3]], [[2, 0], [-2, 0]])) is float
    assert dec.reconstruction_error([[2, 3], [-2, -3]], [[2, 0], [-2, 0]]) == pytest.approx(4.5)
    assert dec.reconstruction_error([[1, 2, 3], [4, 5, 6]], np.zeros((2, 3))) == pytest.approx(91 / 6)
    assert dec.reconstruction_error([["3"]], [["1"]]) == 4.0


def test_mse_no_clipping_and_no_mutation():
    X = np.array([[0., 1., 0.5]])
    X_hat = np.array([[-0.2, 1.1, 0.5]])
    original, predicted = X.copy(), X_hat.copy()
    assert dec.reconstruction_error(X, X_hat) == pytest.approx(0.05 / 3)
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(X_hat, predicted)


@pytest.mark.parametrize("X,X_hat", [
    ([], []), ([[]], [[]]), ([1, 2], [1, 2]),
    ([[1, 2]], [[1], [2]]), ([[1, 2]], [[1, 2], [1, 2]]),
    ([[np.nan]], [[0]]), ([[0]], [[np.inf]]),
])
def test_mse_invalid_inputs(X, X_hat):
    with pytest.raises(ValueError):
        dec.reconstruction_error(X, X_hat)


def test_full_basis_and_nested_error():
    X = np.array([[2., 3., 5.], [-2., -3., 1.], [1., 7., 4.]])
    mean = np.array([1., 2., 3.])
    basis, _ = np.linalg.qr(np.array([[1., 2., 3.], [3., -1., 2.], [2., 3., -1.]]))
    errors = []
    for k in (1, 2, 3):
        components = basis[:k]
        Z = dec.project(X, mean, components)
        X_hat = dec.reconstruct(Z, mean, components)
        errors.append(dec.reconstruction_error(X, X_hat))
        flipped = dec.reconstruct(-Z, mean, -components)
        np.testing.assert_allclose(flipped, X_hat, atol=1e-12)
    assert np.all(np.diff(errors) <= 1e-12)
    np.testing.assert_allclose(X_hat, X, atol=1e-12)
    assert errors[-1] == pytest.approx(0, abs=1e-24)
