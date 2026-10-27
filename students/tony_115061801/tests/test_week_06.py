"""Teacher-provided W6 tests. Run these unchanged; no extra cases required."""
import numpy as np
import pytest
from mini_ml import decomposition as dec


@pytest.mark.parametrize("X", [
    [[1, 2], [2, 3], [3, 4]], [[7, 9]], [[1], [3], [8]],
    [["1", "2"], ["3", "4"]], np.array([[1, 2, 5], [4, 9, 5]]),
])
def test_center_columns(X):
    original = np.array(X, copy=True)
    result = dec.center(X)
    assert isinstance(result, tuple) and len(result) == 2
    Xc, mean = result
    assert Xc.dtype == mean.dtype == np.float64
    assert Xc.shape == np.asarray(X).shape and mean.shape == (Xc.shape[1],)
    np.testing.assert_allclose(mean, np.asarray(X, dtype=float).mean(axis=0))
    np.testing.assert_allclose(Xc.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(Xc + mean, np.asarray(X, dtype=float))
    np.testing.assert_array_equal(X, original)


def test_center_saved_mean_and_mutation():
    X = np.array([[1., 2.], [3., 4.]])
    original = X.copy()
    _, mean = dec.center(X)
    new = np.array([[7., 8.], [100., 200.]])
    saved = new.copy()
    actual, returned_mean = dec.center(new, mean)
    np.testing.assert_allclose(actual, [[5, 5], [98, 197]])
    np.testing.assert_allclose(dec.center(new[:1], mean)[0], actual[:1])
    np.testing.assert_array_equal(returned_mean, mean)
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(new, saved)
    np.testing.assert_array_equal(mean, [2, 3])
    actual[:] = 0
    np.testing.assert_array_equal(new, saved)


@pytest.mark.parametrize("mean", [[1], [[1, 2]], [np.inf, 0], [0, np.nan]])
def test_center_invalid_mean(mean):
    with pytest.raises(ValueError):
        dec.center([[1, 2]], mean)


@pytest.mark.parametrize("function", ["center", "covariance_matrix", "variance_along_direction"])
@pytest.mark.parametrize("X", [[], [[]], [1, 2], [[1, 2], [3]], [[[1]]], [[np.nan]], [[np.inf]]])
def test_invalid_matrices(function, X):
    args = (X, [1]) if function == "variance_along_direction" else (X,)
    with pytest.raises(ValueError):
        getattr(dec, function)(*args)


@pytest.mark.parametrize("Xc", [
    [[-1, -1], [0, 0], [1, 1]], [[-1], [1]], [[0, 0], [0, 0]],
    [[-1, -2, 0], [0, 0, 0], [1, 2, 0]],
    [[-2, 1], [-1, -3], [1, 2], [2, 0]], [["-1", "0"], ["1", "0"]],
])
def test_covariance_reference(Xc):
    original = np.array(Xc, copy=True)
    C = dec.covariance_matrix(Xc)
    values = np.asarray(Xc, dtype=float)
    assert C.dtype == np.float64 and C.shape == (values.shape[1],) * 2
    np.testing.assert_allclose(C, np.atleast_2d(np.cov(values, rowvar=False, ddof=1)), atol=1e-12)
    np.testing.assert_allclose(C, C.T)
    np.testing.assert_allclose(C.diagonal(), np.var(values, axis=0, ddof=1))
    np.testing.assert_array_equal(Xc, original)


@pytest.mark.parametrize("Xc", [[[0, 0]], [[1, 2], [3, 4]], [[-1], [1.000001]]])
@pytest.mark.parametrize("function", ["covariance_matrix", "variance_along_direction"])
def test_covariance_and_direction_require_centered_samples(function, Xc):
    args = (Xc, np.ones(len(Xc[0]))) if function == "variance_along_direction" else (Xc,)
    with pytest.raises(ValueError, match="centered"):
        getattr(dec, function)(*args)


def test_covariance_center_tolerance():
    np.testing.assert_allclose(dec.covariance_matrix([[-1 + 5e-11], [1 + 5e-11]]), [[2]])
    with pytest.raises(ValueError):
        dec.covariance_matrix([[-1 + 2e-10], [1 + 2e-10]])


@pytest.mark.parametrize("direction,expected", [([1, 0], 1), ([0, 1], 1), ([1, 1], 2), ([1, -1], 0)])
def test_direction_hand_calculation(direction, expected):
    actual = dec.variance_along_direction([[-1, -1], [0, 0], [1, 1]], direction)
    assert type(actual) is float
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("factor", [1, -1, 7, 1e-200, 1e200])
def test_direction_reference_scaling_and_no_mutation(factor):
    Xc = np.array([[-2., 1., 0.], [-1., -3., 0.], [1., 2., 0.], [2., 0., 0.]])
    direction = np.array([2., -3., 1.]) * factor
    X_before, v_before = Xc.copy(), direction.copy()
    v = np.array([2., -3., 1.]) / np.sqrt(14)
    actual = dec.variance_along_direction(Xc, direction)
    assert actual == pytest.approx(np.var(Xc @ v, ddof=1))
    np.testing.assert_array_equal(Xc, X_before)
    np.testing.assert_array_equal(direction, v_before)


@pytest.mark.parametrize("direction", [[0, 0], [], [1], [[1, 0]], [np.nan, 0], [0, np.inf]])
def test_direction_invalid(direction):
    with pytest.raises(ValueError):
        dec.variance_along_direction([[-1, 0], [1, 0]], direction)


def test_direction_reuses_covariance(monkeypatch):
    calls = []
    def fake_covariance(Xc):
        calls.append(Xc)
        return np.diag([2., 7.])
    monkeypatch.setattr(dec, "covariance_matrix", fake_covariance)
    assert dec.variance_along_direction([[-1, 0], [1, 0]], [0, 3]) == pytest.approx(7)
    assert len(calls) == 1
