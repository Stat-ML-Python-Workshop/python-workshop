"""Independent teacher acceptance, executed through the isolated student worker."""
import numpy as np
import pytest


def test_center_learns_and_reuses_mean(call):
    Xc, mean = call("decomposition", "center", [[2, 8, 0], [4, 12, 0]])
    assert mean == [3, 10, 0] and Xc == [[-1, -2, 0], [1, 2, 0]]
    Xc, saved = call("decomposition", "center", [[10, 20, 3]], mean=mean)
    assert saved == mean and Xc == [[7, 10, 3]]
    assert call("decomposition", "center", [[7]]) == [[[0]], [7]]


def test_numeric_conversion(call):
    assert call("decomposition", "center", [["1"], ["3"]]) == [[[-1], [1]], [2]]
    assert call("decomposition", "covariance_matrix", [["-1"], ["1"]]) == [[2]]
    assert call("decomposition", "variance_along_direction", [["-1"], ["1"]], ["3"]) == pytest.approx(2)


@pytest.mark.parametrize("Xc", [
    [[-2, 0, 1], [0, 0, -3], [2, 0, 2]], [[-2], [2]], [[0, 0], [0, 0]],
    [[-2, 1], [-1, -3], [1, 2], [2, 0]],
])
def test_covariance_sample_reference(call, Xc):
    C = call("decomposition", "covariance_matrix", Xc)
    expected = np.atleast_2d(np.cov(Xc, rowvar=False, ddof=1))
    np.testing.assert_allclose(C, expected, atol=1e-12)
    np.testing.assert_allclose(C, np.asarray(C).T)


@pytest.mark.parametrize("direction,expected", [([1, 0], 1), ([0, 2], 1), ([-3, -3], 2), ([1, -1], 0)])
def test_direction_hand_values(call, direction, expected):
    assert call("decomposition", "variance_along_direction", [[-1, -1], [0, 0], [1, 1]], direction) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("factor", [1, -5, 1e-200, 1e200])
def test_direction_reference(call, factor):
    Xc = np.array([[-2, 1, 0], [-1, -3, 0], [1, 2, 0], [2, 0, 0]], dtype=float)
    v = np.array([2., -3., 1.])
    expected = np.var(Xc @ (v / np.linalg.norm(v)), ddof=1)
    assert call("decomposition", "variance_along_direction", Xc.tolist(), (factor*v).tolist()) == pytest.approx(expected)


@pytest.mark.parametrize("function,args", [
    ("center", [[]]), ("center", [[1, 2]]), ("center", [[[1, 2], [3]]]),
    ("center", [[[float('nan')]]]), ("center", [[[float('inf')]]]),
    ("center", [[[1, 2]], [0]]), ("center", [[[1, 2]], [[0, 0]]]),
    ("center", [[[1, 2]], [0, float('inf')]]),
    ("covariance_matrix", [[[0, 0]]]), ("covariance_matrix", [[[1], [3]]]),
    ("covariance_matrix", [[[]]]), ("covariance_matrix", [[[float('nan')], [0]]]),
    ("covariance_matrix", [[[-1 + 2e-10], [1 + 2e-10]]]),
    ("variance_along_direction", [[[-1, 0], [1, 0]], [0, 0]]),
    ("variance_along_direction", [[[-1, 0], [1, 0]], [1]]),
    ("variance_along_direction", [[[-1, 0], [1, 0]], [[1, 0]]]),
    ("variance_along_direction", [[[-1, 0], [1, 0]], [float('inf'), 0]]),
    ("variance_along_direction", [[[1, 2], [3, 4]], [1, 0]]),
    ("variance_along_direction", [[[0, 0]], [1, 0]]),
])
def test_invalid_inputs(invoke, function, args):
    assert invoke("decomposition", function, args)["error"] == "ValueError"


def test_center_tolerance(call):
    np.testing.assert_allclose(call("decomposition", "covariance_matrix", [[-1 + 5e-11], [1 + 5e-11]]), [[2]])
