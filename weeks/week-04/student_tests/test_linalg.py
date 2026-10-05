"""Instructor-provided Week 4 tests, copied into your package by sync."""
import pytest

from mini_ml.linalg import matmul, shape, transpose


@pytest.mark.parametrize("matrix, expected", [
    ([[1, 2, 3], [4, 5, 6]], (2, 3)),
    ([[1], [2], [3]], (3, 1)),
    ([[5]], (1, 1)),
])
def test_shape(matrix, expected):
    assert shape(matrix) == expected


@pytest.mark.parametrize("operation", [shape, transpose])
@pytest.mark.parametrize("matrix, message", [
    ([], "matrix must be nonempty"),
    ([[]], "matrix must be nonempty"),
    ([[1, 2], [3]], "ragged matrix"),
    ([[1, 2], []], "ragged matrix"),
])
def test_invalid_matrix(operation, matrix, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        operation(matrix)


@pytest.mark.parametrize("matrix, expected", [
    ([[1, 2, 3], [4, 5, 6]], [[1, 4], [2, 5], [3, 6]]),
    ([[1], [2], [3]], [[1, 2, 3]]),
    ([[5]], [[5]]),
])
def test_transpose(matrix, expected):
    assert transpose(matrix) == expected


@pytest.mark.parametrize("m1, m2, expected", [
    ([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]], [[58, 64], [139, 154]]),
    ([[1, 2], [3, 4]], [[2], [1]], [[4], [10]]),
    ([[3]], [[4]], [[12]]),
    ([[-2, 0, 3]], [[4], [7], [-1]], [[-11]]),
    ([[0.1, 0.2]], [[0.3], [0.4]], [[0.11]]),
    ([[1, 2]], [[0, 0], [0, 0]], [[0, 0]]),
])
def test_matmul(m1, m2, expected):
    result = matmul(m1, m2)
    assert isinstance(result, list)
    assert len(result) == len(expected)
    for row, expected_row in zip(result, expected):
        assert isinstance(row, list)
        assert row == pytest.approx(expected_row)


@pytest.mark.parametrize("m1, m2, message", [
    ([[1, 2]], [[1, 2]], "m1 and m2 dimensions are incompatible"),
    ([], [[1]], "matrix must be nonempty"),
    ([[1]], [], "matrix must be nonempty"),
    ([[]], [[1]], "matrix must be nonempty"),
    ([[1]], [[]], "matrix must be nonempty"),
    ([[1, 2], [3]], [[1], [2]], "ragged matrix"),
    ([[1, 2]], [[1, 2], [3]], "ragged matrix"),
])
def test_invalid_matmul(m1, m2, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        matmul(m1, m2)
