# Week 04 — From Dot Products to Matrix Multiplication

## What you will build

Extend your existing `mini_ml` package with four pure-Python operations:

```python
dot(x, y)
shape(matrix)
transpose(matrix)
matmul(m1, m2)
```

You will implement every function, including its input validation. The central
idea is that one entry of a matrix product is the dot product of one row from
`m1` and one column from `m2`:

```text
dot → shape → transpose → matmul(m1, m2)
```

NumPy is used only after your implementation works, as an independent check
against `np.asarray(m1) @ np.asarray(m2)`. Continue using the same package and
preserve your Week 1–3 implementations.

Replace `<login>` with your GitHub login and `<name_studentnumber>` with your
registered folder. A **repository root** contains `tools/` and `students/`;
your **package root** is `students/<name_studentnumber>/`.

## Step 1 — Start from your Week 3 work

From your package root, confirm that the earlier work still passes:

```sh
uv sync --frozen
uv run pytest
uv run python ../../tools/course.py test <login> 3
git status --short
```

Save unfinished work on its existing branch. Once Week 3 is merged, run from
the repository root:

```sh
git switch main
git pull --ff-only origin main
git switch -c <login>/week-04
cd students/<name_studentnumber>
```

## Step 2 — Get the starter and experiment

From your package root:

```sh
uv run python ../../tools/course.py sync <login> 4
```

Create `src/mini_ml/linalg.py` from the released Week 4 fragment. Its four
functions contain signatures, docstrings, and unfinished student regions. The
experiment belongs in ignored `test_code/`; do not commit that directory.

The biomedical example uses two features from the Wisconsin Diagnostic Breast
Cancer dataset (WDBC): mean radius and mean texture. Each row is one patient,
and the diagnosis target remains separate from the feature matrix.

## Step 3 — Implement `dot`

The dot product combines two equal-length vectors into one scalar:

$$
x \cdot y = \sum_{k=1}^{n} x_k y_k
$$

Implement `dot(x, y)` with an explicit loop:

1. Raise `ValueError("vector dimensions differ")` when the lengths differ.
2. Start an accumulator at `0.0`.
3. Multiply matching entries and add each product to the accumulator.
4. Return the accumulator.

Check these cases:

```python
assert dot([1, 2], [3, 4]) == 11
assert dot([-2, 0, 3], [4, 7, -1]) == -11
assert dot([], []) == 0.0
```

The length check must happen before `zip`: `zip` silently stops at the shorter
input and cannot detect incompatible vectors for you.

## Step 4 — Implement `shape`

Implement `shape(matrix)` for nonempty rectangular lists of rows. A valid
matrix has at least one row, at least one column, and the same number of
columns in every row.

```python
assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)
```

Required errors:

```python
shape([])                 # ValueError("matrix must be nonempty")
shape([[]])               # ValueError("matrix must be nonempty")
shape([[1, 2], [3]])      # ValueError("ragged matrix")
```

Explain why a ragged matrix cannot share one column count or participate in a
well-defined matrix product.

## Step 5 — Implement `transpose`

Transposition turns the columns of a matrix into rows. Implement
`transpose(matrix)` by calling `shape(matrix)` first, then constructing every
column as a new list.

```python
matrix = [[1, 2, 3], [4, 5, 6]]
assert transpose(matrix) == [[1, 4], [2, 5], [3, 6]]
```

The input is `2 × 3`; its transpose is `3 × 2`. Because `transpose` reuses
`shape`, empty and ragged inputs must raise the same `ValueError` cases.

## Step 6 — Understand `m1 @ m2` by hand

Consider a `2 × 3` matrix multiplied by a `3 × 2` matrix:

```python
m1 = [[1, 2, 3], [4, 5, 6]]
m2 = [[7, 8], [9, 10], [11, 12]]
```

The inner dimensions match, so the result has the outer dimensions `2 × 2`.
Transpose `m2` to expose its columns:

```python
columns = transpose(m2)
# [[7, 9, 11], [8, 10, 12]]
```

Each result entry pairs one row with one column:

```text
result[i][j] = dot(m1[i], transpose(m2)[j])
```

For the first row:

```text
58 = dot([1, 2, 3], [7, 9, 11])
64 = dot([1, 2, 3], [8, 10, 12])
```

The complete result is `[[58, 64], [139, 154]]`.

## Step 7 — Implement `matmul(m1, m2)`

Keep the parameter names `m1` and `m2`. Your implementation must:

1. Call `shape` on both inputs.
2. Check that the number of columns in `m1` equals the number of rows in `m2`.
3. Raise `ValueError("m1 and m2 dimensions are incompatible")` if they differ.
4. Call `transpose(m2)` once.
5. Call `dot(row, column)` for every output position.
6. Return a list of rows.

Do not import NumPy or use the `@` operator inside `linalg.py`. The purpose of
this implementation is to make the row-column calculation visible.

```python
assert matmul(m1, m2) == [[58, 64], [139, 154]]
assert matmul([[1, 2], [3, 4]], [[2], [1]]) == [[4], [10]]
```

## Step 8 — Apply matrix multiplication to patient scores

Let `M` contain three patients and let `T` be a single-column weight matrix:

```python
M = [[14.86, 23.21], [8.196, 16.84], [11.22, 19.86]]
T = [[0.4], [0.1]]
bias = -7.0
weighted_sums = matmul(M, T)
scores = [[value + bias for value in row] for row in weighted_sums]
```

`M` is `3 × 2`, `T` is `2 × 1`, and `M @ T` is `3 × 1`. The weighted sums
are approximately `[[8.265], [4.9624], [6.474]]`; after adding the bias, the
linear scores are `[[1.265], [-2.0376], [-0.526]]`.

These instructor-supplied weights are not learned and the scores are not
disease probabilities. Week 9 will learn weights and bias, then apply sigmoid
to interpret a linear score as a probability.

## Step 9 — Write durable tests and compare with NumPy

Create `tests/test_linalg.py`. Cover:

- positive, negative, floating-point, empty, and unequal-length dot products;
- valid non-square, empty, empty-row, and ragged shapes;
- a non-square transpose and invalid transpose inputs;
- `2 × 3 @ 3 × 2`, a single-column result, `1 × 1`, incompatible dimensions,
  and ragged inputs to `matmul`;
- at least one independently calculated example of your own.

Use `pytest.approx` for floating-point results and `pytest.raises(ValueError)`
for invalid inputs. Compare with NumPy only after your own expected values pass:

```python
import numpy as np
from mini_ml.linalg import matmul

assert np.allclose(matmul(m1, m2), np.asarray(m1) @ np.asarray(m2))
assert np.allclose(matmul(M, T), np.asarray(M) @ np.asarray(T))
```

Run:

```sh
uv run pytest
uv run python test_code/week_04.py
uv run python ../../tools/course.py test <login> 4
```

## Submit your Week 4 PR

Use `<login>/week-04` and modify only your own student folder. Commit the
completed `linalg.py` and durable tests, push the branch, and open a PR using
the weekly template. Include the commands you ran and explain:

- why matrix multiplication requires matching inner dimensions;
- how transposition exposes the columns of `m2`;
- how `matmul` reuses `shape`, `transpose`, and `dot`;
- why `M @ T` produces one result row per patient.
