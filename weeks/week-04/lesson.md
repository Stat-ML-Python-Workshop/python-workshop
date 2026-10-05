# Week 04 — Matrix Multiplication

## What you will build

Extend your existing `mini_ml` package with three pure-Python operations:

```python
shape(matrix)
transpose(matrix)
matmul(m1, m2)
```

The main exercise is general matrix multiplication with explicit loops:

```text
shape → transpose → matmul(m1, m2)
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

Create `src/mini_ml/linalg.py` inside your own package. The implementation
blocks in Steps 3, 4, and 6 are the instructor-provided scaffolds for this
lesson; copy them into that file once, then replace each TODO with your code.
They are not completed solutions. Preserve functions you have already written.

**Scaffold source:** `weeks/week-04/fragments/01_linalg.py` in the
repository root. The implementation blocks below reproduce its functions.
**Destination:** `students/<name_studentnumber>/src/mini_ml/linalg.py`
(repository-relative), or `src/mini_ml/linalg.py` (package-relative).

The fragment and implementation blocks below use the same fill-in scaffold.
Keep the validation, loops, and list assembly provided by the instructor.
Replace only the four `_todo("...")` calls with Python expressions:
two dimensions, one transpose entry, and one multiply-and-add expression.

Copy this helper into `src/mini_ml/linalg.py` once, before the three functions.
It produces a clear error when you run an unfinished exercise:

```python
def _todo(hint):
    """Replace each _todo(...) call with your expression."""
    raise NotImplementedError("TODO: " + hint)
```

For example, replace `rows = _todo("number of rows")` with an expression
that counts rows. Do not change the helper to make the tests pass.

**Experiment destination:** `test_code/week_04.py` in your package.
Example values, assertions, and NumPy comparisons belong there while you
explore. Durable pytest cases belong in `tests/test_linalg.py`.
Do not paste experiment code into `src/mini_ml/linalg.py` or commit
`test_code/`.

The biomedical example uses two features from the Wisconsin Diagnostic Breast
Cancer dataset (WDBC): mean radius and mean texture. Each row is one patient.

## Step 3 — Implement `shape`

Before editing the function, explore how a matrix is stored as a list of rows.
**Destination:** `test_code/week_04.py` in your package.

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
]
```

**Try it out:** run these lines one at a time:

1. `print(len(matrix))`
2. `print(matrix)`
3. `print(matrix[0])`
4. `print(len(matrix[0]))`

You should see `2`, `[[1, 2, 3], [4, 5, 6]]`, `[1, 2, 3]`, and `3`.
The outer list contains two rows. The first row contains three entries, so
this matrix has two rows and three columns.

**Implementation scaffold source:** this lesson, adapted from the
`shape` function in `weeks/week-04/fragments/01_linalg.py`.
**Edit:** `src/mini_ml/linalg.py` relative to your package root.
Copy this function scaffold, then replace the TODO:

```python
def shape(matrix):
    """Return (rows, columns) for a nonempty rectangular matrix."""
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be nonempty")
    rows = _todo("number of rows")
    columns = _todo("number of columns")
    for row in matrix:
        if len(row) != columns:
            raise ValueError("ragged matrix")
    return rows, columns
```

Implement `shape(matrix)` for nonempty rectangular lists of rows. A valid
matrix has at least one row, at least one column, and the same number of
columns in every row.

**Experiment check:** after implementing `shape`, add the following code
to `test_code/week_04.py` in your package:

```python
from mini_ml.linalg import shape

assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)
print("shape check passed")
```

Run the file from your package root:

```sh
uv run python test_code/week_04.py
```

If this check passes, it prints `shape check passed`. An incorrect result
raises `AssertionError`; an unfinished function raises
`NotImplementedError`. If another check later in the file fails, fix that
failure too before treating the entire experiment as complete.

**Required error examples:** these calls deliberately raise exceptions.
Check them one at a time, or use `pytest.raises` in `tests/test_linalg.py`.

```python
shape([])                 # ValueError("matrix must be nonempty")
shape([[]])               # ValueError("matrix must be nonempty")
shape([[1, 2], [3]])      # ValueError("ragged matrix")
```

A ragged list has no single column count, so it cannot participate in a
well-defined matrix product.

## Step 4 — Implement `transpose`

**Implementation scaffold source:** this lesson, adapted from the
`transpose` function in `weeks/week-04/fragments/01_linalg.py`.
**Edit:** append this function to `src/mini_ml/linalg.py` in your package.

```python
def transpose(matrix):
    """Return a new matrix whose rows are the input columns."""
    rows, columns = shape(matrix)
    result = []
    for i in range(columns):
        result_row = []
        for j in range(rows):
            result_row.append(_todo("input entry for output row i, column j"))
        result.append(result_row)
    return result
```

Transposition turns the columns of a matrix into rows. Implement
`transpose(matrix)` by calling `shape(matrix)` first, then constructing every
column as a new list. Here, `i` is the output row index and `j` is the
output column index. The output has `columns` rows and `rows` columns,
so the outer `i` loop uses `range(columns)` and the inner `j` loop uses
`range(rows)`. Fill in the input entry corresponding to each output position.

**Experiment check:** `test_code/week_04.py`; import `transpose`
from `mini_ml.linalg`.

```python
matrix = [[1, 2, 3], [4, 5, 6]]
assert transpose(matrix) == [[1, 4], [2, 5], [3, 6]]
```

The input is `2 × 3`; its transpose is `3 × 2`. Because `transpose` reuses
`shape`, empty and ragged inputs must raise the same `ValueError` cases.

## Step 5 — Understand `m1 @ m2` by hand

**Example source:** instructor-created hand-calculation example in this
lesson. **Destination:** `test_code/week_04.py`; reuse these values in
Step 6 and in the NumPy checks.

Consider a `2 × 3` matrix multiplied by a `3 × 2` matrix:

```python
m1 = [[1, 2, 3], [4, 5, 6]]
m2 = [[7, 8], [9, 10], [11, 12]]
```

The inner dimensions match, so the result has the outer dimensions `2 × 2`.
Each result entry combines one row from `m1` with one column from `m2`:

$$
\operatorname{result}_{ij}
= \sum_{k=1}^{n} (m1)_{ik}(m2)_{kj}.
$$

For the first result row:

```text
58 = 1×7 + 2×9 + 3×11
64 = 1×8 + 2×10 + 3×12
```

The complete result is `[[58, 64], [139, 154]]`.

## Step 6 — Implement `matmul(m1, m2)` directly

**Implementation scaffold source:** this lesson's revised matrix-first
version of `matmul`. **Edit:** append to `src/mini_ml/linalg.py`.

```python
def matmul(m1, m2):
    """Return the product of compatible nonempty rectangular matrices."""
    rows_m1, columns_m1 = shape(m1)
    rows_m2, columns_m2 = shape(m2)
    if columns_m1 != rows_m2:
        raise ValueError("m1 and m2 dimensions are incompatible")
    result = [[0.0 for j in range(columns_m2)] for i in range(rows_m1)]
    for i in range(rows_m1):
        for j in range(columns_m2):
            total = 0.0
            for k in range(columns_m1):
                total += _todo("multiply matching entries")
            result[i][j] = total
    return result
```

Keep the parameter names `m1` and `m2`. Your implementation must:

1. Call `shape` on both inputs.
2. Check that the number of columns in `m1` equals the number of rows in `m2`.
3. Raise `ValueError("m1 and m2 dimensions are incompatible")` if they differ.
4. Allocate a zero-filled matrix with `rows_m1` rows and `columns_m2` columns.
5. Visit each output position `(i, j)`.
6. Accumulate `m1[i][k] * m2[k][j]` over the shared inner dimension.
7. Assign the accumulated value to `result[i][j]`, then return the result.

The initialization creates a separate list for each row. For this example,
the result starts as `[[0.0, 0.0], [0.0, 0.0]]`. Each `(i, j)` pair
identifies one cell to fill.

The loop structure is:

```text
create a zero matrix with the output dimensions
for each row i in m1:
    for each column j in m2:
        total = 0.0
        for each inner position k:
            total += m1[i][k] * m2[k][j]
        result[i][j] = total
```

Implement `matmul` using pure Python loops.

**Experiment checks:** `test_code/week_04.py`; import `matmul`
from `mini_ml.linalg` and use `m1`, `m2` from Step 5.

```python
assert matmul(m1, m2) == [[58, 64], [139, 154]]
assert matmul([[1, 2], [3, 4]], [[2], [1]]) == [[4], [10]]
```

## Step 7 — Run the supplied patient-score demo

**Completed demo source:** `weeks/week-04/demo.py` in the repository root.
This is instructor-written application code, ready to run after you finish
`shape`, `transpose`, and `matmul`; there are no additional fill-ins here.
The Week 4 sync command copies it to `test_code/week_04.py` in your package
and copies the training table to `data/week-04/train.csv`.

From your package root, run:

```sh
uv run python test_code/week_04.py
```

If you already have an older `test_code/week_04.py`, sync keeps that existing
file. Save any experiments you want to keep, then copy the updated
`weeks/week-04/demo.py` into `test_code/week_04.py`.

### Read the table and select features

The completed demo uses pandas to read the training table and selects features
by column name, in this order:

```python
training = pd.read_csv(data_dir / "train.csv")
feature_names = ["mean_radius", "mean_texture"]
M = training[feature_names].values.tolist()
```

**Code source:** excerpt from `weeks/week-04/demo.py`, already included in
`test_code/week_04.py`. You do not need to add this excerpt yourself.

Each row of `M` holds one patient's two measurements. The training table has
341 patients, so `M` has shape `341 × 2`.

### Multiply by arbitrary weights

**Code source:** another completed excerpt from the same demo:

```python
T = [[0.4], [0.1]]
bias = -7.0
weighted_sums = matmul(M, T)
scores = [[row[0] + bias] for row in weighted_sums]
```

The instructor chose these arbitrary weights to illustrate the calculation.
The first weight, `0.4`, multiplies `mean_radius`; the second, `0.1`,
multiplies `mean_texture`. For each patient:

```text
weighted_sum = 0.4 × mean_radius + 0.1 × mean_texture
linear_score = weighted_sum - 7.0
```

`T` has shape `2 × 1`, so `matmul(M, T)` returns a `341 × 1` list of
lists: one weighted sum per patient. Adding the bias preserves that shape.

### Expected output

The demo checks your result against NumPy's `@` and prints the first three
patients. Its output should look like:

```text
Features (in order): ['mean_radius', 'mean_texture']
Arbitrary weights: [[0.4], [0.1]]; bias: -7.0
M shape: (341, 2); T shape: (2, 1); result shape: (341, 1)
First 3 patients:
 mean_radius  mean_texture  weighted_sum  linear_score
     14.8600       23.2100        8.2650        1.2650
      8.1960       16.8400        4.9624       -2.0376
     11.2200       19.8600        6.4740       -0.5260
Matrix multiplication checks passed.
```

For example, the first patient's weighted sum is
`14.86 × 0.4 + 23.21 × 0.1 = 8.265`; adding `-7.0` gives `1.265`.
The first three rows of `scores` are approximately
`[[1.265], [-2.0376], [-0.526]]`. The printed table rounds values to four
decimal places; the underlying results remain floating-point numbers.

These weights and the bias are not learned, and the scores are not disease
probabilities. A positive or negative value here is just the result of the
chosen arithmetic. Week 9 will learn weights and bias, then apply sigmoid
to interpret a linear score as a probability.

## Step 8 — Run the supplied pytest tests

**Completed test source:** `weeks/week-04/student_tests/test_linalg.py`
in the repository root.
**Destination after sync:** `tests/test_linalg.py` inside your package.

The Week 4 sync command installs these instructor-written tests automatically.
You do not need to create the test file or fill in any test code. If you synced
before this file was released, run sync again:

```sh
uv run python ../../tools/course.py sync <login> 4
```

The supplied tests cover:

- rectangular matrix shapes;
- empty matrices, empty rows, and ragged inputs;
- non-square, single-column, and scalar transposes;
- `2 × 3 @ 3 × 2`, a single-column result, and `1 × 1`;
- negative, floating-point, and zero entries;
- incompatible dimensions and invalid inputs on either side of `matmul`.

They use `pytest.approx` for floating-point results and
`pytest.raises(ValueError)` for invalid inputs. Keep them in `tests/` and
commit them with your completed functions. You may add more cases if you want.

From your package root, run the Week 4 tests:

```sh
uv run pytest -q tests/test_linalg.py
```

After all four fill-ins are completed correctly, the supplied file should
report `27 passed`. An unfinished `_todo(...)` raises
`NotImplementedError`; use the failing test to find the expression you still
need to complete.

Then run all your package tests, the completed demo, and the course acceptance
tests:

```sh
uv run pytest
uv run python test_code/week_04.py
uv run python ../../tools/course.py test <login> 4
```

The demo already contains the NumPy reference comparisons.

## Submit your Week 4 PR

Use `<login>/week-04` and modify only your own student folder. Commit the
completed `linalg.py` and durable tests, push the branch, and open a PR using
the weekly template. Include the commands you ran and explain:

- why matrix multiplication requires matching inner dimensions;
- how the `i`, `j`, and `k` loops construct each output value;
- why `M @ T` produces one result row per patient.
