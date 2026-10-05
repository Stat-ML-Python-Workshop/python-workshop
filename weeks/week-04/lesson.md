# Week 04 — From One Patient to a Linear Model Score

## What you will build

Extend your existing `mini_ml` package with pure-Python vector and matrix
operations:

```python
dot(x, weights)
shape(matrix)
matvec(matrix, weights)
```

The two core exercises are `dot` and matrix shape validation. The instructor
will guide you through `matvec`, which reuses `dot` for each patient.
Continue using the same package and preserve your Week 1–3 implementations.

By the end of this lesson, you should be able to separate features from a
target, calculate `z = dot(x, weights) + bias`, explain its dimensions, and
recognize invalid inputs. NumPy is used only to check your results this week.

Replace `<login>` with your GitHub login and `<name_studentnumber>` with your
registered folder. A **repository root** contains `tools/` and `students/`;
your **package root** is `students/<name_studentnumber>/`.

## Step 1 — Start from your Week 3 work

**Working directory:** your package root.

```sh
uv sync --frozen
uv run pytest
uv run python ../../tools/course.py test <login> 3
git status --short
```

**Check:** Week 1–3 tests pass. Save any unfinished work on its existing branch
before switching branches. If your Week 3 PR has not been merged, ask the
instructor for help before starting Week 4.

Once Week 3 is merged, run from the repository root:

```sh
git switch main
git pull --ff-only origin main
git switch -c <login>/week-04
cd students/<name_studentnumber>
```

**Check:** your branch is `<login>/week-04`, and your earlier modules and tests
are still present.

## Step 2 — Get this week's experiment

**Working directory:** your package root.

```sh
uv run python ../../tools/course.py sync <login> 4
```

The released experiment belongs in your ignored `test_code/` directory.
`sync` does not complete your source functions. Wait for the instructor if
Week 4 has not been released.

The biomedical example uses two features from the Wisconsin Diagnostic Breast
Cancer dataset (WDBC). The fixed training split contains 341 patients. This
week, begin with one patient; Week 5 will explore the training dataset.

## Step 3 — Represent one patient as a vector

Consider the first row of the training CSV:

| Column | Value | Role |
| --- | ---: | --- |
| `mean_radius` | 14.86 | Feature: mean nucleus radius |
| `mean_texture` | 23.21 | Feature: mean texture, based on variation in grayscale values |
| `malignant` | 1 | Target: 1 = malignant, 0 = benign |

Each row describes one diagnostic sample. In this lesson, we call that row a
patient. The features summarize measurements of cell nuclei in an image.

```python
feature_names = ["mean_radius", "mean_texture"]
patient = [14.86, 23.21]
target = 1
```

A vector is an ordered collection of numbers. Here, position 0 always means
radius and position 1 always means texture. Every patient's vector must use
the same feature order.

The target is the diagnosis we would like to predict. Keep it separate from
the features: a future prediction must be possible before the diagnosis is
known.

**Check:** explain why `[14.86, 23.21, 1]` is inappropriate as this model's
input, and why swapping only the patient's feature order changes the result.

## Step 4 — Calculate a linear score by hand

The instructor supplies a scoring rule:

```python
weights = [0.4, 0.1]
bias = -7.0
```

These weights are chosen for teaching; they have not been learned from data.
They do not establish a clinically useful model or a disease probability.
Week 9 will teach the model to learn weights and bias.

The dot product multiplies matching positions and adds their contributions:

$$
x \cdot w = \sum_{j=1}^{d} x_j w_j
$$

| Feature | Value | Weight | Contribution |
| --- | ---: | ---: | ---: |
| mean radius | 14.86 | 0.4 | 5.944 |
| mean texture | 23.21 | 0.1 | 2.321 |
| Dot product | | | **8.265** |
| Bias | | | −7.000 |
| Linear score | | | **1.265** |

The complete calculation is:

$$
z = x \cdot w + b = 8.265 - 7.0 = 1.265
$$

For this demonstration, use `z >= 0` to predict malignant and `z < 0` to
predict benign. Changing the bias moves this threshold without changing the
individual feature weights.

The score can be negative or greater than 1. In Week 9, logistic regression
will interpret this linear score as a **logit** and apply sigmoid to obtain
a probability. This week, return the score itself.

**Check:** calculate the result with `bias = -9.0`. Explain why a larger
feature value increases the score when its weight is positive and decreases
it when its weight is negative.

## Step 5 — Understand dimensions before coding

| Object | Meaning | Dimensions | Output |
| --- | --- | --- | --- |
| `patient` | One patient's features | Length 2 | Vector |
| `weights` | One weight per feature | Length 2 | Vector |
| `dot(patient, weights)` | Weighted sum | Matching vector lengths | Scalar |
| `X` | Several patients | `n` rows × 2 columns | Matrix |
| `matvec(X, weights)` | One weighted sum per patient | `(n × 2)` with length 2 | Vector of length `n` |

Two vectors can be paired only when their lengths match. A rectangular matrix
must have the same number of columns in every row.

```python
X = [
    [14.86, 23.21],
    [8.196, 16.84],
    [11.22, 19.86],
]
```

This matrix has three rows and two columns. A label vector for these rows
would have length three, while the weight vector still has length two.

**Check:** predict the dimensions for 341 patients with two features, then
for the same patients with 30 features. How many weights would each require?

## Step 6 — Implement `dot`

**Working directory:** package root; create `src/mini_ml/linalg.py` using the
instructor's Week 4 linear algebra starter.

Keep the provided dimension validation. Complete the dot-product student
region with an explicit loop:

1. Start an accumulator at `0.0`.
2. Use `zip(x, weights)` to visit matching positions.
3. Multiply the two numbers and add their contribution to the accumulator.
4. Return the accumulated value.

This is the same accumulation pattern you used for variance and entropy.
Use pure Python for the implementation. The length check must happen before
`zip`, because `zip` stops at the shorter input without raising an error.

**Check:** `dot([1, 2], [3, 4])` returns `11`, and
`dot([-2, 0, 3], [4, 7, -1])` returns `-11`. Define the empty dot product
as `dot([], []) == 0`. Unequal lengths must raise `ValueError`.

In your local experiment, calculate:

```python
from mini_ml.linalg import dot

score = dot(patient, weights) + bias
print(score)
```

**Expected:** approximately `1.265` with the original weights and bias.

## Step 7 — Validate matrix shape

**Edit:** `src/mini_ml/linalg.py`.

Complete `shape(matrix)` so that it returns `(rows, columns)` for a valid,
nonempty rectangular list of rows. Raise `ValueError` for:

- An empty matrix: `[]`.
- An empty first row: `[[]]`.
- Rows of different lengths: `[[1, 2], [3]]`.

Use `len(matrix)` for the number of rows and the first row's length for the
expected number of columns. Check every row against that expected length.
The instructor supplies any additional input-validation scaffolding.

**Check:** `shape([[1, 2], [3, 4], [5, 6]])` returns `(3, 2)`.
Explain why a ragged matrix cannot use one shared weight vector.

## Step 8 — Build `matvec` with the instructor

`matvec` applies the existing `dot` function to every row:

```text
validate the matrix shape
check that the column count equals the weight count
create an empty list of scores
for each patient row:
    append dot(row, weights)
return the scores
```

Follow the instructor's walkthrough and explain how each step reuses an
earlier function. `matvec` returns the weighted sums; add the bias separately
to obtain the complete linear scores.

For the three patients in Step 5, the dot products are approximately
`[8.265, 4.9624, 6.474]`. Adding `-7.0` to each gives
`[1.265, -2.0376, -0.526]`.

Each score belongs to its own patient. The number of scores must equal the
number of rows, and another patient's presence must not change an existing
patient's score.

**Check:** `matvec([[1, 2], [3, 4]], [2, 1])` returns `[4, 10]`.
Passing a length-three weight vector to this matrix must raise `ValueError`.

## Step 9 — Check your implementation and keep durable tests

**Working directory:** package root.

Create `tests/test_linalg.py`. Include a known dot product, a negative-value
case, the empty dot product, unequal vector lengths, a rectangular matrix,
invalid matrix shapes, a known `matvec` result, and incompatible matrix/vector
dimensions. Add at least one example of your own and calculate its expected
result independently. Use `pytest.approx` for floating-point results and
`pytest.raises(ValueError)` for invalid inputs.

In the local experiment, compare your dot product with NumPy:

```python
import numpy as np
from mini_ml.linalg import dot, matvec

assert np.isclose(dot(patient, weights), np.dot(patient, weights))
assert np.allclose(matvec(X, weights), np.asarray(X) @ np.asarray(weights))
```

NumPy provides its own `dot` and matrix multiplication operations. Your
functions belong to `mini_ml.linalg` and use explicit Python loops.

```sh
uv run pytest
uv run python test_code/week_04.py
uv run python ../../tools/course.py test <login> 4
```

**Expected after the revised Week 4 materials are released:** all cumulative
tests and the experiment pass. Keep `test_code/` local; commit the module and
durable tests. Do not modify shared acceptance tests to make a failure pass.

## Submit your Week 4 PR

Use `<login>/week-04` and modify only your own student folder. Review your
changes, commit, push, and open a PR using the repository's weekly template.
Include the commands you ran and a short explanation of features, weights,
bias, and score. Completion follows the published course cutoff policy;
instructor review and merge happen separately.

Before submitting, confirm that you can explain:

- Why the target is stored separately from the features.
- Why feature order and matching dimensions matter.
- How `matvec` reuses `dot`.
- Why the instructor-defined linear score is not yet a learned probability.
