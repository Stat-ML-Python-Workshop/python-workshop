# Week 05 — NumPy, Standardizer, and Visualization

## What you will build

Use NumPy inside your cumulative mini_ml package and keep its Week 1–4 implementations. Build a standardizer with saved training parameters and one vector-distance function, then plot benign/malignant training patients.

| Interface | Input → output | Student calculation |
|---|---|---|
| `Standardizer.fit(X)` | `(n,d) → self` | Feature means, population scales, zero-scale fallback |
| `Standardizer.transform(X)` | `(m,d) → ndarray (m,d)` | Broadcast z-score |
| `euclidean_distance(x,y)` | `(d,), (d,) → float` | Euclidean norm of the difference |

Here n, m, d ≥ 1. Rows are samples; columns are features. x and y each describe **one sample**, with identical feature order and scale. There are three exercises and five expression fill-ins. Teachers supply validation and object structure.

The **repository root** contains tools/, weeks/, students/. Your **package root** is students/<name_studentnumber>/. Replace <login> with your GitHub login. Source paths starting with weeks/ are repository-relative; src/, tests/, data/, test_code/ destinations are package-relative.

## Step 1 — Continue from Week 4

**Directory:** package root. **Edit:** none yet.

```sh
uv sync --frozen
uv run pytest
uv run python ../../tools/course.py test <login> 4
git status --short
```

**Expected:** Week 1–4 tests pass. Save unfinished Week 4 work and ensure its PR is merged. From the **repository root**:

```sh
git switch main
git pull --ff-only origin main
git switch -c <login>/week-05
cd students/<name_studentnumber>
```

If the branch exists, use git switch <login>/week-05. If a check fails, first confirm the package directory and editable installation; resolve previous failures before adding this week's code.

## Step 2 — Sync materials and dependencies

**Directory:** package root. **Edit:** dependency files through uv.

```sh
uv run python ../../tools/course.py sync <login> 5
uv add numpy pandas matplotlib scikit-learn
uv sync --frozen
```

Sync supplies data/week-05/train.csv, test_code/week_05.py and tests/test_week_05.py. It does not integrate source fragments. Keep existing tests and your own added cases. Do not run the complete demo until all three exercises work.

**Expected:** the CSV has 341 training rows and columns mean_radius, mean_texture, malignant. If files are missing, check login/folder mapping and the published week; if imports fail, rerun uv sync from the package root.

## Step 3 — Map Week 4 matrices to NumPy, then integrate once

**Directory:** package root. **Edit:** create test_code/week_05_practice.py for local experiments.

| Your Week 4 function | NumPy expression |
|---|---|
| `shape(matrix)` | `X.shape` |
| `transpose(matrix)` | `X.T` |
| `matmul(m1,m2)` | `A @ B` |

`*` multiplies entries; `@` is matrix multiplication. Converting to an ndarray changes how you express the same mathematics.

**Source:** Instructor-designed hand-calculation example; calls Week 4 linalg functions.

**Purpose:** Runnable practice; create this file.

**Destination:** test_code/week_05_practice.py

```python
import numpy as np
from mini_ml.linalg import shape, transpose, matmul
matrix = [[1, 10], [3, 10]]
X = np.asarray(matrix, dtype=float)
B = np.array([[1], [2]])
assert shape(matrix) == X.shape == (2, 2)
np.testing.assert_allclose(transpose(matrix), X.T)
np.testing.assert_allclose(matmul(matrix, B.tolist()), X @ B)
assert (X @ B).shape == (2, 1)
print("Week 4 and NumPy agree:", (X @ B).tolist())
```

```sh
uv run python test_code/week_05_practice.py
```

**Expected:** `Week 4 and NumPy agree: [[21.0], [23.0]]`. If multiplication fails, check `(2,2) @ (2,1) → (2,1)` and use @ rather than *.

Now append **the complete** weeks/week-05/fragments/01_linalg.py to src/mini_ml/linalg.py. Create src/mini_ml/preprocessing.py from **the complete** weeks/week-05/fragments/02_preprocessing.py. Preserve Week 4 functions and integrate each fragment only once.

**Source:** weeks/week-05/fragments/02_preprocessing.py — Standardizer and _matrix.

**Purpose:** Complete starter; reading copy of the fragment. Integrate the file once, then edit its existing methods.

**Destination:** src/mini_ml/preprocessing.py

```python
"""Training-only feature standardization with NumPy."""
import numpy as np


def _week05_todo(hint):
    """Replace the call with a NumPy expression."""
    raise NotImplementedError("TODO: " + hint)

def _matrix(X):
    array = np.asarray(X, dtype=float)
    if array.ndim != 2 or 0 in array.shape or not np.all(np.isfinite(array)):
        raise ValueError("X must be a nonempty finite (n, d) matrix")
    return array

class Standardizer:
    """fit: (n, d) -> self; transform: (m, d) -> ndarray (m, d).

    mean_, scale_: (d,); population std (ddof=0), zero scale -> 1.
    """

    def fit(self, X):
        """Learn per-feature parameters from training rows only."""
        array = _matrix(X)
        # BEGIN STUDENT: training mean and population scale
        self.mean_ = _week05_todo("feature means, axis=0")
        self.scale_ = _week05_todo("population scales, axis=0")
        self.scale_ = _week05_todo("zero-scale fallback")
        # END STUDENT
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X):
        """Apply saved parameters; preserve X and learned state."""
        if not hasattr(self, "n_features_in_"):
            raise ValueError("fit must be called before transform")
        array = _matrix(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("feature dimensions differ")
        # BEGIN STUDENT: broadcast saved training parameters
        return _week05_todo("broadcast z-score")
        # END STUDENT
```

```sh
uv run python -c "from mini_ml.preprocessing import Standardizer; from mini_ml.linalg import euclidean_distance; print('Imports passed')"
```

**Expected:** Imports passed; calling an unfinished exercise raises NotImplementedError with its hint. If import fails, check indentation and fragment destinations. Do not replace the entire linalg.py file.

## Step 4 — Understand axis, broadcasting, and saved state

**Directory:** package root. **Edit:** append the next practice block; read the existing Standardizer class.

`axis=0` combines rows and produces one value per feature. `(n,d) - (d,) → (n,d)` broadcasts the same feature parameters across all rows. See the [NumPy broadcasting guide](https://numpy.org/doc/stable/user/basics.broadcasting.html).

**Source:** Instructor-designed hand-calculation example; NumPy broadcasting reference linked above.

**Purpose:** Runnable practice; append.

**Destination:** test_code/week_05_practice.py

```python
column_means = X.mean(axis=0)
row_means = X.mean(axis=1)
assert column_means.shape == row_means.shape == (2,)
np.testing.assert_allclose(column_means, [2, 10])
np.testing.assert_allclose(row_means, [5.5, 6.5])
np.testing.assert_allclose(X - column_means, [[-1, 0], [1, 0]])
print("Column means:", column_means.tolist())
```

```sh
uv run python test_code/week_05_practice.py
```

**Expected:** Column means: [2.0, 10.0]. If values differ, inspect axis before shape: square matrices can hide an axis mistake.

`Standardizer` is a class; `scaler = Standardizer()` creates an instance. Inside a method, self is that instance. Calling fit stores attributes on it; transform reads them later. The trailing underscore marks a learned attribute, following sklearn convention. No constructor parameters are needed this week.

```text
scaler.fit(training X) → scaler.mean_, scaler.scale_, scaler.n_features_in_
scaler.transform(new X) → new standardized matrix
```

## Step 5 — Complete fit

**Directory:** package root. **Edit:** Standardizer.fit in src/mini_ml/preprocessing.py. Fill its three placeholders using feature-wise NumPy mean, std, and where.

For training `[[1,10],[3,10]]`, compute by hand:

| Feature | Mean | std(ddof=0) | Saved scale |
|---|---:|---:|---:|
| First | 2 | 1 | 1 |
| Second | 10 | 0 | 1 |

Population scale is `sqrt(sum((x−mean)²)/n)`. Week 3's default ddof=1 divides by n−1; this week's ddof=0 divides by n, matching [NumPy std](https://numpy.org/doc/stable/reference/generated/numpy.std.html) and [sklearn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). Store scale=1 when std=0 to avoid division by zero.

**Source:** Instructor-designed hand-calculation example; interface from weeks/week-05/fragments/02_preprocessing.py — Standardizer.fit.

**Purpose:** Runnable fit checkpoint; append after completing fit.

**Destination:** test_code/week_05_practice.py

```python
from mini_ml.preprocessing import Standardizer
x_train = [[1, 10], [3, 10]]
scaler = Standardizer()
assert scaler.fit(x_train) is scaler
np.testing.assert_allclose(scaler.mean_, [2, 10])
np.testing.assert_allclose(scaler.scale_, [1, 1])
assert scaler.n_features_in_ == 2
print("Fit means:", scaler.mean_.tolist(), "scales:", scaler.scale_.tolist())
```

```sh
uv run python test_code/week_05_practice.py
uv run pytest -q tests/test_week_05.py -k fit_by_feature
```

**Expected:** means [2.0, 10.0], scales [1.0, 1.0]; four fit cases pass. If not, check axis=0, ddof=0 and the zero fallback. fit returns self so `Standardizer().fit(X)` is usable.

## Step 6 — Complete transform

**Directory:** package root. **Edit:** Standardizer.transform in src/mini_ml/preprocessing.py.

Fill the broadcast expression for `Z = (X−mean_train)/scale_train`. Keep the supplied checks. All batches must retain training feature order. New samples can have a different row count; do not calculate new means/scales in transform.

**Source:** Instructor-designed hand-calculation example; weeks/week-05/fragments/02_preprocessing.py — Standardizer.transform.

**Purpose:** Runnable transform checkpoint; append.

**Destination:** test_code/week_05_practice.py

```python
Z = scaler.transform(x_train)
new_Z = scaler.transform([[5, 10], [5, 12]])
assert isinstance(Z, np.ndarray) and Z.shape == (2, 2)
np.testing.assert_allclose(Z, [[-1, 0], [1, 0]])
np.testing.assert_allclose(new_Z, [[3, 0], [3, 2]])
print("Training transformed:", Z.tolist())
print("New samples transformed:", new_Z.tolist())
```

```sh
uv run python test_code/week_05_practice.py
uv run pytest -q tests/test_week_05.py -k 'transform or mutation or unfitted or refit'
```

**Expected:** training [[-1,0],[1,0]]; new samples [[3,0],[3,2]]. A feature constant in training can change later: `(12−10)/1 = 2`. If a single new sample becomes zero, you probably refitted. If broadcasting fails, check the feature count.

## Step 7 — Complete vector distance

**Directory:** package root. **Edit:** euclidean_distance in src/mini_ml/linalg.py. Read this excerpt; the full fragment was already appended in Step 3.

**Source:** weeks/week-05/fragments/01_linalg.py — euclidean_distance.

**Purpose:** Reading excerpt of the integrated starter; do not paste again.

**Destination:** Existing euclidean_distance in src/mini_ml/linalg.py

```python
import numpy as np


def _week05_todo(hint):
    """Replace the call with a NumPy expression."""
    raise NotImplementedError("TODO: " + hint)

def euclidean_distance(x, y):
    """x, y: (d,), d >= 1, matching feature order/scale -> float.

    Example: [0, 0], [3, 4] -> 5.0.
    """
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 1 or y_array.ndim != 1 or x_array.size == 0 or x_array.shape != y_array.shape:
        raise ValueError("nonempty equal-length vectors required")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("finite vector values required")
    # BEGIN STUDENT: Euclidean distance
    return _week05_todo("Euclidean norm of the difference")
    # END STUDENT
```

Fill the norm of the difference: `sqrt(sum((xᵢ−yᵢ)²))`. Use element-wise subtraction and squaring, a sum reduction and a square root; return a Python float.

**Source:** Instructor-designed 3–4–5 hand-calculation example.

**Purpose:** Runnable distance checkpoint; append.

**Destination:** test_code/week_05_practice.py

```python
from mini_ml.linalg import euclidean_distance
assert euclidean_distance([0, 0], [3, 4]) == 5.0
assert euclidean_distance([3, 4], [0, 0]) == 5.0
assert euclidean_distance([3, 4], [3, 4]) == 0.0
try:
    euclidean_distance([0], [3, 4])
except ValueError:
    print("Length mismatch rejected")
else:
    raise AssertionError("Expected ValueError")
```

```sh
uv run python test_code/week_05_practice.py
uv run pytest -q tests/test_week_05.py -k distance
```

**Expected:** known distances pass and length mismatch is rejected. If the result is 25, check the square root. Passing [[0,0]] supplies a matrix rather than a vector; use [0,0].

## Step 8 — Predict the effect of scale

**Directory:** package root. **Edit:** append this synthetic experiment.

Fit [[0,0],[2,200]], which has scale [1,100]. Query [1,100]; A=[3,100], B=[1,200]. Predict the nearer point before running.

**Source:** weeks/week-05/demo.py — scale experiment; instructor-designed synthetic data.

**Purpose:** Runnable experiment; append.

**Destination:** test_code/week_05_practice.py

```python
scale_training = [[0, 0], [2, 200]]
query = [1, 100]
candidates = [[3, 100], [1, 200]]
scale_model = Standardizer().fit(scale_training)
query_scaled = scale_model.transform([query])[0]
candidates_scaled = scale_model.transform(candidates)
raw_distances = [euclidean_distance(query, point) for point in candidates]
scaled_distances = [euclidean_distance(query_scaled, point) for point in candidates_scaled]
assert raw_distances == [2, 100]
assert scaled_distances == [2, 1]
print(f"Candidate distances (A, B): raw={raw_distances}; standardized={scaled_distances}")
print("Nearest candidate: raw=A; standardized=B")
```

```sh
uv run python test_code/week_05_practice.py
```

**Expected:** raw distances [2,100], standardized [2,1]; nearer point changes A → B. If not, ensure the same fitted instance transforms the query and both candidates.

## Step 9 — Plot the patient distribution

**Directory:** package root. **Edit:** read test_code/week_05.py; optionally customize its supplied plotting section after the first successful run.

```sh
uv run python test_code/week_05.py
```

**Source:** weeks/week-05/demo.py — WDBC training preparation, StandardScaler comparison and plotting loop. The complete runnable file is synced to test_code/week_05.py. The CSV is the same teacher-provided training partition used in Week 4; original dataset: [UCI WDBC](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

**Expected:** training and transformed shape (341,2); first-pair distances raw=9.2188, standardized=2.3617; StandardScaler reference checks passed; plot saved as test_code/week_05_scaling.png. The measured training parameters are mean=[14.1634,19.4288], scale=[3.5940,4.3548]. The first three standardized rows, rounded to four decimals, are [[0.1938,0.8683],[-1.6604,-0.5945],[-0.8190,0.0990]]. A headless terminal can open the saved PNG without displaying a window.

The plot uses benign=0 and malignant=1 for color only. Inspect both axes and the legend. Equal-aspect coordinates show relative scale. Standardization changes geometry; it does not guarantee separate clusters or produce a classifier.

If the CSV is missing, check data/week-05/train.csv and sync. If sklearn differs, inspect feature order and ddof. If the plot looks like one color, inspect the label masks. Do not fit separately by diagnosis.

**Looking ahead:** W6 studies variance directions; W7 projects and reconstructs; W8 learns PCA using the same fit/transform idea; W9 learns classifier parameters with gradient descent. PCA labels are used for coloring and classification, not for fitting PCA.

## Step 10 — Verify and submit

**Directory:** package root. **Edit:** add one original distance or standardizer test to tests/test_week_05.py, preserving supplied cases.

**Source:** weeks/week-05/student_tests/test_week_05.py — public contract tests. The file is installed by sync and should be submitted with your implementation. Use its parametrized examples as the pattern for your additional case.

```sh
uv run pytest -q tests/test_week_05.py
uv run pytest -q
uv run python ../../tools/course.py test <login> 5
git status --short
```

**Expected:** all supplied, cumulative and course acceptance tests pass. Measured reference-package counts — supplied W5: 31 passed; cumulative student count: 89 passed; trusted acceptance count: 38 passed (before your W5 extra case; additional tests from earlier weeks can increase the cumulative total). If a previous test fails, check that Step 3 preserved the earlier implementations. Unfinished placeholders fail explicitly.

```sh
git add src/mini_ml/linalg.py src/mini_ml/preprocessing.py tests/test_week_05.py pyproject.toml uv.lock
git commit -m "Implement NumPy standardization and visualize training patients"
git push -u origin <login>/week-05
```

Open a PR to main. Describe input/output shapes, the saved parameters, the raw/scaled nearest candidate, and what overlap remains in the patient plot. Include commands and passing results. test_code/ stays local and ignored; do not commit data obtained outside sync or .venv. If push fails, check the branch and GitHub authorization. If test_code appears in git status, restore its ignore rule before committing.
