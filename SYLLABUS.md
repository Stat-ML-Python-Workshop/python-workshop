# Python Workshop: Build Machine Learning from Scratch

This workshop is designed for Python beginners with backgrounds in life sciences or biotechnology. It runs for 10 weeks, with one 90-minute session per week, alternating between short explanations and immediate hands-on practice.
Students progress from mathematical formulas and pure Python implementations to NumPy matrix operations, a hand-built PCA algorithm, fixed-step logistic regression, and a guided translation of the same model to PyTorch.
Each student maintains one cumulative package. Exploratory work stays in a local `test_code/` directory, while formal implementations, unit tests, and the final example are submitted through pull requests.

This table is the published student-facing summary. The source of truth for the executable course is the instructor repository's `tools/build_course.py`, `tools/course_sources.py`, and `tools/build_tests.py`.

| Week | Core Topics | From-Scratch Focus | Weekly Outcome |
| --- | --- | --- | --- |
| 1 | uv, Python, packages, Git, CI | `hello_world`, public API, unit tests | An editable-installable package that can be imported externally, plus the first pull request |
| 2 | `if`, `for`, counting, probability, Shannon entropy | `class_proportions`, `shannon_entropy` | Analyze immune-cell composition and diversity in bits |
| 3 | `def`, variance, standard deviation, distribution comparison | `variance`, `std`, `cross_entropy`, `kl_divergence`; explicit loops for aggregations | Estimate variation and compare treatment and reference cell-type distributions |
| 4 | vectors, shapes, transposition, matrix multiplication | `shape`, `transpose`, `matmul(m1, m2)` in pure Python | Build matrix multiplication with nested loops and calculate patient scores |
| 5 | visualization, z-scores, Euclidean distance, data leakage | `euclidean_distance`, `fit_standardizer`, `transform_standardizer` | Standardize two-feature training data and compare distances |
| 6 | NumPy arrays, axes, broadcasting, matrix multiplication | Reimplement dot products and standardization with NumPy; compare pure Python `matmul` with `@` | Express loop-based calculations as vectorized matrix operations |
| 7 | 30-dimensional biomedical data, covariance, projection | Center matrices, calculate covariance with `X.T @ X`, and project onto a direction | Move from two visible features to the complete 30-feature dataset |
| 8 | PCA, eigenvectors, explained variance, dimensionality reduction | `fit_pca`, `transform_pca`; use `np.linalg.eigh` for the eigensolver | Implement PCA and project 30 features into a smaller feature space without leakage |
| 9 | sigmoid, binary cross-entropy, fixed-step gradient descent | Stable sigmoid and NumPy `LogisticRegressor` with manual gradients | Train logistic regression on PCA scores and inspect its loss curve |
| 10 | PyTorch, autograd, SGD, Adam, model evaluation | Guided translation of the NumPy model using `nn.Linear` and `BCEWithLogitsLoss` | Compare manual NumPy training with PyTorch SGD and Adam on a held-out test set |

Students submit one pull request per week. The instructor manually publishes solutions on Thursday evening. Work is considered complete when the latest CI run passes by Friday; the instructor reviews and merges it separately.
If the previous week's pull request has not been merged, students should ask the instructor for help rather than maintain dependent pull requests.
Weeks 4–10 use one fixed Wisconsin Diagnostic Breast Cancer train/validation/test split with `1 = malignant`. All preprocessing and PCA state must be fitted on the training partition and then reused for validation and test data.
Each week requires only one or two core implementation tasks. In Week 8, students implement the PCA workflow around `np.linalg.eigh`; implementing an eigensolver is outside the core requirement. Week 10 is an instructor-guided PyTorch rewrite with a small student-completed training loop.
Optional extensions include power iteration, additional biomedical datasets such as Cell Painting or BBBC, and further optimizer experiments.
