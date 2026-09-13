# Python Workshop: Build Machine Learning from Scratch

This workshop is designed for Python beginners with backgrounds in life sciences or biotechnology. It runs for 10 weeks, with one 90-minute session per week, alternating between short explanations and immediate hands-on practice.
Students progress from mathematical formulas to pure Python implementations and then compare their work with NumPy and scikit-learn, while maintaining a loose connection to the main course.
Each student maintains a cumulative package and refactors its architecture in Week 8. Exploratory work stays in a local `test_code/` directory, while formal implementations, unit tests, and the final example are submitted through pull requests.

This table is the published student-facing summary. The source of truth for the executable course is the instructor repository's `tools/build_course.py`, `tools/course_sources.py`, and `tools/build_tests.py`.

| Week | Core Topics | From-Scratch Focus | Weekly Outcome |
| --- | --- | --- | --- |
| 1 | uv, Python, packages, Git, CI | `hello_world`, public API, unit tests | An editable-installable package that can be imported externally, plus the first pull request |
| 2 | `if`, `for`, counting, probability, Shannon entropy | `class_proportions`, `shannon_entropy` | Analyze immune-cell composition and diversity in bits |
| 3 | `def`, distribution comparison, cross-entropy, KL divergence | `cross_entropy`, `kl_divergence`; descriptive statistics provided by the instructor | Compare treatment and reference cell-type distributions |
| 4 | vectors, shapes, NumPy, logits, softmax | `dot`, `stable_softmax`; instructor-guided `matvec` | Convert gene-expression scores into probabilities |
| 5 | visualization, z-scores, Euclidean distance | `euclidean_distance`, `transform_standardizer` | Compare distances and visualizations before and after standardizing the training data |
| 6 | loss, prediction metrics, unit tests | `mse`; instructor-guided `accuracy` | Evaluate model error with reproducible tests |
| 7 | k-NN, training/validation splits, baselines | Neighbor selection and voting | Build a first classifier |
| 8 | refactoring, OOP, gradient descent | Move code into `math/` and `models/`; implement gradient updates | Preserve previous behavior while adding `LinearRegressor` |
| 9 | sigmoid, BCE, training loops | Add logistic regression to the same `linear_models.py` | Implement `LogisticRegressor` and create a version tag |
| 10 | biological application, evaluation | Assemble a complete model workflow | Create `examples/breast_cancer.py`, compare results, and update the README |

Students submit one pull request per week. The instructor manually publishes solutions on Thursday evening. Work is considered complete when the latest CI run passes by Friday; the instructor reviews and merges it separately.
If the previous week's pull request has not been merged, students should ask the instructor for help rather than maintain dependent pull requests.
The Week 8 refactor is instructor-guided, with additional scaffolding provided for linear regression. Each week requires only one or two core implementation tasks.
PyTorch, general matrix multiplication, and additional Cell Painting or BBBC projects are optional extensions.
