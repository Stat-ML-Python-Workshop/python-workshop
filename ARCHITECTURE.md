# Cumulative Package Architecture

Weeks 1–6 have released lesson materials. Weeks 7–10 paths below are planned contracts; implement them only when the corresponding lesson is released.

| Milestone | Package change |
|---|---|
| 1 | greetings.py and public __init__.py |
| 2–3 | statistics.py, probability.py, information.py; append rather than replace |
| 4 | linalg.py with pure-Python shape, transpose and matmul |
| 5 | append NumPy euclidean_distance to linalg.py; create preprocessing.py with Standardizer |
| 6 | create decomposition.py: center, covariance_matrix, variance_along_direction; supplied Python explorer |
| 7–8 (draft) | append projection/reconstruction, then PCA to decomposition.py |
| 9 (draft) | models/__init__.py and models/linear_models.py with multiclass LogisticClassifier |
| 10 (draft) | scanning.py plus examples/pathology.py integration |

Numerical modules depend on NumPy and the earlier numerical helpers, not models, images, sklearn or datasets. The application handles data, references, plotting and serialization. Scanning receives an already fitted pipeline and shared image preparation; it never fits preprocessing or PCA. No math/ migration or parallel numpy_backend rewrite is required.

Stable object contracts: Standardizer.fit/transform; PCA.fit/transform/inverse_transform; LogisticClassifier.fit/predict_proba/predict. Learned attributes end in _. Numeric transforms return ndarrays. Classifier probability columns match classes_. Preserve all earlier functions and tests.

Formal source stays in src/mini_ml/, durable unit tests in tests/, and ignored local experiments in test_code/. Sync supporting materials with tools/course.py and integrate the starter fragments manually. course.json controls which weekly materials are released.
