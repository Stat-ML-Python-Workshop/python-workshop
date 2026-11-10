# Cumulative Package Architecture

Weeks 1–9 have released lesson materials; reference solutions are available for Weeks 1–6. Week 10 paths below are planned contracts; implement them only when that lesson is released. Start the current classifier lab with the [Week 9 lesson](weeks/week-09/lesson.md).

| Milestone | Package change |
|---|---|
| 1 | greetings.py and public __init__.py |
| 2–3 | statistics.py, probability.py, information.py; append rather than replace |
| 4 | linalg.py with pure-Python shape, transpose and matmul |
| 5 | append NumPy euclidean_distance to linalg.py; create preprocessing.py with Standardizer |
| 6 | create decomposition.py: center, covariance_matrix, variance_along_direction; supplied Python explorer |
| 7 | append project, reconstruct and reconstruction_error to decomposition.py; supplied fixed-basis image and WDBC explorers |
| 8 | append fitted PCA object to decomposition.py; ten expressions include covariance and NumPy eigh, with supplied state handling and randomized SVD helper |
| 9 | models/__init__.py, models/linear_models.py and supplied models/_training_utils.py; five tensor softmax/forward expressions with supplied PyTorch autograd/SGD |
| 10 (planned) | New models/manual_linear_models.py: ManualLogisticClassifier reuses Week 9 forward and replaces only gradients/updates with four expressions |

PCA and earlier numerical modules depend on NumPy and earlier helpers. The classifier uses required PyTorch CPU float64 tensors. Its forward method returns scores and probabilities; prediction and saved-model attributes expose NumPy arrays. The application handles data, references, plotting and serialization. The planned optional Week 10 vibe coding scanning app receives an already fitted pipeline and shared image preparation; it never fits preprocessing or PCA. No math/ migration or parallel numpy_backend rewrite is required.

Stable object contracts: Standardizer.fit/transform; PCA.fit/transform/inverse_transform; LogisticClassifier.fit/predict_proba/predict. Learned attributes end in _. Numeric transforms return ndarrays. Classifier probability columns match classes_. Preserve all earlier functions and tests.

Formal source stays in src/mini_ml/, durable unit tests in tests/, and ignored local experiments in test_code/. Sync supporting materials with tools/course.py and integrate the starter fragments manually. course.json controls which weekly materials are released.
