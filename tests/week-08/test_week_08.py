"""Independent W8 acceptance: student code runs only in the isolated worker."""
import numpy as np
import pytest


ATTRIBUTES = ["mean_", "components_", "explained_variance_",
              "explained_variance_ratio_", "n_features_in_", "n_samples_"]
TRAIN = [[-3., 0.], [3., 0.], [0., -1.], [0., 1.]]


def sequence(invoke, k, steps, **parameters):
    response = invoke("decomposition", operation="class_sequence", **{
        "class": "PCA", "init": {"n_components": k, **parameters},
        "attributes": ATTRIBUTES, "steps": steps})
    assert "error" not in response, response
    return response["value"]["steps"]


def step(method, argument):
    return {"method": method, "args": [argument]}


def test_pca_known_training_state_and_total_variance(invoke):
    result, projected, rebuilt = sequence(invoke, 1, [
        step("fit", TRAIN), step("transform", [[6., 5.]]),
        step("inverse_transform", {"result": 1})])
    assert result["is_self"] and all(s["input_unchanged"] for s in (result, projected, rebuilt))
    state = result["attributes"]
    assert state["mean_"] == [0., 0.] and state["n_features_in_"] == 2 and state["n_samples_"] == 4
    np.testing.assert_allclose(state["explained_variance_"], [6.])
    np.testing.assert_allclose(state["explained_variance_ratio_"], [.9])
    assert projected["dtype"] == rebuilt["dtype"] == "float64"
    assert projected["type"] == rebuilt["type"] == "ndarray"
    assert projected["shape"] == [1, 1] and rebuilt["shape"] == [1, 2]
    np.testing.assert_allclose(rebuilt["value"], [[6., 0.]], atol=1e-12)


def test_pca_correlated_covariance_eigenpairs_and_reconstruction(invoke):
    Xc = np.array([[2., 2.], [-2., -2.], [1., -1.], [-1., 1.]])
    mean = np.array([11., -7.])
    X = Xc + mean
    C = np.array([[10 / 3, 2.], [2., 10 / 3]])
    results = sequence(invoke, 1, [step("fit", X.tolist()), step("transform", X.tolist()),
                                   step("inverse_transform", {"result": 1})], solver="eigh")
    state = results[0]["attributes"]
    component = np.asarray(state["components_"])
    np.testing.assert_allclose(state["mean_"], mean, atol=1e-12)
    np.testing.assert_allclose(state["explained_variance_"], [16 / 3], atol=1e-12)
    np.testing.assert_allclose(state["explained_variance_ratio_"], [.8], atol=1e-12)
    np.testing.assert_allclose(C @ component.T, (16 / 3) * component.T, atol=1e-12)
    np.testing.assert_allclose(component.T @ component, np.full((2, 2), .5), atol=1e-12)
    expected = np.array([[2., 2.], [-2., -2.], [0., 0.], [0., 0.]]) + mean
    np.testing.assert_allclose(results[2]["value"], expected, atol=1e-12)
    assert np.mean((X - results[2]["value"]) ** 2) == pytest.approx(.5)
    assert all(item["input_unchanged"] for item in results)


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("solver", ["auto", "eigh", "randomized"])
def test_pca_rotated_distinct_spectrum_reference(invoke, k, solver):
    X = np.array([[7., 1., -2.], [2., 5., 4.], [-3., 2., 1.], [4., -2., 8.], [0., 1., 3.]])
    query = np.array([[10., -3., 7.], [-8., 6., 5.]])
    mean = X.mean(axis=0)
    _, singular, basis = np.linalg.svd(X - mean, full_matrices=False)
    variances = singular ** 2 / (len(X) - 1)
    expected_projector = basis[:k].T @ basis[:k]
    results = sequence(invoke, k, [step("fit", X.tolist()), step("transform", query.tolist()),
                                  step("inverse_transform", {"result": 1})], solver=solver)
    state = results[0]["attributes"]
    components = np.asarray(state["components_"])
    assert components.shape == (k, 3)
    np.testing.assert_allclose(components @ components.T, np.eye(k), atol=1e-10)
    np.testing.assert_allclose(components.T @ components, expected_projector, atol=1e-10)
    np.testing.assert_allclose(state["explained_variance_"], variances[:k], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(state["explained_variance_ratio_"], variances[:k] / variances.sum(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(results[1]["value"], (query - mean) @ components.T, atol=1e-10)
    np.testing.assert_allclose(results[2]["value"], (query - mean) @ expected_projector + mean, atol=1e-10)
    assert all(item["input_unchanged"] for item in results)


def test_pca_batch_independence_and_training_mean(invoke):
    shifted = (np.asarray(TRAIN) + [11., 23.]).tolist()
    results = sequence(invoke, 1, [step("fit", shifted), step("transform", [[21., 30.]]),
        step("transform", [[21., 30.], [-1000., 5000.]]), step("inverse_transform", [[0.]])])
    np.testing.assert_allclose(results[1]["value"], np.asarray(results[2]["value"])[:1], atol=1e-12)
    np.testing.assert_allclose(np.abs(results[1]["value"]), [[10.]])
    np.testing.assert_allclose(results[3]["value"], [[11., 23.]])
    assert results[0]["attributes"] == results[3]["attributes"]


def test_pca_zero_variance_and_single_feature(invoke):
    constant = sequence(invoke, 2, [step("fit", [[7., -2., 3.]] * 4),
        step("transform", [[7., -2., 3.]]), step("inverse_transform", {"result": 1})])
    state = constant[0]["attributes"]
    assert state["explained_variance_"] == state["explained_variance_ratio_"] == [0., 0.]
    np.testing.assert_allclose(np.asarray(state["components_"]) @ np.asarray(state["components_"]).T, np.eye(2), atol=1e-12)
    assert constant[1]["value"] == [[0., 0.]] and constant[2]["value"] == [[7., -2., 3.]]
    single = sequence(invoke, 1, [step("fit", [["1"], ["3"], ["5"]])])[0]["attributes"]
    assert single["mean_"] == [3.] and single["explained_variance_"] == [4.]
    assert single["explained_variance_ratio_"] == [1.]


def test_pca_tied_cutoff_checks_metric_without_assuming_unique_basis(invoke):
    X = [[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]
    results = sequence(invoke, 1, [step("fit", X), step("transform", X), step("inverse_transform", {"result": 1})])
    state = results[0]["attributes"]
    np.testing.assert_allclose(state["explained_variance_"], [2 / 3])
    np.testing.assert_allclose(state["explained_variance_ratio_"], [.5])
    assert np.linalg.norm(state["components_"]) == pytest.approx(1.)
    assert np.mean((np.asarray(X) - results[2]["value"]) ** 2) == pytest.approx(.25)


def test_pca_refit_success_and_failure_are_state_consistent(invoke):
    replacement = [[2., 10., 4.], [5., 12., 9.], [8., 11., 3.]]
    results = sequence(invoke, 1, [step("fit", TRAIN), step("transform", [[8., 9.]]),
        step("fit", [[0., 0.]]), step("transform", [[8., 9.]]),
        step("fit", replacement), step("transform", [[2., 10., 4.]]),
        step("inverse_transform", {"result": 5}), step("transform", [[8., 9.]])])
    assert results[2]["error"] == results[7]["error"] == "ValueError"
    assert results[0]["attributes"] == results[2]["attributes"] == results[3]["attributes"]
    assert results[1]["value"] == results[3]["value"]
    assert results[4]["is_self"]
    state = results[4]["attributes"]
    assert state["n_features_in_"] == state["n_samples_"] == 3
    np.testing.assert_allclose(state["mean_"], np.mean(replacement, axis=0))
    assert results[5]["shape"] == [1, 1] and results[6]["shape"] == [1, 3]
    assert all(item["input_unchanged"] for item in results)


@pytest.mark.parametrize("parameters", [
    {"n_components": 0}, {"n_components": -1}, {"n_components": True},
    {"n_components": 1.0}, {"n_components": "1"}, {"n_components": None},
    {"n_components": 1, "solver": "svd"}, {"n_components": 1, "solver": None},
    {"n_components": 1, "random_state": -1}, {"n_components": 1, "random_state": True},
])
def test_pca_invalid_constructor_parameters(invoke, parameters):
    response = invoke("decomposition", **{"class": "PCA", "init": parameters,
        "fit": [TRAIN], "method": "transform", "predict": TRAIN})
    assert response["error"] == "ValueError"


@pytest.mark.parametrize("train,k", [
    ([], 1), ([[]], 1), ([1, 2], 1), ([[1, 2]], 1), ([[1], [2, 3]], 1),
    ([[1], [float("nan")]], 1), ([[1], [float("inf")]], 1),
    ([[1], [2], [3]], 2), ([[1, 2, 3], [4, 5, 6]], 2),
])
def test_pca_invalid_training_data(invoke, train, k):
    result = sequence(invoke, k, [step("fit", train)])[0]
    assert result["error"] == "ValueError" and result["input_unchanged"]


def test_pca_unfitted_and_bad_transform_inputs(invoke):
    before = sequence(invoke, 1, [step("transform", [[1, 2]]), step("inverse_transform", [[1]])])
    assert all(item["error"] == "ValueError" for item in before)
    results = sequence(invoke, 1, [step("fit", TRAIN), step("transform", [[1]]),
        step("transform", []), step("transform", [[float("nan"), 0]]),
        step("inverse_transform", [[1, 2]]), step("inverse_transform", []),
        step("inverse_transform", [[float("inf")]])])
    assert all(item["error"] == "ValueError" and item["input_unchanged"] for item in results[1:])


def test_pca_randomized_low_rank_reference(invoke):
    rng = np.random.default_rng(352)
    X = rng.normal(size=(80, 12)) @ rng.normal(size=(12, 40)) + rng.normal(size=40)
    _, singular, axes = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    results = sequence(invoke, 3, [step("fit", X.tolist()), step("transform", X[:2].tolist()),
        step("fit", X.tolist()), step("transform", X[:2].tolist())], solver="randomized")
    state = results[0]["attributes"]
    np.testing.assert_allclose(state["explained_variance_"], singular[:3] ** 2 / 79, rtol=1e-10)
    np.testing.assert_allclose(state["explained_variance_ratio_"], singular[:3] ** 2 / np.sum(singular ** 2), rtol=1e-10)
    components = np.asarray(state["components_"])
    np.testing.assert_allclose(components.T @ components, axes[:3].T @ axes[:3], atol=1e-10)
    assert state == results[2]["attributes"] and results[1]["value"] == results[3]["value"]
