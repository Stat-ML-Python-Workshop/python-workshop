"""Teacher-provided W8 tests. Run unchanged; no original cases are required."""
import numpy as np
import pytest
from sklearn.decomposition import PCA as ReferencePCA
from mini_ml import decomposition as dec


TRAIN = np.array([[-3., 0.], [3., 0.], [0., -1.], [0., 1.]])
ATTRIBUTES = ("mean_", "components_", "explained_variance_",
              "explained_variance_ratio_", "n_features_in_", "n_samples_")


def snapshot(model):
    return {key: getattr(model, key).copy() if isinstance(getattr(model, key), np.ndarray)
            else getattr(model, key) for key in ATTRIBUTES}


def assert_state_equal(model, saved):
    for key, value in saved.items():
        np.testing.assert_array_equal(getattr(model, key), value)


def test_fit_hand_values_sample_denominator_and_ratio():
    model = dec.PCA(1)
    assert model.fit(TRAIN) is model
    np.testing.assert_allclose(model.mean_, [0, 0], atol=1e-12)
    np.testing.assert_allclose(model.components_.T @ model.components_, [[1, 0], [0, 0]], atol=1e-12)
    np.testing.assert_allclose(model.explained_variance_, [6.])
    np.testing.assert_allclose(model.explained_variance_ratio_, [.9])
    assert model.n_features_in_ == 2 and model.n_samples_ == 4
    for name, shape in (("mean_", (2,)), ("components_", (1, 2)),
                        ("explained_variance_", (1,)), ("explained_variance_ratio_", (1,))):
        value = getattr(model, name)
        assert isinstance(value, np.ndarray) and value.dtype == np.float64 and value.shape == shape


@pytest.mark.parametrize("solver", ["auto", "eigh"])
def test_fit_calls_eigh_on_centered_sample_covariance(monkeypatch, solver):
    centered = np.array([[2., 2.], [-2., -2.], [1., -1.], [-1., 1.]])
    X = centered + [11., -7.]
    expected_covariance = np.array([[10 / 3, 2.], [2., 10 / 3]])
    calls = []
    covariance, eigh = dec.covariance_matrix, np.linalg.eigh
    def track_covariance(Xc):
        calls.append("covariance")
        np.testing.assert_array_equal(Xc, centered)
        return covariance(Xc)
    def track_eigh(C):
        calls.append("eigh")
        np.testing.assert_allclose(C, expected_covariance, atol=1e-12)
        values, axes = eigh(C)
        np.testing.assert_allclose(values, [4 / 3, 16 / 3], atol=1e-12)
        np.testing.assert_allclose(C @ axes, axes * values, atol=1e-12)
        return values, axes
    monkeypatch.setattr(dec, "covariance_matrix", track_covariance)
    monkeypatch.setattr(np.linalg, "eigh", track_eigh)
    model = dec.PCA(1, solver=solver).fit(X)
    assert calls == ["covariance", "eigh"]
    np.testing.assert_array_equal(X, centered + [11., -7.])
    np.testing.assert_allclose(model.mean_, [11., -7.])
    np.testing.assert_allclose(model.explained_variance_, [16 / 3])
    np.testing.assert_allclose(model.explained_variance_ratio_, [.8])


def test_correlated_hand_example_eigenvalues_scores_and_reconstruction():
    X = np.array([[2., 2.], [-2., -2.], [1., -1.], [-1., 1.]])
    model = dec.PCA(1).fit(X)
    np.testing.assert_allclose(model.components_.T @ model.components_, np.full((2, 2), .5), atol=1e-12)
    np.testing.assert_allclose(np.abs(model.transform(X)).ravel(), [2 * np.sqrt(2), 2 * np.sqrt(2), 0, 0], atol=1e-12)
    rebuilt = model.inverse_transform(model.transform(X))
    np.testing.assert_allclose(rebuilt, [[2, 2], [-2, -2], [0, 0], [0, 0]], atol=1e-12)
    assert np.mean((X - rebuilt) ** 2) == pytest.approx(.5)


@pytest.mark.parametrize("solver", ["auto", "eigh", "randomized"])
def test_fit_matches_reference_subspace_and_variance(solver):
    X = np.array([[4., 1., 2.], [-1., -3., 1.], [2., 5., 8.], [-3., 2., -2.], [1., 0., 4.]])
    actual = dec.PCA(2, solver=solver).fit(X)
    reference = ReferencePCA(2, svd_solver="full").fit(X)
    np.testing.assert_allclose(actual.explained_variance_, reference.explained_variance_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual.explained_variance_ratio_, reference.explained_variance_ratio_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual.components_.T @ actual.components_,
                               reference.components_.T @ reference.components_, atol=1e-10)
    np.testing.assert_allclose(actual.components_ @ actual.components_.T, np.eye(2), atol=1e-12)
    assert np.all(np.diff(actual.explained_variance_) <= 0)


def test_fit_numeric_strings_and_single_feature():
    model = dec.PCA(np.int64(1), random_state=np.int64(42)).fit([["1"], ["3"], ["5"]])
    np.testing.assert_allclose(model.mean_, [3.])
    np.testing.assert_allclose(model.explained_variance_, [4.])
    np.testing.assert_allclose(model.explained_variance_ratio_, [1.])


@pytest.mark.parametrize("k", [0, -1, 1.5, 2.0, True, np.bool_(False), "2", None])
def test_fit_invalid_component_parameter(k):
    with pytest.raises(ValueError):
        dec.PCA(k)


@pytest.mark.parametrize("solver", ["full", "svd", "", None, 1])
def test_fit_invalid_solver(solver):
    with pytest.raises(ValueError):
        dec.PCA(1, solver=solver)


@pytest.mark.parametrize("seed", [-1, 1.5, True, None, "42"])
def test_fit_invalid_seed(seed):
    with pytest.raises(ValueError):
        dec.PCA(1, random_state=seed)


@pytest.mark.parametrize("X,k", [
    ([], 1), ([[]], 1), ([1, 2], 1), ([[1, 2]], 1),
    ([[1, 2], [3]], 1), ([[1], [np.nan]], 1), ([[1], [np.inf]], 1),
    ([[1], [2], [3]], 2), ([[1, 2, 3], [4, 5, 6]], 2),
])
def test_fit_invalid_training_shape_and_component_bound(X, k):
    with pytest.raises(ValueError):
        dec.PCA(k).fit(X)


@pytest.mark.parametrize("solver", ["auto", "eigh", "randomized"])
def test_fit_constant_data_zero_ratio(solver):
    X = np.tile([2., -3., 9., 0.], (6, 1))
    model = dec.PCA(3, solver=solver).fit(X)
    np.testing.assert_array_equal(model.explained_variance_, np.zeros(3))
    np.testing.assert_array_equal(model.explained_variance_ratio_, np.zeros(3))
    np.testing.assert_allclose(model.components_ @ model.components_.T, np.eye(3), atol=1e-12)
    np.testing.assert_array_equal(model.transform(X), np.zeros((6, 3)))
    np.testing.assert_allclose(model.inverse_transform(model.transform(X)), X)


def test_fit_repeated_eigenvalue_cutoff_compares_metrics_not_axes():
    X = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
    model = dec.PCA(1).fit(X)
    np.testing.assert_allclose(model.explained_variance_, [2 / 3])
    np.testing.assert_allclose(model.explained_variance_ratio_, [.5])
    np.testing.assert_allclose(model.components_ @ model.components_.T, [[1.]], atol=1e-12)
    reconstructed = model.inverse_transform(model.transform(X))
    assert np.mean((X - reconstructed) ** 2) == pytest.approx(.25)


def test_transform_saved_mean_and_batch_independence():
    X = TRAIN + [10., 20.]
    model = dec.PCA(1).fit(X)
    query = np.array([[17., 22.], [-100., 1000.]])
    saved = snapshot(model)
    actual = model.transform(query)
    assert actual.dtype == np.float64 and actual.shape == (2, 1)
    np.testing.assert_allclose(actual, (query - model.mean_) @ model.components_.T)
    np.testing.assert_allclose(model.transform(query[:1]), actual[:1])
    assert_state_equal(model, saved)


def test_inverse_restores_training_origin_and_full_basis():
    X = TRAIN + [10., 20.]
    model = dec.PCA(2).fit(X)
    Z = model.transform(X)
    restored = model.inverse_transform(Z)
    assert restored.dtype == np.float64 and restored.shape == X.shape
    np.testing.assert_allclose(restored, X, atol=1e-12)
    np.testing.assert_allclose(model.inverse_transform([[0., 0.]]), model.mean_[None, :])


@pytest.mark.parametrize("method,values", [("transform", [[1, 2]]), ("inverse_transform", [[1]])])
def test_transform_and_inverse_require_fit(method, values):
    with pytest.raises(ValueError):
        getattr(dec.PCA(1), method)(values)


@pytest.mark.parametrize("method,values", [
    ("transform", []), ("transform", [1, 2]), ("transform", [[1]]),
    ("transform", [[np.nan, 0]]), ("transform", [[1, np.inf]]),
    ("inverse_transform", []), ("inverse_transform", [1]),
    ("inverse_transform", [[1, 2]]), ("inverse_transform", [[np.inf]]),
])
def test_transform_and_inverse_invalid_inputs(method, values):
    with pytest.raises(ValueError):
        getattr(dec.PCA(1).fit(TRAIN), method)(values)


def test_fit_transform_inverse_preserve_inputs_and_state_storage():
    X = TRAIN.copy()
    original = X.copy()
    model = dec.PCA(1).fit(X)
    saved = snapshot(model)
    np.testing.assert_array_equal(X, original)
    X[:] = 999
    assert_state_equal(model, saved)
    query = np.array([[10., 20.], [30., 40.]])
    before_query = query.copy()
    Z = model.transform(query)
    before_Z = Z.copy()
    reconstructed = model.inverse_transform(Z)
    np.testing.assert_array_equal(query, before_query)
    np.testing.assert_array_equal(Z, before_Z)
    reconstructed[:] = 0
    Z[:] = 0
    assert_state_equal(model, saved)


def test_refit_replaces_state_and_failed_refit_preserves_it():
    model = dec.PCA(1).fit(TRAIN)
    replacement = np.array([[10., 20., 30.], [11., 22., 33.], [13., 21., 35.]])
    assert model.fit(replacement) is model
    assert model.n_features_in_ == 3 and model.n_samples_ == 3
    np.testing.assert_allclose(model.mean_, replacement.mean(axis=0))
    saved = snapshot(model)
    expected = model.transform(replacement[:1])
    for invalid in ([[0., 0., 0.]], [[np.nan, 0., 0.], [1., 2., 3.]]):
        with pytest.raises(ValueError):
            model.fit(invalid)
        assert_state_equal(model, saved)
        np.testing.assert_array_equal(model.transform(replacement[:1]), expected)
    with pytest.raises(ValueError):
        model.transform([[1., 2.]])


def test_transform_and_inverse_reuse_week7_functions(monkeypatch):
    model = dec.PCA(1).fit(TRAIN)
    calls = []
    project, reconstruct = dec.project, dec.reconstruct
    def track_project(X, mean, components):
        calls.append("project")
        return project(X, mean, components)
    def track_reconstruct(Z, mean, components):
        calls.append("reconstruct")
        return reconstruct(Z, mean, components)
    monkeypatch.setattr(dec, "project", track_project)
    monkeypatch.setattr(dec, "reconstruct", track_reconstruct)
    model.inverse_transform(model.transform(TRAIN))
    assert calls == ["project", "reconstruct"]


def test_randomized_low_rank_reference_and_seed_reproducibility():
    rng = np.random.default_rng(123)
    # k+10=13 < min(80,40); rank12 exercises the real randomized path.
    X = rng.normal(size=(80, 12)) @ rng.normal(size=(12, 40)) + rng.normal(size=40)
    model = dec.PCA(3, solver="randomized", random_state=42).fit(X)
    repeat = dec.PCA(3, solver="randomized", random_state=42).fit(X)
    reference = ReferencePCA(3, svd_solver="full").fit(X)
    np.testing.assert_allclose(model.explained_variance_, reference.explained_variance_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(model.explained_variance_ratio_, reference.explained_variance_ratio_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(model.components_.T @ model.components_, reference.components_.T @ reference.components_, atol=1e-10)
    np.testing.assert_array_equal(model.components_, repeat.components_)
    np.testing.assert_allclose(dec.PCA(3).fit(X).components_, model.components_, atol=1e-12)


def test_randomized_exact_fallback_matches_reference():
    X = np.random.default_rng(17).normal(size=(12, 35))
    model = dec.PCA(3, solver="randomized").fit(X)
    reference = ReferencePCA(3, svd_solver="full").fit(X)
    np.testing.assert_allclose(model.explained_variance_, reference.explained_variance_, rtol=1e-10)
    np.testing.assert_allclose(model.components_.T @ model.components_, reference.components_.T @ reference.components_, atol=1e-10)
