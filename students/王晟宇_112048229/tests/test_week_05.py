"""Public Week 5 contracts: NumPy arrays and training-only state."""
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from mini_ml.linalg import euclidean_distance
from mini_ml.preprocessing import Standardizer

@pytest.mark.parametrize("x,y,expected", [
    ([0,0],[3,4],5), ([2,-1],[2,-1],0), ([0.1,0.2],[0.4,0.6],0.5),
    (np.array([1,2,3]),np.array([4,2,-1]),5),
])
def test_distance(x,y,expected):
    result = euclidean_distance(x,y)
    assert isinstance(result, float)
    assert result == pytest.approx(expected)
    assert result == pytest.approx(euclidean_distance(y,x))

@pytest.mark.parametrize("x,y", [([],[]), ([1],[1,2]), ([[1]],[[2]]), ([np.nan],[0]), ([0],[np.inf])])
def test_invalid_distance(x,y):
    with pytest.raises(ValueError):euclidean_distance(x,y)

@pytest.mark.parametrize("X,mean,scale", [
    ([[1,10],[3,10]],[2,10],[1,1]),
    ([[1],[2],[3]],[2],[(2/3)**0.5]),
    ([[1,100],[3,300]],[2,200],[1,100]),
    ([[7,9]],[7,9],[1,1]),
])
def test_fit_by_feature(X,mean,scale):
    model = Standardizer()
    assert model.fit(X) is model
    assert isinstance(model.mean_, np.ndarray) and isinstance(model.scale_, np.ndarray)
    np.testing.assert_allclose(model.mean_, mean)
    np.testing.assert_allclose(model.scale_, scale)
    assert model.n_features_in_ == len(mean)

@pytest.mark.parametrize("X", [[],[[]],[[1,2],[3]],[1,2],[[np.nan]],[[np.inf]]])
@pytest.mark.parametrize("method", ["fit","transform"])
def test_invalid_matrices(X,method):
    model = Standardizer().fit([[1,10],[3,10]]) if method == "transform" else Standardizer()
    with pytest.raises(ValueError):getattr(model,method)(X)

def test_transform_saved_parameters():
    model = Standardizer().fit([[1,10],[3,10]])
    Z = model.transform([[1,10],[3,10]])
    assert isinstance(Z, np.ndarray) and Z.shape == (2,2)
    np.testing.assert_allclose(Z,[[-1,0],[1,0]])
    np.testing.assert_allclose(model.transform([[5,10],[5,12]]),[[3,0],[3,2]])

def test_no_refit_or_mutation():
    X = np.array([[1.,10.],[3.,10.]])
    before = X.copy()
    model = Standardizer().fit(X)
    np.testing.assert_array_equal(X,before)
    mean, scale = model.mean_.copy(), model.scale_.copy()
    query = np.array([[5.,10.],[1000.,-100.]])
    saved = query.copy()
    alone = model.transform(query[:1])
    batch = model.transform(query)
    np.testing.assert_array_equal(query,saved)
    np.testing.assert_array_equal(model.mean_,mean)
    np.testing.assert_array_equal(model.scale_,scale)
    np.testing.assert_allclose(alone[0],batch[0])
    batch[0,0] = 99
    np.testing.assert_array_equal(query,saved)

def test_unfitted_and_feature_mismatch():
    with pytest.raises(ValueError,match="fit"):Standardizer().transform([[1]])
    with pytest.raises(ValueError,match="feature"):Standardizer().fit([[1,2]]).transform([[1]])

def test_refit_replaces_state():
    model = Standardizer().fit([[1,2],[3,4]])
    assert model.fit([[7]]) is model
    assert model.n_features_in_ == 1
    np.testing.assert_allclose(model.transform([[9]]),[[2]])

def test_sklearn_reference():
    X = [[-2,10,1],[0,10,5],[8,10,2]]
    query = [[3,12,9],[-1,10,2]]
    model, reference = Standardizer().fit(X), StandardScaler().fit(X)
    np.testing.assert_allclose(model.mean_, reference.mean_)
    np.testing.assert_allclose(model.scale_, reference.scale_)
    np.testing.assert_allclose(model.transform(query), reference.transform(query))

def test_scale_changes_nearest_candidate():
    model = Standardizer().fit([[0,0],[2,200]])
    query, candidates = [1,100], [[3,100],[1,200]]
    assert [euclidean_distance(query,c) for c in candidates] == [2,100]
    zq, zc = model.transform([query])[0], model.transform(candidates)
    assert [euclidean_distance(zq,c) for c in zc] == [2,1]
