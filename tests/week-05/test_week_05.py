"""Teacher-controlled checks through the isolated student worker."""
import pytest

def standardize(invoke, train, query):
    response = invoke("preprocessing", **{"class":"Standardizer", "fit":[train],
        "method":"transform", "predict":query,
        "attributes":["mean_","scale_","n_features_in_"]})
    assert "error" not in response, response
    return response["value"]

@pytest.mark.parametrize("x,y,expected", [([0,0],[3,4],5),([1,2],[1,2],0),([0.1,0.2],[0.4,0.6],0.5)])
def test_distance(call,x,y,expected):
    assert call("linalg","euclidean_distance",x,y) == pytest.approx(expected)
    assert call("linalg","euclidean_distance",y,x) == pytest.approx(expected)

@pytest.mark.parametrize("x,y", [([],[]),([1],[1,2]),([[1]],[[2]])])
def test_invalid_distance(invoke,x,y):
    assert invoke("linalg","euclidean_distance",[x,y])["error"] == "ValueError"

def test_training_state(invoke):
    value = standardize(invoke,[[1,10],[3,10]],[[5,10],[5,12]])
    assert value["mean_"] == [2,10] and value["scale_"] == [1,1]
    assert value["n_features_in_"] == 2
    assert value["prediction"] == [[3,0],[3,2]]

def test_axis_and_population(invoke):
    value = standardize(invoke,[[1,100],[3,300]],[[5,400]])
    assert value["mean_"] == [2,200] and value["scale_"] == [1,100]
    assert value["prediction"] == [[3,2]]
    value = standardize(invoke,[[1],[2],[3]],[[2]])
    assert value["scale_"] == pytest.approx([(2/3)**0.5])

def test_batch_independence(invoke):
    first = standardize(invoke,[[1,10],[3,10]],[[5,10]])
    second = standardize(invoke,[[1,10],[3,10]],[[5,10],[1000,-100]])
    assert first["prediction"][0] == second["prediction"][0] == [3,0]

@pytest.mark.parametrize("train,query", [([],[[1]]),([[1,2],[3]],[[1,2]]),([[1,2]],[[1]]),([[1,2]],[])])
def test_invalid_standardizer(invoke,train,query):
    response=invoke("preprocessing", **{"class":"Standardizer","fit":[train],"method":"transform","predict":query})
    assert response["error"] == "ValueError"

def test_unfitted(invoke):
    response=invoke("preprocessing", **{"class":"Standardizer","method":"transform","predict":[[1]]})
    assert response["error"] == "ValueError"
