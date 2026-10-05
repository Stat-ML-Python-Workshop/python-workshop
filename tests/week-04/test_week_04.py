import pytest
def test_dot(call):
    assert call("linalg","dot",[1,2],[3,4])==pytest.approx(11)
    assert call("linalg","dot",[-2,0,3],[4,7,-1])==pytest.approx(-11)
    assert call("linalg","dot",[],[])==0
def test_matvec(call):
    assert call("linalg","matvec",[[1,2],[3,4]],[2,1])==pytest.approx([4,10])
def test_dimensions(invoke):
    assert invoke("linalg","dot",[[1],[1,2]])["error"]=="ValueError"
    assert invoke("linalg","matvec",[[[1],[1,2]],[1]])["error"]=="ValueError"
def test_stable_softmax(call,invoke):
    result=call("probability","stable_softmax",[1000,999,-1000])
    assert sum(result)==pytest.approx(1)
    assert result[0]>result[1]>result[2]
    shifted=call("probability","stable_softmax",[1017,1016,-983])
    assert shifted==pytest.approx(result)
    assert call("probability","stable_softmax",[0,0])==pytest.approx([0.5,0.5])
    assert invoke("probability","stable_softmax",[[]])["error"]=="ValueError"
