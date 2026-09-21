import pytest
@pytest.mark.parametrize("values,expected",[([1,2,3],6),([-2,0,4],2),([],0),([0.1,0.2],0.3)])
def test_sum(call,values,expected):
    assert call("statistics","sum_values",values)==pytest.approx(expected)
@pytest.mark.parametrize("values,expected",[([2,4,6],4),([-1,2],0.5),([7],7)])
def test_mean(call,values,expected):
    assert call("statistics","mean",values)==pytest.approx(expected)
def test_empty_mean(invoke):
    assert invoke("statistics","mean",[[]])["error"]=="ValueError"
def test_class_proportions(call,invoke):
    assert call("probability","class_proportions",["T","T"])==pytest.approx({"T":1.0})
    assert call("probability","class_proportions",["T","T","B","M"])==pytest.approx({"T":0.5,"B":0.25,"M":0.25})
    result=call("probability","class_proportions",[0,0,1])
    # The isolated worker uses JSON, which serializes integer dictionary keys.
    assert result==pytest.approx({"0":2/3,"1":1/3})
    assert sum(result.values())==pytest.approx(1)
    assert invoke("probability","class_proportions",[[]])["error"]=="ValueError"
def test_class_proportions_unbalanced(call):
    result=call("probability","class_proportions",["A","A","A","A","B","C","C"])
    assert result==pytest.approx({"A":4/7,"B":1/7,"C":2/7})
def test_shannon_entropy(call,invoke):
    assert call("information","shannon_entropy",[0.5,0.5])==pytest.approx(1)
    assert call("information","shannon_entropy",[1,0])==pytest.approx(0)
    assert call("information","shannon_entropy",[0.25]*4)==pytest.approx(2)
    assert invoke("information","shannon_entropy",[[0.2,0.2]])["error"]=="ValueError"
    assert invoke("information","shannon_entropy",[[]])["error"]=="ValueError"
    assert invoke("information","shannon_entropy",[[-0.1,1.1]])["error"]=="ValueError"
def test_shannon_entropy_boundaries(call,invoke):
    assert call("information","shannon_entropy",[0.75,0.25])==pytest.approx(0.8112781244591328)
    assert call("information","shannon_entropy",[0.0,0.5,0.5,0.0])==pytest.approx(1)
    assert call("information","shannon_entropy",[0.1]*10)==pytest.approx(3.321928094887362)
    assert invoke("information","shannon_entropy",[[0.6,0.6]])["error"]=="ValueError"
