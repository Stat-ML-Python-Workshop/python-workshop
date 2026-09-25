import pytest
def test_variance_and_std(call):
    assert call("statistics","variance",[2,4,6])==pytest.approx(4)
    assert call("statistics","variance",[2,4,6],ddof=0)==pytest.approx(8/3)
    assert call("statistics","std",[2,4,6])==pytest.approx(2)
    assert call("statistics","std",[2,4,6],ddof=0)==pytest.approx((8/3)**0.5)
    assert call("statistics","variance",[7,7,7])==0
def test_ddof(invoke):
    assert invoke("statistics","variance",[[],0])["error"]=="ValueError"
    assert invoke("statistics","variance",[[2]])["error"]=="ValueError"
    assert invoke("statistics","variance",[[2,4],-1])["error"]=="ValueError"
def test_cross_entropy_identity(call):
    assert call("information","cross_entropy",[0.75,0.25],[0.5,0.5])==pytest.approx(1)
    p=[0.5,0.3,0.2]
    assert call("information","cross_entropy",p,p)==pytest.approx(call("information","shannon_entropy",p))
def test_kl_identity_direction_and_asymmetry(call):
    p=[0.5,0.3,0.2];q=[0.6,0.25,0.15]
    assert call("information","kl_divergence",p,p)==pytest.approx(0,abs=1e-12)
    forward=call("information","kl_divergence",p,q)
    reverse=call("information","kl_divergence",q,p)
    assert forward==pytest.approx(0.030400618689010048)
    assert forward!=pytest.approx(reverse)
def test_zero_probability_and_validation(call,invoke):
    assert call("information","cross_entropy",[0,1],[0,1])==0
    assert call("information","cross_entropy",[1,0],[0,1])==float("inf")
    assert call("information","kl_divergence",[1,0],[0,1])==float("inf")
    assert invoke("information","cross_entropy",[[0.5,0.5],[1.0]])["error"]=="ValueError"
    assert invoke("information","kl_divergence",[[0.5,0.5],[1.0]])["error"]=="ValueError"
