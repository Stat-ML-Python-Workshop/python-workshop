"""Week 9 supplied checks: forward, PyTorch training, and public prediction API."""
import numpy as np
import pytest
import torch
from mini_ml.models.linear_models import LogisticClassifier, TrainingDivergedError, softmax


def test_softmax_values_shift_and_extreme_scores():
    X = np.array([[0., 0., 0.], [1000., 1001., 999.]])
    before = X.copy()
    actual = softmax(X)
    np.testing.assert_allclose(actual[0], [1/3]*3)
    np.testing.assert_allclose(actual[1], np.array([1., np.e, 1/np.e]) / (1 + np.e + 1/np.e))
    np.testing.assert_allclose(actual.sum(axis=1), 1.)
    np.testing.assert_allclose(softmax(X + 10000), actual)
    np.testing.assert_array_equal(X, before)
    np.testing.assert_allclose(softmax([[1e308, -1e308]]), [[1., 0.]])
    assert actual.dtype == torch.float64 and actual.device.type == "cpu"


@pytest.mark.parametrize('bad', [[], [[]], [1,2], [[1]], [[0, np.nan]], [[0, np.inf]], [[1,2],[3]]])
def test_softmax_invalid(bad):
    with pytest.raises(ValueError):
        softmax(bad)


def test_hand_update_and_bias_update():
    model = LogisticClassifier(learning_rate=.1, epochs=1, batch_size=2)
    assert model.fit([[0.],[2.]], [0,1]) is model
    np.testing.assert_allclose(model.coef_, [[-.05],[.05]])
    np.testing.assert_allclose(model.intercept_, [0,0], atol=1e-15)
    np.testing.assert_allclose(model.loss_history_[0], np.log(2))
    assert len(model.loss_history_) == 2
    bias = LogisticClassifier(learning_rate=.3, epochs=1, batch_size=3).fit([[0.]]*3, ['a','a','b'])
    np.testing.assert_allclose(bias.intercept_, [.05,-.05])


def test_three_classes_predict_and_input_preservation():
    X = np.array([[-2.,0.],[-1.,0.],[0.,2.],[0.,1.],[2.,0.],[1.,0.]])
    y = np.array(['left','left','up','up','right','right'])
    before, labels_before = X.copy(), y.copy()
    model = LogisticClassifier(epochs=40, learning_rate=.2, batch_size=4).fit(X,y, (X,y))
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_array_equal(model.classes_, ['left','right','up'])
    np.testing.assert_array_equal(X,before)
    np.testing.assert_array_equal(y,labels_before)
    assert model.coef_.shape == (3,2) and model.intercept_.shape == (3,)
    assert model.n_features_in_ == 2 and model.n_samples_ == 6 and model.n_iter_ == 40
    assert model.loss_history_[-1] < model.loss_history_[0]
    np.testing.assert_allclose(model.loss_history_, model.validation_loss_history_)
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)


def test_validation_is_evaluation_only_and_refit_resets():
    X, y = [[-2.],[-1.],[1.],[2.],[3.]], [0,0,1,1,1]
    model = LogisticClassifier(epochs=4, batch_size=2, learning_rate=.1)
    model.fit(X,y)
    weights, bias, history = model.coef_.copy(), model.intercept_.copy(), model.loss_history_.copy()
    model.fit(X,y, ([[8.],[-8.]], [0,1]))
    np.testing.assert_array_equal(model.coef_,weights)
    np.testing.assert_array_equal(model.intercept_,bias)
    assert model.loss_history_ == history and len(model.validation_loss_history_) == 5
    assert model.validation_loss_history_[-1] > model.validation_loss_history_[0]
    model.fit(X,y)
    assert model.validation_loss_history_ == []
    np.testing.assert_array_equal(model.coef_,weights)
    with pytest.raises(ValueError):
        model.fit(X,[0]*5)
    np.testing.assert_array_equal(model.coef_,weights)


@pytest.mark.parametrize('parameters', [
    {'learning_rate':0}, {'learning_rate':-1}, {'learning_rate':np.inf}, {'learning_rate':True},
    {'epochs':0}, {'epochs':1.5}, {'epochs':True}, {'batch_size':0}, {'batch_size':False},
    {'random_state':-1}, {'random_state':True}, {'random_state':1.5},
])
def test_bad_parameters(parameters):
    with pytest.raises(ValueError):
        LogisticClassifier(**parameters)


@pytest.mark.parametrize('X,y', [([],[]), ([[0],[1]], [0]), ([[0],[1]], [[0],[1]]),
    ([[0],[1]], [0,0]), ([[0],[np.inf]], [0,1]), ([[0],[1]], [0,np.nan])])
def test_bad_training_data(X,y):
    with pytest.raises(ValueError):
        LogisticClassifier().fit(X,y)


def test_prediction_guards_and_validation_labels():
    model = LogisticClassifier(epochs=1)
    with pytest.raises(ValueError):
        model.predict([[0.]])
    model.fit([[0.],[1.]], ['a','b'])
    for query in [[], [[1,2]], [[np.nan]]]:
        with pytest.raises(ValueError):
            model.predict_proba(query)
    for validation in [([[1,2]], ['a']), ([[1]], ['unknown']), ([[1]], [])]:
        with pytest.raises(ValueError):
            model.fit([[0.],[1.]], ['a','b'], validation)


def test_divergence_preserves_fitted_state():
    model = LogisticClassifier(epochs=1).fit([[-10.],[10.]], [0,1])
    before = model.coef_.copy()
    model.learning_rate = 1e308
    with pytest.raises(TrainingDivergedError) as caught:
        model.fit([[-10.],[10.]], [0,1])
    np.testing.assert_array_equal(model.coef_, before)
    assert len(caught.value.loss_history) == 1


@pytest.fixture(autouse=True, scope='module')
def small_cpu_matrices():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_softmax_preserves_autograd_graph():
    scores = torch.tensor([[1.,2.,3.],[-1.,0.,2.]], dtype=torch.float64, requires_grad=True)
    probabilities = softmax(scores)
    expected = torch.softmax(scores, dim=1)
    torch.testing.assert_close(probabilities, expected)
    weights = torch.tensor([[1.,2.,4.],[3.,1.,2.]], dtype=torch.float64)
    actual_gradient = torch.autograd.grad((probabilities*weights).sum(),scores)[0]
    reference_gradient = torch.autograd.grad((expected*weights).sum(),scores)[0]
    torch.testing.assert_close(actual_gradient,reference_gradient)


def test_forward_scores_shapes_and_parameter_gradients():
    model = LogisticClassifier(epochs=1).fit([[-1.,2.],[1.,-2.],[2.,3.]],[0,1,2])
    X = torch.tensor([[.5,2.],[-1.,1.]],dtype=torch.float64)
    scores, probabilities = model.forward(X)
    torch.testing.assert_close(scores,X@model.W.T+model.b)
    torch.testing.assert_close(probabilities,torch.softmax(scores,dim=1))
    assert scores.shape == probabilities.shape == (2,3)
    loss = torch.nn.functional.cross_entropy(scores,torch.tensor([0,2]))
    dw,db = torch.autograd.grad(loss,[model.W,model.b])
    assert dw.shape == model.W.shape and db.shape == model.b.shape
    assert torch.isfinite(dw).all() and torch.isfinite(db).all()
    assert isinstance(model.predict_proba(X),np.ndarray)
    exported = model.coef_
    exported[:] = 0
    assert not np.all(model.coef_ == 0)  # Public arrays are copies.


def test_training_calls_forward_and_sgd_once_per_batch(monkeypatch):
    forward_calls, step_calls = [], []
    original_forward = LogisticClassifier.forward
    original_step = torch.optim.SGD.step
    def forward(self, X):
        forward_calls.append(torch.is_grad_enabled())
        return original_forward(self,X)
    def step(self,*args,**kwargs):
        step_calls.append(1)
        return original_step(self,*args,**kwargs)
    monkeypatch.setattr(LogisticClassifier,'forward',forward)
    monkeypatch.setattr(torch.optim.SGD,'step',step)
    model = LogisticClassifier(epochs=2,batch_size=2).fit([[-2.],[-1.],[1.]],[0,0,1])
    assert len(step_calls)==4 and sum(forward_calls)==4
    assert len(model.loss_history_)==3
