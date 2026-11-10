"""Independent numerical acceptance through the isolated student worker."""
import numpy as np
import pytest

MODULE = 'models.linear_models'
ATTRIBUTES = ['coef_', 'intercept_', 'classes_', 'loss_history_',
              'validation_loss_history_', 'n_iter_', 'n_features_in_', 'n_samples_']


def sequence(invoke, steps, **parameters):
    response = invoke(MODULE, operation='class_sequence', **{
        'class': 'LogisticClassifier', 'init': parameters,
        'attributes': ATTRIBUTES, 'steps': steps})
    assert 'error' not in response, response
    return response['value']['steps']


def fit(X,y,validation=None):
    return {'method':'fit','args':[X,y] if validation is None else [X,y,validation]}


def reference(X, y, epochs, rate, batch_size, seed):
    """Scalar example accumulation, independent of the vectorized student update."""
    X=np.asarray(X, dtype=float)
    classes, encoded=np.unique(y,return_inverse=True)
    W=np.zeros((len(classes),X.shape[1])); b=np.zeros(len(classes))
    rng=np.random.default_rng(seed)
    history=[]
    for epoch in range(epochs+1):
        logits=X@W.T+b
        log_prob=logits-np.logaddexp.reduce(logits,axis=1)[:,None]
        history.append(float(-np.mean(log_prob[np.arange(len(X)),encoded])))
        if epoch==epochs:break
        order=rng.permutation(len(X))
        for start in range(0,len(X),batch_size):
            indices=order[start:start+batch_size]
            dw=np.zeros_like(W); db=np.zeros_like(b)
            for i in indices:
                scores=W@X[i]+b
                probs=np.exp(scores-np.logaddexp.reduce(scores))
                for c in range(len(classes)):
                    error=probs[c] - (encoded[i]==c)
                    dw[c] += error*X[i]/len(indices)
                    db[c] += error/len(indices)
            W-=rate*dw; b-=rate*db
    return W,b,history


def test_softmax_reference(call):
    X=np.array([[10000.,10002.,9998.],[-10000.,-9999.,-10002.]])
    expected=np.exp(X-np.logaddexp.reduce(X,axis=1)[:,None])
    actual=call(MODULE,'softmax',X.tolist())
    np.testing.assert_allclose(actual,expected,rtol=1e-11)


@pytest.mark.parametrize('batch_size',[1,2,5,20])
def test_minibatch_updates_partial_batch_and_histories(invoke,batch_size):
    X=[[1.,2.],[-2.,1.],[0.,3.],[2.,-1.],[-1.,-2.]]
    y=['b','a','c','b','a']
    W,b,history=reference(X,y,3,.07,batch_size,17)
    trained,predicted=sequence(invoke,[fit(X,y),{'method':'predict_proba','args':[X]}],
                               epochs=3,learning_rate=.07,batch_size=batch_size,random_state=17)
    assert trained['is_self'] and trained['input_unchanged'] and predicted['input_unchanged']
    state=trained['attributes']
    assert state['classes_']==['a','b','c'] and state['n_iter_']==3
    assert state['n_features_in_']==2 and state['n_samples_']==5
    np.testing.assert_allclose(state['coef_'],W,atol=1e-12)
    np.testing.assert_allclose(state['intercept_'],b,atol=1e-12)
    np.testing.assert_allclose(state['loss_history_'],history,atol=1e-12)
    assert state['validation_loss_history_']==[]
    logits=np.array(X)@W.T+b
    np.testing.assert_allclose(predicted['value'],np.exp(logits-np.logaddexp.reduce(logits,axis=1)[:,None]))
    assert predicted['dtype']=='float64' and predicted['shape']==[5,3]


def test_validation_does_not_train_and_refit_is_reproducible(invoke):
    X=[[-2.],[-1.],[1.],[2.],[3.]];y=[0,0,1,1,1]
    results=sequence(invoke,[fit(X,y),fit(X,y,[[[10.],[-10.]],[0,1]]),
        fit(X,[0]*5),{'method':'predict','args':[X]},fit(X,y)],
        epochs=5,learning_rate=.1,batch_size=2)
    first,validated,failed,predicted,refit=results
    for key in ['coef_','intercept_','loss_history_']:
        assert first['attributes'][key]==validated['attributes'][key]==refit['attributes'][key]
    assert failed['error']=='ValueError' and failed['attributes']==validated['attributes']
    assert predicted['value']==y
    assert validated['attributes']['validation_loss_history_'][-1]>np.log(2)
    assert refit['attributes']['validation_loss_history_']==[]
    assert all(item['input_unchanged'] for item in results)


def test_validation_loss_matches_same_checkpoint(invoke):
    X=[[-2.],[-1.],[1.],[2.]];y=[0,0,1,1]
    result=sequence(invoke,[fit(X,y,[X,y])],epochs=4,learning_rate=.1,batch_size=3)[0]['attributes']
    np.testing.assert_allclose(result['loss_history_'],result['validation_loss_history_'])


@pytest.mark.parametrize('parameters',[{'epochs':0},{'batch_size':0},{'random_state':-1},
    {'learning_rate':0},{'learning_rate':float('nan')},{'epochs':True}])
def test_invalid_parameters(invoke,parameters):
    response=invoke(MODULE,**{'class':'LogisticClassifier','init':parameters,
        'fit':[[[0],[1]],[0,1]],'predict':[[0]]})
    assert response['error']=='ValueError'


def test_unfitted_shapes_and_unknown_validation_labels(invoke):
    results=sequence(invoke,[{'method':'predict_proba','args':[[[0]]]},
        fit([[0],[1]],['a','b']), {'method':'predict_proba','args':[[[0,1]]]},
        fit([[0],[1]],['a','b'],[[[0]],['c']])],epochs=1)
    assert results[0]['error']==results[2]['error']==results[3]['error']=='ValueError'
    assert results[1]['attributes']==results[3]['attributes']
