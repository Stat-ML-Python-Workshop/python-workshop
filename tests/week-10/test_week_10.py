"""Week 10 acceptance: the frozen PCA + classifier behave consistently for batch scanning.

A scan predicts thousands of tiles at once. Batch results must equal one-tile-at-a-time
results, and every probability row must be a valid distribution.
"""
import numpy as np


def sequence(invoke, module, cls, init, attributes, steps):
    response = invoke(module, operation='class_sequence', **{
        'class': cls, 'init': init, 'attributes': attributes, 'steps': steps})
    assert 'error' not in response, response
    return response['value']['steps']


RNG = np.random.default_rng(10)
X_TRAIN = RNG.normal(size=(24, 6)).round(3).tolist()
Y_TRAIN = [i % 3 for i in range(24)]
X_SCAN = RNG.normal(size=(5, 6)).round(3).tolist()


def test_pca_batch_transform_matches_single_rows(invoke):
    steps = [{'method': 'fit', 'args': [X_TRAIN]}, {'method': 'transform', 'args': [X_SCAN]}]
    steps += [{'method': 'transform', 'args': [[row]]} for row in X_SCAN]
    results = sequence(invoke, 'decomposition', 'PCA', {'n_components': 3}, ['mean_'], steps)
    batch = np.array(results[1]['value'])
    singles = np.vstack([np.array(r['value']) for r in results[2:]])
    assert batch.shape == (5, 3)
    np.testing.assert_allclose(batch, singles, atol=1e-10)
    assert all(r['input_unchanged'] for r in results)


def test_classifier_batch_probabilities_match_single_rows(invoke):
    steps = [{'method': 'fit', 'args': [X_TRAIN, Y_TRAIN]}, {'method': 'predict_proba', 'args': [X_SCAN]}]
    steps += [{'method': 'predict_proba', 'args': [[row]]} for row in X_SCAN]
    results = sequence(invoke, 'models.linear_models', 'LogisticClassifier',
                       {'epochs': 20, 'learning_rate': 0.1, 'batch_size': 8, 'random_state': 42},
                       ['classes_'], steps)
    batch = np.array(results[1]['value'])
    singles = np.vstack([np.array(r['value']) for r in results[2:]])
    assert results[1]['attributes']['classes_'] == [0, 1, 2]
    assert batch.shape == (5, 3)
    np.testing.assert_allclose(batch.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(batch >= 0)
    np.testing.assert_allclose(batch, singles, atol=1e-12)
    assert all(r['input_unchanged'] for r in results)
