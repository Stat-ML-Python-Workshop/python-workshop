"""Week 9: train the student forward with PyTorch autograd and SGD.

From package root: uv run python test_code/week_09.py --headless
Reuses the verified Week 8 offline data pack and loader; no downloads.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import time

import matplotlib
import numpy as np
import torch
from mini_ml.decomposition import PCA
from mini_ml.models.linear_models import LogisticClassifier, TrainingDivergedError

LEARNING_RATES = (1e-4, 1e-3, 1e-2, 1e-1)


def week08_support():
    """Resolve the existing demo after student sync or in teacher curriculum."""
    here = Path(__file__).resolve()
    sibling = here.with_name('week_08.py')
    path = sibling if sibling.is_file() else here.parents[1] / 'week-08/demo.py'
    spec = importlib.util.spec_from_file_location('week08_data_support', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_hand_update():
    model = LogisticClassifier(learning_rate=0.1, epochs=1, batch_size=2)
    model.fit([[0.], [2.]], [0, 1])
    np.testing.assert_allclose(model.coef_, [[-0.05], [0.05]], atol=1e-12)
    np.testing.assert_allclose(model.intercept_, [0., 0.], atol=1e-12)
    np.testing.assert_allclose(model.loss_history_[0], np.log(2), atol=1e-12)


def select_training_subset(data, per_class, seed):
    """Sample original training rows only; preserve labels and validation data."""
    if per_class is None:
        return data
    if not isinstance(per_class, (int, np.integer)) or isinstance(per_class, (bool, np.bool_)) or per_class < 1:
        raise ValueError('train-per-class must be a positive integer')
    training = data['tissue_train']
    labels = training['labels']
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError(f'Class {label} has fewer than {per_class} training images')
        selected.extend(rng.choice(candidates, per_class, replace=False))
    indices = np.sort(selected)
    if len(indices) <= 30:
        raise ValueError('Need at least 31 training images for 30 PCs')
    subset = dict(training)
    for key in ('images', 'labels', 'image_ids', 'filenames'):
        subset[key] = training[key][indices].copy()
    return {**data, 'tissue_train': subset}


def run_sweep(X, y, Xv, yv, learning_rates=LEARNING_RATES, epochs=100, batch_size=128, seed=42):
    """Same initialization/order; validation never participates in an update."""
    results, best_model, best_key = [], None, None
    for rate in learning_rates:
        model = LogisticClassifier(learning_rate=rate, epochs=epochs,
                                   batch_size=batch_size, random_state=seed)
        started = time.perf_counter()
        result = {'learning_rate': rate, 'requested_epochs': epochs, 'batch_size': batch_size}
        try:
            model.fit(X, y, validation_data=(Xv, yv))
            train_losses = list(model.loss_history_)
            validation_losses = list(model.validation_loss_history_)
            if len(train_losses) != epochs + 1 or len(validation_losses) != epochs + 1:
                raise AssertionError('Histories must include epoch 0 and every complete epoch')
            prediction = model.predict(Xv)
            balanced_accuracy = float(np.mean([
                np.mean(prediction[yv == label] == label) for label in np.unique(yv)]))
            result.update(status='completed', error=None, validation_balanced_accuracy=balanced_accuracy)
            key = (validation_losses[-1], rate)
            if best_key is None or key < best_key:
                best_key, best_model = key, model
        except TrainingDivergedError as error:
            train_losses = error.loss_history
            validation_losses = error.validation_loss_history
            result.update(status='diverged', error=str(error), validation_balanced_accuracy=None)
        result.update(fit_seconds=time.perf_counter() - started,
                      completed_epochs=len(train_losses) - 1,
                      train_loss=train_losses, validation_loss=validation_losses)
        results.append(result)
    return results, best_model


def save_results(output_dir, results, best_model, pca, data, settings):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'week_09_history.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['learning_rate', 'epoch', 'training_loss', 'validation_loss'])
        for result in results:
            for epoch, (train, valid) in enumerate(zip(result['train_loss'], result['validation_loss'])):
                writer.writerow([result['learning_rate'], epoch, train, valid])
    with (output_dir / 'week_09_results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['learning_rate', 'requested_epochs', 'completed_epochs', 'final_train_ce',
                         'final_validation_ce', 'validation_balanced_accuracy', 'fit_seconds', 'status'])
        for result in results:
            completed = result['status'] == 'completed'
            writer.writerow([result['learning_rate'], result['requested_epochs'], result['completed_epochs'],
                             result['train_loss'][-1] if completed else '',
                             result['validation_loss'][-1] if completed else '',
                             result['validation_balanced_accuracy'], result['fit_seconds'], result['status']])
    state = {'settings': settings, 'selection': 'lowest final validation CE; ties favor smaller learning rate',
             'selected_learning_rate': None if best_model is None else best_model.learning_rate,
             'data_files': data['provenance']['files'], 'runs': results}
    if settings.get('train_per_class') is not None:
        state['training_ids'] = data['tissue_train']['image_ids'].tolist()
    (output_dir / 'week_09_state.json').write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    target = output_dir / 'week_09_best_model.npz'
    if best_model is None:
        target.unlink(missing_ok=True)
    else:
        np.savez_compressed(target, pca_mean=pca.mean_, pca_components=pca.components_,
                            coef=best_model.coef_, intercept=best_model.intercept_,
                            classes=best_model.classes_, class_names=data['tissue_train']['classes'],
                            training_ids=data['tissue_train']['image_ids'],
                            validation_ids=data['tissue_validation']['image_ids'],
                            learning_rate=best_model.learning_rate, epochs=best_model.epochs,
                            batch_size=best_model.batch_size, seed=best_model.random_state,
                            input_shape=np.array([28, 28, 3]), intensity_divisor=255.0)


def plot_curves(results, output_dir, settings):
    import matplotlib.pyplot as plt
    columns = min(2, len(results))
    rows = (len(results) + columns - 1) // columns
    figsize = (10, 5.5) if len(results) == 1 else (12, 3.8 * rows)
    fig, axes = plt.subplots(rows, columns, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor('#fffdf7')
    for axis, result in zip(axes.flat, results):
        epoch = np.arange(len(result['train_loss']))
        marker = 'o' if len(epoch) == 1 else None
        axis.plot(epoch, result['train_loss'], color='#008782', label='Training', marker=marker)
        axis.plot(epoch, result['validation_loss'], color='#bd5d35', label='Validation', marker=marker)
        if settings.get('train_per_class') is not None and result['validation_loss']:
            best_epoch = int(np.argmin(result['validation_loss']))
            axis.axvline(best_epoch, color='#646b76', linestyle='--', linewidth=1,
                         label=f'Lowest validation CE: epoch {best_epoch}')
        axis.set_title(f"Learning rate = {result['learning_rate']:g}  |  {result['status']}")
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Cross-entropy (nats)')
        axis.grid(alpha=0.2)
        axis.legend()
    for axis in list(axes.flat)[len(results):]:
        axis.set_visible(False)
    training_size, validation_size = settings['train_shape'][0], settings['validation_shape'][0]
    mode = 'Full-batch gradient descent' if settings['batch_size'] >= training_size else 'Mini-batch SGD'
    fig.suptitle(f'PyTorch {mode} | 30 training-only PCs\n'
                 f'Training: {training_size:,} images | Validation: {validation_size:,} images | Seed: {settings["seed"]}')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / 'week_09_loss_curves.png', dpi=160)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rates', type=float, nargs='+', default=list(LEARNING_RATES))
    parser.add_argument('--train-per-class', type=int,
                        help='Sample this many images per training class before fitting PCA')
    parser.add_argument('--data-dir', type=Path, help='Existing Week 8 verified data directory')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.seed < 0:
        parser.error('epochs/batch-size must be positive; seed must be nonnegative')
    if any(not np.isfinite(rate) or rate <= 0 for rate in args.learning_rates):
        parser.error('learning rates must be finite and positive')
    if len(set(args.learning_rates)) != len(args.learning_rates):
        parser.error('learning rates must be distinct')
    if args.headless:
        matplotlib.use('Agg')
    torch.set_num_threads(1)  # Small CPU matrices: avoid thread-launch overhead.
    check_hand_update()
    support = week08_support()
    data = support.load_data(args.data_dir)
    try:
        data = select_training_subset(data, args.train_per_class, args.seed)
    except ValueError as error:
        parser.error(str(error))
    X = support.image_features(data['tissue_train']['images'])
    Xv = support.image_features(data['tissue_validation']['images'])
    y = data['tissue_train']['labels']
    yv = data['tissue_validation']['labels']
    if not np.array_equal(data['tissue_train']['classes'], data['tissue_validation']['classes']):
        raise ValueError('Training and validation class orders differ')
    started = time.perf_counter()
    pca = PCA(30, random_state=args.seed).fit(X)
    train_scores, validation_scores = pca.transform(X), pca.transform(Xv)
    print(f'PCA training-only fit: {time.perf_counter() - started:.2f}s', flush=True)
    results, best_model = run_sweep(train_scores, y, validation_scores, yv,
                                   args.learning_rates, args.epochs, args.batch_size, args.seed)
    settings = {'backend': 'pytorch-autograd-sgd', 'torch_version': torch.__version__,
                'device': 'cpu', 'dtype': 'float64', 'torch_threads': 1, 'pcs': 30, 'epochs': args.epochs, 'batch_size': args.batch_size,
                'seed': args.seed, 'train_shape': list(train_scores.shape),
                'validation_shape': list(validation_scores.shape),
                'train_per_class': args.train_per_class}
    save_results(args.output_dir, results, best_model, pca, data, settings)
    figure = plot_curves(results, args.output_dir, settings)
    for result in results:
        print(f"LR={result['learning_rate']:g}: {result['status']}, "
              f"{result['completed_epochs']} epochs, {result['fit_seconds']:.2f}s", flush=True)
    print(f"Selected learning rate: {None if best_model is None else best_model.learning_rate}")
    import matplotlib.pyplot as plt
    if not args.headless:
        plt.show()
    plt.close(figure)
    return 0 if best_model is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())
