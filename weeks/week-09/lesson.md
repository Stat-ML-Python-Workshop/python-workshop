# Week 09 — Forward Pass with PyTorch

Complete **five expressions**: softmax and the forward pass. The supplied code trains your model with PyTorch autograd and SGD. Then compare training/validation loss curves.

## 1. Set up

Start from your completed Week 8 package on `<login>/week-09`. From `students/<name_studentnumber>/`, after the lesson is published:

```sh
uv add "torch==2.15.0"
uv run python ../../tools/course.py test <login> 8
uv run python ../../tools/course.py sync <login> 9
```

Commit `pyproject.toml` and `uv.lock`. Sync supplies tests, demo and missing Week 8 support/data; it preserves existing files and never edits your source.

Create `src/mini_ml/models/` and copy:

| Supplied fragment | Destination | Work |
|---|---|---|
| [01___init__.py](fragments/01___init__.py) | `__init__.py` | Copy |
| [02_linear_models.py](fragments/02_linear_models.py) | `linear_models.py` | Fill **1–5** |
| [03__training_utils.py](fragments/03__training_utils.py) | `_training_utils.py` | Copy |

## 2. Implement forward

For a batch: $X:(B,k)$, $W:(c,k)$, $b:(c,)$, scores/probabilities: $(B,c)$.

$$
S=XW^\top+b,\qquad m_i=\max_j S_{ij},\qquad
P_{ij}=\frac{e^{S_{ij}-m_i}}{\sum_\ell e^{S_{i\ell}-m_i}}.
$$

| Exercise | Implement |
|---|---|
| 1 | Subtract each row's maximum; retain `(B, 1)` for broadcasting |
| 2 | Exponentiate the shifted scores |
| 3 | Divide by each row's sum |
| 4 | Compute scores using `X_batch`, `self.W`, `self.b` |
| 5 | Apply your `softmax(scores)` |

Use PyTorch tensor operations (`dim`, `keepdim`). `forward(X)` returns **scores, probabilities**. Do not convert to NumPy or detach inside forward.

Training is supplied: `zero_grad()` clears old gradients, `loss.backward()` computes new gradients, and `optimizer.step()` updates W/b. `cross_entropy` receives **scores**, not probabilities. `predict_proba` and `predict` are supplied too; hand-written gradients and updates are next week.

```sh
uv run pytest -q
uv run python ../../tools/course.py test <login> 9
```

## 3. Compare learning rates

```sh
uv run python test_code/week_09.py --headless
```

Use the same 3,000 training / 1,000 validation images, RGB flatten /255, **training-only PCA(30)**, zero parameters, seed 42, batch 128 and 100 epochs. The demo fits PCA once and compares **1e-4, 1e-3, 1e-2, 1e-1**. CPU/float64, no momentum or weight decay. Test data stays unused.

Loss is recorded at epoch 0 and after each epoch on both complete splits using the same parameters. Validation only evaluates. Outputs in ignored `test_code/`: `week_09_loss_curves.png`, history/results CSVs, settings JSON and the frozen best model NPZ. Select the completed finite run with lowest final validation loss; ties favor smaller LR. Keep failed runs in the report.

If both curves still decrease, try a shared longer budget, for example `--epochs 200 --output-dir test_code/week09_200`.

**Separate overfitting demo:**

```sh
uv run python test_code/week_09.py --headless --train-per-class 10 --learning-rates 0.01 --batch-size 80 --epochs 5000 --output-dir test_code/week09_overfit
```

This uses 80 training images, fits PCA only on those 80 and retains all 1,000 validation images. It changes several settings; keep it separate from the four-rate comparison. The dashed minimum is retrospective, not early stopping.

## 4. Bring your results

| LR | Final train loss | Final validation loss | Status |
|---|---|---|---|
| 1e-4 | | | |
| 1e-3 | | | |
| 1e-2 | | | |
| 1e-1 | | | |

1. **Does your loss curve suggest overfitting, underfitting, or not yet converged?** Point to both curves.
2. **How does learning rate affect your loss curve?** Compare speed, oscillation and final validation loss.

Commit source, supplied tests and dependency locks. Attach curves/table to your PR; keep experiments and generated files in ignored `test_code/`. Preserve your Week 9 forward for Week 10.
