# Week 08 — Learn a PCA Basis, Keep Its State

Complete **ten short expressions** in `PCA.fit`, `transform` and `inverse_transform`. In `fit`, you will build a covariance matrix and explicitly call **`np.linalg.eigh(C)`** to learn its eigenvectors. Keep the Week 6–7 functions in the same `src/mini_ml/decomposition.py`. The instructor supplies input checks, state management, an image SVD helper, data, plotting and tests. You use NumPy's numerical eigensolver; you do not write an eigensolver or SVD algorithm, or add original tests this week.

Use the [17-slide offline lecture](../../teaching/week08-pca/index.html) and its printable PDF. The browser shows precomputed results from the complete course PCA; the Python demo uses **your installed package**. Neither browser controls nor the offline demo download data.

## 1. Start from your completed Week 7 package

The repository root contains `students/`, `tools/` and `weeks/`. Your **package root** is `students/<name_studentnumber>/`. Replace `<login>` with your GitHub login. Commands below run from package root unless stated otherwise. Week 8 sync/test commands become available only after the teacher publishes the lesson; teacher readiness does not publish it.

```sh
uv sync --frozen
uv run pytest -q
uv run python ../../tools/course.py test <login> 7
git status --short
```

Expected: earlier tests pass. Save your changes and confirm the previous PR is merged. From **repository root**, start your next branch:

```sh
git switch main
git pull --ff-only origin main
git switch -c <login>/week-08
cd students/<name_studentnumber>
uv run python ../../tools/course.py sync <login> 8
uv sync --frozen
```

Expected: `test_code/week_08.py`, `tests/test_week_08.py` and `data/week-08/` exist. Sync adds missing supporting files, preserves existing local files and never edits your source. If a local demo is old, compare it with `weeks/week-08/demo.py` before copying changes manually. No shell activation command is needed on Windows, macOS or Linux when using `uv run`.

## 2. Append the starter once

Append [fragments/01_decomposition.py](fragments/01_decomposition.py) **after** the completed W6 and W7 code in `src/mini_ml/decomposition.py`. Keep `center`, `covariance_matrix`, `variance_along_direction`, `project`, `reconstruct`, `reconstruction_error` and all their helpers. Do not replace the file or paste the class twice. No file moves are required.

```sh
uv run python -c "from mini_ml.decomposition import PCA; print('PCA imports')"
```

Expected: `PCA imports`. An unfinished expression raises `NotImplementedError: TODO: ...`. Replace the **ten TODO calls inside the methods**; keep the supplied helper and guards.

| Interface | Input → result |
|---|---|
| `PCA(k, solver="auto", random_state=42)` | constructor; `k` is an integer, not a boolean |
| `fit(X)` | finite `(n,d)`, `n≥2`, `1≤k≤min(n−1,d)` → the same object |
| `transform(X)` | finite `(m,d)` → scores `(m,k)` |
| `inverse_transform(Z)` | finite `(m,k)` → reconstruction `(m,d)` |

Empty arrays, nonfinite values, incompatible widths and transforms before fit raise `ValueError`. Inputs are unchanged. There is no whitening. A successful refit replaces the state; a failed refit preserves the last successful state.

## 3. Why learning a direction becomes an eigenvalue problem

### From centered data to covariance

Each of the `n` rows of `X (n,d)` is an observation; each of the `d` columns is a feature. Subtract the **training feature means** `mean (d,)` to obtain `Xc (n,d)`. The sample covariance is

$$
C=\frac{X_c^\top X_c}{n-1}, \qquad C\in\mathbb{R}^{d\times d}.
$$

The diagonal entries are feature variances; off-diagonal entries describe how pairs of features vary together. Because $C^\top=C$, covariance is **symmetric**. For any direction $v$,

$$
v^\top Cv=\frac{\lVert X_cv\rVert^2}{n-1}\geq 0,
$$

so it is also **positive semidefinite**: its variance cannot be negative.

### From a direction to score variance

Choose a **unit** direction $v\in\mathbb{R}^d$, with $v^\top v=1$. Its training scores are $z=X_cv$, shape `(n,)`. These scores have mean zero because the columns of `Xc` have mean zero. Their sample variance is therefore

$$
\operatorname{Var}(z)=\frac{z^\top z}{n-1}
=\frac{v^\top X_c^\top X_cv}{n-1}=v^\top Cv.
$$

PCA chooses a unit direction that makes this variance as large as possible. The unit constraint matters: otherwise simply doubling $v$ would multiply the variance by four. At an optimum, turning the direction slightly cannot improve the variance; $Cv$ must point along $v$. This gives the **eigenvalue equation**

$$
Cv=\lambda v.
$$

Here $v$ is an eigenvector (a direction), and $\lambda$ is its eigenvalue. For a unit eigenvector, its training score variance is $v^\top Cv=\lambda$.

### Why the largest eigenvalue comes first

A real symmetric covariance matrix has an orthonormal eigenvector basis:

$$
C=Q\operatorname{diag}(\lambda_1,\ldots,\lambda_d)Q^\top,
\qquad Q^\top Q=I.
$$

The eigenvectors are the **columns** of `Q (d,d)`, paired with the entries of `lambda (d,)`. Any unit direction can be written $v=Qa$, where $\sum_j a_j^2=1$. Thus

$$
v^\top Cv=\sum_{j=1}^d\lambda_j a_j^2\leq\max_j\lambda_j.
$$

This is a weighted average of eigenvalues. Choosing an eigenvector with the largest eigenvalue reaches the maximum; it is the first principal component. The next orthogonal eigenvector with the largest remaining eigenvalue is the second component, and so on.

The total training variance includes **all** features, or equivalently **all** covariance eigenvalues:

$$
\operatorname{tr}(C)=\sum_{j=1}^d\lambda_j
=\frac{\sum_{i=1}^n\sum_{j=1}^d (X_c)_{ij}^2}{n-1}.
$$

Here $\operatorname{tr}(C)$ means the sum of diagonal entries. A component's explained variance ratio is its eigenvalue divided by this total, not by the sum of only the retained eigenvalues.

### The NumPy operation you implement

Use **`np.linalg.eigh(C)`** for this real symmetric matrix. It returns eigenvalues in **ascending order** and the corresponding eigenvectors as **columns**. The more general `eig` routine is unnecessary here. Pass the covariance `C (d,d)`, not the centered observations `Xc (n,d)`, even if `Xc` happens to be square. We implement the PCA calculation using NumPy's eigendecomposition routine; implementing a numerical eigensolver is outside this week's task. See [NumPy eigh](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html).

## 4. Work through a correlated example

Use `X = [[2,2], [-2,-2], [1,-1], [-1,1]]`. Both feature means are zero. Compute the products before continuing:

$$
X_c^\top X_c=
\begin{bmatrix}10&6\\6&10\end{bmatrix},
\qquad
C=\begin{bmatrix}10/3&2\\2&10/3\end{bmatrix}.
$$

For this small example, find the eigenvalues by hand:

$$
\det(C-\lambda I)=(10/3-\lambda)^2-4=0,
\qquad \lambda=10/3\pm2.
$$

In descending order they are $16/3$ and $4/3$. Substituting $16/3$ into $Cv=\lambda v$ gives $v_1=v_2$; substituting $4/3$ gives $v_1=-v_2$. Normalize the two directions:

$$
q_1=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix},
\qquad
q_2=\frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}.
$$

The first direction follows the longer diagonal cloud. Its scores are $[2\sqrt{2},-2\sqrt{2},0,0]$. For `k=1`, retained variance is `16/3`, total variance is `20/3`, and explained variance ratio is **0.8**. Reconstructions are `[[2,2],[-2,-2],[0,0],[0,0]]`. The squared residuals sum to 4, so MSE over **eight scalar entries** is **0.5**.

Create **`test_code/week_08_practice.py`**, an ignored local experiment. This first checkpoint works before `PCA` is complete:

```python
import numpy as np
from mini_ml.decomposition import center, covariance_matrix

X = np.array([[2., 2.], [-2., -2.], [1., -1.], [-1., 1.]])
Xc, mean = center(X)
C = covariance_matrix(Xc)
values, Q = np.linalg.eigh(C)
assert Xc.shape == (4, 2) and C.shape == (2, 2)
assert values.shape == (2,) and Q.shape == (2, 2)
np.testing.assert_allclose(mean, [0., 0.])
np.testing.assert_allclose(C, [[10 / 3, 2], [2, 10 / 3]])
np.testing.assert_allclose(values, [4 / 3, 16 / 3])  # ascending
np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-12)
np.testing.assert_allclose(C @ Q, Q * values, atol=1e-12)
np.testing.assert_allclose(Q @ np.diag(values) @ Q.T, C, atol=1e-12)
assert np.isclose(np.trace(C), values.sum())
assert np.isclose(np.trace(C), 20 / 3)
assert np.isclose(np.var(Xc @ Q[:, -1], ddof=1), values[-1])
print("Covariance, eigenvector residual and orthogonality checks passed")
```

```sh
uv run python test_code/week_08_practice.py
```

In `C @ Q == Q * values`, broadcasting multiplies **column j** by **eigenvalue j**. It checks all the eigenvalue equations at once; `Q.T @ Q` checks unit lengths and perpendicular directions.

A direction and its negative are equivalent. A sign flip changes scores and component together, leaving reconstruction unchanged. Do not compare component entries directly across implementations. When a cutoff splits repeated eigenvalues, valid bases can even span different optimal subspaces. Compare captured variance, reconstruction MSE and orthonormality in that case.

## 5. Fit: eight expressions, one consistent shape convention

The `eigh` branch is visible **inside your `fit` method**: center the training array, build `C`, then call NumPy. For this branch, `candidate_variance (d,)` contains every covariance eigenvalue and `axes (d,d)` contains the matching direction columns. The separate image helper returns the same kind of pair with shapes `(r,)` and `(d,r)`, where `r≥k`; it may return fewer than `d` directions. The shared code sorts and saves the learned state.

| Blank | Inputs → output | Hand checkpoint above | Common error |
|---|---|---|---|
| 1: center | `array (n,d)` → `Xc (n,d), mean (d,)` | mean `[0,0]` | returning only one value; centering each row |
| 2: covariance | `Xc (n,d)` → `C (d,d)` | `[[10/3,2],[2,10/3]]` | using raw data or the wrong denominator |
| 3: eigendecomposition | `C (d,d)` → values `(d,)`, columns `(d,d)` | ascending values `[4/3,16/3]` | calling `eigh` on `Xc`; treating vectors as rows |
| 4: order | candidate variances `(r,)` → indices `(r,)` | order `[1,0]` | ascending order keeps the least variation |
| 5: components | columns `(d,r)`, order, `k` → rows `(k,d)` | `[[1,1]] / sqrt(2)` up to sign | selecting rows before transposing |
| 6: variance | candidates and order → nonnegative `(k,)` | `[16/3]` | selecting different indices from the components |
| 7: total | all centered entries `(n,d)` → scalar | `20/3` | dividing by `n`, or summing only retained variances |
| 8: ratio | retained variance `(k,)`, total → `(k,)` | `[0.8]` | every model falsely reports 100% |

Reuse `center` and `covariance_matrix` for blanks 1–2 and call `np.linalg.eigh(C)` for blank 3. Sort largest variance first for blank 4. Select the same first `k` indices for blanks 5–6, and clamp tiny negative numerical variances to zero. Total variance equals the sum of squared centered entries divided by **`n−1`**, including every feature. Compute it from all of `Xc`, because the randomized helper need not return every eigenvalue. The teacher supplies the zero-total branch: constant data has zero variances and zero ratios.

The constructor and method body save these attributes only after the computation succeeds:

| Saved training state | Shape / meaning |
|---|---|
| `mean_` | `(d,)`, training feature means |
| `components_` | `(k,d)`, orthonormal directions as rows |
| `explained_variance_` | `(k,)`, descending sample variances |
| `explained_variance_ratio_` | `(k,)`, fractions of all training variance |
| `n_features_in_` | integer `d` |
| `n_samples_` | integer `n` |

After completing the eight fit expressions, check just the fitted state before implementing the last two methods:

```sh
uv run pytest -q tests/test_week_08.py -k "fit_hand_values or fit_matches_reference or fit_numeric_strings"
```

Expected: the selected fit checks pass. The full supplied file will still report TODO failures until transform and inverse_transform are complete.

## 6. Transform and reconstruct: two delegation expressions

Blank 9 calls your W7 `project` using new `X` and the **saved** training mean/components. Shape `(m,d) → (m,k)`. Blank 10 calls your W7 `reconstruct` using `Z` and the same saved state. Shape `(m,k) → (m,d)`. Neither method calls `fit` or computes a new batch mean.

For the hand example, a validation row `[[3,5]]` has score $4\sqrt{2}$ (or its sign-flipped equivalent), reconstructs to `[[4,4]]`, and has MSE **1**. Re-centering that one validation row would incorrectly produce zero scores.

Append these checks to **the same `test_code/week_08_practice.py`** after completing the class:

```python
from mini_ml.decomposition import PCA, reconstruction_error

model = PCA(1, solver="eigh")
assert model.fit(X) is model
Z = model.transform(X)
X_hat = model.inverse_transform(Z)
assert Z.shape == (4, 1)
np.testing.assert_allclose(model.explained_variance_, [16 / 3])
np.testing.assert_allclose(model.explained_variance_ratio_, [0.8])
np.testing.assert_allclose(X_hat, [[2, 2], [-2, -2], [0, 0], [0, 0]], atol=1e-12)
assert np.isclose(reconstruction_error(X, X_hat), 0.5)
W = model.components_
np.testing.assert_allclose(W @ W.T, np.eye(1), atol=1e-12)
np.testing.assert_allclose(C @ W.T, W.T * model.explained_variance_, atol=1e-12)
np.testing.assert_allclose(model.inverse_transform(model.transform([[3., 5.]])), [[4., 4.]])
np.testing.assert_allclose(model.mean_, [0., 0.])
assert model.n_samples_ == 4
full = PCA(2, solver="eigh").fit(X)
np.testing.assert_allclose(full.inverse_transform(full.transform(X)), X, atol=1e-12)
print("Hand calculation and saved-state checks passed")
```

```sh
uv run python test_code/week_08_practice.py
uv run pytest -q tests/test_week_08.py
```

Expected: both printed checkpoints and all supplied tests pass. Check orientation first for matrix multiplication errors; check frozen means for validation failures.

## 7. Connect the explicit covariance path to the image helper

`auto` uses the covariance + `np.linalg.eigh` expressions **you implement inside `fit`** when `d≤30`. Thus WDBC's 30 features use your explicit eigendecomposition path. The helper does not hide this operation.

For more than 30 features, `auto` uses the teacher's **NumPy-only** randomized SVD helper. It uses a local seed, 10 oversampling directions and 7 QR-stabilized power iterations, with thin SVD when the sketch reaches the smaller matrix dimension. For an exact thin SVD `Xc = U @ diag(s) @ Q.T`, `Q` holds right singular-vector **columns**, and `Xc.T @ Xc = Q @ diag(s²) @ Q.T`. Thus the matching covariance eigenvalue is `s²/(n−1)`. The helper uses this conversion with its approximate singular values; selected columns of `Q` still become the rows of `components_`. See the parameter concepts in [randomized SVD documentation](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html); sklearn is used only for external comparisons, not inside `mini_ml`.

Explicit `solver="eigh"` and `solver="randomized"` are also supported. Randomized results approximate the leading subspace and its variances; the stored tail variances can differ slightly from exact variances of transformed training scores. A fixed seed makes this course implementation reproducible. Different numerical libraries may choose different signs or tied directions. Compare reference results with a tolerance, not identical component entries or identical final decimal places.

## 8. Run your PCA on real data

```sh
uv run python test_code/week_08.py --headless
uv run python test_code/week_08.py --dataset wdbc
```

Inspect and rotate the WDBC plots, then close their windows before starting the image explorer:

```sh
uv run python test_code/week_08.py
```

`--headless` checks both datasets and saves figures and a numerical summary without opening windows. Each demo launch fits both models once; the dataset option selects which native windows to open. A first image fit takes some time; wait for it to finish before using controls. In a notebook/server without a desktop, use `--headless`. Use `--help` for explicit data/output directories.

The approximately **8.7 MB** offline pack contains `wdbc.npz`, `kather_train.npz`, `kather_validation.npz`, `kather_examples.npz` and `provenance.json`. It contains no pretrained PCA or test images. NPZ files load with `allow_pickle=False`.

**WDBC:** 341 training and 114 validation records, all 30 named features. The split uses row indices into sklearn's bundled WDBC table; the training indices and first two features exactly match W7. These indices are not the source dataset's clinical ID column. The W5 Standardizer fits training only, then PCA fits standardized training only. Rotate the PC1–3 plot and inspect all 30 coefficients for each of PC1, PC2 and PC3. Both split plots share axis limits; axis variance percentages refer to training. Labels (malignant=1) color plots and never enter PCA. Loadings are combination weights; they are not diagnostic feature importance.

**Kather:** 3,000 training and 1,000 validation RGB thumbnails, 28×28×3. Convert to float64, flatten C-order to 2,352 features and divide by 255. Do not standardize each pixel. The eight original 150×150 validation tiles are the same W7 examples. Compare reconstructions against the 28×28 inputs, not the originals.

The demo fits **one 300-component image model**. Choices `k=3,10,30,100,300` use prefixes of that same basis; changing a control does not fit again. Nested orthogonal projections cannot increase MSE. View signed component maps with a shared diverging scale and mean ± `2 sqrt(variance)` × component as RGB images. These are directions, not literal tissue templates. RGB clipping is for display only; calculate MSE on unclipped reconstructions. The summary distinguishes all **1,000 validation images** from the **eight displayed examples**.

Expected: WDBC 3D scores and 30-feature loadings, variance curves, signed image components, mean/component images, eight reconstructions and a decreasing MSE curve. No training occurs in the browser slides, which use saved visual results; Python outputs are computed by your implementation.

Headless outputs go to `test_code/` by default:

| Output | Files |
|---|---|
| Eight figures | `week_08_wdbc_scores.png`, `week_08_wdbc_loadings.png`, `week_08_variance.png`, `week_08_tissue_scores.png`, `week_08_tissue_components.png`, `week_08_tissue_reconstruction.png`, `week_08_tissue_mse.png`, `week_08_tissue_explorer.png` |
| Three tables | `week_08_scores.csv` (PC1–3, both datasets/splits), `week_08_wdbc_coefficients.csv` (all 30 components), `week_08_tissue_mse.csv` (each k; named eight/all-1,000 means) |
| State summary | `week_08_state.json` (fit dimensions/timing, explained variances/ratios, data hashes and MSE populations) |

## 9. Validate and submit

```sh
uv run pytest -q
uv run python ../../tools/course.py test <login> 8
git diff --check
git status --short
```

Expected: cumulative W1–W8 acceptance passes. Checks cover state, both solvers, sorting/transpose, total-variance denominator, constant data, invalid inputs, input preservation, transforms before fit, refit and validation using training state. The independent acceptance runner calls your installed package in a separate worker; changing a local test does not change acceptance.

Submit your updated `src/mini_ml/decomposition.py` and supplied `tests/test_week_08.py` in your own student directory. Keep experiments, generated plots, data and large binaries out of the PR. Describe your results in the PR: PC1–3 plot, one component's named weights, image reconstructions and variance/MSE observations (attach selected figures when useful). Keep your existing PR convention and deadline.

Answer these questions:

1. Why does maximizing training score variance over unit directions lead to the largest covariance eigenvalue?
2. What matrix goes into `np.linalg.eigh`, and why do we sort its eigenvalues and transpose selected eigenvector columns?
3. Why can the first component explain 80%, rather than 100%, in our hand example?
4. Why must a validation batch use the training mean and scale?
5. What changes and what stays unchanged after a simultaneous component/score sign flip?
6. Why do lower reconstruction error and higher explained variance not establish diagnostic accuracy?

## 90-minute route and sources

0–10 min: W7 recap, centered data and shapes. 10–30: covariance, direction variance and eigendecomposition. 30–40: correlated hand example and NumPy checks. 40–60: fill eight fit expressions. 60–67: transform/reconstruct and saved-state checks. 67–76: WDBC exploration. 76–84: image basis, nested k and interpretation. 84–90: cumulative checks and exit questions.

[WDBC source](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), CC BY 4.0; [Kather 2016 tiles](https://doi.org/10.5281/zenodo.53169), CC BY 4.0. The pack's provenance records IDs, filenames, classes, preprocessing, fixed stratified seed-42 splits and SHA-256 hashes. Kather uses 375 training and 125 validation tiles per class; the 1,000 held-out test tiles are not included. This tile split does not establish patient-independent validation. No classification or test evaluation is required this week. Week 9 will use the learned scores to train a classifier.

## Extra material: visualize image PC1–PC30

This optional viewer extends the image-component view from three to **30 learned directions**. It uses your completed `PCA.fit`, the existing `test_code/week_08.py`, and the Week 8 data already on your computer. It adds no assignment requirements or dependencies.

**Already working in class?** Stay on your current branch. Copy the complete block below into a **new** file, `test_code/week_08_extra_30pcs.py`, beside `week_08.py`. You can copy it directly from this online lesson without pulling or switching branches. Keep your existing demo, tests and `src/mini_ml/decomposition.py` as they are. If your Week 8 demo already runs, you have everything needed. If data or `week_08.py` are missing, run the existing Week 8 sync command from section 1 first.

From your **package root** (`students/<name_studentnumber>/`), run:

```sh
uv run python test_code/week_08_extra_30pcs.py
```

For a terminal, server or notebook environment without native plot windows, save the same figures without opening them:

```sh
uv run python test_code/week_08_extra_30pcs.py --headless
```

```python
"""Optional Week 8 viewer. Save beside the existing test_code/week_08.py."""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
from week_08 import PAPER, fit_models, load_data


def component_grid(model, signed):
    import matplotlib.pyplot as plt

    # Reuse the first 30 directions of one 300-component training fit.
    components = model.components_[:30].reshape(30, 28, 28, 3)
    mean = model.mean_.reshape(28, 28, 3)
    bound = max(float(np.abs(components).max()), np.finfo(float).eps)
    fig, axes = plt.subplots(6, 5, figsize=(15, 11), facecolor=PAPER,
                             layout="constrained")
    for i, ax in enumerate(axes.flat):
        if signed:
            # Three scalar channel maps preserve positive AND negative weights.
            panel = np.concatenate([components[i, :, :, c] for c in range(3)], axis=1)
            image = ax.imshow(panel, cmap="RdBu_r", vmin=-bound, vmax=bound,
                              interpolation="nearest")
        else:
            delta = 2 * np.sqrt(model.explained_variance_[i]) * components[i]
            panel = np.concatenate([mean - delta, mean, mean + delta], axis=1)
            ax.imshow(np.clip(panel, 0, 1), interpolation="nearest")
        ax.set_title(f"PC{i + 1} | {model.explained_variance_ratio_[i]:.2%}", fontsize=10)
        ax.set_axis_off()
    if signed:
        fig.suptitle("PC1–PC30: signed component weights\n"
                     "Each panel: R / G / B channels", fontsize=16)
        fig.colorbar(image, ax=axes.ravel().tolist(), orientation="horizontal",
                     shrink=.7, pad=.025,
                     label="Shared coefficient scale: blue negative, white zero, red positive")
    else:
        fig.suptitle("PC1–PC30: isolated changes around the training mean\n"
                     "Each panel: mean − 2σ × PC / mean / mean + 2σ × PC", fontsize=16)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Loading the existing offline pack and fitting your PCA. Please wait...", flush=True)
    models = fit_models(load_data())
    model = models["tissue"]
    output = Path(__file__).resolve().parent
    figures = []
    for signed, filename in (
        (True, "week_08_extra_30pcs_signed.png"),
        (False, "week_08_extra_30pcs_mean_modes.png"),
    ):
        fig = component_grid(model, signed)
        fig.savefig(output / filename, dpi=130, facecolor=PAPER)
        figures.append(fig)
        print(f"Saved {output / filename}")
    print(f"PC1–PC30 retain {model.explained_variance_ratio_[:30].sum():.2%} "
          "of total training pixel variance.")
    if args.headless or figures[0].canvas.required_interactive_framework is None:
        plt.close("all")
        print("Open the two saved PNG files to view all 30 components.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
```

Expected: two windows, or two PNG files in `test_code/`, each with **30 panels ordered PC1 through PC30**, left to right and then top to bottom. The program always saves both PNGs, including when you use native windows. The existing helper also performs its small WDBC fit; the image PCA is fitted **once with 300 components**, and both figures reuse its first 30. This preserves the basis used by the original image demo rather than fitting a separate 30-component randomized approximation. A new launch fits again; it does not load a pretrained PCA.

| Figure | How to read one panel |
|---|---|
| `week_08_extra_30pcs_signed.png` | R, G and B **coefficient maps**, side by side. All 90 channel maps share one symmetric color scale. These diverging colors represent signed weights, not tissue RGB colors. |
| `week_08_extra_30pcs_mean_modes.png` | Three RGB images: `mean − 2σ × PC`, the training mean, and `mean + 2σ × PC`, with `σ = sqrt(explained_variance_[i])`. Only this PC changes; all other scores stay zero. RGB values clip to `[0,1]` for display. These are synthesized variations, not individual image reconstructions. |

Each panel's percentage is that PC's fraction of **all training pixel variance**. The first 30 percentages do not have to sum to 100%. Component signs can flip across valid implementations: that swaps the minus/plus images but preserves the learned direction. Components describe pixel combinations, not tissue classes or diagnostic importance.

If you see `NotImplementedError: TODO`, finish the remaining expressions in **your `PCA.fit`**. If Python cannot import `week_08`, check that the new file is in `test_code/` beside the already-synced `week_08.py`; run the command from package root. Keep this optional script and its generated images in ignored `test_code/`, outside your submission.
