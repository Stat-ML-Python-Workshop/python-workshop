# Week 7: Projection and Reconstruction

Open [index.html](index.html) directly in a browser. The twelve-slide English deck works offline, including geometry controls, eight tissue examples, and all five component settings. References open external source pages only when clicked.

## Presenting

- Use the arrow keys, Previous/Next buttons, or slide picker. The URL retains the current slide number.
- Press **N** or select **Teacher notes** for the current slide. Escape closes notes. Controls keep their normal keyboard behavior.
- Choose **Print** for twelve landscape pages. The current control settings appear on printed pages. Teacher notes stay out of the printout.
- Use **Fullscreen** when the browser permits it. On narrow windows, content stacks vertically and the page scrolls.

## Teaching sequence: 90 minutes

| Slides | Activity | Minutes |
|---|---|---:|
| 1–2 | Outcomes, shared components, signed scores, Week 6 recap | 10 |
| 3 | One-observation hand calculation | 9 |
| 4 | Batch shapes, complete `project` | 10 |
| 5 | Restore the mean, complete `reconstruct` | 10 |
| 6 | One/full-basis experiments and checks | 11 |
| 7 | MSE hand calculation, complete `reconstruction_error` | 10 |
| 8–10 | WDBC and tissue experiments | 18 |
| 11–12 | Verify, interpret, submit, connect to Week 8 | 12 |

The student worksheet supplies commands, starter integration, tests, and submission details. In the instructor checkout it is under `curriculum/week-07/lesson.md`; in the student repository it is under `weeks/week-07/lesson.md`.

## Reading the reference figures

Browser geometry diagrams are explicit instructor examples. WDBC contains 341 training observations and the same two standardized features used in Week 6. Pathology figures use the fixed Week 7 training basis and eight validation tiles, one per class. The mean MSE curve averages **these eight tiles**, not the full validation set. The orange dashed line follows the currently selected tile.

Reconstruction error compares 28 × 28 RGB inputs after division by 255 with the numerical reconstruction **before clipping**. PNGs clip to `[0, 1]` for display. The original 150 × 150 images illustrate the detail already lost during resizing. Each image stores its own scores; the mean and basis are shared and also need storage.

The browser figures are reference calculations. Students run their own three functions through the supplied Python demo. The deck includes formulas and hand calculations, without complete callable student solution functions.

Sources: [Kather et al. (2016), CC BY 4.0](https://doi.org/10.5281/zenodo.53169), [UCI WDBC](https://doi.org/10.24432/C5DW2B), and [PCA geometry documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca). Dataset hashes, selected tile IDs, display policy, and reference MSE values are in [provenance.json](provenance.json).

## Instructor reproduction

From the instructor repository, with NumPy and Pillow installed:

```sh
.venv/bin/python teaching/week07-projection/build_assets.py
.venv/bin/python teaching/week07-projection/verify_assets.py
```

These scripts read `tools/templates/week-07/data/pca_basis.npz` and `wdbc_basis.npz` with `allow_pickle=False`. They generate all reference images and `data.js` without downloading data or fitting a new model. Private builders, check scripts and `reports/` are excluded from the student publication bundle.
