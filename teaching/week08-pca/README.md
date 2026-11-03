# Week 8 · Learning PCA

Open [index.html](index.html) in a browser. Keep this folder and its `assets/` directory together. The 17-slide lecture runs offline without a web server, installation, data downloads or model training. [slides.pdf](slides.pdf) contains the A4 landscape print edition.

Use ←/→ or Page Up/Page Down to navigate; Home/End jump to the first/last slide. Press **N** for teacher notes and Escape to close them. Arrow keys keep their normal behavior while an input, select or button has focus. The slide picker, notes, fullscreen and print buttons provide equivalent mouse controls.

The [student worksheet](../../weeks/week-08/lesson.md) supplies the ten expression exercises, commands and expected results. The 90-minute lesson progresses from a correlated hand example, a 3D variance-bound exploration, covariance eigendecomposition, NumPy `eigh` and fitted state to WDBC score/loadings views and image reconstruction. Student Python demos calculate results with the student's own PCA; this deck contains frozen, verified reference views.

- Slide 2 flips a component and its scores together.
- Slide 4 lets students move a unit direction through a centered synthetic 3D cloud. Maximum/PC1 and Minimum/PC3 buttons reveal the variance bounds and concentration of squared PC-basis weights. Camera rotation changes the view independently. This is a complete three-feature example, so PC3 is PCN.
- Slide 5 rotates a candidate direction in the two-feature hand example; slide 7 changes the retained variance fraction.
- Slide 12 rotates WDBC PC1–3 with dragging or the camera sliders. Reset restores the initial view; the split selector shows 341 training or 114 validation observations.
- Slide 13 selects PC1–3 and flips all 30 named coefficients together.
- Slide 15 selects an image component and moves between −2 and +2 training standard deviations. Signed R/G/B maps use a shared symmetric scale. RGB reconstructions clip only for display.
- Slide 16 selects the same eight validation tiles as Week 7 and k = 3, 10, 30, 100 or 300. All reconstructions use prefixes of one fitted 300-component basis. MSE is calculated before display clipping and averaged over the eight shown images.

The 3D lead-in uses 240 reproducible synthetic observations with sample-covariance eigenvalues 9, 3 and 0.6. Its weights are squared coefficients of the moving direction in the PC basis, not feature loadings. The arrow is enlarged for display while numerical calculations always use a unit direction.

WDBC uses all 30 features standardized with the training mean and population scale, followed by PCA fitted on training rows only. Display IDs are source-row indices, not clinical identifiers. Kather uses 3,000 training and 1,000 validation 28 × 28 RGB thumbnails; the eight originals are 150 × 150. Labels color or name examples and are not inputs to PCA.

Data attribution: [UCI WDBC](https://doi.org/10.24432/C5DW2B), CC BY 4.0; [Kather et al. (2016)](https://doi.org/10.5281/zenodo.53169), CC BY 4.0. [provenance.json](provenance.json) records source and asset hashes, transformations and reference settings. Numerical interfaces follow [NumPy eigh](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html) and the [randomized SVD reference](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html). External reference links open only when deliberately selected.
