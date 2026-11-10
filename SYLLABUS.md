# Python Workshop: PCA Pathology Classification from Scratch

Ten 90-minute sessions for life-science Python beginners, maintaining one cumulative mini_ml package. The required goal is RGB tiles → student PCA → forward classification → manual backpropagation and SGD. An optional Week 10 vibe coding bonus adds overlapping large-image prediction maps.

**Release status:** Weeks 1–9 have lesson materials in this release; reference solutions are available for Weeks 1–6. Week 10 and its optional bonus are planned and not yet released. `course.json` records released lessons and solutions; the roadmap is not a promise that later sync or test commands already work.

| Week | Concepts | Student implementation | Visible outcome | Status |
|---|---|---|---|---|
| 1 | uv, package, Git, CI | hello_world, public API and tests | Importable editable package and PR | Released |
| 2 | Counting, probability, Shannon entropy | class_proportions, shannon_entropy | Immune-cell composition and entropy in bits | Released |
| 3 | Variance, std, cross-entropy, KL | variance, std, cross_entropy, kl_divergence | Variation and distribution comparison | Released |
| 4 | Shapes, transpose, matrix multiplication | shape, transpose, matmul | Hand-built patient-score matrix product | Released |
| 5 | NumPy, axes, broadcasting, minimal objects, scale | Standardizer.fit, transform, euclidean_distance | Label-colored WDBC plots and scale experiment | Released |
| 6 | Centering, covariance, variance directions; PCA preview | center, covariance_matrix, variance_along_direction | Native variance explorer: Synthetic/WDBC, Center only/Standardize | Released |
| 7 | Projection, reconstruction, information loss | project, reconstruct, reconstruction_error | Fixed-basis reconstruction and MSE curve | Released |
| 8 | Covariance eigendecomposition, PCA state, explained variance | PCA.fit, transform, inverse_transform | 3D variance-bound explorer, PC1–3 scores, loadings and image reconstruction | Released |
| 9 | Forward, PyTorch tensors, experimental design and metrics | Five expressions: softmax and forward; supplied autograd + SGD | Four learning rates with training/validation loss curves | Released |
| 10 | Backpropagation, batch gradients and SGD | Four expressions: manual W/b gradients and updates in PyTorch | Automatic/manual curves and numerical agreement | Planned |

W6–8 supply all student tests and independent CI acceptance; students run them without adding cases. In W8, students complete ten short expressions, including calls to `covariance_matrix(Xc)` and `np.linalg.eigh(C)`, then sort and select the principal directions. The teacher supplies input validation, object state handling, a NumPy randomized SVD helper for images, and plotting. Later scaffolds supply label encoding and stable loss. NumPy is used inside mini_ml from Week 5. Week 8 appends PCA to the existing decomposition.py module without moving earlier code. See the [Week 8 lesson](weeks/week-08/lesson.md), [17-slide offline lecture](teaching/week08-pca/README.md) and [print edition](teaching/week08-pca/slides.pdf).

The [Week 9 lesson](weeks/week-09/lesson.md) supplies tests, a complete training loop, autograd/SGD updates and prediction. Students fill only the five forward expressions and analyze loss curves. PyTorch is a required dependency from Week 9 onward.

## Data progression and final assessment

Weeks 4–7 use WDBC: fixed training patient indices, malignant=1, the same two radius/texture features throughout W4–7. A 30-feature heatmap is optional enrichment, not a W6 implementation task. Standardizer fits training only. W7 also supplies a training-only mean, 300 components and eight validation examples from Kather 2016; students do not download the full image dataset. The image path uses Kather 2016's 5,000 original eight-class RGB tiles, seed 42, stratified 60/20/20, BILINEAR 150→28 resize, C-order flatten /255 (2352 features), then training-only PCA centering. /255 is not stain normalization and does not apply WDBC feature-wise z-scores to pixels.

Week 8 expands WDBC to all 30 named features with the same 341 training and 114 validation observations. Its offline image pack contains 3,000 training and 1,000 validation thumbnails, plus the same eight original display images. It contains no test images or pretrained PCA. Students fit 300 image components once and reuse prefixes of that basis when changing k.

PCA receives no labels. Labels color plots and train/evaluate the classifier. Week 8 explores PC counts; Weeks 9–10 fix 30 training-only PCs, CPU float64, zero parameters and identical seed-42 mini-batches. Week 9 uses supplied PyTorch autograd/SGD; the planned Week 10 lab replaces gradients and updates with manual tensor expressions. Required experiments use training/validation only.

Required results include the Week 8 reconstruction and PCA explorations, Week 9 learning-rate loss curves, and Week 10 agreement between automatic and manual updates. Assessment covers correctness, reproducibility and evidence-based interpretation, not a minimum accuracy or visibly separated clusters.

## Planned Week 10 Bonus — Vibe Coding

The optional scanning task will use AI coding assistance to turn the completed classifier into a pathology-image application. It is part of Week 10 and does not add a required grading milestone. Its instructions are not yet released. Freeze the model before final held-out test evaluation, and inspect original/input crops and overlapping tissue/tumor/confidence maps. Tile splits do not establish patient independence; an unannotated large image cannot establish localization accuracy.

Lessons are English step-by-step documents with directory, destination, command, expected output, failure checks and public code provenance. Explorations stay in ignored test_code/; source, unit tests and dependency locks are submitted through weekly PRs. Existing release deadlines and student review/merge rules remain in force.
