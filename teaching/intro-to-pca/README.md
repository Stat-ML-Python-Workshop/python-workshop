# Intro to PCA

An interactive Week 6 lecture: **16 slides, approximately 36 minutes**.
It introduces the full PCA workflow, then returns to Week 6's three functions (`center`, `covariance_matrix`, and `variance_along_direction`).

## Open locally

Open `index.html` locally in your browser, or serve from the repository root:

```sh
python3 -m http.server 8876 --bind 127.0.0.1
```

Open <http://127.0.0.1:8876/teaching/intro-to-pca/index.html>.
The complete deck works offline: no CDN, analytics, remote scripts, runtime downloads or Python execution in the browser.

- Use Previous/Next, arrow keys or the slide menu. Slide URLs use `#slide-1` through `#slide-16`.
- **Teacher notes** (or `N`) opens the notes for the current slide.
- Sliders retain their normal arrow-key controls. String animation is opt-in, pauses on leaving the slide, and honors reduced-motion preferences.
- **Print** includes all slides; interactive figures retain their current settings.
- Follow the [Week 6 lesson](../../weeks/week-06/lesson.md) for the coding assignment.

## References

- [AT&T Laboratories Cambridge face database](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) and [scikit-learn Olivetti documentation](https://scikit-learn.org/stable/datasets/real_world.html#the-olivetti-faces-dataset).
- [Turk & Pentland, *Face Recognition Using Eigenfaces* (1991)](https://www.cs.utexas.edu/~grauman/courses/spring2007/395T/papers/turk_pentland_1991.pdf).
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) and [NumPy `eigh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html).
- [OpenStax: The Hydrogen Atom](https://openstax.org/books/university-physics-volume-3/pages/8-1-the-hydrogen-atom) and [Standing Waves and Resonance](https://openstax.org/books/university-physics-volume-1/pages/16-6-standing-waves-and-resonance).
- [Kather et al. (2016), colorectal histology image collection](https://doi.org/10.5281/zenodo.53169), CC BY 4.0.
