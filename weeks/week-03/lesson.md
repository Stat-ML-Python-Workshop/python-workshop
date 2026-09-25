# Week 03 — Cross-Entropy and KL Divergence

## What you will build

Extend the same `mini_ml` package with four functions. Variance,
cross-entropy, and KL divergence use explicit Python loops; standard deviation
reuses variance:

```python
variance(values, ddof=1)
std(values, ddof=1)
cross_entropy(p, q)
kl_divergence(p, q)
```

Variance and standard deviation prepare the package for feature scaling in
Week 5. Cross-entropy and KL divergence compare two probability distributions
and prepare the idea of a model loss used later in the workshop. Use NumPy only
to check your answers, not to implement these functions. Keep all working Week
1–2 code.

Replace `<login>` with your GitHub login and `<name_studentnumber>` with your
registered folder from `roster.json`. Do not type the angle brackets. A
**repository root** contains `tools/` and `students/`; your **package root** is
`students/<name_studentnumber>/`. Run each command in the stated directory.

### Optional: start an Antigravity conversation

Paste one of these prompts at the beginning of a new conversation. You only
need one language.

**中文**

```text
請先說明你準備進行的操作，等我確認後再執行。保留現有變更，只修改我的學生資料夾，也不要直接完成 lesson 中的核心 TODO。
```

**English**

```text
Explain what you plan to do and wait for my confirmation before acting. Preserve existing changes, modify only my student folder, and do not complete the lesson’s core TODOs for me.
```

### If Week 2 is unfinished

Once `reference/week-02/` has been released, you may use it to catch up. Stay on
your Week 2 branch, integrate the reference into your own student folder, run
the Week 1–2 tests, and update your existing Week 2 PR. Add
`I used the released Week 2 reference solution to catch up before starting Week 3.`
to the PR description. After the tests pass, create your Week 3 branch from
that work. Using the reference does not change the saved Week 2 cutoff result.
If the reference is not public yet, ask the instructor rather than looking for
a private copy.

**Optional Antigravity prompt — 中文**

```text
我的 Week 2 還沒完成。請依照 reference/week-02 的說明，把解答整合到我的學生資料夾、執行 Week 1–2 測試，並協助我更新原本的 PR。
```

**Optional Antigravity prompt — English**

```text
My Week 2 work is unfinished. Follow reference/week-02, integrate the answer into my student folder, run the Week 1–2 tests, and help me update the existing PR.
```

## Step 1 — Start from your Week 2 work

**Working directory:** your package root, then the repository root.

```sh
uv sync --frozen
uv run pytest
uv run python ../../tools/course.py test <login> 2
git status --short
cd ../..
```

**Expected:** Week 1–2 tests pass and status is empty. Commit unfinished work on
its existing branch before continuing; do not discard it.

If your Week 2 PR is merged:

```sh
git switch main
git pull --ff-only origin main
git switch -c <login>/week-03
```

If Week 2 is not merged, start from its working code:

```sh
git switch <login>/week-02
git fetch origin
git merge origin/main
git switch -c <login>/week-03
```

Resolve merge conflicts with help. Do not replace your package with another
student's files. **Check:** `git branch --show-current` prints
`<login>/week-03`, and all Week 2 modules and tests remain present.

**Optional Antigravity prompt — 中文**

```text
請檢查目前的 Git 狀態，依照 Week 3 Step 1 幫我建立 <login>/week-03 branch。
```

**Optional Antigravity prompt — English**

```text
Check my Git status and follow Week 3 Step 1 to help me create the <login>/week-03 branch.
```

## Step 2 — Get this week's experiment

**Working directory:** repository root; then enter your package root.

```sh
cd students/<name_studentnumber>
uv run python ../../tools/course.py sync <login> 3
```

`sync` copies the released demo into your ignored `test_code/` directory. It
does not create, complete, or overwrite `src/mini_ml/` modules. It preserves
existing practice files.

**Expected/check:** `test_code/week_03.py` exists. If Week 3 is not released,
wait for the instructor; do not edit shared course configuration to bypass the
release check.

## Step 3 — Calculate variance and standard deviation by hand

**Working directory:** package root; no file changes needed.

For `values = [2, 4, 6]`, the mean is `4`, the deviations are `[-2, 0, 2]`,
and the squared deviations are `[4, 0, 4]`. Their sum is `8`.

The `ddof` argument changes the denominator:

| Call | Denominator | Variance | Use |
| --- | ---: | ---: | --- |
| `variance(values, ddof=0)` | `n = 3` | `8/3` | Describe this complete array; Week 5 scaling |
| `variance(values, ddof=1)` | `n-1 = 2` | `4` | Estimate population variance from a sample |

The population mean is usually unknown, so a sample calculation substitutes
the sample mean. Because that center was selected from the same observations,
it is especially close to them. The squared deviations around it are therefore
systematically too small for estimating population variance. Dividing by
`n-1` rather than `n` is **Bessel's correction**; it makes the sample variance
an unbiased estimator of population variance.

This correction applies exactly to the **variance** estimate. Taking the square
root is nonlinear, so `std(values, ddof=1)` still has a small downward bias as
an estimator of population standard deviation. You do not need to derive that
bias in this workshop.

Standard deviation is the square root of variance. Therefore the default sample
standard deviation for `[2, 4, 6]` is `2`. In Week 5, the package will call
`std(column, ddof=0)` to match NumPy and scikit-learn feature scaling.

**Check:** explain why estimating one center leaves `n-1` freely varying
deviations, and why `ddof=0` is appropriate when describing the observed
training array itself.

## Step 4 — Compare two biological distributions by hand

Suppose a treatment sample has the observed cell-type distribution `p`, while
`q` is a reference or model distribution. The positions must describe the same
cell types in the same order.

| Cell type | `p` treatment | `q` reference/model |
| --- | ---: | ---: |
| T cell | `0.75` | `0.50` |
| B cell | `0.25` | `0.50` |

Shannon entropy describes one distribution:

`H(p) = -Σ p_i log2(p_i) ≈ 0.811278 bits`.

Cross-entropy asks how many bits are needed on average when events follow `p`
but are described using `q`:

`H(p, q) = -Σ p_i log2(q_i) = 1 bit`.

KL divergence is the extra cost of using `q` instead of the true `p`:

`D_KL(p || q) = H(p, q) - H(p) ≈ 0.188722 bits`.

Direction matters. For these values, `D_KL(q || p) ≈ 0.207519`, which is not
the same result. KL divergence is not a distance metric.

Use these boundary rules:

- If `p_i == 0`, that event contributes `0`; do not evaluate its logarithm.
- If `p_i > 0` but `q_i == 0`, the result is infinity because `q` says an
  observed event is impossible.
- Both inputs must be valid distributions of equal length.

**Check:** predict `cross_entropy(p, p)` and `kl_divergence(p, p)` before coding.
Cross-entropy will return later as a model-training loss. Week 9 uses a binary
version with natural logarithms and reports nats rather than bits.

## Step 5 — Append two fragments without replacing Week 2

**Working directory:** package root. Open the target files in your editor.

Append each fragment to the matching existing module:

| Fragment relative to package root | Append to |
| --- | --- |
| `../../weeks/week-03/fragments/01_statistics.py` | `src/mini_ml/statistics.py` |
| `../../weeks/week-03/fragments/02_information.py` | `src/mini_ml/information.py` |

Do not replace either complete file. Preserve `sum_values`, `mean`,
`_validate_distribution`, and `shannon_entropy`. The new functions depend on
those earlier definitions.

```sh
uv run python -c "from mini_ml.statistics import variance, std; from mini_ml.information import cross_entropy, kl_divergence; print('imports passed')"
```

**Expected:** `imports passed`. The functions may still raise
`NotImplementedError` when called; that is expected before Steps 6–8.

**Optional Antigravity prompt — 中文**

```text
請依照 Week 3 Step 5，把兩個 fragments append 到正確的既有 modules，保留原本內容，並執行 import check。
```

**Optional Antigravity prompt — English**

```text
Follow Week 3 Step 5, append the two fragments to the existing modules without replacing earlier code, and run the import check.
```

## Step 6 — Complete `variance` and `std`

**Working directory:** package root; edit `src/mini_ml/statistics.py`.

The provided validation requires `0 <= ddof < n`. Keep it unchanged. The
sample mean, accumulator, and explicit loop are already present.

Inside the `variance` student region:

1. Replace the loop TODO with an update that adds this observation's squared
   deviation from `center` to `total`.
2. Replace the normalization TODO with a return value using the denominator
   `len(values) - ddof`.

Inside `std`, replace its TODO with the square root of the variance calculated
using the same `values` and `ddof`. Do not duplicate the variance calculation.

Use loops, not NumPy, comprehensions, or the built-in `sum` for the core
variance calculation.

```sh
uv run python -c "from mini_ml.statistics import variance, std; print(variance([2,4,6]), variance([2,4,6], ddof=0), std([2,4,6]))"
```

**Expected:** approximately `4.0 2.6666666666666665 2.0`. Also check that
`variance([7,7,7])` is `0`, while `variance([2])` with the default `ddof=1`
raises `ValueError`.

## Step 7 — Complete `cross_entropy`

**Working directory:** package root; edit `src/mini_ml/information.py`.

The supplied validation checks both distributions and their lengths. Keep it.
The accumulator and `for actual, model in zip(p, q)` loop are provided.

Replace the loop TODO with logic that:

1. skips an event whose actual probability is zero;
2. returns infinity when the actual probability is positive but the model
   probability is zero;
3. otherwise updates the accumulator using the Step 4 cross-entropy formula.

Keep the final return. Use the explicit loop rather than NumPy, a comprehension,
or a generator expression.

```sh
uv run python -c "from mini_ml.information import cross_entropy; print(cross_entropy([0.75,0.25], [0.5,0.5]))"
```

**Expected:** `1.0`. Also check that `cross_entropy([1,0], [0,1])` returns
`inf`, not a math-domain exception.

## Step 8 — Complete `kl_divergence`

**Working directory:** package root; continue editing `information.py`.

Use the provided accumulator and loop. Apply the same two zero-probability
rules, then update the accumulator with the directed formula from Step 4. Do
not swap `actual` and `model`, and do not replace the exercise with a one-line
call to the other information functions.

```sh
uv run python -c "from mini_ml.information import kl_divergence; print(kl_divergence([0.75,0.25], [0.5,0.5])); print(kl_divergence([0.5,0.5], [0.75,0.25]))"
```

**Expected:** approximately `0.1887218755` and `0.2075187496`. They differ
because KL divergence is directed. `kl_divergence(p, p)` must be approximately
zero.

## Step 9 — Add tests and run all checks

**Working directory:** package root.

Create `tests/test_statistics.py`:

```python
import pytest
from mini_ml.statistics import std, variance


@pytest.mark.parametrize("values, ddof, expected", [
    ([2, 4, 6], 1, 4.0),
    ([2, 4, 6], 0, 8 / 3),
    ([7, 7, 7], 1, 0.0),
    # STUDENT TODO: add one original (values, ddof, expected) case.
])
def test_variance(values, ddof, expected):
    assert variance(values, ddof=ddof) == pytest.approx(expected)


def test_standard_deviation():
    assert std([2, 4, 6]) == pytest.approx(2.0)


@pytest.mark.parametrize("values, ddof", [([], 0), ([2], 1), ([2, 4], -1)])
def test_invalid_variance(values, ddof):
    with pytest.raises(ValueError):
        variance(values, ddof=ddof)
```

In `tests/test_composition.py`, extend the existing information import:

```python
from mini_ml.information import cross_entropy, kl_divergence, shannon_entropy
```

Then append these tests. Add one original valid `(p, q)` pair to each marked
parametrized list.

```python
@pytest.mark.parametrize("p, q, expected", [
    ([0.75, 0.25], [0.5, 0.5], 1.0),
    ([0.5, 0.5], [0.5, 0.5], 1.0),
    # STUDENT TODO: add one original cross-entropy (p, q, expected) case.
])
def test_cross_entropy(p, q, expected):
    assert cross_entropy(p, q) == pytest.approx(expected)


@pytest.mark.parametrize("p, q, expected", [
    ([0.75, 0.25], [0.5, 0.5], 0.18872187554086717),
    ([0.5, 0.5], [0.5, 0.5], 0.0),
    # STUDENT TODO: add one original KL (p, q, expected) case.
])
def test_kl_divergence(p, q, expected):
    assert kl_divergence(p, q) == pytest.approx(expected, abs=1e-12)


def test_information_boundaries():
    assert cross_entropy([0, 1], [0, 1]) == pytest.approx(0.0)
    assert cross_entropy([1, 0], [0, 1]) == float("inf")
    assert kl_divergence([1, 0], [0, 1]) == float("inf")
    with pytest.raises(ValueError):
        kl_divergence([0.5, 0.5], [1.0])
```

Run the three levels of checks:

```sh
uv run pytest
uv run python test_code/week_03.py
uv run python ../../tools/course.py test <login> 3
```

**Expected:** your student tests pass, the demo prints
`NumPy reference checks passed.`, and cumulative Week 1–3 acceptance passes.
NumPy is a reference only; the four core functions must retain explicit Python
loops.

Your package tests are yours to extend and commit. Repository-root trusted tests
are instructor files; do not edit or copy them to bypass grading.

**Optional Antigravity prompt — 中文**

```text
請依照 Week 3 Step 9 建立並延伸測試，再執行三個檢查；若失敗，解釋第一個錯誤並給我提示，不要直接完成核心 TODO。
```

**Optional Antigravity prompt — English**

```text
Follow Week 3 Step 9 to create and extend the tests, then run all three checks. If one fails, explain the first error and give me a hint without completing the core TODOs.
```

## Step 10 — Commit, push, and update the PR

**Working directory:** repository root.

```sh
cd ../..
git status --short
git diff -- students/<name_studentnumber>
git add students/<name_studentnumber>/src/mini_ml/statistics.py
git add students/<name_studentnumber>/src/mini_ml/information.py
git add students/<name_studentnumber>/tests/test_statistics.py
git add students/<name_studentnumber>/tests/test_composition.py
git commit -m "Complete week 3 statistics and divergence exercises"
git push -u origin <login>/week-03
```

Do not use `git add .`. Inspect the staged diff with `git diff --cached`. Do not
commit `.venv/`, `test_code/`, shared `weeks/`, or repository-root tests. Open
or update a PR titled `[Week 03] <login>` and report:

- why `ddof=1` changes the variance denominator;
- why KL divergence has a direction;
- the result of the three checks from Step 9;
- whether a released Week 2 reference was used.

If CI fails, keep the same PR, fix the first real error locally, rerun all three
checks, commit, and push again.

**Optional Antigravity prompt — 中文**

```text
請依照 Week 3 Step 10 檢查 diff，協助我精確 stage、commit 和 push；不要使用 git add .。
```

**Optional Antigravity prompt — English**

```text
Follow Week 3 Step 10 to review the diff and help me stage, commit, and push the changes. Do not use git add .
```

## Completed package and submission checklist

```text
students/<name_studentnumber>/
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
├── src/mini_ml/
│   ├── __init__.py
│   ├── greetings.py
│   ├── statistics.py          # mean plus completed variance and std
│   ├── probability.py
│   └── information.py         # entropy plus completed divergence functions
├── tests/
│   ├── test_greetings.py
│   ├── test_composition.py    # retained Week 2 tests plus Week 3 tests
│   └── test_statistics.py
└── test_code/week_03.py       # local only, ignored
```

- All four Week 3 functions are complete and follow the required loop/reuse structure.
- Week 1–2 functions and required Week 2 test cases remain unchanged and pass.
- Each new parametrized group includes one original student case.
- `test_code/` and `.venv/` remain uncommitted.
- Only the registered student folder appears in the PR diff.

## Deadline and reference release

The weekly cutoff is **Thursday 23:59, Asia/Taipei**. The latest PR commit's
trusted automatic-test result at the cutoff determines `tests_passed`; leave
time for CI to finish. The scheduled reference release starts at that cutoff.
Later fixes may still be reviewed and merged, but do not rewrite the saved cutoff
result. If you use a released reference later, disclose that in your PR.

## Common problems

| Symptom | What to check |
| --- | --- |
| `NotImplementedError` | Replace every Week 3 TODO without deleting validation or earlier functions. |
| Wrong population variance | `ddof=0` divides by `n`, not `n-1`. |
| Wrong sample variance | Use squared deviations from the sample mean and divide by `n-1`. |
| `math domain error` | Skip `p_i == 0` before evaluating a logarithm. |
| Expected infinity but got an error | Return infinity when `p_i > 0` and `q_i == 0`. |
| Negative KL value | Check the `p/q` direction inside the logarithm. |
| Length error | `p` and `q` must describe the same ordered categories. |
| Week 2 tests disappeared | Append to modules and tests; do not replace complete files. |
| NumPy comparison passes but CI fails | The required implementation must use explicit Python loops. |
| CI scope failure | Only files inside your registered student folder may change. |

## 90-minute session

- 0–5: review Week 2 entropy and confirm branch/sync.
- 5–12: calculate variance/std and explain `ddof` without deriving it.
- 12–25: compare `p` and `q`, cross-entropy, KL direction, and boundary cases.
- 25–30: map fragments, tests, and later Week 5/9 connections.
- 30–55: implement variance and std.
- 55–75: implement cross-entropy and KL divergence.
- 75–85: write tests and run all three checks.
- 85–90: inspect the diff, commit, push, and update the PR.
