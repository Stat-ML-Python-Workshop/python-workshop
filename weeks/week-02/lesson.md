# Week 02 — From Biological Counts to Shannon Entropy

## What you will build

Extend your existing `mini_ml` package with two functions written using Python loops:
`class_proportions(labels)` and `shannon_entropy(probabilities)`. Use NumPy only to
check your results, not to implement these functions. The instructor provides the
statistics helpers and input validation. Keep all working Week 1 code.

Replace `<login>` with your GitHub login and `<name_studentnumber>` with your
registered folder from `roster.json`. Do not type the angle brackets. A **repository
root** contains `tools/` and `students/`; your **package root** is
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

### If Week 1 is unfinished

Once `reference/week-01/` has been released, you may use it to catch up. Stay on
your Week 1 branch, integrate the reference into your own student folder, run the
Week 1 tests, and update your existing Week 1 PR (or open it if necessary). Add
`I used the released Week 1 reference solution to catch up before starting Week 2.`
to the PR description. After the tests pass, create your Week 2 branch from that
work. Using the reference does not change the saved Week 1 cutoff result. If the
reference is not public yet, ask the instructor rather than looking for a private copy.

**Optional Antigravity prompt — 中文**

```text
我的 Week 1 還沒完成。請閱讀 reference/week-01 的說明和 fragments，幫我把解答整合到自己的學生資料夾、執行 Week 1 測試，並更新原本的 PR。
```

**Optional Antigravity prompt — English**

```text
My Week 1 work is unfinished. Read the instructions and fragments in reference/week-01, integrate the solution into my student folder, run the Week 1 tests, and help me update the existing PR.
```

## Step 1 — Start from your Week 1 work

**Working directory:** your package root, then the repository root.

```sh
uv sync --frozen
uv run pytest
uv run python ../../tools/course.py test <login> 1
git status --short
cd ../..
```

**Expected:** Week 1 passes and status is empty. If you have unfinished changes,
commit your own work on its existing branch before continuing; do not discard it.

If your Week 1 PR is merged:

```sh
git switch main
git pull --ff-only origin main
git switch -c <login>/week-02
```

If Week 1 is not merged, keep that PR open and start from its working code:

```sh
git switch <login>/week-01
git fetch origin
git merge origin/main
git switch -c <login>/week-02
```

Resolve any merge conflicts with help before proceeding. Do not replace your
package with another student's files. **Check:** `git branch --show-current`
prints `<login>/week-02` and your Week 1 files are still present. The unmerged
path includes Week 1 changes in the Week 2 PR until Week 1 reaches main.

**Optional Antigravity prompt — 中文**

```text
請檢查目前的 Git 狀態，幫我依照 Week 2 Step 1 建立 <login>/week-02 branch。
```

**Optional Antigravity prompt — English**

```text
Check my current Git status and help me create the <login>/week-02 branch by following Week 2 Step 1.
```

## Step 2 — Get this week's experiment

**Working directory:** repository root; enter your package root:

```sh
cd students/<name_studentnumber>
uv run python ../../tools/course.py sync <login> 2
```

`sync` copies released demos and any data into your local practice area. It does
not create, complete, or overwrite your `src/mini_ml/` modules. It also preserves
existing practice files; ask the instructor if you need a refreshed demo.

**Expected/check:** `test_code/week_02.py` exists. Keep `test_code/` and `.venv/`
ignored. If Week 2 is not released, wait for the instructor; do not edit
`course.json` or shared course tools to bypass the check.

## Step 3 — Add NumPy for reference checks

**Working directory:** package root.

```sh
uv add numpy
uv run python -c "import numpy; print(numpy.__version__)"
git status --short
```

**Expected:** NumPy imports; `pyproject.toml` and `uv.lock` are updated.
**Check:** keep both changes for your commit. NumPy is a project dependency,
not a replacement for the loop exercise. Continue using the Week 1 uv environment.

## Step 4 — Predict the cell composition by hand

**Working directory:** package root (no file changes needed).

Imagine four observed immune cells: `['T', 'T', 'B', 'M']`. Count each label,
then divide its count by the total number of cells. The result is
`{'T': 0.5, 'B': 0.25, 'M': 0.25}`; proportions sum to 1.

Shannon entropy is `H(p) = −Σ pᵢ log₂(pᵢ)`. It describes uncertainty in a
randomly selected cell's label. Base-2 logarithms give **bits**. Greater entropy
means greater compositional diversity, not better sample quality.

**Action:** calculate these values on paper before writing code.

| Probabilities | Expected entropy (bits) |
| --- | ---: |
| `[0.5, 0.25, 0.25]` | 1.5 |
| `[0.75, 0.25]` | approximately 0.811278 |
| `[0.5, 0.5]` | 1.0 |
| `[1.0, 0.0]` | 0.0 |

**Check:** explain why balanced two-class composition is more uncertain than
skewed composition. A zero-probability class contributes zero; never evaluate
`log2(0)`. This convention does not make an empty distribution valid.

## Step 5 — Create three modules, preserving Week 1

**Working directory:** package root. Open the files in your editor.

Copy the full contents of each provided fragment to its target **new** file:

| Fragment relative to package root | Target relative to package root |
| --- | --- |
| `../../weeks/week-02/fragments/01_statistics.py` | `src/mini_ml/statistics.py` |
| `../../weeks/week-02/fragments/02_probability.py` | `src/mini_ml/probability.py` |
| `../../weeks/week-02/fragments/03_information.py` | `src/mini_ml/information.py` |

If a target already exists, inspect and integrate it; do not overwrite your work.
Keep `greetings.py`, `__init__.py`, and `tests/test_greetings.py` unchanged.
The statistics helpers `sum_values` and `mean` are already complete. Do not edit
the provided `_validate_distribution` helper or change the public function names.

```sh
uv run python -c "from mini_ml.probability import class_proportions; from mini_ml.information import shannon_entropy; print('Imports OK')"
```

**Expected/check:** imports succeed, but calling an unfinished core function with
valid input raises `NotImplementedError`. Import success is not task completion.

**Optional Antigravity prompt — 中文**

```text
請依照 Week 2 Step 5，把三個 fragments 複製到正確的 mini_ml 路徑，並執行 import check。
```

**Optional Antigravity prompt — English**

```text
Follow Week 2 Step 5, copy the three fragments to the correct mini_ml paths, and run the import check.
```

## Step 6 — Complete `class_proportions`

**Working directory:** package root; edit `src/mini_ml/probability.py`.

Input is a nonempty list of hashable labels (for example strings or integers).
Output is a dictionary mapping each observed label to its floating-point
proportion. Preserve label types: integer labels must remain integer keys.
An empty list must raise the provided `ValueError`.

Inside the `BEGIN STUDENT` / `END STUDENT` region:

1. In the first loop, replace its TODO exception with a count update. Think about
   both the first occurrence of a label and later occurrences.
2. In the second loop, replace its TODO exception with a normalized dictionary
   entry. Which denominator represents all observed cells?
3. Keep the initialized dictionaries, loop structure, and returned dictionary.

Use loops, not `Counter`, comprehensions, or NumPy counting functions.

```sh
uv run python -c "from mini_ml.probability import class_proportions; print(class_proportions(['T', 'T', 'B', 'M']))"
```

**Expected:** the Step 4 dictionary (key order does not matter).
**Check:** one class has proportion 1; `[0, 0, 1]` has integer keys and
proportions approximately 2/3 and 1/3. Do not hard-code these examples.

## Step 7 — Complete `shannon_entropy`

**Working directory:** package root; edit `src/mini_ml/information.py`.

Input is a nonempty list of nonnegative probabilities summing approximately to 1.
Output is a floating-point entropy in bits. The supplied validation rejects
empty, negative, and incorrectly normalized distributions with `ValueError`.
Do not normalize invalid input silently.

The accumulator and loop are provided. Replace the TODO exception with a
condition that selects positive probabilities and, inside that condition, an
update to the accumulator using the formula from Step 4. Skip zero terms.
Keep the final return. Use an explicit loop, not a generator, comprehension,
`sum`, or NumPy reduction for the core calculation.

```sh
uv run python -c "from mini_ml.information import shannon_entropy; print(shannon_entropy([0.5, 0.25, 0.25]))"
```

**Expected:** `1.5`. **Check:** `[1.0, 0.0]` returns zero without a logarithm
error; `[0.2, 0.2]` raises `ValueError`. Do not remove validation to make tests pass.

## Step 8 — Add your own unit tests

**Working directory:** package root. Create `tests/test_composition.py` with this
executable starting set. Add one new label/proportion example of your own to the
first parametrized list and one new probability/entropy example of your own to
the second:

```python
import pytest
from mini_ml.probability import class_proportions
from mini_ml.information import shannon_entropy


@pytest.mark.parametrize("labels, expected", [
    (["T", "T", "B", "M"], {"T": 0.5, "B": 0.25, "M": 0.25}),
    (["B", "B"], {"B": 1.0}),
    ([0, 0, 1], {0: 2 / 3, 1: 1 / 3}),
    # STUDENT TODO: add one more (labels, expected) example here.
])
def test_proportions(labels, expected):
    result = class_proportions(labels)
    assert result == pytest.approx(expected)
    assert sum(result.values()) == pytest.approx(1.0)


def test_empty_labels():
    with pytest.raises(ValueError):
        class_proportions([])


@pytest.mark.parametrize("probabilities, expected", [
    ([0.5, 0.5], 1.0),
    ([1.0], 0.0),
    ([1.0, 0.0], 0.0),
    ([0.5, 0.25, 0.25], 1.5),
    # STUDENT TODO: add one more (probabilities, expected) example here.
])
def test_entropy(probabilities, expected):
    assert shannon_entropy(probabilities) == pytest.approx(expected)


@pytest.mark.parametrize("probabilities", [[], [-0.1, 1.1], [0.2, 0.2]])
def test_invalid_distribution(probabilities):
    with pytest.raises(ValueError):
        shannon_entropy(probabilities)
```

```sh
uv run pytest
```

**Expected/check:** your new tests and the Week 1 greeting test pass. Your two
additions should exercise one new proportion case and one new entropy case. Use
`pytest.approx` for floating-point results. The `sum` call above checks an output;
it does not implement your core functions. Your package's `tests/` are yours to
extend and commit. The repository-root `tests/week-XX/` are instructor acceptance
tests: do not edit or copy them into your package to bypass grading.

Trusted CI verifies that `tests/test_composition.py` retains the supplied cases,
contains at least one original case in each of the two marked parametrized lists,
and that all student tests pass.

**Optional Antigravity prompt for Steps 8–9 — 中文**

```text
請依照 Week 2 Step 8 建立測試檔，再執行 Step 9 的三個檢查；若失敗，請解釋第一個錯誤並給我提示。
```

**Optional Antigravity prompt for Steps 8–9 — English**

```text
Create the test file from Week 2 Step 8 and run the three checks in Step 9. If one fails, explain the first error and give me a hint.
```

## Step 9 — Compare with NumPy and run cumulative acceptance

**Working directory:** package root.

```sh
uv run python test_code/week_02.py
uv run pytest
uv run python ../../tools/course.py test <login> 2
```

**Expected:** the demo prints approximately 0.811278 bits for skewed composition,
1.000000 bits for balanced composition, and `NumPy reference checks passed.`
Dictionary display order may differ. NumPy is an independent reference in the
demo; your core functions must still use loops. The course command checks
Week 1–2 cumulatively as well as your own tests, not just the new functions.

**Check:** all three commands succeed. Before both TODO regions are completed,
failures are expected. Read the first traceback, fix the function, and rerun;
do not delete a test or change the expected result to hide a failure.

## Step 10 — Submit the Week 2 PR

**Working directory:** package root for the final tests, then repository root.

```sh
uv run pytest
uv run python ../../tools/course.py test <login> 2
cd ../..
git status --short
git diff -- students/<name_studentnumber>/
git add students/<name_studentnumber>/src/mini_ml/statistics.py students/<name_studentnumber>/src/mini_ml/probability.py students/<name_studentnumber>/src/mini_ml/information.py students/<name_studentnumber>/tests/test_composition.py students/<name_studentnumber>/pyproject.toml students/<name_studentnumber>/uv.lock
git diff --cached --name-only
git diff --cached
git commit -m "Complete Week 02 cell composition and entropy"
git push -u origin <login>/week-02
```

Before committing, confirm every staged file belongs to your own folder. If
you staged something accidentally, use `git restore --staged <exact-file>`;
this unstages it without deleting your working copy. Never force-add `test_code/`
or `.venv/`, and never commit someone else's work or shared course infrastructure.

Open GitHub → Pull requests → New pull request. Select base `main` and compare
`<login>/week-02`. Use title **`Week 02 — <login> — Cell composition and entropy`**.
Describe your implementation, commands tested, and interpretation of the demo.
If Week 1 is still unmerged, link that PR and explain the dependency.

**Expected/check:** the PR changes only your folder and automatic checks pass.
If CI fails, inspect its log, reproduce locally, fix and commit on the same
Week 2 branch, then `git push`. The same PR updates automatically; do not open
a replacement PR merely to hide a failure. Ask for review after checks pass.

**Optional Antigravity prompt — 中文**

```text
請依照 Week 2 Step 10 檢查 diff，並協助我 stage、commit 和 push；不要使用 git add .。
```

**Optional Antigravity prompt — English**

```text
Follow Week 2 Step 10 to review the diff and help me stage, commit, and push the changes. Do not use git add .
```

## Completed package and submission checklist

```text
students/<name_studentnumber>/
├── pyproject.toml                 # includes NumPy
├── uv.lock
├── README.md
├── .gitignore
├── src/mini_ml/
│   ├── __init__.py                # preserve Week 1 API
│   ├── greetings.py
│   ├── statistics.py              # instructor-provided
│   ├── probability.py             # complete the loop exercise
│   └── information.py             # complete the loop exercise
├── tests/
│   ├── test_greetings.py
│   └── test_composition.py
└── test_code/week_02.py           # local only, ignored
```

- Both core functions use loops and have no unfinished TODO exceptions.
- The three new modules, your unit tests, `pyproject.toml`, and `uv.lock` are committed.
- Week 1 behavior and all Week 1–2 checks pass; the demo passes independently.
- `test_code/` and `.venv/` are not committed; shared files are unchanged.
- Your PR explains what entropy tells you and links any unmerged Week 1 PR.

## Deadline and reference release

The weekly cutoff is **Thursday 23:59, Asia/Taipei**. The latest PR commit's
trusted automatic-test result at the cutoff determines `tests_passed`; leave
time for CI to finish. The scheduled reference release starts at that cutoff
and appears after its publication checks succeed. Instructor review and eventual
merge are tracked separately, with merge reconciliation Friday at 00:30.
Later fixes may still be reviewed and merged, but do not rewrite the saved
cutoff result. If you use the released reference later, disclose that in your PR.

## Common problems

| Symptom | What to check |
| --- | --- |
| `ModuleNotFoundError: mini_ml` | Work in your package root; run `uv sync --frozen`; preserve Week 1 src layout and build configuration. |
| NumPy cannot import | Run `uv add numpy` in your package, not at the repository root. |
| `NotImplementedError` | A core exercise is unfinished; replace the TODO, not the test. |
| `math domain error` | Zero probabilities must be skipped before calling `log2`. |
| Negative entropy for a valid distribution | Revisit the sign in the entropy formula. |
| Incorrect counts | Check repeated labels and the first occurrence of each label. |
| `ValueError` for a distribution | It must be nonempty, nonnegative, and sum approximately to 1. |
| Sync appears to do nothing | It preserves existing demo files and never fills in modules. |
| CI scope failure | Inspect the PR diff; only your registered student folder may change. |

## 90-minute session

- 0–10: confirm Week 1, branch, sync, and add NumPy (Steps 1–3).
- 10–25: predict cell proportions and entropy by hand (Step 4).
- 25–35: map fragments to modules and confirm imports (Step 5).
- 35–60: complete the two loop exercises (Steps 6–7).
- 60–80: write tests, compare with NumPy, and run cumulative checks (Steps 8–9).
- 80–90: inspect the diff, commit, push, and open the PR (Step 10).
