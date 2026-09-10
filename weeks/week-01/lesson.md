# Week 01 — 建立第一個可安裝的 Python Package

## 本週完成目標

今天不先學機器學習公式。我們先建立一個之後能連續成長十週的 Python package。下課前你要能：

1. 使用 uv 建立 Python 3.12 環境。
2. 在 `students/<姓名_學號>/` 親手建立 src layout。
3. 以 editable mode 安裝自己的 package。
4. 從 package 外部 import `hello_world()` 並印出結果。
5. 跑過第一個 unit test，送出第一個 PR，看到 CI 結果。

AI 可以協助建立檔案骨架，但你需要能指出每個檔案的用途，並自行修正到本機測試及 CI 通過。

## 上課前檢查表

- [ ] 建立 GitHub 帳號，申請加入 `Stat-ML-Python-Workshop`，並接受 repo Write 權限邀請。
- [ ] 安裝 Git；終端機執行 `git --version` 有顯示版本。
- [ ] 安裝 uv；終端機執行 `uv --version` 有顯示版本。
- [ ] 設定 Git 的姓名與 email。
- [ ] 安裝 IDE。推薦 Antigravity，也可使用 VS Code 或其他 IDE。
- [ ] 建議安裝 Git Graph；Markdown 可直接使用 IDE 內建預覽。

Antigravity 與教育優惠是選配，不影響課程驗收。

## 90 分鐘：每學一段就操作

| 時間 | 概念 | 立即操作與檢查點 |
| --- | --- | --- |
| 0–5 | 今日目標與最後會完成的成品 | 確認 Git、uv、IDE 可使用 |
| 5–17 | repository、main 與 branch | clone repo，建立自己的 Week 1 branch，以 Git Graph 確認 |
| 17–27 | Python 如何執行、語法、動態型別與取捨 | 建立簡短 `.py`，修改字串後再執行 |
| 27–39 | interpreter、virtual environment、dependency；uv 與 Anaconda | 在自己的 package 根目錄建立 `.venv`，確認 Python 3.12 |
| 39–57 | package、module、public API、src layout 與責任分離 | 親手建立自己的 package 結構 |
| 57–68 | editable installation、`uv sync` 與外部 import | 安裝 package，從 `test_code/` 印出 `Hello, world!` |
| 68–76 | unit test、expected/actual 與 pytest | 先看到一次本機失敗，再修正至通過 |
| 76–84 | commit、push 與 Pull Request | 建立 `[Week 01] <login>` PR |
| 84–90 | GitHub Actions、Docker 與 CI | 打開 Actions，觀察 queued、running、passed/failed |

## Step 1：建立 branch 和自己的資料夾

請把 `<login>` 換成 GitHub 帳號，`<folder>` 換成教師登錄的姓名學號資料夾。

```sh
git clone https://github.com/Stat-ML-Python-Workshop/python-workshop.git
cd python-workshop
git switch -c <login>/week-01
mkdir -p students/<folder>
cd students/<folder>
```

完成後確認目前 branch 是 `<login>/week-01`，不要直接在 `main` 修改。

## Step 2：第一個 Python 檔案

先建立 `test_code/python_basics.py`：

```python
language = "Python"
print("Hello from", language)
```

```sh
mkdir -p test_code
uv run --no-project --python 3.12 python test_code/python_basics.py
```

把 `language` 改成自己的名字後再執行一次。這裡先觀察變數、字串、`print()` 與「修改原始碼後重新交給 interpreter 執行」；細部語法會在後續週次逐步補上。

## Step 3：建立自己的環境

```sh
uv venv --python 3.12
uv pip list
```

`.venv` 是這位學生、這份 package 的隔離環境。Anaconda／conda 也能管理環境，但本課統一使用 uv，避免同一專案混用兩套安裝方式。

## Step 4：建立 codebase structure

使用 IDE 親手建立以下結構。`uv.lock` 會由稍後的 `uv lock` 產生，不要手寫；`test_code/` 只留在本機，不提交。

```text
students/<姓名_學號>/
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock                  # uv 產生
├── src/
│   └── mini_ml/
│       ├── __init__.py
│       └── greetings.py
├── tests/
│   └── test_greetings.py
└── test_code/
    └── try_package.py
```

`pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=75,<81"]
build-backend = "setuptools.build_meta"

[project]
name = "workshop-mini-ml"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = []

[dependency-groups]
dev = ["pytest>=8.3,<9"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`：

```gitignore
.venv/
__pycache__/
.pytest_cache/
*.pyc
*.egg-info/
test_code/
```

`README.md` 至少寫下 package 名稱、本週完成內容，以及如何執行測試。

## Step 5：建立第一個 public API

依序建立 `src/mini_ml/greetings.py`、`src/mini_ml/__init__.py` 與 `tests/test_greetings.py`。內容直接放在本講義中，可自行輸入、copy/paste，或請 AI 協助建立後逐行核對。

`src/mini_ml/greetings.py`：

```python
def hello_world() -> str:
    return "Hello, world!"
```

`src/mini_ml/__init__.py`：

```python
from .greetings import hello_world

__all__ = ["hello_world"]
```

`tests/test_greetings.py`：

```python
from mini_ml import hello_world

def test_hello_world():
    result = hello_world()
    assert result == "Hello, world!"
```

`hello_world()` 回傳資料；外部程式才負責 `print()`。這讓相同函式能被終端機、測試或未來其他程式重用。

在 `test_code/try_package.py` 寫下：

```python
from mini_ml import hello_world

print(hello_world())
```

## Step 6：安裝與執行

```sh
uv pip install -e .
uv lock
uv sync --frozen
uv run python test_code/try_package.py
```

預期最後一行顯示：

```text
Hello, world!
```

- `uv pip install -e .` 明確示範 editable installation：修改 `src/mini_ml/` 後不必每次重新複製 package。
- `uv sync` 是之後每週用來依 `pyproject.toml` 與 `uv.lock` 同步環境的日常指令。

## Step 7：第一次 unit test

先暫時把 test 中的 expected value 改錯，執行一次：

```sh
uv run pytest
```

閱讀 expected 與 actual，修正後再執行，直到測試通過。unit test 留在 `tests/` 並提交；本機試算留在 `test_code/`。

## Step 8：提交第一個 PR

```sh
git status
git add students/<folder>
git commit -m "Complete week 01 package setup"
git push -u origin <login>/week-01
```

PR 標題使用 `[Week 01] <login>`。CI 失敗時，在相同 branch 修正並再次 push，不需要另開 PR。

## GitHub Actions、Docker 與 CI

```text
學生 push
    ↓
Pull Request 更新
    ↓
GitHub Actions 啟動
    ↓
檢查帳號、branch、週次與修改範圍
    ↓
建立隔離的 Docker container
    ↓
執行教師 unit tests
    ↓
回報 Passed／Failed
```

- GitHub Actions 安排自動驗收流程。
- Docker image 是標準測試環境的藍圖；container 是每次 CI 建立的臨時環境。
- 第一週不要求安裝 Docker，也不教 Dockerfile 指令。
- 本課目前使用 CI 自動測試，沒有自動部署的 CD。

## 驗收

- branch 為 `<GitHub login>/week-01`，且只修改自己的 `students/<姓名_學號>/`。
- `uv.lock`、package source 與 `tests/test_greetings.py` 必須提交；`test_code/` 不得提交。
- package 能從專案外部安裝及 import，`hello_world()` 回傳 `Hello, world!`，本機 pytest 與可信 CI 均通過。
- 星期四晚上發布解答；星期五以 PR 最新 commit 的 CI 為完成依據，教師 review 與 merge 可稍後進行。
