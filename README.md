# Python Workshop — Build Machine Learning from Scratch

給生科／生技背景的 Python 初學者。10 週，每週 60 分鐘授課＋30 分鐘操作。
每個人建造一份持續長大的 mini ML package：公式 → 自己的 Python → 正式工具比較。

## 先認識資料夾

```text
students/<姓名_學號>/
├── pyproject.toml
├── uv.lock
├── README.md
├── src/mini_ml/       # 正式函式與模型，提交
├── tests/            # 自己的 unit tests，提交
├── test_code/        # 本機試算、畫圖、debug，忽略且不提交
├── examples/         # 第 10 週正式 biological application，提交
└── data/
```

所有人使用相同結構，但各有獨立環境和實作。GitHub login 與資料夾名稱不同，由教師在 `roster.json` 設定對照。
沒有每週複製的 package：第 3 週擴充原本的 statistics.py；第 8 週移入 math/、models/；第 9 週擴充既有 linear_models.py。
詳細架構見 [ARCHITECTURE.md](ARCHITECTURE.md)，課程路線見 [SYLLABUS.md](SYLLABUS.md)。

## 第一次開始（repo 根目錄）

先安裝 [Git](https://git-scm.com/downloads) 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
請教師登錄 GitHub login 與姓名學號資料夾，並給予公開 repo Write 權限。
`<login>` 替換成 GitHub 帳號，`<folder>` 替換成教師登錄的姓名學號資料夾。

```sh
git clone https://github.com/Stat-ML-Python-Workshop/python-workshop.git
cd python-workshop
git switch -c <login>/week-01
uv run --no-project --python 3.12 python tools/course.py init <login> 1
uv sync --frozen --project students/<folder>
uv run --frozen --project students/<folder> python tools/course.py test <login> 1
```

第 1 週 CI 驗收 package 可 import 及必要專案檔案。基本算術在本機 test_code/ 練習，不要求將試算內容提交。

## 自己試算與畫圖（學生 package 目錄）

```sh
cd students/<folder>
uv run python test_code/week_01.py
uv run pytest
```

可以自行建立 `test_code/try_mean.py`，呼叫自己的 mini_ml。測試用資料、圖片及暫存輸出也放這裡。
`uv run pytest` 只收集自己的 `tests/`；正式驗收則使用 repo 根目錄的 course.py。
可重用的計算寫進 src/mini_ml；test_code/ 的內容不會同步給教師，也不列入正式測試。

## 每週同步與新增程式（回到 repo 根目錄）

前週 PR 合併後才開始下一週：

```sh
git switch main
git pull --ff-only
git switch -c <login>/week-02
uv run --no-project --python 3.12 python tools/course.py sync <login> 2
```

sync 只新增本機示範和資料，**不替你新增或覆蓋 module**。
閱讀 `weeks/week-02/lesson.md`，依「開始結構 → 新增片段 → 完成結構」手動整合。
片段位於 `weeks/week-XX/fragments/`：create 建立檔案；append 在原檔案追加；第 8 週另有 move 與 import 調整。
第 3 週不可覆蓋先前的 sum_values／mean，第 9 週不可覆蓋 LinearRegressor。

```sh
uv run --frozen --project students/<folder> python tools/course.py test <login> 2
git status
git diff
git add students/<folder>
git commit -m "Complete week 02 core exercise"
git push -u origin <login>/week-02
```

PR 標題 `[Week 02] <login>`，填写實作、測試與參考解答情況。不要以 `git add -f` 提交 test_code/；CI 會拒絕這些檔案。
共用驗收測試在 `tests/week-XX/`，只讀不要修改；自己的測試可放在學生 package 的 tests/。

## 每週節奏

| 時間 | 流程 |
| --- | --- |
| 星期二上課後 | 自行實作並提交當週 PR |
| 星期四晚上 | 教師手動發布 reference/week-XX/；可取用需要的片段整合到自己的 codebase |
| 星期五 | PR 最新 commit 的正式 workshop/acceptance 通過即算完成 |
| 後續 | 教師 review 與 merge 另外處理；未通過者在原 PR 持續修正 |

正式狀態由教師工具核對可信 workflow 與 commit，不以任意同名綠燈判定。

## 第 10 週可重現成果

在自己的 package 目錄執行：

```sh
uv sync --frozen
uv run python examples/breast_cancer.py
```

正式範例要提交；JSON 和圖形預設輸出至本機 test_code/。README 記錄執行方法、結果摘要及限制。
資料使用 [Wisconsin Breast Cancer Diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)（CC BY 4.0；Wolberg、Mangasarian、Street、Street；DOI: 10.24432/C5DW2B）。
1=malignant、0=benign，固定切分；scaler 只 fit train、validation 選設定、test 留到最後。
未發布的教材與解答不在公開 repo；不放私人研究資料或憑證。
