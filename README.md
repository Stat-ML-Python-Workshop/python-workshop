# Python Workshop — Build Machine Learning from Scratch

給生科／生技背景的 Python 初學者。10 週，每週 60 分鐘授課＋30 分鐘操作。
公式 → 自己的 Python → NumPy／sklearn 核對；每人逐步完成自己的 mini ML library。

## 第一次開始

先安裝 [Git](https://git-scm.com/downloads)、[uv](https://docs.astral.sh/uv/getting-started/installation/) 及支援 notebook 的編輯器。GitHub Desktop 可以替代部分 Git 操作。
請教師先把你的 **GitHub login** 加入名單並給予本 repo Write 權限。
以下指令都在 repo 根目錄執行；把 `<login>` 換成自己的帳號。

```sh
git clone https://github.com/Stat-ML-Python-Workshop/python-workshop.git
cd python-workshop
git switch -c <login>/week-01
uv run --no-project --python 3.12 python tools/course.py init <login> 1
uv sync --project students/<login>
```

在編輯器開啟 `students/<login>/notebooks/week-01.ipynb`，選擇
`students/<login>/.venv` 的 Python kernel。先閱讀 `weeks/week-01/lesson.md`。
第一週就會看到 `src/`、`tests/`、`pyproject.toml`、README；先會使用，不要求立即了解打包細節。

## 實作、測試、提交

```sh
uv run --project students/<login> python tools/course.py test <login> 1
git status
git diff
git add students/<login>
git commit -m "Complete week 01 core exercise"
git push -u origin <login>/week-01
```

在 GitHub 開 PR 到 main，標題 `[Week 01] <login>`，填寫 PR 模板。
原始骨架包含 `NotImplementedError`，第一次測試失敗是預期；完成填空再測。
`tests/week-XX` 是當週公開驗收測試；你可以另外在自己的 `tests/` 撰寫測試。
不要修改共用測試來通過驗收。CI 使用受保護的驗收版本，不執行學生的打包設定。

## 每週往下累積

等前週 PR 合併後，切回 main，pull，再建立下一週分支：

```sh
git switch main
git pull --ff-only
git switch -c <login>/week-02
uv run --no-project --python 3.12 python tools/course.py sync <login> 2
uv sync --project students/<login>
uv run --project students/<login> python tools/course.py test <login> 2
```

sync 只新增檔案，不覆蓋你已有的實作。後續週次自行替換數字。
前週還沒合併時，先請教師協助；不要同時維護互相依賴的作業 PR。

## 每週節奏

| 時間 | 要做的事 |
| --- | --- |
| 星期二上課後 | 自行實作、提交一份 PR、依 CI 修正 |
| 星期四晚上 | 教師手動發布當週 `reference/week-XX/` 解答；需要時複製到自己 package |
| 星期五 | 最新 commit 的正式 CI 通過即算當週完成，與 merge 分開記錄 |
| 後續 | 教師審核並 merge；未通過者在原 PR 繼續補交 |

解答是協助學習的工具；請在 PR 說明參考情況。程式與 PR 都是公開的，不放學號、私人研究資料或憑證。

## 十週路線

1. 環境、Python 與 package
2. 迴圈、加總與平均
3. 函式、module、variance、SD、covariance
4. NumPy、dot 與矩陣
5. 圖形、標準化與距離
6. 機率比例、loss 與測試
7. k-NN 分類
8. OOP、linear regression 與梯度
9. logistic regression 與訓練
10. 真實 biological dataset 整合

詳見 [課綱](SYLLABUS.md)。未發布的週次教材與解答不在此 repo。

## 資料與引用

期末使用 [Wisconsin Breast Cancer Diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)，透過 sklearn 載入。569 筆、30 個細胞核形態特徵，主線先選 mean radius／mean texture。資料為 CC BY 4.0；作者 Wolberg、Mangasarian、Street、Street，DOI: 10.24432/C5DW2B。
教學載入器將標籤轉成 **1=malignant、0=benign**。固定切分，scaler 僅 fit train，validation 選模型，test 最後評估。

