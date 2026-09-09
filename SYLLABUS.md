# Python Workshop：Build Machine Learning from Scratch

儲存日期：2026-09-09。來源：使用者共編的文件區塊 73916；保留共編內容，僅整理 Markdown 格式。

**對象：**生科／生技背景的 Python 初學者  
**安排：**10 週，每週 60 分鐘授課＋30 分鐘 hands-on  
**目標：**逐步完成自己的 mini ML library，最後應用於真實 biological dataset。與正課 detailed syllabus 保持弱連結。

每週採用「**數學公式 → 自己的 Python implementation → NumPy／sklearn 正式版本比較**」。教師提供骨架，學生完成核心運算。

| 週次／主題 | 核心概念 | 手刻 function／class | 可用現成 library | Biological example | 當週小成果 |
| --- | --- | --- | --- | --- | --- |
| **1｜環境、Python 入門與 package 結構** | 認識一個 Python package 的基本構造（`src/`、`tests/`、`pyproject.toml`、README）；使用 uv 建立與管理環境；notebook、變數、運算、list、import；Git 與 GitHub 基礎；將公式拆成步驟；clone workshop repository 並建立本地環境 | 建立並使用之後持續沿用的 basic package template；先寫三筆資料的加總、平均運算式；預覽 `sum_values()`、`mean()` | uv；notebook；內建 `sum()` 作核對；Git、GitHub、GitHub Desktop 或命令列工具 | 三次模擬細胞培養量測 | 成功 clone workshop repository，使用 uv 建立本地環境並執行範例；理解 package 基本結構；建立可安裝、可 import 的 basic package template，完成第一次 commit 並推送至 GitHub |
| **2｜Programming basics** | 索引、for、if、計數與累加；Git 基本流程（clone、status、add、commit、push、pull） | 補完 `sum_values()`、`mean()` 的迴圈；函式外殼先提供；以 commit 記錄每次練習 | Python 內建語法；Git、GitHub | 不同培養條件的量測資料 | 能處理任意長度資料，篩選後計算平均；能從 GitHub 同步、提交並查看程式碼變更 |
| **3｜函式、資料結構與 module** | def、return、dict、CSV、缺失值；variance、SD、covariance；repository、README 與 `.gitignore` | **必寫：**`variance()`、`std()`；教師帶寫 `covariance()`；整理 module 並提交版本 | pandas 讀檔；NumPy 核對；GitHub README、`.gitignore` | 比較兩組量測的變異；兩項指標是否共同變化 | 建立可 import 的統計 module，輸出分組摘要；在 GitHub 補上 README 與專案說明 |
| **4｜NumPy 與線性代數** | vector、shape、樣本 × 特徵、dot、矩陣乘法；branch、merge 與版本管理 | **必寫：**`dot()`；帶寫 `matvec()`；小型 `matmul()` 示範；以 branch 開發並合併 | NumPy array、`@` 作比較；Git branch、GitHub Pull Request | 以細胞核形態特徵計算加權分數 | 純 Python 與 NumPy 算出一致的加權和；完成一次 branch、merge 或 Pull Request |
| **5｜Matplotlib、標準化與距離** | 散布圖、直方圖、尺度、normalization、Euclidean distance；GitHub issue 與程式碼審查 | **必寫：**`euclidean_distance()`、標準化核心式；提供 scaler 骨架；依 issue 修正程式 | matplotlib；`StandardScaler` 作比較；GitHub Issues、Pull Requests | Breast Cancer 的細胞核半徑、紋理 | 畫出標準化前後的資料，解釋距離如何改變；能建立 issue、提交修正並回應 review |
| **6｜機率與 loss** | 計數 → 比例 → 經驗機率；預測誤差、MSE；測試、commit 與可追蹤的實驗紀錄 | `class_proportions()`、`mse()`；為核心函式撰寫簡單測試 | NumPy、sklearn metrics；SciPy 檢定保留短示範；GitHub Actions 可作延伸 | 良性／惡性比例；模擬螢光讀值的預測誤差 | 完成比例表與 loss 計算，理解模型如何被評分；提交測試與實驗結果，保持專案可重現 |
| **7｜第一個 ML：k-NN** | 距離、排序、多數決；train／validation／test、baseline；從 GitHub 取得資料與執行專案 | **必寫：**`knn_predict_one()` 核心；提供批次預測骨架；以 issue 記錄模型問題 | sklearn 切分資料、`KNeighborsClassifier` 作比較；GitHub repository | Breast Cancer 二分類 | 自寫分類器產生預測，與 baseline、sklearn 比較；能依 README 從 GitHub 下載並重現結果 |
| **8｜OOP 與 linear regression** | class、instance、method；線性模型、MSE、gradient descent；協作開發與 Pull Request | 提供 `LinearRegressor.fit/predict`；**必寫梯度與參數更新**；示範將 k-NN 包成 class | sklearn `LinearRegression`；PyTorch 僅延伸示範；GitHub Pull Request | 模擬濃度與螢光讀值的校正曲線 | 模型學到斜率與截距，畫出 loss 曲線；完成一個具說明與 review 紀錄的 Pull Request |
| **9｜訓練與 logistic regression** | sigmoid、binary cross-entropy、learning rate、training loop、分類閾值；版本標籤與 release | 提供 `LogisticRegressor`；**必寫 sigmoid 核心與更新片段**；BCE 帶寫；以 tag 標記可重現版本 | matplotlib；sklearn `LogisticRegression` 作比較；Git tag、GitHub Release | 用細胞核形態預測良性／惡性 | 訓練 logistic 模型，改 learning rate 並解釋結果；建立一個可重現的 Git tag 或 GitHub Release |
| **10｜Biological application** | 完整資料流程、評估、ROC AUC、可重現性、結果限制；ROC AUC 衡量模型在所有分類閾值下區分陽性與陰性的能力：ROC 曲線以假陽性率（FPR）為 x 軸、真陽性率（TPR）為 y 軸，AUC 是曲線下面積，數值越接近 1 代表排序區分能力越好，0.5 約等於隨機猜測；應使用預測機率或 decision score 計算，並注意類別不平衡與資料洩漏；整理 GitHub 專案、README 與版本紀錄 | 整合自己的 scaler、k-NN 或 logistic；不新增演算法；計算 confusion matrix、precision、recall、ROC AUC，並比較不同模型與分類閾值；整理完整程式碼與文件 | pandas、matplotlib、sklearn 評估與參考模型；GitHub repository、README、Release | Wisconsin Breast Cancer 真實資料 | 一份可完整重跑的 notebook、模型比較表與結果解釋；將成果整理至 GitHub，附上安裝方式、資料來源、執行步驟與限制說明 |

## 教學範圍

- 每週完成 1–2 個核心程式片段，其餘由教師帶寫或提供骨架。
- 第 3 週開始整理自己的 module；第 7 週認識模型的 `fit/predict`；第 8 週正式定義 class。
- 最低共同成果為完整的自寫 k-NN 分類流程；logistic regression 在骨架支援下完成。
- 標準化只由訓練資料估計參數；validation 用於調整設定，test 留到最後評估。
- PyTorch 為延伸示範，不列必修。
- Cell Painting／BBBC 作為延伸 project，優先使用教師整理好的形態特徵表。
