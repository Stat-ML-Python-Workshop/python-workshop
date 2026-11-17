# Week 10：Vibe Coding 病理大圖掃描 Demo 網頁

先看投影片：[Vibe Coding 重要概念（PDF）](../../teaching/week10-vibe-coding/slides.pdf)

## 🎯 目標

用 vibe coding 做一個 **demo 網頁**：上傳一張病理大圖，就能輸出掃描預測結果，像 [README 封面那張圖](../../docs/assets/final-project/slide-scan.png)。

模型請用你自己實作的 PCA 和 Week 9 訓練好的分類器。

## 📦 Dataset（請課前準備）

Week 8 的資料包**沒有大圖**，請另外下載測試用的圖：

- 來源：Kather, Zöllner & Bianconi (2016), *Collection of textures in colorectal cancer histology*，[Zenodo 53169](https://doi.org/10.5281/zenodo.53169)
- 檔案：`Kather_texture_2016_larger_images_10`（10 張 5000×5000 的 TIF，檔案有好幾百 MB，請提早下載）
- 大圖和模型檔請放在 `test_code/` 等被 ignore 的資料夾，**不要 commit**。
- 授權是 CC BY 4.0，網頁上請註明資料來源。

## 🛠 環境

- **技術不限**：可以用 Gradio、Streamlit，也可以用 JavaScript 自己寫網頁。
- 用 Python 套件的話，請用 `uv add` 安裝，並把 `pyproject.toml` 和 `uv.lock` 一起 commit。
- demo 程式請放在你資料夾的 `examples/` 裡，**不要放進 `src/mini_ml/`**，以免影響自動評分。
- 請保留 Week 9 存下來的模型檔，網頁要載入它來預測。如果選擇純 JS，需要自己把模型參數匯出成網頁讀得到的格式。

## 🔧 提醒

- 請用自己訓練的模型，不要用 scikit-learn。結果跟封面不一樣是正常的。
- 怎麼切圖、怎麼前處理、怎麼呈現結果，請自己跟 AI 一起摸索。
  💡 提示：掃描時餵給模型的資料，要跟訓練時「長得一樣」。
- 大圖跑起來可能要一點時間，可以先拿小的裁切圖測試。
- 大圖沒有標準答案，網頁顯示的是模型的預測，不代表正確答案。

## 🚀 開始

從 `students/<name_studentnumber>/`，開 `<login>/week-10` 分支後：

```sh
uv run python ../../tools/course.py test <login> 9
uv run python ../../tools/course.py sync <login> 10
uv run python test_code/week_10.py --image <大圖路徑>
```

`week_10.py` 只會印出你 Week 9 模型檔的內容和大圖尺寸，用來確認檔案都準備好了。掃描和網頁要自己做。

## 📤 繳交

- PR 內容：`examples/` 裡的 demo 程式，以及新增的 `pyproject.toml`、`uv.lock` 變更。
- PR 說明裡附上網頁截圖，或是一張大圖的掃描結果。
- 自動測試只檢查你的 PCA 和分類器「整批預測」跟「一張一張預測」的結果一致，網頁本身由老師看 PR。

## ⭐ Bonus（選做）：手刻 backprop

把 Week 9 用 PyTorch autograd 和 SGD 做的訓練，改成自己手寫 gradient 和參數更新，再跟 autograd 的 loss curve 比較，看兩者是否一致。

