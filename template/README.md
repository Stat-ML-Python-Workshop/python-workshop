# 我的 mini ML library

一份 package 持續累積十週。src/mini_ml 放實作，tests 放 unit tests，test_code 僅本機試算。

```sh
uv sync --frozen
uv run pytest
uv run python test_code/week_01.py
```

第 8 週依教材將統計與線代移到 math/、模型移到 models/；先前測試仍須通過。
第 10 週執行 `uv run python examples/breast_cancer.py`，將 JSON 與圖形留在本機 test_code/。

請逐週補上：新增的函式、如何執行、測試结果、參考解答情況。
期末補上資料來源、模型比較結果、precision／recall／ROC AUC 的解釋與限制。
