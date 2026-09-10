# 一份 codebase，十週持續長大

前七週使用單層 modules。按責任分檔，沒有 week_01_package、week_02_package 的副本。

| 週 | 成長目標 |
| --- | --- |
| 1 | greetings.py、public API、editable install 與 unit test；試算留在 test_code/ |
| 2 | statistics.py：sum_values、mean |
| 3 | 同一 statistics.py：加入 variance、std、covariance |
| 4 | linalg.py：dot、matvec |
| 5 | preprocessing.py：標準化與距離，重用 statistics |
| 6 | metrics.py：比例、loss、評分 |
| 7 | neighbors.py：以函式完成 k-NN |
| 8 | 重構 math/ 與 models/，加入 class 與線性回歸 |
| 9 | 擴充 models/linear_models.py，加入 logistic regression |
| 10 | examples/breast_cancer.py：完整生物資料應用 |

第 8 週起的結構：

```text
src/mini_ml/
├── __init__.py
├── math/
│   ├── __init__.py
│   ├── statistics.py
│   └── linalg.py
├── models/
│   ├── __init__.py
│   ├── neighbors.py
│   └── linear_models.py
├── preprocessing.py
└── metrics.py
```

依賴方向：examples 使用 mini_ml；models 使用 math、preprocessing、metrics；math 不依賴模型、讀檔或 dataset。
公式與模型放 src；資料載入、正式工具比較、繪圖、完整流程放 examples 或本機 test_code。

重構順序：先跑第 7 週測試 → 移動檔案 → 調整 import → 跑第 8 週累積測試 → 補完新模型。
第 8 週測試會要求 math/、models/ 的新位置，並沿用前七週的行為案例。不要保留舊路徑的重複實作。
舊的本機試算程式如果使用舊 import，自己改成新路徑即可；不需要把它們提交。

同學的檔案結構相同，實作可不同。新的函式必須符合教材指定介面；不要求逐行複製教師解答。
