# Python Workshop：Build Machine Learning from Scratch

生科／生技背景的 Python beginner；10 週，每週 90 分鐘，以短講解與立即操作交錯進行。
以公式 → 純 Python → NumPy／sklearn 比較逐步學習，和正課保持弱連結。
所有人維護一份累積式 package，第 8 週進行架構重構。日常試算放本機 test_code/，正式實作、unit tests 與期末範例提交 PR。

| 週 | 核心 | 手刻重點 | 小成果 |
| --- | --- | --- | --- |
| 1 | uv、Python、package、Git、CI | hello_world、public API、unit test | 可 editable install、外部 import 的 package 與第一次 PR |
| 2 | if、for、list、累加 | sum_values、mean | statistics.py |
| 3 | def、dict、CSV、缺失值 | variance、std；covariance 帶寫 | 擴充相同 statistics.py |
| 4 | vector、shape、NumPy | dot；matvec 帶寫 | 細胞核形態的加權分數 |
| 5 | 圖形、normalization、distance | 距離與 transform | 訓練資料標準化與圖形比較 |
| 6 | 機率、loss、unit tests | class_proportions、mse | 機率摘要與可重跑的測試 |
| 7 | k-NN、train/validation、baseline | 鄰居選擇與投票 | 第一個分類器 |
| 8 | 重構、OOP、gradient descent | 移動到 math/、models/；梯度更新 | 前週行為不變，新增 LinearRegressor |
| 9 | sigmoid、BCE、training loop | 在相同 linear_models.py 加入 logistic | LogisticRegressor 與版本標記 |
| 10 | biological application、評估 | 組裝自己的模型流程 | examples/breast_cancer.py、比較結果、README |

每週一份 PR；星期四晚上教師手動發布解答；星期五最新 CI 通過即算完成，教師另行 review／merge。
前週未合併時先請教師協助，不要求初學者維護相依 PR。
第 8 週重構由教師引導，線性回歸提供更多骨架；每週只要求 1–2 個核心實作片段。
PyTorch、通用矩陣乘法、Cell Painting／BBBC 額外專題為延伸。
