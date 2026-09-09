# Week 01 — 環境、package 與 Python 入門

## 這週要完成
三次細胞培養量測。核心操作：**三筆加總與平均**。只補 `BEGIN STUDENT` 區塊；其餘先閱讀。

## 公式 → Python → 正式版本
平均 = (a+b+c)/3

先用三到五筆數據逐步計算，再執行 notebook。正式版本僅用來核對，不代替學生核心運算。

## 90 分鐘節奏
- 0–10 分：回顧前週與預測小例子的輸出。
- 10–25 分：公式、輸入／輸出與 biological example。
- 25–45 分：逐行示範與讀錯誤訊息；保留核心填空。
- 45–60 分：正式工具比較、當週 Git 操作：Git clone、uv sync、commit、push 與第一次 PR。
- 60–85 分：完成 1–2 個核心片段，執行當週累積測試。
- 85–90 分：提交／更新 PR，寫下結果與一個問題。

## 操作
先依根目錄 README 同步本週 starter（不覆蓋既有檔案），再執行：
`uv run --project students/<你的帳號> python tools/course.py test <你的帳號> 1`

本週 notebook 位於 `students/<你的帳號>/notebooks/week-01.ipynb`。
不直接執行尚未填好的下一週；每週正式驗收只包括已發布且截至這週的必修測試。

## PR 驗收
分支 `<你的帳號>/week-01`，PR 標題 `[Week 01] <你的帳號>`。
寫出使用的公式、測試結果、是否參考解答。星期四晚上教師發布解答，星期五最新 commit CI 通過即算完成；merge 由教師另外審核。

## 延伸（不影響必修）
修改數據或尺度、增加一個邊界案例；比較手刻與正式版本的差異。PyTorch、通用矩陣乘法、額外 biological datasets 不列入必修。
