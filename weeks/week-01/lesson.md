# Week 01 — 環境、package 與本機試算

## 本週核心
建立 package；在 test_code/ 用三筆量測算平均。公式：**mean=(a+b+c)/3**。
同一份 package 持續擴充，不建立 week_XX 版本的 mini_ml。

## 開始時的 codebase
```text
src/mini_ml/__init__.py
```

## 本週操作與新增介面
初始化工具已提供 package。自行在 test_code/ 試算，確認 import mini_ml 成功。

片段中的 `BEGIN STUDENT` 是核心填空，其餘由教師帶寫或提供。
append 表示加入新函式，不能覆蓋整份檔案；第 3 週保留 sum_values／mean，第 9 週保留 LinearRegressor。

## 完成後的 codebase
```text
src/mini_ml/__init__.py
```

## 練習與正式比較
同步後，在自己的 package 根目錄執行 `uv run python test_code/week_01.py`。
此程式呼叫自己的 package 並與正式工具核對，只在本機，不列入 PR。
正式 unit tests 放在 tests/，需要提交；共用驗收使用 repo 的 course.py 指令。

## 90 分鐘安排
- 0–10 分：前週回顧與小例子的輸出預測。
- 10–30 分：公式、生物案例與輸入輸出。
- 30–60 分：教師展示片段、import 與正式版本比較。
- 60–85 分：完成 1–2 個核心片段、跑累積測試。
- 85–90 分：commit、push，更新當週 PR。

新增一小段就跑測試，前週功能也必須通過。

## 驗收
分支 `<GitHub login>/week-01`，只修改自己的 students/<姓名_學號>/。
星期四晚上手動發布解答，星期五最新 commit CI 通過即算完成；教師另行 review／merge。
通用矩陣運算、PyTorch、其他 biological datasets 為延伸，不阻擋必修。
