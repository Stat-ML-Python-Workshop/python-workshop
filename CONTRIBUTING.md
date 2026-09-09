# 每週 PR 與常見問題

- 教師先在 roster.json 登錄 GitHub login → 姓名學號資料夾，並給學生 Write 權限。
- 分支仍是 `<GitHub login>/week-XX`；不要把姓名學號當成 GitHub 帳號。
- 所有變更限定自己的 `students/<姓名_學號>/`。test_code/ 只在本機，不能強制加入 PR。
- tests/ 是正式 unit tests，要提交；課程會用自己的共用驗收測試，忽略學生 pytest 設定。
- 第 3、9 週片段是追加內容，不能整檔覆蓋；第 8 週移動既有檔案並修正 import。
- 首週測試確認 package 與專案檔案；後續 NotImplementedError 代表核心填空尚未完成。
- 本機正式驗收使用 repo 根目錄 course.py；學生自己寫的測試在 package 目錄 `uv run pytest`。
- CI 通過即算完成；教師審核與合併稍後進行。開始下一週前請先確認前週已合併。
- 未通過者維持原 PR 補交。參考解答後 commit、push 到原分支，不另開 PR。
- 第 10 週提交 examples/breast_cancer.py 與 README 結果；暫存 JSON、圖形放 test_code/。
