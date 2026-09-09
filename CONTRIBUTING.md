# 每週 PR 與常見問題

- 分支格式：`<GitHub login>/week-XX`。請先讓教師將帳號加入 roster。
- 第一次執行出現 `NotImplementedError`：這是核心填空尚未完成；依 lesson 找到 `BEGIN STUDENT`。
- CI 提示修改範圍不符：只提交自己的 `students/<login>/`，不要修改共用測試或別人的資料夾。
- 本機通過但 CI 失敗：檢查是否提交最新檔案；正式驗收忽略自訂 pytest 設定，請使用 README 的 course.py 指令。
- 教師發布解答後：需要時複製至自己的 package，再測試、commit、push 到原分支；不必重開 PR。
- 星期五 CI 尚未完成：維持原 PR 補交，通過後完成狀態才更新。
- 已通過但尚未 merge：作業驗收與教師審核分開；開始下一週前請教師協助合併。
- 尚未加入課程名單的帳號或 fork PR 不會通過正式作業驗收。

只有教師可合併 main；學生仍可互相閱讀、提問及 review。教師未發布的解答不在這個 repo。
