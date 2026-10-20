# HANDOFF — P0 → P1

日期：2026-10-19

## 已完成
- 使用者建立 `Seoul_Bike_Sharing_Project` 初始目錄並完成 CSV 下載。
- Kaggle `joebeachcapital/seoul-bike-sharing` Version 1，SHA-256 `373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60`。
- P0 實測資料 8760×14、12 個原始輸入、One-Hot 後 14 維、缺失／重複／無效日期均為零。
- 研究情境經使用者同意：在已知營運狀態與氣象條件下，預測每小時實際租借量。

## P1 待確認
1. 什麼時間點進行預測？`Functioning Day` 和天氣觀測值能否於該時間點取得？
2. 預測對象是否包括停駛時段的零租借量？
3. 統一定義 Baseline、MAE／RMSE／R² 的評估用途，但不自行設定硬性分數。
4. 明確說明時間切分與評估場景。
5. 記錄 Kaggle／原始 UCI 資料出處與再散布條款。

## 禁止提前執行
P1 未確認前不啟動 P2～P5；不覆寫原始 CSV，不擅自把 Laptop Price 舊專案改名或刪除。
