# PROJECT PLAN — Seoul Bike Sharing Project

## 研究題目
基於 CRISP-DM 之首爾共享自行車每小時租借需求預測：多元線性迴歸模型分析。

## 課程作業與本專案的區分
- **教授明文**：Kaggle、多元線性迴歸、10 < 特徵數 < 30。
- **專案自主要求**：原始選用和 One-Hot 後維度均維持 12～29；採 CRISP-DM，確保可重現和稽核。

## P0 — Dataset Verification（已完成）
- 來源：Kaggle `joebeachcapital/seoul-bike-sharing` Version 1。
- CSV：`SeoulBikeData.csv`，cp1252，8,760 × 14。
- 目標：`Rented Bike Count`；Date 暫作時間排序，不當直接模型輸入。
- 選用特徵：9 數值 + 2 二元 + Seasons 1 類別 = 12；後者以 drop-first One-Hot 得 3 欄，合計 14 維。
- 缺失值 0、完整重複列 0、日期解析失敗 0；測試時序 7,008/1,752。
- `Functioning Day` 條件保留，須持續追蹤預測時可得性。

## P1 — Business Understanding（下一步）
釐清使用情境、模型預測時間點、營運狀態及氣象資訊可得性、研究問題、Baseline 與模型評估方式；產出 `results/P1/` 研究定義報告。

## P2 — Data Understanding
EDA：租借量分布、零租借時段、季節與天氣、時間分布、相關性、異常值及資料品質；輸出 CSV/PNG/MD/稽核紀錄。

## P3 — Data Preparation
時序切分；在訓練集擬合季節 One-Hot、假日與營運狀態的 0/1 映射、必要尺度轉換；記錄 12→14 維或修改後的實測結果，嚴防資料洩漏。

## P4 — Modeling
建立可重現的多元線性迴歸基準模型；保存完整 Pipeline、係數、預測值及執行設定。

## P5 — Evaluation
以 MAE、MSE、RMSE、R²、Baseline 和殘差圖評估；不預設教授未要求的通過分數。

## P6 — Delivery
整理可重現流程、報告、圖表、模型與限制說明；不預設需要架設網站／API。

## 跨階段驗收
每一階段都須先檢查輸出、撰寫人工審查與交接記錄，再宣告完成。
