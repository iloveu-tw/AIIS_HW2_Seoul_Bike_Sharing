# Seoul Bike Sharing P0-2 驗證報告

- 結果：**STRUCTURE_PASS_REVIEW_REQUIRED**
- Kaggle：`joebeachcapital/seoul-bike-sharing` Version 1
- SHA-256：`373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60`
- 資料筆數／欄位數：8760／14
- 目標：`Rented Bike Count`
- 缺失值／重複列：0／0
- 選用原始輸入特徵：12
- 編碼後模型維度：14
- 時間序切分：訓練 7008 筆，測試 1752 筆
- 12～29 維檢查：PASS

## 尚待人工審查
- `Functioning Day` 在實際預測時能否事先取得；若排除，原始選用特徵將只剩 11 個，應重新設計特徵。
- 時間型資料需確認切分策略，不宜直接套用一般隨機切分。
- One-Hot 使用訓練資料類別、drop=first；本階段只核對維度，沒有訓練模型。
- 不覆寫或更動原始 CSV。
