# P0 最終驗收與選題決策

- 專案：Seoul_Bike_Sharing_Project
- 日期：2026-10-19
- 狀態：**P0 候選題目可行性驗證通過**；不等於模型已驗證。

## 來源追蹤
- Kaggle Dataset ID：`joebeachcapital/seoul-bike-sharing`
- Version：1
- Kaggle URL：https://www.kaggle.com/datasets/joebeachcapital/seoul-bike-sharing/versions/1
- 下載時間（UTC）：2026-10-19T09:44:59.183667+00:00
- 原始檔：`data/raw/SeoulBikeData.csv`
- 檔案位元組：604166
- SHA-256：`373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60`

## P0-2 實測
| 指標 | 結果 |
|---|---|
| 資料筆數／欄位 | 8760／14 |
| 目標變數 | Rented Bike Count |
| 編碼 | cp1252 |
| 缺失值／完全重複列／無效日期 | 0／0／0 |
| 選用原始輸入特徵 | 12 |
| 編碼後欄位數 | 14 |
| 時序 80/20 | 訓練 7008、測試 1752 |
| 特徵數專案限制 | PASS：12～29（原始、編碼後） |

原始輸入：`Hour`、`Temperature(°C)`、`Humidity(%)`、`Wind speed (m/s)`、`Visibility (10m)`、`Dew point temperature(°C)`、`Solar Radiation (MJ/m2)`、`Rainfall(mm)`、`Snowfall (cm)`、`Holiday`、`Functioning Day`、`Seasons`。季節 4 類，以 drop-first One-Hot 展開 3 維；總計 14 維。

## 人工研究情境決策
使用者同意：「在**已知營運狀態與氣象條件下**，預測每小時實際租借量」。據此 P0 可保留 `Functioning Day`；是否屬真實未來預測，須於 P1 確認資料可取得的時點，不能把事後觀測的天氣當成預測前已有資訊。

## 授權提醒
Kaggle 頁面標示為 Database / Contents 對應的資料庫授權，若要公開原始檔仍需核對授權全文及引用要求；不在此做商業再散布授權判斷。

## 結論
P0 以結構與維度驗證為 PASS，同意正式採用此題目；P1 尚未開始。原 Laptop_Price_Project 保留不動。
