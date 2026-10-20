# HW2 選題複核：Seoul Bike 與 Appliances Energy Prediction

日期：2026-10-20<br>
性質：選題可行性檢查；不是正式 CRISP-DM P4／P5 模型成果，也不修改 `docs/spec.md` 的既定題目。文末對照教授提供的示範題目，僅用來釐清欄位與資料集身分。

## 1. 要回答的問題

作業要用多元線性迴歸理解「特徵數量」及「具體特徵組合」如何影響未見資料的預測，並以兩個預測問題、RMSE／R² 曲線與對應表格呈現。比較資料集時，考量：合法且有意義的候選特徵數、可解釋的預測時點、線性模型的基本可用性、網站端資料量、資料來源與可重現性。

## 2. 來源與檔案

| 資料 | 來源 | 本機檔案 | SHA-256 |
|---|---|---|---|
| Appliances 官方封存 | [UCI Dataset 374](https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction)，由 `https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip` 下載；[Kaggle 鏡像](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction)僅作來源候選，尚未比對內容 | `data/candidates/appliances_energy/uci_374.zip` | `2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a` |
| Appliances CSV | 取自上述官方封存 | `data/candidates/appliances_energy/energydata_complete.csv` | `2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d` |
| Seoul Bike CSV | 既有 Kaggle 下載檔；[UCI 原始資料說明](https://archive.ics.uci.edu/dataset/560/seoul%2Bbike%2Bsharing%2Bdemand) | `data/raw/SeoulBikeData.csv` | `373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60` |

UCI 將 Appliances 標示為 CC BY 4.0；若最終選用 Kaggle 鏡像作正式下載來源，仍須另行核對鏡像的資料版本、檔案雜湊及引用方式。既有 Seoul Bike 原檔的雜湊未變。

## 3. 資料結構與預測任務

| 檢查 | Seoul Bike | Appliances |
|---|---:|---:|
| 列數／欄數 | 8,760／14 | 19,735／29 |
| 時間範圍 | 2017-12-01 至 2018-11-30 | 2016-01-11 至 2016-05-27 |
| 時間間隔 | 連續每小時 | 連續每 10 分鐘 |
| 缺值／完全重複列 | 0／0 | 0／0 |
| 原始候選特徵 | 12；本次 One-Hot 後 14 個模型欄位 | 24 個數值欄位，排除日期、`Appliances`、`lights`、官方為檢驗特徵選擇而加入的 `rv1`、`rv2` |
| 第一個任務 | 當小時租借量 | 當期電器用電量 |
| 第二個任務 | 用 `t` 的資料預測 `t+1` 小時租借量 | 用 `t` 的資料預測 `t+1` 個 10 分鐘的電器用電量 |

Appliances 的 `lights` 雖可另作連續目標，但本次資料檢查發現 **77.28% 為 0**；若直接用普通線性迴歸預測它，須特別處理大量零值的解讀，故不把它當成本輪的優先第二任務。24 個候選欄位中最高的兩欄絕對相關係數約為 **0.975**，係數解讀需考量共線性。

以上當期任務只在該期感測／氣象值已取得、用電量或租借量尚待估計的情境成立。真正事前預報時，還須逐欄確認輸入在預測時點可得。Appliances 官方說明指出部分外部天氣資料以較疏時間間隔取得後插值；若作即時下一期預報，插值資料的取得時點需額外核對。

## 4. 可重做的初步篩選設定

- 使用本機 Python 3.13、pandas 3.0.5、scikit-learn 1.9.0；沒有改寫正式分析主程式。計算摘要在 `data/candidates/appliances_energy/screening_results.json`。
- 各資料依時間排序；最後 20% 為測試期，前 80% 用 `TimeSeriesSplit(n_splits=3)`。Seoul Bike 類別欄位在此輪做 One-Hot，類別清單未依正式 pipeline 在各折內擬合，因此這輪只作選題篩查。
- 比較 `SelectKBest(f_regression)` 後接 `LinearRegression` 的每個可得模型欄位數 `k`，以三折平均驗證 RMSE 選出 `11 ≤ k < 30` 中的一列；另列全特徵 `LinearRegression` 測試結果。未執行 LassoCV、RFECV、巢狀驗證或完整殘差檢查。
- 表內 Seoul Bike 的 `k` 是 **One-Hot 後模型欄位數**，不是原始 12 欄的數量；課程計數口徑仍待確認。兩資料的 RMSE 單位不同，不能直接比較大小；R² 亦只可作這兩個測試期間的粗略參考，不能推論跨資料集普遍優劣。

| 資料／任務 | 驗證選出的 `k` | 平均驗證 RMSE | 平均驗證 R² | 選擇後測試 RMSE | 選擇後測試 R² | 全特徵測試 R² |
|---|---:|---:|---:|---:|---:|---:|
| Appliances 當期 | 16／24 | 99.970 Wh | 0.010 | 87.134 Wh | 0.084 | 0.097 |
| Appliances 下一個 10 分鐘 | 17／24 | 100.196 Wh | 0.007 | 88.087 Wh | 0.064 | 0.085 |
| Seoul Bike 當小時 | 13／14 編碼欄位 | 591.339 輛 | 0.079 | 424.140 輛 | 0.522 | 0.553 |
| Seoul Bike 下一小時 | 14／14 編碼欄位 | 593.202 輛 | 0.075 | 427.438 輛 | 0.514 | 0.514 |

Seoul Bike 的驗證 R² 與測試 R² 差距明顯；可能與不同時間區間的季節、租借量分布有關，不能單憑最後測試期的較高 R² 宣稱模型穩定。Appliances 有較寬的特徵數範圍，但這組線性基準在時間驗證中幾乎無法解釋目標變異。這是模型與任務的實測觀察，不代表資料本身品質差，也不排除經合理特徵工程後改善。

另以 Seoul Bike 的日期衍生週末、時刻正弦／餘弦及年週期正弦／餘弦作一次探索：原始候選數可增至 17，編碼後 19 欄；但這輪驗證選出的 13 欄在兩個測試任務的 R² 約 0.155／0.153，低於上述原始方案。故**不能只為擴大曲線範圍而宣稱衍生特徵帶來改善**。此探索不納入正式結果包。

## 5. 選題結論與關卡

**在目前兩個已實際檢查的選項中，建議保留 Seoul Bike 作正式題目。** 理由是預測目標直觀、涵蓋完整一年、原始資料和既有規格已備妥，且本輪線性迴歸篩查對未見期間仍有較清楚的預測訊號。Appliances 適合作為自訂 CSV 功能或 skill 的第二資料集演練；它的 24 個候選特徵能測試通用分析是否真的支援較寬的 `k` 範圍，但本輪不建議直接取代正式題目。

**尚未解除的關鍵條件**：教授的 `10 < 特徵數 < 30` 究竟要求候選輸入數，還是最後選入模型的特徵數？若是後者且以原始欄位計數，Seoul Bike 的正式合規 `k` 僅有 11、12，無法靠本輪實測不佳的衍生欄位簡單解決。屆時應先確認可否呈現「探索性 1～12 曲線、正式選擇 11～12」；若不接受，就需要另找具 15～25 個有效原始特徵、且線性回歸效果可接受的第三個資料集，不能因 Appliances 欄位較多便自動改題。

教授是否接受同一資料的下一時點作第二個 Problem，也尚待課程方確認。此報告確認的是技術可行性，不代替課程要求的正式核准。

## 6. 教授示範題目「Bike Sharing Demand」的欄位對照

使用者已釐清：教授依先前提示條件搜尋並列出五個題目，這段描述是**示範選項，不是指定必須採用的資料檔或欄位格式**。其中目標欄 `count` 與 `workingday`、`weather`、`atemp`（體感溫度）逐一對應 [Kaggle Bike Sharing Demand 競賽資料](https://www.kaggle.com/c/bike-sharing-demand/data/)及其 [Washington, D.C. 專案說明](https://www.kaggle.com/c/bike-sharing-demand/overview/description)；目前 Seoul Bike 原始 CSV 的目標名為 `Rented Bike Count`，且沒有上述三欄。兩者是同類主題的不同資料集，不可混用欄位，但**不因此排除 Seoul Bike**。

Kaggle 的 `train.csv` 資料欄說明列出 `datetime`、`season`、`holiday`、`workingday`、`weather`、`temp`、`atemp`、`humidity`、`windspeed`、`casual`、`registered`、`count`：合計 **12 個總欄位**。其中 `count = casual + registered`，`casual` 與 `registered` 是結果的組成部分，不可在預測 `count` 時當輸入；`count` 本身是目標。因此直接可用的原始預測欄位為 **9 個（含時間戳記 `datetime`）**，而非 12 個。若從時間戳記衍生年、月、星期與小時，模型欄位可超過 10，但必須明確區分「原始資料欄位數」、「衍生特徵數」與「編碼後模型維度」。

該競賽官方說明指出訓練資料是**每月前 19 天**，測試資料是第 20 天至月底，不能把其 `train.csv` 相鄰兩列一律當作連續下一小時來建立第二預測任務。若題目允許使用同來源的完整記錄，[UCI Bike Sharing `hour.csv`](https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset)是另一份按小時的公開資料，但欄位命名與數量不同，不能冒稱 Kaggle 競賽原檔。

**選題判斷**：示範文字並未要求改用 Washington, D.C. 競賽資料；因此第 5 節的「保留 Seoul Bike」建議維持。是否符合 `10 < 特徵數 < 30`，仍取決於課程對候選／最終入模特徵數及計數單位的定義；不能借用示範資料的「12 欄」替 Seoul Bike 證明合規。
