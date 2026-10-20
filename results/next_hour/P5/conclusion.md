# next_hour 訓練結論（P5 Evaluation）

訓練區間三折時間驗證推薦 **RFECV**，入模 **6** 個編碼後欄位：Hour, Temperature(°C), Humidity(%), Seasons_Spring, Seasons_Winter, Functioning Day_Yes。驗證 RMSE 590.73 輛，R² 0.077。

留出期 MAE 319.41、RMSE 438.07、R² 0.490；全特徵基準留出期 RMSE 427.44。營運日 R² 0.406；停駛時段 223 筆；負租借量預測 98 筆。

此推薦只在已評估候選中成立；題目總候選欄位數為 14，符合課程大於 10、小於 30 的條件。留出期曾在選題推演被查看，並非完全盲測。氣象與營運欄位可得性是條件式研究前提，不能解讀為因果關係。模型截距、係數與其他方案的完整分數見 evaluation.json、modeling.json。
