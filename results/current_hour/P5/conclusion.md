# current_hour 訓練結論（P5 Evaluation）

訓練區間三折時間驗證推薦 **SelectKBest**，入模 **13** 個編碼後欄位：Hour, Temperature(°C), Humidity(%), Wind speed (m/s), Visibility (10m), Dew point temperature(°C), Solar Radiation (MJ/m2), Rainfall(mm), Snowfall (cm), Seasons_Summer, Seasons_Winter, Holiday_No Holiday, Functioning Day_Yes。驗證 RMSE 591.34 輛，R² 0.079。

留出期 MAE 304.36、RMSE 424.14、R² 0.522；全特徵基準留出期 RMSE 409.91。營運日 R² 0.441；停駛時段 223 筆；負租借量預測 112 筆。

此推薦只在已評估候選中成立；題目總候選欄位數為 14，符合課程大於 10、小於 30 的條件。留出期曾在選題推演被查看，並非完全盲測。氣象與營運欄位可得性是條件式研究前提，不能解讀為因果關係。模型截距、係數與其他方案的完整分數見 evaluation.json、modeling.json。
