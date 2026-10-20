# Seoul Bike Sharing — P0-2 執行手冊

## 目的
驗證 Kaggle Version 1 CSV 的結構、原始特徵數與 **使用訓練集學習類別後** 的 One-Hot 模型維度；不修改原始 CSV、不訓練模型。

## 本機位置
- 程式放在 `src/00_verify_seoul_bike_dataset.py`
- 必須已有 `data/raw/SeoulBikeData.csv` 與 `results/P0/P0_download_metadata.json`

## 執行前套件
```bash
cd /Users/summer/NCHU_project/Seoul_Bike_Sharing_Project
/usr/local/bin/python3 -c "import pandas, sklearn; print('dependencies OK')"
```

如果缺少套件，請先回報訊息，不要自行更換整套 Python 環境。

## 語法檢查、執行與檢視
```bash
/usr/local/bin/python3 -m py_compile src/00_verify_seoul_bike_dataset.py
/usr/local/bin/python3 src/00_verify_seoul_bike_dataset.py
ls -lah results/P0/
```

## 輸出
- `results/P0/P0_report.md`：結構與模型維度摘要
- `results/P0/P0_verification.json`：機器可讀驗證紀錄
- `results/P0/P0_column_inventory.csv`：全部 14 欄位盤點

## 預期判定
- 8,760 列、14 欄；預測目標 `Rented Bike Count`
- 12 個非日期的原始輸入候選特徵（9 個數值、1 個四類季節、2 個二元欄位）
- One-Hot `Seasons` 採 `drop='first'`，並且只在依日期／小時排序後的前 80% 訓練資料上擬合；若四季皆出現，編碼後 14 維
- `STRUCTURE_PASS_REVIEW_REQUIRED` 代表**結構與維度符合，但仍須人工審查** `Functioning Day` 的預測時可用性，以及時間切分方式

不要將 `STRUCTURE_PASS_REVIEW_REQUIRED` 視為 P0 最終結案；P0 的正式研究適用性須另行確認。
