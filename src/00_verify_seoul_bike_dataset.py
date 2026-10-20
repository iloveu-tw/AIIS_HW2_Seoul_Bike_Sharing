"""P0-2：首爾共享單車資料集的唯讀結構與模型維度驗證。"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/SeoulBikeData.csv'
META = ROOT / 'results/P0/P0_download_metadata.json'
OUTPUT = ROOT / 'results/P0'
TARGET = 'Rented Bike Count'
DATE = 'Date'
NUMERIC = [
    'Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
    'Visibility (10m)', 'Dew point temperature(°C)',
    'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)',
]
CATEGORICAL = ['Seasons']
BINARY = ['Holiday', 'Functioning Day']
EXPECTED = [DATE, TARGET, *NUMERIC, *CATEGORICAL, *BINARY]


def main():
    # 先核對來源內容與下載時的 SHA-256，嚴格禁止在此階段修改 CSV。
    metadata = json.loads(META.read_text(encoding='utf-8'))
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != metadata['sha256']:
        raise ValueError('原始 CSV 的 SHA-256 與 P0 下載紀錄不一致')
    if metadata.get('dataset_id') != 'joebeachcapital/seoul-bike-sharing' or metadata.get('dataset_version') != 1:
        raise ValueError('資料來源或版本不符預期')

    # 部分 Kaggle 版本使用 cp1252（度數符號），讀取成功才進行欄位檢查。
    try:
        df = pd.read_csv(SOURCE, encoding='utf-8-sig')
        encoding = 'utf-8-sig'
    except UnicodeDecodeError:
        df = pd.read_csv(SOURCE, encoding='cp1252')
        encoding = 'cp1252'
    df.columns = df.columns.str.strip()
    if len(df.columns) != len(set(df.columns)):
        raise ValueError('CSV 包含重複欄位名稱')
    missing_cols = set(EXPECTED) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED)
    if missing_cols or extra_cols:
        raise ValueError(f'欄位與預期不一致：缺少 {sorted(missing_cols)}；多出 {sorted(extra_cols)}')

    # 檢查日期可讀性、目標欄位及九個數值輸入是否真的是數值。
    timestamps = pd.to_datetime(df[DATE], format='%d/%m/%Y', errors='coerce')
    invalid_dates = int(timestamps.isna().sum())
    numeric_invalid = {}
    for col in [TARGET, *NUMERIC]:
        parsed = pd.to_numeric(df[col], errors='coerce')
        numeric_invalid[col] = int((parsed.isna() & df[col].notna()).sum())
    quality_ok = not invalid_dates and not any(numeric_invalid.values()) and df[TARGET].notna().all()
    missing_cells = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    # 獨立盤點所有原始欄位，提供 dtype、缺失及唯一值數量作為 P0 證據。
    inventory = pd.DataFrame([
        {'column': c, 'dtype': str(df[c].dtype), 'unique_count': int(df[c].nunique(dropna=True)),
         'missing_count': int(df[c].isna().sum()),
         'role': 'target' if c == TARGET else ('date' if c == DATE else 'input')}
        for c in df.columns
    ])

    # 只對時間前 80% 的訓練資料擬合編碼器；測試資料不參與學習類別。
    ordered = df.assign(_timestamp=timestamps).sort_values(['_timestamp', 'Hour'], kind='stable')
    train_count = int(len(ordered) * 0.8)
    train = ordered.iloc[:train_count]
    one_hot_dim = None
    category_levels = {}
    encoding_error = None
    if quality_ok and missing_cells == 0 and train_count > 0:
        try:
            valid_binary = {
                c: sorted(df[c].astype(str).str.strip().unique().tolist()) for c in BINARY
            }
            if any(len(v) != 2 for v in valid_binary.values()):
                raise ValueError(f'二元欄位出現非 2 種類別：{valid_binary}')
            encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
            encoder.fit(train[CATEGORICAL].astype(str).apply(lambda s: s.str.strip()))
            category_levels = {c: list(map(str, v)) for c, v in zip(CATEGORICAL, encoder.categories_)}
            one_hot_dim = int(len(encoder.get_feature_names_out(CATEGORICAL)))
        except ValueError as exc:
            encoding_error = str(exc)

    selected_raw_count = len(NUMERIC) + len(CATEGORICAL) + len(BINARY)
    encoded_count = None if one_hot_dim is None else len(NUMERIC) + len(BINARY) + one_hot_dim
    feature_range_ok = (12 <= selected_raw_count <= 29 and encoded_count is not None and 12 <= encoded_count <= 29)
    structural_ok = (len(df) == 8760 and len(df.columns) == 14 and quality_ok and missing_cells == 0
                     and duplicates == 0 and feature_range_ok)
    # Functioning Day 是否可在預測時取得，屬於研究情境決策；因此不自動宣告完全合格。
    result = {
        'stage': 'P0-2', 'status': 'STRUCTURE_PASS_REVIEW_REQUIRED' if structural_ok else 'CHECK_FAILED',
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_id': metadata['dataset_id'], 'dataset_version': metadata['dataset_version'],
        'sha256': digest, 'encoding': encoding, 'rows': int(len(df)), 'columns': int(len(df.columns)),
        'target': TARGET, 'missing_cells': missing_cells, 'duplicate_rows': duplicates,
        'invalid_dates': invalid_dates, 'numeric_invalid_counts': numeric_invalid,
        'selected_original_features': selected_raw_count, 'train_rows': train_count,
        'test_rows': len(df) - train_count, 'split': 'chronological_80_20',
        'numeric_features': NUMERIC, 'binary_features': BINARY,
        'categorical_features': CATEGORICAL, 'training_category_levels': category_levels,
        'encoded_model_features': encoded_count, 'feature_range_pass': feature_range_ok,
        'encoding_error': encoding_error,
        'manual_review': ['Functioning Day 是否在預測時已知', '時間切分是否符合研究目的', 'Kaggle 授權與引用條件'],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUTPUT / 'P0_column_inventory.csv', index=False, encoding='utf-8-sig')
    (OUTPUT / 'P0_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = [
        '# Seoul Bike Sharing P0-2 驗證報告', '',
        f'- 結果：**{result["status"]}**', f'- Kaggle：`{result["dataset_id"]}` Version {result["dataset_version"]}',
        f'- SHA-256：`{digest}`', f'- 資料筆數／欄位數：{len(df)}／{len(df.columns)}',
        f'- 目標：`{TARGET}`', f'- 缺失值／重複列：{missing_cells}／{duplicates}',
        f'- 選用原始輸入特徵：{selected_raw_count}',
        f'- 編碼後模型維度：{encoded_count if encoded_count is not None else "無法確認"}',
        f'- 時間序切分：訓練 {train_count} 筆，測試 {len(df)-train_count} 筆',
        f'- 12～29 維檢查：{"PASS" if feature_range_ok else "FAIL"}', '',
        '## 尚待人工審查',
        '- `Functioning Day` 在實際預測時能否事先取得；若排除，原始選用特徵將只剩 11 個，應重新設計特徵。',
        '- 時間型資料需確認切分策略，不宜直接套用一般隨機切分。',
        '- One-Hot 使用訓練資料類別、drop=first；本階段只核對維度，沒有訓練模型。',
        '- 不覆寫或更動原始 CSV。',
    ]
    (OUTPUT / 'P0_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'P0-2：{result["status"]}；原始特徵 {selected_raw_count}；編碼後維度 {encoded_count}；輸出 {OUTPUT}')


if __name__ == '__main__':
    main()
