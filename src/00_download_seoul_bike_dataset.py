"""P0：從已確認的 Kaggle 資料集下載首爾自行車租借資料。"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULT_DIR = PROJECT_ROOT / "results" / "P0"
DATASET_ID = "joebeachcapital/seoul-bike-sharing"
DATASET_VERSION = 1
CSV_NAME = "SeoulBikeData.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    # 固定 Kaggle Dataset ID 與版本，避免誤下載 Notebook 或其他資料集。
    source_dir = Path(
        kagglehub.dataset_download(f"{DATASET_ID}/versions/{DATASET_VERSION}")
    )
    matches = list(source_dir.rglob(CSV_NAME))
    if len(matches) != 1:
        raise RuntimeError(f"預期恰有一個 {CSV_NAME}，實際找到 {len(matches)} 個")

    source = matches[0]
    source_hash = sha256(source)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / CSV_NAME

    # 已存在同名原始資料時，只容許內容完全一致；不覆寫不同版本。
    if destination.exists():
        if sha256(destination) != source_hash:
            raise FileExistsError(f"檔案已存在且內容不同，拒絕覆寫：{destination}")
        action = "already_exists_identical"
    else:
        shutil.copy2(source, destination)
        action = "copied"

    # 記錄可追溯的來源與檔案指紋；資料欄位及 One-Hot 維度留給 P0 驗證程式。
    metadata = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "dataset_url": f"https://www.kaggle.com/datasets/{DATASET_ID}/versions/{DATASET_VERSION}",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "file": str(destination.relative_to(PROJECT_ROOT)),
        "file_size_bytes": destination.stat().st_size,
        "sha256": source_hash,
        "action": action,
    }
    log_file = RESULT_DIR / "P0_download_metadata.json"
    log_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"P0 下載成功：{destination}")
    print(f"Kaggle Dataset：{DATASET_ID}，Version {DATASET_VERSION}")
    print(f"SHA-256：{source_hash}")
    print(f"下載紀錄：{log_file}")


if __name__ == "__main__":
    main()
