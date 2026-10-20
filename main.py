"""AIIS HW2: Seoul Bike CRISP-DM P1–P6 in one reproducible Python file.

Run ``python main.py`` for the official data. Import ``analyze_encoded`` from this
same file for a browser-side Pyodide analysis after dataset-specific preparation.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import RFE, RFECV, SelectKBest, f_regression
from sklearn.linear_model import Lasso, LinearRegression, lasso_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/raw/SeoulBikeData.csv"
EXPECTED_SHA256 = "373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60"
METHODS = ("All Features", "SelectKBest", "LassoCV", "RFECV")
EXPLORATION = {"LassoCV": "Lasso path", "RFECV": "RFE path"}
MIN_TOTAL_FEATURES, MAX_TOTAL_FEATURES = 11, 29
N_SPLITS = 3
LASSO_ALPHAS = np.geomspace(1000.0, 0.1, 20)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """共同評分函式：所有方案、時間折和留出期使用同一口徑。"""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def dump_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def selector_indices(method: str, x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    if method == "All Features":
        return np.arange(x.shape[1])
    if method == "SelectKBest":
        return SelectKBest(f_regression, k=k).fit(x, y).get_support(indices=True)
    scaled = StandardScaler().fit_transform(x)
    return RFE(LinearRegression(), n_features_to_select=k).fit(scaled, y).get_support(indices=True)


def lasso_path_indices(x: np.ndarray, y: np.ndarray) -> dict[int, np.ndarray]:
    """只記錄 Lasso 路徑真正出現的 k；這不是 LassoCV 正式選擇。"""
    scaled = StandardScaler().fit_transform(x)
    centered = y - y.mean()
    alpha_max = max(float(np.max(np.abs(scaled.T @ centered)) / len(y)), 1e-7)
    alphas = np.geomspace(alpha_max, max(alpha_max * 1e-5, 1e-7), 80)
    _, coefficients, _ = lasso_path(scaled, centered, alphas=alphas, max_iter=20000)
    result = {}
    for vector in coefficients.T:
        chosen = np.flatnonzero(np.abs(vector) > 1e-7)
        if len(chosen) and len(chosen) not in result:
            result[len(chosen)] = chosen
    return result


def exact_selector(method: str, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    """在目前訓練區間內重新選擇；內層標準化不得預看驗證折。"""
    inner = TimeSeriesSplit(n_splits=N_SPLITS)
    if method == "LassoCV":
        # GridSearchCV(Pipeline) 是無內層尺度洩漏的 Lasso CV 實作。
        search = GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("lasso", Lasso(max_iter=20000))]),
            {"lasso__alpha": LASSO_ALPHAS}, cv=inner,
            scoring="neg_root_mean_squared_error", refit=True,
        ).fit(x, y)
        chosen = np.flatnonzero(np.abs(search.best_estimator_.named_steps["lasso"].coef_) > 1e-7)
        return chosen, {"alpha": float(search.best_params_["lasso__alpha"]),
                        "inner_cv_rmse": float(-search.best_score_)}
    if method == "RFECV":
        selector = RFECV(
            estimator=Pipeline([("scale", StandardScaler()), ("reg", LinearRegression())]),
            importance_getter="named_steps.reg.coef_", cv=inner,
            scoring="neg_root_mean_squared_error", min_features_to_select=1,
        ).fit(x, y)
        return selector.get_support(indices=True), {"inner_selected_k": int(selector.n_features_)}
    raise ValueError(f"Unknown exact selector: {method}")


def evaluate_path(method: str, x: np.ndarray, y: np.ndarray,
                  names: list[str], folds: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """P4：每個 k 在相同外層時間折評估；路徑與正式 CV 選擇分開。"""
    by_k: dict[int, list[dict]] = {}
    for train_idx, valid_idx in folds:
        train_x, train_y = x[train_idx], y[train_idx]
        path = lasso_path_indices(train_x, train_y) if method == "LassoCV" else None
        ks = [x.shape[1]] if method == "All Features" else range(1, x.shape[1] + 1)
        for k in ks:
            chosen = path.get(k) if path is not None else selector_indices(
                "RFE" if method == "RFECV" else method, train_x, train_y, k
            )
            if chosen is None:
                continue
            model = LinearRegression().fit(train_x[:, chosen], train_y)
            fold_score = metrics(y[valid_idx], model.predict(x[valid_idx][:, chosen]))
            by_k.setdefault(k, []).append({"metrics": fold_score, "selected": chosen.tolist()})
    full_path = lasso_path_indices(x, y) if method == "LassoCV" else None
    rows = []
    for k, fold_results in sorted(by_k.items()):
        if len(fold_results) != len(folds):
            continue
        chosen = full_path.get(k) if full_path is not None else selector_indices(
            "RFE" if method == "RFECV" else method, x, y, k
        )
        if chosen is None:
            continue
        scores = [item["metrics"] for item in fold_results]
        rows.append({
            "k": k,
            "features": [names[i] for i in chosen],
            "cv_rmse": float(np.mean([score["rmse"] for score in scores])),
            "cv_r2": float(np.mean([score["r2"] for score in scores])),
            "cv_rmse_sd": float(np.std([score["rmse"] for score in scores], ddof=1)),
            "folds": scores,
            "selection_frequency": {
                names[i]: sum(i in item["selected"] for item in fold_results)
                for i in chosen
            },
        })
    return rows


def evaluate_exact(method: str, x: np.ndarray, y: np.ndarray,
                   names: list[str], folds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """P4：LassoCV/RFECV 的外層驗證分數，不能借用探索路徑。"""
    results = []
    for train_idx, valid_idx in folds:
        chosen, setup = exact_selector(method, x[train_idx], y[train_idx])
        if not len(chosen):
            return {"available": False, "reason": "某時間折未選出任何特徵"}
        model = LinearRegression().fit(x[train_idx][:, chosen], y[train_idx])
        results.append({"k": int(len(chosen)), "features": [names[i] for i in chosen],
                        "metrics": metrics(y[valid_idx], model.predict(x[valid_idx][:, chosen])),
                        "selector": setup})
    chosen, setup = exact_selector(method, x, y)
    if not len(chosen):
        return {"available": False, "reason": "完整訓練區間未選出任何特徵"}
    return {
        "available": True, "k": int(len(chosen)),
        "features": [names[i] for i in chosen],
        "cv_rmse": float(np.mean([item["metrics"]["rmse"] for item in results])),
        "cv_r2": float(np.mean([item["metrics"]["r2"] for item in results])),
        "folds": results, "selector": setup,
    }


def analyze_encoded(x: np.ndarray, y: np.ndarray, names: list[str],
                    functioning: np.ndarray | None = None) -> dict:
    """P3–P5 共用分析核心；輸入須先按時間排序並完成固定領域編碼。"""
    if len(x) != len(y) or len(x) < 80 or x.shape[1] < 2:
        raise ValueError("需要至少 80 筆資料、兩個特徵，且 X/y 筆數相同")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("分析輸入含缺失或非有限數值")
    cut = int(len(y) * 0.8)
    train_x, test_x, train_y, test_y = x[:cut], x[cut:], y[:cut], y[cut:]
    folds = list(TimeSeriesSplit(n_splits=N_SPLITS).split(train_x))
    methods = {}
    candidates = []
    for method in METHODS:
        rows = evaluate_path(method, train_x, train_y, names, folds)
        path_best = min(rows, key=lambda row: (row["cv_rmse"], row["k"])) if rows else None
        exact = evaluate_exact(method, train_x, train_y, names, folds) if method in EXPLORATION else None
        methods[method] = {"rows": rows, "path_label": EXPLORATION.get(method, method),
                           "best_path_k": path_best["k"] if path_best else None,
                           "exact": exact}
        if method in EXPLORATION:
            if exact and exact["available"]:
                candidates.append({"method": method, "k": exact["k"], "features": exact["features"],
                                   "cv_rmse": exact["cv_rmse"], "cv_r2": exact["cv_r2"],
                                   "source": "nested selector"})
        elif path_best:
            candidates.append({"method": method, "k": path_best["k"],
                               "features": path_best["features"],
                               "cv_rmse": path_best["cv_rmse"], "cv_r2": path_best["cv_r2"],
                               "source": "validated k path"})
    # P5 Evaluation：只依訓練區間 CV 排名；測試分數不參與推薦。
    candidates.sort(key=lambda item: (item["cv_rmse"], item["k"], item["method"]))
    baseline_prediction = np.full(len(test_y), float(train_y.mean()))
    result = {"n_train": cut, "n_test": len(y) - cut, "encoded_feature_count": len(names),
              "course_total_feature_count_ok": MIN_TOTAL_FEATURES <= len(names) <= MAX_TOTAL_FEATURES,
              "cv_folds": N_SPLITS, "methods": methods, "candidates": candidates,
              "mean_baseline_test": metrics(test_y, baseline_prediction),
              "winner": None, "holdout_was_seen_in_feasibility": True}
    if not candidates:
        return result
    for candidate in candidates:
        positions = [names.index(name) for name in candidate["features"]]
        model = LinearRegression().fit(train_x[:, positions], train_y)
        predictions = model.predict(test_x[:, positions])
        candidate["test"] = metrics(test_y, predictions)
        candidate["equation"] = {
            "intercept": float(model.intercept_),
            "coefficients": {name: float(coef) for name, coef in zip(candidate["features"], model.coef_)},
        }
        candidate["negative_prediction_count"] = int(np.sum(predictions < 0))
        if functioning is not None:
            mask = functioning[cut:].astype(bool)
            candidate["functioning_test"] = metrics(test_y[mask], predictions[mask]) if mask.any() else None
            candidate["nonfunctioning_test_count"] = int((~mask).sum())
        if candidate == candidates[0]:
            result["winner_predictions"] = predictions.tolist()
            result["winner_actual"] = test_y.tolist()
    result["winner"] = candidates[0]
    return result


def prepare_seoul() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """P2/P3：驗證來源、時間順序及已知領域類別，防止目標洩漏。"""
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"原始 CSV 雜湊不符：{digest}")
    raw = pd.read_csv(SOURCE, encoding="cp1252")
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True)
    raw = raw.sort_values(["Date", "Hour"]).reset_index(drop=True)
    times = raw["Date"] + pd.to_timedelta(raw["Hour"], unit="h")
    if len(raw) != 8760 or raw.isna().any().any() or not times.is_unique:
        raise ValueError("Seoul 原始資料筆數、缺值或時間唯一性異常")
    if not times.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Seoul 資料有非連續小時，不可直接建立下一小時目標")
    domain = {"Seasons": ["Autumn", "Spring", "Summer", "Winter"],
              "Holiday": ["Holiday", "No Holiday"],
              "Functioning Day": ["No", "Yes"]}
    for name, categories in domain.items():
        if not set(raw[name]) <= set(categories):
            raise ValueError(f"未知的類別值：{name}")
        raw[name] = pd.Categorical(raw[name], categories=categories)
    candidates = raw.drop(columns=["Date", "Rented Bike Count"])
    encoded = pd.get_dummies(candidates, drop_first=True, dtype=float)
    if candidates.shape[1] != 12 or encoded.shape[1] != 14:
        raise ValueError("Seoul 候選／編碼後特徵數與 P0 不符")
    return raw, encoded, times


def plot_pair(task: str, method: str, rows: list[dict], path: Path) -> None:
    """P4：正式圖表保留所有真實 k，標出該路徑驗證誤差最低點。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for axis, key, title in zip(axes, ("cv_rmse", "cv_r2"),
                                ("Validation RMSE by feature count", "Validation R-squared by feature count")):
        axis.plot([r["k"] for r in rows], [r[key] for r in rows], "o-", color="#2278b4")
        best = min(rows, key=lambda row: (row["cv_rmse"], row["k"]))
        axis.scatter(best["k"], best[key], color="#12aa83", s=65, zorder=3)
        axis.set(title=title, xlabel="Encoded model columns (k)",
                 ylabel="RMSE (bikes)" if key == "cv_rmse" else "R-squared")
        axis.set_xticks([r["k"] for r in rows])
        axis.grid(alpha=.2)
    fig.suptitle(f"{task} | {method} | green = lowest path CV RMSE")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_diagnostics(task: str, actual: list[float], predicted: list[float], path: Path) -> None:
    """P5：留出期實際值／預測值與時間順序殘差。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    axes[0].scatter(actual, predicted, s=6, alpha=.3)
    high = max(max(actual), max(predicted))
    axes[0].plot([0, high], [0, high], "--", color="#ce7254")
    axes[0].set(title="Held-out actual vs predicted", xlabel="Actual rentals", ylabel="Predicted rentals")
    axes[1].scatter(range(len(actual)), np.asarray(actual) - np.asarray(predicted), s=5, alpha=.3)
    axes[1].axhline(0, linestyle="--", color="#ce7254")
    axes[1].set(title="Residuals over held-out hours", xlabel="Held-out hour index", ylabel="Actual minus predicted")
    fig.suptitle(f"{task} | selected by training CV | holdout previously seen in feasibility")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_official() -> dict:
    """命令列入口：P1 Business → P2 Data → P3 Prepare → P4 Model → P5 Evaluate → P6 Deploy data."""
    # P1 Business Understanding：固定研究問題、預測時點、基準及合規口徑。
    p1_text = (
        "# P1 Business Understanding\n\n"
        "- Problem 1：已知當小時氣象與營運狀態、租借量尚未揭露時，估計當小時租借量。\n"
        "- Problem 2：已知 t 時點氣象與營運狀態，以 t 的欄位預測 t+1 租借量；不使用 t+1 實測欄位。\n"
        "- 停駛時段保留；另外報告目標時段為營運日的成績。\n"
        "- 作業特徵數條件檢查總候選欄位：One-Hot 後 14 欄，符合大於 10、小於 30；最佳入模 k 可少於 11。\n"
        "- 預測基準：訓練目標平均值；特徵基準：全部合格欄位 LinearRegression。\n"
        "- 主指標為時間驗證 RMSE；相同時選較少 k。測試期曾在選題推演被看過，並非完全盲測。\n"
        "- 欄位可得性是本研究的條件式假設，真實事前預報須另核對資料取得時間。\n"
    )
    # P2 Data Understanding：只驗證與摘要，保留原檔不改。
    raw, encoded, times = prepare_seoul()
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    p2 = ROOT / "results/current_hour/P2"
    data_summary = {
        "rows": len(raw), "columns": len(raw.columns), "raw_candidate_features": 12,
        "encoded_features": len(encoded.columns), "missing_cells": int(raw.isna().sum().sum()),
        "start": str(times.iloc[0]), "end": str(times.iloc[-1]),
        "zero_target_rows": int((raw["Rented Bike Count"] == 0).sum()),
        "source_sha256": source_hash,
    }
    # P3 Data Preparation：y_(t+1) 只由連續小時形成；按目標時間切分。
    report = {"status": "formal_analysis", "run_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": source_hash, "dataset": "Kaggle joebeachcapital/seoul-bike-sharing v1",
              "dataset_summary": data_summary,
              "python": platform.python_version(), "pandas": pd.__version__,
              "scikit_learn": sklearn.__version__, "feature_count_rule": "total encoded candidate columns: 10 < p < 30; selected k may be 1..p",
              "tasks": {}}
    for task, horizon in (("current_hour", 0), ("next_hour", 1)):
        x = encoded.to_numpy(dtype=float)[:-horizon] if horizon else encoded.to_numpy(dtype=float)
        y = raw["Rented Bike Count"].to_numpy(dtype=float)[horizon:]
        target_times = times.iloc[horizon:].reset_index(drop=True)
        target_functioning = raw["Functioning Day"].astype(str).eq("Yes").to_numpy()[horizon:]
        names = encoded.columns.tolist()
        cut = int(len(y) * .8)
        stage_root = ROOT / "results" / task
        (stage_root / "P1").mkdir(parents=True, exist_ok=True)
        (stage_root / "P1/research_definition.md").write_text(p1_text, encoding="utf-8")
        dump_json(stage_root / "P2/data_summary.json", data_summary)
        dump_json(stage_root / "P3/split.json", {
            "target": "Rented Bike Count" if horizon == 0 else "target_next_hour",
            "horizon_hours": horizon, "n_train": cut, "n_test": len(y) - cut,
            "first_test_target_time": str(target_times.iloc[cut]),
            "last_train_target_time": str(target_times.iloc[cut - 1]),
            "candidate_raw_features": 12, "encoded_columns": names,
            "split": "ordered 80/20; 3 expanding TimeSeriesSplit folds in training",
        })
        # P4 Modeling / P5 Evaluation：同一分析核心，測試集只評估 CV 先選出的候選。
        detail = analyze_encoded(x, y, names, target_functioning)
        detail["target_first_test_time"] = str(target_times.iloc[cut])
        for method, method_result in detail["methods"].items():
            slug = method.lower().replace(" ", "_")
            rows = method_result["rows"]
            if rows:
                plot_pair(task, method_result["path_label"], rows, stage_root / f"P4/{slug}.png")
                pd.DataFrame([{k: v for k, v in row.items() if k not in ("folds", "selection_frequency")}
                              for row in rows]).to_csv(stage_root / f"P4/{slug}.csv", index=False)
        dump_json(stage_root / "P4/modeling.json", {"methods": detail["methods"],
                                                     "candidates": detail["candidates"]})
        dump_json(stage_root / "P5/evaluation.json", {
            "winner": detail["winner"], "mean_baseline_test": detail["mean_baseline_test"],
            "holdout_was_seen_in_feasibility": True,
            "limitations": ["條件式可得氣象／營運資訊", "曾在選題階段查看留出期", "線性模型可能預測負租借量"],
        })
        if detail["winner"]:
            plot_diagnostics(task, detail["winner_actual"], detail["winner_predictions"],
                             stage_root / "P5/diagnostics.png")
            winner = detail["winner"]
            baseline = next(item for item in detail["candidates"] if item["method"] == "All Features")
            (stage_root / "P5/conclusion.md").write_text(
                f"# {task} 訓練結論（P5 Evaluation）\n\n"
                f"訓練區間三折時間驗證推薦 **{winner['method']}**，入模 **{winner['k']}** 個編碼後欄位："
                f"{', '.join(winner['features'])}。驗證 RMSE {winner['cv_rmse']:.2f} 輛，R² {winner['cv_r2']:.3f}。\n\n"
                f"留出期 MAE {winner['test']['mae']:.2f}、RMSE {winner['test']['rmse']:.2f}、"
                f"R² {winner['test']['r2']:.3f}；全特徵基準留出期 RMSE {baseline['test']['rmse']:.2f}。"
                f"營運日 R² {winner['functioning_test']['r2']:.3f}；"
                f"停駛時段 {winner['nonfunctioning_test_count']} 筆；"
                f"負租借量預測 {winner['negative_prediction_count']} 筆。\n\n"
                "此推薦只在已評估候選中成立；題目總候選欄位數為 14，符合課程大於 10、小於 30 的條件。"
                "留出期曾在選題推演被查看，並非完全盲測。"
                "氣象與營運欄位可得性是條件式研究前提，不能解讀為因果關係。"
                "模型截距、係數與其他方案的完整分數見 evaluation.json、modeling.json。\n",
                encoding="utf-8",
            )
        report["tasks"][task] = detail
        print(task, "winner:", detail["winner"]["method"] if detail["winner"] else "none")
    # P6 Deployment：產出前端可讀資料包；GitHub Pages 發布仍由部署流程執行。
    dump_json(ROOT / "site/data/results.json", report)
    dump_json(ROOT / "results/run_manifest.json", {
        "source_sha256": source_hash, "run_utc": report["run_utc"],
        "result_file": "site/data/results.json", "status": "ready_for_site_review",
    })
    return report


if __name__ == "__main__":
    run_official()
