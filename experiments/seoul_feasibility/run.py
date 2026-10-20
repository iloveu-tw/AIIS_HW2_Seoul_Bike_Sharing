"""Exploratory Seoul Bike feasibility run; these are not final HW2 results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, RFECV, SelectKBest, f_regression
from sklearn.linear_model import LassoCV, LinearRegression, lasso_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/SeoulBikeData.csv"
OUT = Path(__file__).resolve().parent / "results"
METHODS = ("All Features", "SelectKBest", "Lasso Path", "RFE")


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def lasso_subsets(x: np.ndarray, y: np.ndarray) -> dict[int, np.ndarray]:
    """Return only feature counts actually reached by the Lasso alpha path."""
    scaled = StandardScaler().fit_transform(x)
    centered = y - y.mean()
    alpha_max = np.max(np.abs(scaled.T @ centered)) / len(y)
    alphas = np.geomspace(alpha_max, max(alpha_max * 1e-5, 1e-7), 80)
    _, coefficients, _ = lasso_path(
        scaled, centered, alphas=alphas, max_iter=20000, tol=1e-5
    )
    subsets: dict[int, np.ndarray] = {}
    for vector in coefficients.T:
        selected = np.flatnonzero(np.abs(vector) > 1e-7)
        if len(selected) and len(selected) not in subsets:
            subsets[len(selected)] = selected
    return subsets


def choose_indices(method: str, x: np.ndarray, y: np.ndarray, k: int,
                   lasso: dict[int, np.ndarray] | None = None) -> np.ndarray | None:
    if method == "All Features":
        return np.arange(x.shape[1]) if k == x.shape[1] else None
    if method == "SelectKBest":
        selector = SelectKBest(f_regression, k=k).fit(x, y)
        return selector.get_support(indices=True)
    if method == "Lasso Path":
        return None if lasso is None else lasso.get(k)
    scaled = StandardScaler().fit_transform(x)
    selector = RFE(LinearRegression(), n_features_to_select=k).fit(scaled, y)
    return selector.get_support(indices=True)


def exact_selector_check(name: str, train_x: np.ndarray, train_y: np.ndarray,
                         test_x: np.ndarray, test_y: np.ndarray,
                         feature_names: pd.Index) -> dict:
    """Check the two exact sklearn selector classes against the time holdout."""
    def select(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if name == "LassoCV":
            scaled = StandardScaler().fit_transform(x)
            selector = LassoCV(
                alphas=60, cv=TimeSeriesSplit(n_splits=3),
                max_iter=20000, tol=1e-4,
            ).fit(scaled, y)
            return np.flatnonzero(np.abs(selector.coef_) > 1e-7)
        pipeline = Pipeline([("scale", StandardScaler()), ("reg", LinearRegression())])
        selector = RFECV(
            estimator=pipeline, importance_getter="named_steps.reg.coef_",
            cv=TimeSeriesSplit(n_splits=3), scoring="neg_root_mean_squared_error",
            step=1, min_features_to_select=1,
        ).fit(x, y)
        return selector.get_support(indices=True)

    folds = []
    for train_index, valid_index in TimeSeriesSplit(n_splits=3).split(train_x):
        chosen = select(train_x[train_index], train_y[train_index])
        if len(chosen) == 0:
            folds.append({"k": 0, "score": None})
            continue
        model = LinearRegression().fit(train_x[train_index][:, chosen], train_y[train_index])
        folds.append({
            "k": int(len(chosen)),
            "score": score(train_y[valid_index], model.predict(train_x[valid_index][:, chosen])),
        })
    chosen = select(train_x, train_y)
    if len(chosen) == 0:
        return {"k": 0, "folds": folds, "test": None}
    model = LinearRegression().fit(train_x[:, chosen], train_y)
    return {
        "k": int(len(chosen)),
        "features": feature_names[chosen].tolist(),
        "folds": folds,
        "cv_rmse_mean": float(np.mean([v["score"]["rmse"] for v in folds if v["score"]])),
        "cv_r2_mean": float(np.mean([v["score"]["r2"] for v in folds if v["score"]])),
        "test": score(test_y, model.predict(test_x[:, chosen])),
    }


def chart(task: str, method: str, rows: list[dict]) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    x = [r["k"] for r in rows]
    for axis, key, title in zip(
        axes,
        ("cv_rmse_mean", "cv_r2_mean"),
        ("Validation RMSE by Number of Features", "Validation R-squared by Number of Features"),
    ):
        axis.plot(x, [r[key] for r in rows], marker="o", color="#1976a3")
        axis.set(title=title, xlabel="Encoded feature count", ylabel="RMSE" if key == "cv_rmse_mean" else "R-squared")
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
    fig.suptitle(f"{task} | {method} | 3 time validation folds")
    fig.savefig(OUT / f"{task}_{method.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


def diagnostic_chart(task: str, target_times: pd.Series,
                     actual: np.ndarray, predicted: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].scatter(actual, predicted, s=7, alpha=0.28, color="#1976a3")
    high = max(float(actual.max()), float(predicted.max()))
    axes[0].plot([0, high], [0, high], color="#d05a3f", linestyle="--")
    axes[0].set(title="Held-out actual vs predicted", xlabel="Actual rentals", ylabel="Predicted rentals")
    axes[1].scatter(target_times, actual - predicted, s=5, alpha=0.3, color="#1976a3")
    axes[1].axhline(0, color="#d05a3f", linestyle="--")
    axes[1].set(title="Held-out residuals over time", xlabel="Target time", ylabel="Actual minus predicted")
    fig.autofmt_xdate()
    fig.suptitle(f"{task} | CV-selected scheme | held-out period")
    fig.savefig(OUT / f"{task}_diagnostics.png", dpi=150)
    plt.close(fig)


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    # P2 Data Understanding: verify source and hourly chronology.
    raw = pd.read_csv(SOURCE, encoding="cp1252")
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True)
    raw = raw.sort_values(["Date", "Hour"]).reset_index(drop=True)
    times = raw["Date"] + pd.to_timedelta(raw["Hour"], unit="h")
    assert len(raw) == 8760 and raw.isna().sum().sum() == 0
    assert times.is_unique and times.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    # P3 Data Preparation: fixed domain categories make every time fold comparable.
    ordered = {
        "Seasons": ["Autumn", "Spring", "Summer", "Winter"],
        "Holiday": ["Holiday", "No Holiday"],
        "Functioning Day": ["No", "Yes"],
    }
    for column, categories in ordered.items():
        assert set(raw[column].unique()) <= set(categories)
        raw[column] = pd.Categorical(raw[column], categories=categories)
    candidates = raw.drop(columns=["Date", "Rented Bike Count"])
    encoded = pd.get_dummies(candidates, drop_first=True, dtype=float)
    assert candidates.shape[1] == 12 and encoded.shape[1] == 14
    return raw, encoded, times


def run_task(task: str, x_all: pd.DataFrame, target: pd.Series,
             times: pd.Series, horizon: int) -> dict:
    # P3 Data Preparation: X_t predicts y_t or y_(t+1); no y_t input in either task.
    x = x_all.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    target_functioning = x_all["Functioning Day_Yes"].to_numpy(dtype=bool)
    if horizon:
        x, y = x[:-horizon], y[horizon:]
        target_functioning = target_functioning[horizon:]
        target_times = times.iloc[horizon:].reset_index(drop=True)
    else:
        target_times = times.reset_index(drop=True)
    cut = int(len(y) * 0.8)
    train_x, test_x, train_y, test_y = x[:cut], x[cut:], y[:cut], y[cut:]
    folds = list(TimeSeriesSplit(n_splits=3).split(train_x))
    result = {
        "task": task,
        "horizon_hours": horizon,
        "n_train": cut,
        "n_test": len(y) - cut,
        "test_start_target_time": str(target_times.iloc[cut]),
        "feature_count_raw": 12,
        "feature_count_encoded": x.shape[1],
        "methods": {},
    }
    # P4 Modeling / P5 Evaluation: selectors fit within each training fold.
    for method in METHODS:
        by_k: dict[int, list[dict]] = {}
        for train_index, valid_index in folds:
            fold_x, fold_y = train_x[train_index], train_y[train_index]
            path = lasso_subsets(fold_x, fold_y) if method == "Lasso Path" else None
            available = [x.shape[1]] if method == "All Features" else range(1, x.shape[1] + 1)
            for k in available:
                selected = choose_indices(method, fold_x, fold_y, k, path)
                if selected is None:
                    continue
                model = LinearRegression().fit(fold_x[:, selected], fold_y)
                validation = score(train_y[valid_index], model.predict(train_x[valid_index][:, selected]))
                by_k.setdefault(k, []).append({"score": validation, "selected": selected.tolist()})
        rows = []
        full_path = lasso_subsets(train_x, train_y) if method == "Lasso Path" else None
        for k, fold_results in sorted(by_k.items()):
            if len(fold_results) != len(folds):
                continue  # No invented Lasso points when a fold never reached k.
            selected = choose_indices(method, train_x, train_y, k, full_path)
            if selected is None:
                continue
            metrics = [item["score"] for item in fold_results]
            names = x_all.columns[selected].tolist()
            rows.append({
                "k": k,
                "selected_features_full_train": names,
                "cv_rmse_mean": float(np.mean([m["rmse"] for m in metrics])),
                "cv_rmse_sd": float(np.std([m["rmse"] for m in metrics], ddof=1)),
                "cv_r2_mean": float(np.mean([m["r2"] for m in metrics])),
                "cv_folds": metrics,
                "feature_stability": {name: sum(int(int(idx) in item["selected"]) for item in fold_results)
                                      for idx, name in enumerate(x_all.columns) if name in names},
            })
        if not rows:
            result["methods"][method] = {"rows": [], "chosen": None}
            continue
        chosen = min(rows, key=lambda r: (r["cv_rmse_mean"], r["k"]))
        selected_names = chosen["selected_features_full_train"]
        positions = x_all.columns.get_indexer(selected_names)
        model = LinearRegression().fit(train_x[:, positions], train_y)
        chosen_test = score(test_y, model.predict(test_x[:, positions]))
        result["methods"][method] = {
            "rows": rows,
            "chosen_k": chosen["k"],
            "chosen_cv_rmse": chosen["cv_rmse_mean"],
            "chosen_cv_r2": chosen["cv_r2_mean"],
            "chosen_features": selected_names,
            "test": chosen_test,
        }
        chart(task, method, rows)
        flat = [{key: value for key, value in row.items() if key not in ("cv_folds", "feature_stability")}
                for row in rows]
        pd.DataFrame(flat).to_csv(OUT / f"{task}_{method.lower().replace(' ', '_')}.csv", index=False)
    winner_name = min(
        result["methods"],
        key=lambda name: result["methods"][name].get("chosen_cv_rmse", float("inf")),
    )
    winner = result["methods"][winner_name]
    positions = x_all.columns.get_indexer(winner["chosen_features"])
    selected_model = LinearRegression().fit(train_x[:, positions], train_y)
    predicted = selected_model.predict(test_x[:, positions])
    diagnostic_chart(task, target_times.iloc[cut:], test_y, predicted)
    result["overall_winner"] = {
        "method": winner_name,
        "k": winner["chosen_k"],
        "features": winner["chosen_features"],
        "test": score(test_y, predicted),
        "negative_prediction_count": int(np.sum(predicted < 0)),
        "negative_prediction_pct": float(np.mean(predicted < 0) * 100),
        "test_zero_actual_pct": float(np.mean(test_y == 0) * 100),
        "functioning_test": score(test_y[target_functioning[cut:]], predicted[target_functioning[cut:]]),
        "nonfunctioning_test_count": int(np.sum(~target_functioning[cut:])),
        "nonfunctioning_test_rmse": float(np.sqrt(mean_squared_error(
            test_y[~target_functioning[cut:]], predicted[~target_functioning[cut:]]
        ))),
    }
    result["exact_selectors"] = {
        name: exact_selector_check(name, train_x, train_y, test_x, test_y, x_all.columns)
        for name in ("LassoCV", "RFECV")
    }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, encoded, times = prepare()
    report = {
        "status": "exploratory_feasibility_only",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "encoding": "fixed domain One-Hot; first category dropped",
        "split": "time ordered 80/20 holdout, 3 TimeSeriesSplit folds in train",
        "selection": "lowest mean validation RMSE; test assessed after each method k was fixed",
        "caveat": "Curves use the actual Lasso alpha path and scaled RFE path. Exact LassoCV/RFECV checks are separate points; this is not the final nested-CV implementation.",
        "tasks": {},
    }
    for name, horizon in (("current_hour", 0), ("next_hour", 1)):
        report["tasks"][name] = run_task(name, encoded, raw["Rented Bike Count"], times, horizon)
    (OUT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for task, detail in report["tasks"].items():
        print(task, "test_start", detail["test_start_target_time"])
        for method, summary in detail["methods"].items():
            print(method, "points", len(summary["rows"]), "k", summary.get("chosen_k"),
                  "cv_rmse", round(summary.get("chosen_cv_rmse", float("nan")), 2),
                  "cv_r2", round(summary.get("chosen_cv_r2", float("nan")), 3),
                  "test_r2", round(summary.get("test", {}).get("r2", float("nan")), 3))


if __name__ == "__main__":
    main()
