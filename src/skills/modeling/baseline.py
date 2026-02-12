from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.agent.data_manager import DataManager
from src.skills.reporting.artifacts import save_csv, save_json, upsert_artifact_index


def _choose_target(df: pd.DataFrame, target_column: Optional[str]) -> str:
    if target_column and target_column in df.columns:
        return target_column
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric:
        return numeric[-1]
    return df.columns[-1]


def _prepare_xy(
    df: pd.DataFrame,
    task_type: str,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    y = df[target_column]
    x = df.drop(columns=[target_column], errors="ignore")
    x = x.select_dtypes(include=[np.number]).copy()
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    meta: Dict[str, Any] = {}

    if task_type == "classification":
        if not pd.api.types.is_numeric_dtype(y):
            encoder = LabelEncoder()
            y = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index, name=target_column)
            meta["label_classes"] = encoder.classes_.tolist()
        else:
            y = y.fillna(0).astype(int)
    else:
        y = pd.to_numeric(y, errors="coerce").fillna(0.0)

    return x, y, meta


def _split(
    x: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    time_column_series: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if task_type == "timeseries" and time_column_series is not None:
        order = time_column_series.sort_values().index
        x_sorted = x.loc[order]
        y_sorted = y.loc[order]
        split_idx = int(len(x_sorted) * 0.8) if len(x_sorted) > 1 else 1
        return x_sorted.iloc[:split_idx], x_sorted.iloc[split_idx:], y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
    return train_test_split(x, y, test_size=0.2, random_state=42)


def run_baseline_model(
    data_manager: DataManager,
    df_key: str,
    run_id: str,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    df = data_manager.get(df_key).copy()
    task_type = str(plan.get("task_type", "eda"))
    target_column = _choose_target(df, plan.get("target_column"))

    x, y, meta = _prepare_xy(df, "regression" if task_type == "timeseries" else task_type, target_column)
    time_column = plan.get("time_column")
    time_series = df[time_column] if time_column in df.columns else None
    x_train, x_test, y_train, y_test = _split(x, y, task_type, time_series)

    if task_type in {"regression", "timeseries"}:
        models = {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(n_estimators=120, random_state=42),
        }
        metric_fn = lambda yy, pp: {
            "rmse": float(np.sqrt(mean_squared_error(yy, pp))),
            "mae": float(mean_absolute_error(yy, pp)),
            "r2": float(r2_score(yy, pp)),
        }
    elif task_type == "classification":
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest_classifier": RandomForestClassifier(n_estimators=120, random_state=42),
        }

        def metric_fn(yy: pd.Series, pp: np.ndarray, proba: Optional[np.ndarray] = None) -> Dict[str, float]:
            result = {
                "accuracy": float(accuracy_score(yy, pp)),
                "f1": float(f1_score(yy, pp, average="weighted")),
            }
            if proba is not None and len(np.unique(yy)) == 2:
                result["auc"] = float(roc_auc_score(yy, proba))
            return result

    else:
        stage_result = {
            "stage": "modeling",
            "run_id": run_id,
            "df_key_in": df_key,
            "df_key_out": df_key,
            "actions": ["skip_baseline_for_eda_or_survival"],
            "metrics": {},
            "artifacts": [],
        }
        stage_path = save_json(run_id, "modeling/baseline", "baseline_stage_result.json", stage_result)
        upsert_artifact_index(run_id, "modeling.baseline.stage_result", stage_path)
        return {"stage_result": stage_result, "best_model": None, "metrics": {}}

    model_results = {}
    best_name = None
    best_score = -np.inf
    best_model_obj = None
    predictions_out = []

    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        if task_type == "classification":
            proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
            metrics = metric_fn(y_test, pred, proba)
            score = metrics.get("f1", 0.0)
            predictions_out.append(
                pd.DataFrame({"y_true": y_test.values, "y_pred": pred, "proba": proba if proba is not None else np.nan})
            )
        else:
            metrics = metric_fn(y_test, pred)
            score = metrics.get("r2", -np.inf)
            predictions_out.append(pd.DataFrame({"y_true": y_test.values, "y_pred": pred}))

        model_results[name] = metrics
        if score > best_score:
            best_score = score
            best_name = name
            best_model_obj = model

    from pathlib import Path
    import os

    baseline_dir = Path(os.getenv("AGENT_STORAGE_ROOT", "storage/runs")) / run_id / "modeling" / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    model_path = baseline_dir / "best_model.joblib"
    joblib.dump(best_model_obj, model_path)
    upsert_artifact_index(run_id, "modeling.baseline.best_model", model_path.as_posix())

    pred_df = pd.concat(predictions_out, ignore_index=True)
    pred_path = save_csv(run_id, "modeling/baseline", "predictions.csv", pred_df)
    upsert_artifact_index(run_id, "modeling.baseline.predictions", pred_path)

    metrics_payload = {
        "task_type": task_type,
        "target_column": target_column,
        "best_model": best_name,
        "metrics": model_results,
        "meta": meta,
    }
    metrics_path = save_json(run_id, "modeling/baseline", "metrics.json", metrics_payload)
    upsert_artifact_index(run_id, "modeling.baseline.metrics", metrics_path)

    stage_result = {
        "stage": "modeling",
        "run_id": run_id,
        "df_key_in": df_key,
        "df_key_out": df_key,
        "actions": ["run_baseline_model"],
        "metrics": {"best_model": best_name, "best_score": float(best_score)},
        "artifacts": [model_path.as_posix(), pred_path, metrics_path],
    }
    stage_path = save_json(run_id, "modeling/baseline", "baseline_stage_result.json", stage_result)
    upsert_artifact_index(run_id, "modeling.baseline.stage_result", stage_path)
    return {"stage_result": stage_result, "best_model": best_name, "metrics": metrics_payload}

