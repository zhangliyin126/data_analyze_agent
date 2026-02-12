from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.agent.data_manager import DataManager
from src.skills.reporting.artifacts import save_json, upsert_artifact_index


def _detect_time_columns(df: pd.DataFrame) -> List[str]:
    time_cols: List[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            time_cols.append(col)
            continue
        if series.dtype == "object" and ("date" in col.lower() or "time" in col.lower()):
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                time_cols.append(col)
    return time_cols


def _detect_group_candidates(df: pd.DataFrame) -> List[str]:
    candidates: List[str] = []
    n_rows = max(len(df), 1)
    for col in df.columns:
        uniq_ratio = df[col].nunique(dropna=True) / n_rows
        if uniq_ratio <= 0.5:
            if pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
                candidates.append(col)
            elif pd.api.types.is_integer_dtype(df[col]) and uniq_ratio <= 0.2:
                candidates.append(col)
    return candidates


def profile_dataset(
    data_manager: DataManager,
    df_key: str,
    run_id: str,
    schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    schema = schema or {}
    df = data_manager.get(df_key)
    n_rows, n_cols = df.shape
    memory_bytes = int(df.memory_usage(deep=True).sum())

    missing_ratio = {col: float(df[col].isna().mean()) for col in df.columns}
    unique_ratio = {col: float(df[col].nunique(dropna=True) / max(n_rows, 1)) for col in df.columns}
    column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    distribution_stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            outlier_ratio = 0.0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_ratio = float(((series < lower) | (series > upper)).mean())
        distribution_stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "skew": float(series.skew()) if len(series) > 2 else 0.0,
            "kurtosis": float(series.kurtosis()) if len(series) > 3 else 0.0,
            "outlier_ratio": outlier_ratio,
        }

    time_columns = _detect_time_columns(df)
    group_candidates = _detect_group_candidates(df)
    avg_obs_per_group = 0.0
    if group_candidates:
        group_col = group_candidates[0]
        avg_obs_per_group = float(df.groupby(group_col, dropna=False).size().mean())

    data_structure_signals = {
        "hierarchical_likely": bool(group_candidates and avg_obs_per_group >= 2.0),
        "repeated_measure_likely": bool(group_candidates and time_columns and avg_obs_per_group >= 2.0),
        "high_d_low_n": bool(n_rows > 0 and (n_cols / max(n_rows, 1)) >= 1.5),
        "feature_row_ratio": float(n_cols / max(n_rows, 1)),
    }

    profile = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "memory_bytes": memory_bytes,
        "column_types": column_types,
        "missing_ratio": missing_ratio,
        "unique_ratio": unique_ratio,
        "numeric_columns": numeric_cols,
        "time_columns": time_columns,
        "group_candidates": group_candidates,
        "avg_obs_per_group": avg_obs_per_group,
        "distribution_stats": distribution_stats,
        "data_structure_signals": data_structure_signals,
        "schema_columns": list((schema or {}).get("columns", {}).keys()),
    }

    profile_path = save_json(run_id, "profiling", "profile.json", profile)
    upsert_artifact_index(run_id, "profiling.profile", profile_path)

    stage_result = {
        "stage": "profiling",
        "run_id": run_id,
        "df_key_in": df_key,
        "df_key_out": df_key,
        "actions": ["profile_dataset"],
        "metrics": {"n_rows": n_rows, "n_cols": n_cols, "memory_bytes": memory_bytes},
        "artifacts": [profile_path],
    }
    stage_path = save_json(run_id, "profiling", "stage_result.json", stage_result)
    upsert_artifact_index(run_id, "profiling.stage_result", stage_path)
    return {"profile": profile, "stage_result": stage_result}
