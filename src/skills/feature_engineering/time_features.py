from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.agent.data_manager import DataManager
from src.skills.reporting.artifacts import save_json, upsert_artifact_index


def run_time_features(
    data_manager: DataManager,
    df_key: str,
    run_id: str,
    time_column: Optional[str] = None,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    df = data_manager.get(df_key).copy()
    actions = []

    if time_column and time_column in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
        df = df.sort_values(time_column)
        actions.append(f"sort_by_time:{time_column}")

    if target_column and target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
        df[f"{target_column}_lag1"] = df[target_column].shift(1)
        df[f"{target_column}_rolling_mean_3"] = df[target_column].rolling(3, min_periods=1).mean()
        actions.extend([f"lag1:{target_column}", f"rolling_mean_3:{target_column}"])
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:2]
        for col in numeric_cols:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_rolling_mean_3"] = df[col].rolling(3, min_periods=1).mean()
            actions.extend([f"lag1:{col}", f"rolling_mean_3:{col}"])

    df = df.bfill().fillna(0)
    new_key = data_manager.register(df, source_name=f"{df_key}_time_feat", stage="feat")
    stage_result = {
        "stage": "feature_engineering",
        "run_id": run_id,
        "df_key_in": df_key,
        "df_key_out": new_key,
        "actions": actions or ["noop_time_features"],
        "metrics": {"columns_out": int(df.shape[1])},
        "artifacts": [],
    }
    stage_path = save_json(run_id, "feature_engineering", "time_features_stage_result.json", stage_result)
    upsert_artifact_index(run_id, "feature_engineering.time_features_stage_result", stage_path)
    return {"df_key": new_key, "stage_result": stage_result}
