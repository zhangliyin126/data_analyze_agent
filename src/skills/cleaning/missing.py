from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.agent.data_manager import DataManager
from src.skills.reporting.artifacts import save_json, upsert_artifact_index


def run_missing_imputation(data_manager: DataManager, df_key: str, run_id: str) -> Dict[str, Any]:
    df = data_manager.get(df_key).copy()
    before = float(df.isna().mean().mean())
    actions: List[str] = []

    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if np.issubdtype(df[col].dtype, np.number):
            value = float(df[col].median()) if not df[col].dropna().empty else 0.0
            df[col] = df[col].fillna(value)
            actions.append(f"fill_median:{col}")
        elif np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = df[col].ffill().bfill()
            actions.append(f"fill_time_ffill_bfill:{col}")
        else:
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "UNKNOWN"
            df[col] = df[col].fillna(fill_value)
            actions.append(f"fill_mode:{col}")

    after = float(df.isna().mean().mean())
    new_key = data_manager.register(df, source_name=f"{df_key}_missing", stage="clean")

    stage_result = {
        "stage": "cleaning",
        "run_id": run_id,
        "df_key_in": df_key,
        "df_key_out": new_key,
        "actions": actions or ["noop_missing"],
        "metrics": {"missing_before": before, "missing_after": after},
        "artifacts": [],
    }
    path = save_json(run_id, "cleaning", "missing_stage_result.json", stage_result)
    upsert_artifact_index(run_id, "cleaning.missing_stage_result", path)
    return {"df_key": new_key, "stage_result": stage_result}

