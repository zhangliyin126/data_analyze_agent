from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.agent.data_manager import DataManager
from src.skills.reporting.artifacts import save_json, upsert_artifact_index


def run_type_fix(data_manager: DataManager, df_key: str, run_id: str) -> Dict[str, Any]:
    df = data_manager.get(df_key).copy()
    actions: List[str] = []

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_object_dtype(series):
            lname = col.lower()
            if "date" in lname or "time" in lname:
                converted = pd.to_datetime(series, errors="coerce")
                if converted.notna().mean() >= 0.8:
                    df[col] = converted
                    actions.append(f"parse_datetime:{col}")
                    continue

            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().mean() >= 0.9:
                df[col] = numeric
                actions.append(f"to_numeric:{col}")

    new_key = data_manager.register(df, source_name=f"{df_key}_type_fix", stage="clean")
    stage_result = {
        "stage": "cleaning",
        "run_id": run_id,
        "df_key_in": df_key,
        "df_key_out": new_key,
        "actions": actions or ["noop_type_fix"],
        "metrics": {"columns": int(df.shape[1])},
        "artifacts": [],
    }
    path = save_json(run_id, "cleaning", "type_fix_stage_result.json", stage_result)
    upsert_artifact_index(run_id, "cleaning.type_fix_stage_result", path)
    return {"df_key": new_key, "stage_result": stage_result}

