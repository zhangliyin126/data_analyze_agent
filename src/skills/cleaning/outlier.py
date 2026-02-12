from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.agent.data_manager import DataManager
from src.skills.reporting.artifacts import save_json, upsert_artifact_index


def run_outlier_winsorize(
    data_manager: DataManager,
    df_key: str,
    run_id: str,
    iqr_multiplier: float = 1.5,
) -> Dict[str, Any]:
    df = data_manager.get(df_key).copy()
    actions: List[str] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_count = 0

    for col in numeric_cols:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        mask = (series < lower) | (series > upper)
        count = int(mask.sum())
        if count > 0:
            df[col] = series.clip(lower=lower, upper=upper)
            outlier_count += count
            actions.append(f"winsorize:{col}:{count}")

    new_key = data_manager.register(df, source_name=f"{df_key}_outlier", stage="clean")
    stage_result = {
        "stage": "cleaning",
        "run_id": run_id,
        "df_key_in": df_key,
        "df_key_out": new_key,
        "actions": actions or ["noop_outlier"],
        "metrics": {"winsorized_values": outlier_count},
        "artifacts": [],
    }
    path = save_json(run_id, "cleaning", "outlier_stage_result.json", stage_result)
    upsert_artifact_index(run_id, "cleaning.outlier_stage_result", path)
    return {"df_key": new_key, "stage_result": stage_result}

