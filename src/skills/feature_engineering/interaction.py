from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.agent.data_manager import DataManager
from src.skills.reporting.artifacts import save_json, upsert_artifact_index


def run_interaction_features(
    data_manager: DataManager,
    df_key: str,
    run_id: str,
    target_column: Optional[str] = None,
    enable: bool = True,
) -> Dict[str, Any]:
    df = data_manager.get(df_key).copy()
    actions = []

    if enable:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        if len(numeric_cols) >= 2:
            c1, c2 = numeric_cols[:2]
            inter_name = f"{c1}__x__{c2}"
            df[inter_name] = df[c1] * df[c2]
            actions.append(f"interaction:{inter_name}")

    new_key = data_manager.register(df, source_name=f"{df_key}_interaction", stage="feat")
    stage_result = {
        "stage": "feature_engineering",
        "run_id": run_id,
        "df_key_in": df_key,
        "df_key_out": new_key,
        "actions": actions or ["noop_interaction"],
        "metrics": {"columns_out": int(df.shape[1])},
        "artifacts": [],
    }
    stage_path = save_json(run_id, "feature_engineering", "interaction_stage_result.json", stage_result)
    upsert_artifact_index(run_id, "feature_engineering.interaction_stage_result", stage_path)
    return {"df_key": new_key, "stage_result": stage_result}

