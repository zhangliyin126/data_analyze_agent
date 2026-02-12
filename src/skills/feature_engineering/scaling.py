from __future__ import annotations

from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.agent.data_manager import DataManager
from src.skills.reporting.artifacts import save_json, upsert_artifact_index


def run_scaling(
    data_manager: DataManager,
    df_key: str,
    run_id: str,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    df = data_manager.get(df_key).copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)  # target should not be scaled in place

    scaler = StandardScaler()
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    new_key = data_manager.register(df, source_name=f"{df_key}_scaling", stage="feat")

    scaler_path = f"{run_id}_scaler.joblib"
    # keep model files under modeling folder for easier retrieval
    from pathlib import Path
    import os

    root = Path(os.getenv("AGENT_STORAGE_ROOT", "storage/runs")) / run_id / "feature_engineering"
    root.mkdir(parents=True, exist_ok=True)
    scaler_file = root / scaler_path
    joblib.dump(scaler, scaler_file)
    upsert_artifact_index(run_id, "feature_engineering.scaler", scaler_file.as_posix())

    stage_result = {
        "stage": "feature_engineering",
        "run_id": run_id,
        "df_key_in": df_key,
        "df_key_out": new_key,
        "actions": [f"standard_scale:{len(numeric_cols)}"] if numeric_cols else ["noop_scaling"],
        "metrics": {"scaled_columns": len(numeric_cols)},
        "artifacts": [scaler_file.as_posix()],
    }
    stage_path = save_json(run_id, "feature_engineering", "scaling_stage_result.json", stage_result)
    upsert_artifact_index(run_id, "feature_engineering.scaling_stage_result", stage_path)
    return {"df_key": new_key, "stage_result": stage_result}

