from __future__ import annotations

from typing import Any, Dict

from src.agent.data_manager import DataManager
from src.skills.reporting.artifacts import save_json, upsert_artifact_index


def run_dedup(data_manager: DataManager, df_key: str, run_id: str) -> Dict[str, Any]:
    df = data_manager.get(df_key).copy()
    before_rows = int(df.shape[0])
    df = df.drop_duplicates()
    after_rows = int(df.shape[0])
    removed = before_rows - after_rows
    new_key = data_manager.register(df, source_name=f"{df_key}_dedup", stage="clean")

    stage_result = {
        "stage": "cleaning",
        "run_id": run_id,
        "df_key_in": df_key,
        "df_key_out": new_key,
        "actions": [f"drop_duplicates:{removed}"] if removed else ["noop_dedup"],
        "metrics": {"rows_before": before_rows, "rows_after": after_rows},
        "artifacts": [],
    }
    path = save_json(run_id, "cleaning", "dedup_stage_result.json", stage_result)
    upsert_artifact_index(run_id, "cleaning.dedup_stage_result", path)
    return {"df_key": new_key, "stage_result": stage_result}

