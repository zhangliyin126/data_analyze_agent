from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.agent.data_manager import DataManager
from src.skills.reporting.artifacts import save_json, upsert_artifact_index


def run_feature_select(
    data_manager: DataManager,
    df_key: str,
    run_id: str,
    target_column: Optional[str] = None,
    variance_threshold: float = 1e-10,
    corr_threshold: float = 0.98,
) -> Dict[str, Any]:
    df = data_manager.get(df_key).copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column and target_column in numeric_cols:
        numeric_cols.remove(target_column)

    removed_low_var = []
    for col in list(numeric_cols):
        if float(df[col].var()) <= variance_threshold:
            removed_low_var.append(col)
    if removed_low_var:
        df = df.drop(columns=removed_low_var)

    numeric_cols_after = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != target_column]
    corr = df[numeric_cols_after].corr(numeric_only=True).abs() if numeric_cols_after else None
    removed_corr = []
    if corr is not None and not corr.empty:
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        removed_corr = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        if removed_corr:
            df = df.drop(columns=removed_corr)

    new_key = data_manager.register(df, source_name=f"{df_key}_fselect", stage="feat")
    manifest = {"removed_low_variance": removed_low_var, "removed_high_correlation": removed_corr}
    manifest_path = save_json(run_id, "feature_engineering", "feature_select_manifest.json", manifest)
    upsert_artifact_index(run_id, "feature_engineering.feature_select_manifest", manifest_path)

    stage_result = {
        "stage": "feature_engineering",
        "run_id": run_id,
        "df_key_in": df_key,
        "df_key_out": new_key,
        "actions": [f"drop_low_var:{len(removed_low_var)}", f"drop_high_corr:{len(removed_corr)}"],
        "metrics": {"columns_out": int(df.shape[1])},
        "artifacts": [manifest_path],
    }
    stage_path = save_json(run_id, "feature_engineering", "feature_select_stage_result.json", stage_result)
    upsert_artifact_index(run_id, "feature_engineering.feature_select_stage_result", stage_path)
    return {"df_key": new_key, "stage_result": stage_result}

