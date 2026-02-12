from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.agent.data_manager import DataManager
from src.skills.reporting.artifacts import save_json, upsert_artifact_index


def run_encoding(
    data_manager: DataManager,
    df_key: str,
    run_id: str,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    df = data_manager.get(df_key).copy()
    manifest: Dict[str, Any] = {"one_hot": [], "frequency": []}

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    protected = {target_column} if target_column else set()
    cat_cols = [c for c in cat_cols if c not in protected]

    for col in cat_cols:
        n_unique = int(df[col].nunique(dropna=True))
        if n_unique > 50:
            freq = df[col].value_counts(dropna=False, normalize=True)
            df[f"{col}__freq"] = df[col].map(freq).fillna(0.0)
            df = df.drop(columns=[col])
            manifest["frequency"].append(col)
        else:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            manifest["one_hot"].append(col)

    new_key = data_manager.register(df, source_name=f"{df_key}_encoding", stage="feat")
    manifest_path = save_json(run_id, "feature_engineering", "encoding_manifest.json", manifest)
    upsert_artifact_index(run_id, "feature_engineering.encoding_manifest", manifest_path)

    stage_result = {
        "stage": "feature_engineering",
        "run_id": run_id,
        "df_key_in": df_key,
        "df_key_out": new_key,
        "actions": [f"one_hot:{c}" for c in manifest["one_hot"]] + [f"frequency:{c}" for c in manifest["frequency"]] or ["noop_encoding"],
        "metrics": {"encoded_columns": len(manifest["one_hot"]) + len(manifest["frequency"])},
        "artifacts": [manifest_path],
    }
    stage_path = save_json(run_id, "feature_engineering", "encoding_stage_result.json", stage_result)
    upsert_artifact_index(run_id, "feature_engineering.encoding_stage_result", stage_path)
    return {"df_key": new_key, "stage_result": stage_result, "manifest": manifest}

