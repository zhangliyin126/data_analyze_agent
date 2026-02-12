from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.agent.data_manager import DataManager
from src.skills.reporting.artifacts import save_csv, save_json, upsert_artifact_index


def run_basic_stats(data_manager: DataManager, df_key: str, run_id: str) -> Dict[str, Any]:
    df = data_manager.get(df_key)

    numeric = df.select_dtypes(include=[np.number])
    numeric_stats = numeric.describe().T.reset_index().rename(columns={"index": "column"})
    numeric_path = save_csv(run_id, "profiling", "basic_stats_numeric.csv", numeric_stats)
    upsert_artifact_index(run_id, "profiling.basic_stats_numeric", numeric_path)

    cat = df.select_dtypes(exclude=[np.number])
    if not cat.empty:
        cat_stats = pd.DataFrame(
            {
                "column": cat.columns,
                "n_unique": [int(cat[c].nunique(dropna=True)) for c in cat.columns],
                "top_value": [str(cat[c].mode(dropna=True).iloc[0]) if not cat[c].mode(dropna=True).empty else "" for c in cat.columns],
            }
        )
    else:
        cat_stats = pd.DataFrame(columns=["column", "n_unique", "top_value"])
    cat_path = save_csv(run_id, "profiling", "basic_stats_categorical.csv", cat_stats)
    upsert_artifact_index(run_id, "profiling.basic_stats_categorical", cat_path)

    corr = numeric.corr(numeric_only=True)
    corr_path = save_csv(run_id, "profiling", "correlation_matrix.csv", corr.reset_index().rename(columns={"index": "column"}))
    upsert_artifact_index(run_id, "profiling.correlation_matrix", corr_path)

    stage_result = {
        "stage": "profiling",
        "run_id": run_id,
        "df_key_in": df_key,
        "df_key_out": df_key,
        "actions": ["run_basic_stats"],
        "metrics": {
            "numeric_columns": int(numeric.shape[1]),
            "categorical_columns": int(cat.shape[1]),
        },
        "artifacts": [numeric_path, cat_path, corr_path],
    }
    stage_path = save_json(run_id, "profiling", "basic_stats_stage_result.json", stage_result)
    upsert_artifact_index(run_id, "profiling.basic_stats_stage_result", stage_path)
    return stage_result

