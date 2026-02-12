from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.agent.data_manager import DataManager
from src.skills.reporting.artifacts import save_csv, save_json, upsert_artifact_index


def _choose_group_column(df: pd.DataFrame, group_column: Optional[str]) -> str:
    if group_column and group_column in df.columns:
        return group_column
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
            if df[col].nunique(dropna=True) < len(df):
                return col
    return df.columns[0]


def _choose_target_column(df: pd.DataFrame, target_column: Optional[str], group_column: str) -> str:
    if target_column and target_column in df.columns and target_column != group_column:
        return target_column
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != group_column]
    if numeric:
        return numeric[-1]
    raise ValueError("Mixed effect requires at least one numeric target column.")


def run_mixed_effect_model(
    data_manager: DataManager,
    df_key: str,
    run_id: str,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    df = data_manager.get(df_key).copy()
    out_dir = Path(os.getenv("AGENT_STORAGE_ROOT", "storage/runs")) / run_id / "modeling" / "mixed_effect"
    out_dir.mkdir(parents=True, exist_ok=True)

    group_column = _choose_group_column(df, plan.get("group_column"))
    target_column = _choose_target_column(df, plan.get("target_column"), group_column)

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
    if group_column in feature_cols:
        feature_cols.remove(group_column)
    feature_cols = feature_cols[:20]

    clean_df = df[[group_column, target_column] + feature_cols].dropna().copy()
    if clean_df.empty:
        raise ValueError("No valid rows for mixed effect model.")

    y = pd.to_numeric(clean_df[target_column], errors="coerce")
    x = clean_df[feature_cols] if feature_cols else pd.DataFrame(index=clean_df.index)
    x = sm.add_constant(x, has_constant="add")
    groups = clean_df[group_column].astype(str)

    try:
        mixed = sm.MixedLM(endog=y, exog=x, groups=groups)
        mixed_result = mixed.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        status = "ok"
        error_message = ""
    except Exception as exc:
        # Fallback to deterministic small optimization path.
        mixed = sm.MixedLM(endog=y, exog=x, groups=groups)
        mixed_result = mixed.fit(reml=False, method="powell", maxiter=200, disp=False)
        status = "ok_with_fallback"
        error_message = str(exc)

    model_path = out_dir / "mixed_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(mixed_result, f)
    upsert_artifact_index(run_id, "modeling.mixed_effect.model", model_path.as_posix())

    fe_table = pd.DataFrame(
        {
            "feature": mixed_result.fe_params.index.tolist(),
            "coef": mixed_result.fe_params.values.tolist(),
        }
    )
    fe_path = save_csv(run_id, "modeling/mixed_effect", "fe_table.csv", fe_table)
    upsert_artifact_index(run_id, "modeling.mixed_effect.fe_table", fe_path)

    cov_re = mixed_result.cov_re
    if isinstance(cov_re, pd.DataFrame):
        re_var = float(np.diag(cov_re.values).mean()) if cov_re.size else 0.0
        re_payload = {"matrix": cov_re.to_dict(), "avg_random_variance": re_var}
    else:
        re_var = float(cov_re) if cov_re is not None else 0.0
        re_payload = {"avg_random_variance": re_var}

    re_path = save_json(run_id, "modeling/mixed_effect", "re_variance.json", re_payload)
    upsert_artifact_index(run_id, "modeling.mixed_effect.re_variance", re_path)

    residual_var = float(mixed_result.scale) if mixed_result.scale is not None else 0.0
    denom = re_var + residual_var
    icc = float(re_var / denom) if denom > 0 else 0.0
    icc_path = save_json(run_id, "modeling/mixed_effect", "ICC.json", {"ICC": icc, "random_variance": re_var, "residual_variance": residual_var})
    upsert_artifact_index(run_id, "modeling.mixed_effect.icc", icc_path)

    ols_result = sm.OLS(y, x).fit()
    comparison = {
        "mixed_aic": float(mixed_result.aic) if mixed_result.aic is not None else None,
        "mixed_bic": float(mixed_result.bic) if mixed_result.bic is not None else None,
        "ols_aic": float(ols_result.aic) if ols_result.aic is not None else None,
        "ols_bic": float(ols_result.bic) if ols_result.bic is not None else None,
        "ols_r2": float(ols_result.rsquared),
        "status": status,
        "fit_warning": error_message,
    }
    cmp_path = save_json(run_id, "modeling/mixed_effect", "comparison_with_ols.json", comparison)
    upsert_artifact_index(run_id, "modeling.mixed_effect.comparison_with_ols", cmp_path)

    stage_result = {
        "stage": "modeling",
        "run_id": run_id,
        "df_key_in": df_key,
        "df_key_out": df_key,
        "actions": ["run_mixed_effect_model"],
        "metrics": {"icc": icc, "group_column": group_column, "target_column": target_column},
        "artifacts": [model_path.as_posix(), fe_path, re_path, icc_path, cmp_path],
    }
    stage_path = save_json(run_id, "modeling/mixed_effect", "mixed_effect_stage_result.json", stage_result)
    upsert_artifact_index(run_id, "modeling.mixed_effect.stage_result", stage_path)
    return {"stage_result": stage_result, "comparison": comparison, "icc": icc}
