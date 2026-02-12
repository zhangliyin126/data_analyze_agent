from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .state import DataStructure, PlannerOutput, TaskType


def _schema_target_candidates(schema: Dict[str, Any]) -> List[str]:
    columns = schema.get("columns", {})
    targets = []
    for col, spec in columns.items():
        if str(spec.get("suggested_role", "")).lower() == "target":
            targets.append(col)
    return targets


def _intent_task_hint(user_intent: str) -> TaskType | None:
    text = (user_intent or "").lower()
    if any(k in text for k in ["分类", "classification", "classify"]):
        return "classification"
    if any(k in text for k in ["时序", "time series", "forecast", "预测未来"]):
        return "timeseries"
    if any(k in text for k in ["生存", "survival"]):
        return "survival"
    if any(k in text for k in ["探索", "eda", "可视化", "describe"]):
        return "eda"
    if any(k in text for k in ["回归", "regression", "连续"]):
        return "regression"
    return None


def infer_task_candidates(
    profile: Dict[str, Any],
    schema: Dict[str, Any],
    user_intent: str,
) -> List[Dict[str, Any]]:
    candidates: List[Tuple[TaskType, int, str]] = []
    intent_hint = _intent_task_hint(user_intent)
    if intent_hint:
        candidates.append((intent_hint, 100, "matched user intent"))

    target_candidates = _schema_target_candidates(schema)
    column_types = profile.get("column_types", {})
    if target_candidates:
        target = target_candidates[0]
        dtype = str(column_types.get(target, "")).lower()
        if "int" in dtype or "float" in dtype:
            candidates.append(("regression", 85, "schema target appears numeric"))
        else:
            candidates.append(("classification", 85, "schema target appears categorical"))

    if profile.get("time_columns"):
        candidates.append(("timeseries", 70, "profile contains time columns"))

    if not candidates:
        candidates.append(("eda", 60, "fallback to exploratory data analysis"))

    dedup: Dict[str, Dict[str, Any]] = {}
    for task, score, reason in candidates:
        cur = dedup.get(task)
        if cur is None or score > cur["score"]:
            dedup[task] = {"task_type": task, "score": score, "reason": reason}

    return sorted(dedup.values(), key=lambda x: x["score"], reverse=True)


def infer_data_structure(profile: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    n_rows = max(int(profile.get("n_rows", 0)), 1)
    n_cols = int(profile.get("n_cols", 0))
    ratio = n_cols / n_rows

    schema_group_cols = []
    for col, spec in schema.get("columns", {}).items():
        if str(spec.get("suggested_role", "")).lower() == "group":
            schema_group_cols.append(col)

    group_candidates = profile.get("group_candidates", [])
    avg_obs = float(profile.get("avg_obs_per_group", 0.0))
    time_columns = profile.get("time_columns", [])

    hierarchical = bool(schema_group_cols or group_candidates) and avg_obs >= 2.0
    repeated = hierarchical and bool(time_columns)
    high_d_low_n = ratio >= 1.5

    if repeated:
        structure: DataStructure = "repeated_measure"
    elif hierarchical:
        structure = "hierarchical"
    elif high_d_low_n:
        structure = "high_d_low_n"
    else:
        structure = "flat_iid"

    return {
        "data_structure": structure,
        "signals": {
            "schema_group_cols": schema_group_cols,
            "group_candidates": group_candidates,
            "avg_obs_per_group": avg_obs,
            "time_columns": time_columns,
            "feature_row_ratio": ratio,
            "high_d_low_n": high_d_low_n,
        },
    }


def must_use_mixed_effect(profile: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    structure = infer_data_structure(profile, schema).get("data_structure")
    return structure in {"hierarchical", "repeated_measure"}


def build_planner_candidates(
    profile: Dict[str, Any],
    schema: Dict[str, Any],
    user_intent: str,
) -> Dict[str, Any]:
    task_candidates = infer_task_candidates(profile, schema, user_intent)
    structure = infer_data_structure(profile, schema)
    return {
        "task_candidates": task_candidates,
        "data_structure": structure,
        "target_candidates": _schema_target_candidates(schema),
        "must_use_mixed_effect": must_use_mixed_effect(profile, schema),
    }


def enforce_plan_constraints(
    plan: Dict[str, Any],
    profile: Dict[str, Any],
    schema: Dict[str, Any],
    user_intent: str,
) -> PlannerOutput:
    candidates = build_planner_candidates(profile, schema, user_intent)
    task_candidates = candidates.get("task_candidates", [])
    top_task = task_candidates[0]["task_type"] if task_candidates else "eda"
    structure = candidates["data_structure"]["data_structure"]
    mixed_required = bool(candidates["must_use_mixed_effect"])

    normalized = dict(plan or {})
    if normalized.get("task_type") not in {"regression", "classification", "timeseries", "survival", "eda"}:
        normalized["task_type"] = top_task

    normalized["data_structure"] = structure
    normalized["use_mixed_effect"] = bool(normalized.get("use_mixed_effect", False) or mixed_required)
    normalized["reasoning"] = list(normalized.get("reasoning", []))

    if mixed_required and "forced mixed effect by decision rules" not in normalized["reasoning"]:
        normalized["reasoning"].append("forced mixed effect by decision rules")

    targets = candidates.get("target_candidates", [])
    if not normalized.get("target_column") and targets:
        normalized["target_column"] = targets[0]

    if not normalized.get("group_column"):
        schema_groups = candidates["data_structure"]["signals"].get("schema_group_cols", [])
        if schema_groups:
            normalized["group_column"] = schema_groups[0]
        elif profile.get("group_candidates"):
            normalized["group_column"] = profile["group_candidates"][0]

    if not normalized.get("time_column") and profile.get("time_columns"):
        normalized["time_column"] = profile["time_columns"][0]

    normalized.setdefault("feature_strategy", {})
    normalized.setdefault("modeling_strategy", {})
    normalized.setdefault("evaluation_strategy", {})
    normalized.setdefault("only_eda", normalized.get("task_type") == "eda")

    return PlannerOutput.model_validate(normalized)

