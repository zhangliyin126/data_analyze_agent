from src.agent.decision_rules import (
    build_planner_candidates,
    enforce_plan_constraints,
    infer_data_structure,
    infer_task_candidates,
    must_use_mixed_effect,
)


def test_task_candidates_from_intent_and_schema() -> None:
    profile = {"column_types": {"target": "float64"}, "time_columns": []}
    schema = {"columns": {"target": {"suggested_role": "target"}}}
    cands = infer_task_candidates(profile, schema, "做回归预测")
    assert cands[0]["task_type"] == "regression"


def test_data_structure_hierarchical_detection() -> None:
    profile = {
        "n_rows": 100,
        "n_cols": 10,
        "group_candidates": ["patient_id"],
        "avg_obs_per_group": 3.2,
        "time_columns": ["visit_time"],
    }
    schema = {"columns": {"patient_id": {"suggested_role": "group"}}}
    result = infer_data_structure(profile, schema)
    assert result["data_structure"] == "repeated_measure"
    assert must_use_mixed_effect(profile, schema) is True


def test_enforce_plan_constraints_overrides_invalid_plan() -> None:
    profile = {
        "n_rows": 60,
        "n_cols": 8,
        "group_candidates": ["user_id"],
        "avg_obs_per_group": 2.4,
        "time_columns": [],
        "column_types": {"y": "float64"},
    }
    schema = {
        "columns": {
            "user_id": {"suggested_role": "group"},
            "y": {"suggested_role": "target"},
        }
    }
    candidates = build_planner_candidates(profile, schema, "做分析")
    assert candidates["must_use_mixed_effect"] is True

    raw_plan = {
        "task_type": "regression",
        "data_structure": "flat_iid",
        "target_column": "y",
        "use_mixed_effect": False,
        "reasoning": [],
    }
    plan = enforce_plan_constraints(raw_plan, profile, schema, "做分析")
    assert plan.use_mixed_effect is True
    assert plan.data_structure in {"hierarchical", "repeated_measure"}
    assert "forced mixed effect by decision rules" in plan.reasoning

