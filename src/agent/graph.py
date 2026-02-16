from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.skills.cleaning.dedup import run_dedup
from src.skills.cleaning.missing import run_missing_imputation
from src.skills.cleaning.outlier import run_outlier_winsorize
from src.skills.cleaning.type_fix import run_type_fix
from src.skills.eda.basic_stats import run_basic_stats
from src.skills.eda.visualization import run_visualization
from src.skills.feature_engineering.encoding import run_encoding
from src.skills.feature_engineering.feature_select import run_feature_select
from src.skills.feature_engineering.interaction import run_interaction_features
from src.skills.feature_engineering.scaling import run_scaling
from src.skills.feature_engineering.time_features import run_time_features
from src.skills.modeling.baseline import run_baseline_model
from src.skills.modeling.evaluation import run_evaluation
from src.skills.modeling.mixed_effect import run_mixed_effect_model
from src.skills.profiling.dataset_profile import profile_dataset
from src.skills.reporting.artifacts import save_json, upsert_artifact_index

from .data_manager import DataManager
from .decision_rules import build_planner_candidates, enforce_plan_constraints
from .llm_orchestration import LLMOrchestrator
from .observer import Observer
from .state import AgentState, ArtifactRef, PlannerOutput, StageCard


def _stage_card(
    llm_orchestrator: LLMOrchestrator,
    run_id: str,
    stage: str,
    stage_result: Dict[str, Any],
) -> StageCard:
    key_findings = [f"{stage} completed"]
    next_step = "continue"
    try:
        card = llm_orchestrator.interpret_stage(run_id, stage, stage_result)
        key_findings = card.get("key_findings", key_findings)
        next_step = card.get("next_step", next_step)
    except Exception:
        pass
    artifacts = [ArtifactRef(name=Path(p).name, path=p) for p in stage_result.get("artifacts", [])]
    return StageCard(stage=stage, key_findings=key_findings, artifacts=artifacts, next_step=next_step)


def _storage_root() -> Path:
    return Path(os.getenv("AGENT_STORAGE_ROOT", "storage/runs"))


def build_graph(use_mock_llm: bool = False) -> Dict[str, Any]:
    # Lightweight graph spec used both for documentation and tests.
    return {
        "nodes": [
            "ingest",
            "profiling",
            "plan",
            "cleaning",
            "feature_engineering",
            "modeling",
            "evaluation",
            "final_summary",
            "qa",
        ],
        "edges": [
            ("ingest", "profiling"),
            ("profiling", "plan"),
            ("plan", "cleaning"),
            ("cleaning", "feature_engineering"),
            ("feature_engineering", "modeling"),
            ("modeling", "evaluation"),
            ("evaluation", "final_summary"),
        ],
        "flags": {"use_mock_llm": use_mock_llm},
    }


def run_pipeline(
    initial_state: AgentState,
    data_manager: DataManager,
    observer: Observer,
    llm_client: Any,
    schema: Optional[Dict[str, Any]] = None,
) -> AgentState:
    schema = schema or {}
    state = initial_state.model_copy(deep=True)
    run_id = state.run_id
    llm_orchestrator = LLMOrchestrator(llm_client=llm_client, observer=observer)

    # Profiling
    observer.on_node_start(run_id, "profiling", state.df_key)
    profile_res = profile_dataset(data_manager, state.df_key, run_id, schema=schema)
    run_basic_stats(data_manager, state.df_key, run_id)
    run_visualization(data_manager, state.df_key, run_id)
    state.profile_summary = profile_res["profile"]
    profile_card = _stage_card(llm_orchestrator, run_id, "profiling", profile_res["stage_result"])
    state.stage_cards.append(profile_card)
    observer.emit_stage_card(run_id, profile_card.model_dump())
    observer.on_node_end(run_id, "profiling", {"n_rows": state.profile_summary.get("n_rows"), "n_cols": state.profile_summary.get("n_cols")})

    # Plan
    observer.on_node_start(run_id, "plan", state.df_key)
    candidates = build_planner_candidates(state.profile_summary, schema, state.user_intent)
    observer.emit_stream_event(run_id, {"event": "tool_call", "node": "plan", "message": "build_planner_candidates"})
    raw_plan = llm_orchestrator.plan(run_id, state.profile_summary, schema, candidates, state.user_intent)
    plan: PlannerOutput = enforce_plan_constraints(raw_plan, state.profile_summary, schema, state.user_intent)
    state.plan = plan
    plan_path = save_json(run_id, "reporting", "plan.json", plan.model_dump())
    upsert_artifact_index(run_id, "reporting.plan", plan_path)
    observer.emit_stream_event(run_id, {"event": "artifact_saved", "node": "plan", "path": plan_path})
    observer.on_node_end(run_id, "plan", {"task_type": plan.task_type, "use_mixed_effect": plan.use_mixed_effect})

    current_key = state.df_key

    # Cleaning
    if not plan.only_eda:
        observer.on_node_start(run_id, "cleaning", current_key)
        clean_steps = [
            run_type_fix(data_manager, current_key, run_id),
        ]
        current_key = clean_steps[-1]["df_key"]
        clean_steps.append(run_missing_imputation(data_manager, current_key, run_id))
        current_key = clean_steps[-1]["df_key"]
        clean_steps.append(run_outlier_winsorize(data_manager, current_key, run_id))
        current_key = clean_steps[-1]["df_key"]
        clean_steps.append(run_dedup(data_manager, current_key, run_id))
        current_key = clean_steps[-1]["df_key"]

        clean_actions = [a for step in clean_steps for a in step["stage_result"].get("actions", [])]
        clean_artifacts = [p for step in clean_steps for p in step["stage_result"].get("artifacts", [])]
        clean_stage = {
            "stage": "cleaning",
            "run_id": run_id,
            "df_key_in": state.df_key,
            "df_key_out": current_key,
            "actions": clean_actions,
            "metrics": {"steps": len(clean_steps)},
            "artifacts": clean_artifacts,
        }
        clean_path = save_json(run_id, "cleaning", "stage_result.json", clean_stage)
        upsert_artifact_index(run_id, "cleaning.stage_result", clean_path)
        clean_card = _stage_card(llm_orchestrator, run_id, "cleaning", clean_stage)
        state.stage_cards.append(clean_card)
        observer.emit_stage_card(run_id, clean_card.model_dump())
        observer.on_node_end(run_id, "cleaning", {"df_key_out": current_key})

        # Feature engineering
        observer.on_node_start(run_id, "feature_engineering", current_key)
        feat_steps = [run_encoding(data_manager, current_key, run_id, target_column=plan.target_column)]
        current_key = feat_steps[-1]["df_key"]
        if plan.time_column:
            feat_steps.append(run_time_features(data_manager, current_key, run_id, time_column=plan.time_column, target_column=plan.target_column))
            current_key = feat_steps[-1]["df_key"]
        feat_steps.append(run_interaction_features(data_manager, current_key, run_id, target_column=plan.target_column, enable=True))
        current_key = feat_steps[-1]["df_key"]
        feat_steps.append(run_scaling(data_manager, current_key, run_id, target_column=plan.target_column))
        current_key = feat_steps[-1]["df_key"]
        feat_steps.append(run_feature_select(data_manager, current_key, run_id, target_column=plan.target_column))
        current_key = feat_steps[-1]["df_key"]

        feat_actions = [a for step in feat_steps for a in step["stage_result"].get("actions", [])]
        feat_artifacts = [p for step in feat_steps for p in step["stage_result"].get("artifacts", [])]
        feat_stage = {
            "stage": "feature_engineering",
            "run_id": run_id,
            "df_key_in": clean_stage["df_key_out"],
            "df_key_out": current_key,
            "actions": feat_actions,
            "metrics": {"steps": len(feat_steps)},
            "artifacts": feat_artifacts,
        }
        feat_path = save_json(run_id, "feature_engineering", "stage_result.json", feat_stage)
        upsert_artifact_index(run_id, "feature_engineering.stage_result", feat_path)
        feat_card = _stage_card(llm_orchestrator, run_id, "feature_engineering", feat_stage)
        state.stage_cards.append(feat_card)
        observer.emit_stage_card(run_id, feat_card.model_dump())
        observer.on_node_end(run_id, "feature_engineering", {"df_key_out": current_key})

    state.df_key = current_key

    # Modeling
    observer.on_node_start(run_id, "modeling", state.df_key)
    baseline = run_baseline_model(data_manager, state.df_key, run_id, state.plan.model_dump() if state.plan else {})
    mixed = None
    if state.plan and state.plan.use_mixed_effect:
        observer.emit_stream_event(run_id, {"event": "tool_call", "node": "modeling", "message": "run_mixed_effect_model"})
        mixed = run_mixed_effect_model(data_manager, state.df_key, run_id, state.plan.model_dump())

    model_stage = {
        "stage": "modeling",
        "run_id": run_id,
        "df_key_in": state.df_key,
        "df_key_out": state.df_key,
        "actions": ["run_baseline_model"] + (["run_mixed_effect_model"] if mixed else []),
        "metrics": {"baseline_best": baseline.get("best_model"), "mixed_enabled": bool(mixed)},
        "artifacts": baseline["stage_result"].get("artifacts", []) + (mixed["stage_result"].get("artifacts", []) if mixed else []),
    }
    model_path = save_json(run_id, "modeling", "stage_result.json", model_stage)
    upsert_artifact_index(run_id, "modeling.stage_result", model_path)
    model_card = _stage_card(llm_orchestrator, run_id, "modeling", model_stage)
    state.stage_cards.append(model_card)
    observer.emit_stage_card(run_id, model_card.model_dump())
    observer.on_node_end(run_id, "modeling", {"mixed_effect": bool(mixed)})

    # Evaluation
    observer.on_node_start(run_id, "evaluation", state.df_key)
    eval_res = run_evaluation(
        run_id,
        task_type=state.plan.task_type if state.plan else "eda",
        baseline_metrics=baseline.get("metrics", {}),
        mixed_effect_metrics=mixed,
    )
    eval_card = _stage_card(llm_orchestrator, run_id, "evaluation", eval_res["stage_result"])
    state.stage_cards.append(eval_card)
    observer.emit_stage_card(run_id, eval_card.model_dump())
    observer.on_node_end(run_id, "evaluation", {"best_section": eval_res["evaluation"]["best_section"]})

    # Final summary
    summary_lines = [
        f"# Run Summary: {run_id}",
        f"- task_type: {state.plan.task_type if state.plan else 'unknown'}",
        f"- data_structure: {state.plan.data_structure if state.plan else 'unknown'}",
        f"- mixed_effect: {state.plan.use_mixed_effect if state.plan else False}",
        f"- total_stage_cards: {len(state.stage_cards)}",
        f"- evaluation_best_section: {eval_res['evaluation']['best_section']}",
    ]
    summary_text = "\n".join(summary_lines)
    state.final_summary = summary_text

    report_dir = _storage_root() / run_id / "reporting"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_md = report_dir / "summary.md"
    summary_md.write_text(summary_text, encoding="utf-8")
    upsert_artifact_index(run_id, "reporting.summary_md", summary_md.as_posix())
    final_summary_path = save_json(run_id, "reporting", "final_summary.json", {"final_summary": summary_text})
    upsert_artifact_index(run_id, "reporting.final_summary", final_summary_path)

    artifact_index_path = report_dir / "artifact_index.json"
    if artifact_index_path.exists():
        state.artifact_index = json.loads(artifact_index_path.read_text(encoding="utf-8"))
    return state
