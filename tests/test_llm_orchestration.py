import json
from pathlib import Path
import uuid

import pandas as pd

from src.agent.data_manager import DataManager
from src.agent.graph import run_pipeline
from src.agent.llm_orchestration import LLMOrchestrator
from src.agent.observer import Observer
from src.agent.state import AgentState


class _CaptureObserver:
    def __init__(self) -> None:
        self.events = []

    def emit_stream_event(self, run_id: str, event):
        self.events.append({"run_id": run_id, **event})


class _GoodLLM:
    def plan(self, profile, schema, candidates, user_intent):
        return {
            "task_type": "regression",
            "data_structure": "flat_iid",
            "target_column": "y",
            "only_eda": False,
            "use_mixed_effect": False,
            "group_column": None,
            "time_column": None,
            "feature_strategy": {},
            "modeling_strategy": {},
            "evaluation_strategy": {},
            "reasoning": ["ok"],
        }

    def interpret_stage(self, stage, stage_result):
        return {"key_findings": [f"{stage} ok"], "risks": [], "next_step": "continue"}

    def answer(self, question, artifact_summaries, final_summary):
        return "ok"


class _BrokenLLM:
    def plan(self, profile, schema, candidates, user_intent):
        raise RuntimeError("planner unavailable")

    def interpret_stage(self, stage, stage_result):
        raise RuntimeError("interpreter unavailable")

    def answer(self, question, artifact_summaries, final_summary):
        raise RuntimeError("qa unavailable")


def test_orchestrator_emits_success_events() -> None:
    observer = _CaptureObserver()
    orchestrator = LLMOrchestrator(llm_client=_GoodLLM(), observer=observer)
    profile = {"time_columns": [], "group_candidates": []}
    schema = {"columns": {"y": {"suggested_role": "target"}}}
    candidates = {
        "task_candidates": [{"task_type": "regression", "score": 100, "reason": "intent"}],
        "data_structure": {"data_structure": "flat_iid", "signals": {}},
        "target_candidates": ["y"],
        "must_use_mixed_effect": False,
    }

    plan = orchestrator.plan("run_success", profile, schema, candidates, "回归")
    card = orchestrator.interpret_stage("run_success", "profiling", {"actions": []})
    answer = orchestrator.answer("run_success", "best model?", [], "summary")

    assert plan["task_type"] == "regression"
    assert card["key_findings"] == ["profiling ok"]
    assert answer == "ok"
    event_names = [e["event"] for e in observer.events]
    assert event_names.count("llm_call_start") == 3
    assert event_names.count("llm_call_success") == 3


def test_orchestrator_fallback_when_llm_fails() -> None:
    observer = _CaptureObserver()
    orchestrator = LLMOrchestrator(llm_client=_BrokenLLM(), observer=observer)
    profile = {"time_columns": ["dt"], "group_candidates": ["gid"]}
    schema = {"columns": {"y": {"suggested_role": "target"}, "gid": {"suggested_role": "group"}}}
    candidates = {
        "task_candidates": [{"task_type": "regression", "score": 90, "reason": "schema"}],
        "data_structure": {"data_structure": "hierarchical", "signals": {"schema_group_cols": ["gid"]}},
        "target_candidates": ["y"],
        "must_use_mixed_effect": True,
    }

    plan = orchestrator.plan("run_fallback", profile, schema, candidates, "回归")
    stage = orchestrator.interpret_stage("run_fallback", "modeling", {"actions": []})
    qa = orchestrator.answer("run_fallback", "?", [], "summary text")

    assert plan["target_column"] == "y"
    assert plan["use_mixed_effect"] is True
    assert "fallback" in plan["reasoning"][0]
    assert stage["next_step"] == "continue"
    assert "Fallback" in qa
    failed = [e for e in observer.events if e["event"] == "llm_call_failed"]
    assert len(failed) == 3


def test_pipeline_continues_with_broken_llm(monkeypatch) -> None:
    local_root = Path("test_output") / f"orchestration_{uuid.uuid4().hex[:8]}"
    local_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AGENT_STORAGE_ROOT", str(local_root / "runs"))

    dm = DataManager()
    df = pd.DataFrame({"x": [0.1, 0.2, 0.4, 0.8, 1.2], "y": [1.0, 1.2, 1.6, 2.0, 2.5]})
    df_key = dm.register(df, source_name="unit_orchestration", stage="raw")
    observer = Observer()
    run_id = "run_test_orchestration_fallback"
    state = AgentState(run_id=run_id, df_key=df_key, user_intent="请做回归分析")
    schema = {"columns": {"y": {"suggested_role": "target"}}}

    out = run_pipeline(
        initial_state=state,
        data_manager=dm,
        observer=observer,
        llm_client=_BrokenLLM(),
        schema=schema,
    )

    assert out.plan is not None
    assert out.plan.task_type == "regression"
    assert out.final_summary is not None

    events_path = local_root / "runs" / run_id / "logs" / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(e.get("event") == "llm_call_failed" for e in events)
