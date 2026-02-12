from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is expected in runtime
    pd = None  # type: ignore[assignment]

TaskType = Literal["regression", "classification", "timeseries", "survival", "eda"]
DataStructure = Literal["hierarchical", "repeated_measure", "high_d_low_n", "flat_iid"]
StageName = Literal["profiling", "cleaning", "feature_engineering", "modeling", "evaluation"]


class ArtifactRef(BaseModel):
    name: str
    path: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class StageCard(BaseModel):
    stage: StageName
    key_findings: List[str]
    artifacts: List[ArtifactRef]
    next_step: str


class PlannerOutput(BaseModel):
    task_type: TaskType
    data_structure: DataStructure
    target_column: Optional[str] = None
    only_eda: bool = False
    use_mixed_effect: bool = False
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    feature_strategy: Dict[str, Any] = Field(default_factory=dict)
    modeling_strategy: Dict[str, Any] = Field(default_factory=dict)
    evaluation_strategy: Dict[str, Any] = Field(default_factory=dict)
    reasoning: List[str] = Field(default_factory=list)


def _contains_dataframe(value: Any) -> bool:
    if pd is not None and isinstance(value, pd.DataFrame):
        return True
    if isinstance(value, dict):
        return any(_contains_dataframe(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_dataframe(v) for v in value)
    if isinstance(value, BaseModel):
        return _contains_dataframe(value.model_dump())
    return False


class AgentState(BaseModel):
    run_id: str
    df_key: str
    user_intent: str
    schema_path: Optional[str] = None
    profile_summary: Dict[str, Any] = Field(default_factory=dict)
    plan: Optional[PlannerOutput] = None
    stage_cards: List[StageCard] = Field(default_factory=list)
    artifact_index: Dict[str, str] = Field(default_factory=dict)
    stream_events: List[Dict[str, Any]] = Field(default_factory=list)
    final_summary: Optional[str] = None

    @model_validator(mode="after")
    def _validate_no_dataframe(self) -> "AgentState":
        if _contains_dataframe(self):
            raise ValueError("AgentState must not contain pandas DataFrame.")
        return self

