from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.agent.data_manager import DataManager
from src.agent.graph import run_pipeline
from src.agent.observer import Observer
from src.agent.state import AgentState
from src.agent.utils import generate_run_id

from .local_llm import LocalLLMClient
from .mock_llm import MockLLMClient


def _load_schema(schema_path: Optional[str]) -> Dict[str, Any]:
    if not schema_path:
        return {}
    path = Path(schema_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dataframe(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data format: {path.suffix}. Use CSV or Excel.")


def run_agent_from_df(
    df: pd.DataFrame,
    intent: str,
    schema: Optional[Dict[str, Any]] = None,
    use_mock_llm: bool = True,
    run_id: Optional[str] = None,
) -> AgentState:
    schema = schema or {}
    run_id = run_id or generate_run_id()
    data_manager = DataManager()
    observer = Observer()
    df_key = data_manager.register(df, source_name="in_memory_df", stage="raw")
    state = AgentState(run_id=run_id, df_key=df_key, user_intent=intent, schema_path=None)
    llm = MockLLMClient() if use_mock_llm else LocalLLMClient()
    return run_pipeline(state, data_manager=data_manager, observer=observer, llm_client=llm, schema=schema)


def run_agent(
    data_path: str,
    intent: str,
    schema_path: Optional[str] = None,
    use_mock_llm: bool = False,
    run_id: Optional[str] = None,
) -> AgentState:
    df = _load_dataframe(data_path)
    schema = _load_schema(schema_path)
    return run_agent_from_df(df=df, intent=intent, schema=schema, use_mock_llm=use_mock_llm, run_id=run_id)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Data analysis agent runner")
    parser.add_argument("--data", required=True, help="Input CSV/Excel file path")
    parser.add_argument("--schema", required=False, default=None, help="Optional column_schema.json path")
    parser.add_argument("--intent", required=True, help="User analysis intent")
    parser.add_argument("--use-mock-llm", action="store_true", help="Use mock LLM instead of local LLM API")
    parser.add_argument("--run-id", required=False, default=None, help="Optional explicit run_id")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    state = run_agent(
        data_path=args.data,
        intent=args.intent,
        schema_path=args.schema,
        use_mock_llm=args.use_mock_llm,
        run_id=args.run_id,
    )
    print(f"run_id={state.run_id}")
    print(state.final_summary or "No summary generated.")


if __name__ == "__main__":
    main()

