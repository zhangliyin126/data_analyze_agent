import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from src.app.agent_runner import build_arg_parser, run_agent_from_df


def test_pipeline_smoke(monkeypatch) -> None:
    local_root = Path("test_output") / f"smoke_{uuid.uuid4().hex[:8]}"
    local_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AGENT_STORAGE_ROOT", str(local_root / "runs"))
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(8):
        bias = rng.normal(0, 1)
        for t in range(5):
            x = rng.normal(0, 1)
            y = 1.5 * x + bias + rng.normal(0, 0.2)
            rows.append({"user_id": f"u{uid}", "visit_time": f"2025-01-{t+1:02d}", "x": x, "y": y})
    df = pd.DataFrame(rows)
    schema = {
        "columns": {
            "user_id": {"suggested_role": "group"},
            "visit_time": {"suggested_role": "time"},
            "y": {"suggested_role": "target"},
        }
    }
    state = run_agent_from_df(df=df, intent="请做回归分析", schema=schema, use_mock_llm=True, run_id="run_test_smoke")

    run_dir = local_root / "runs" / "run_test_smoke"
    assert (run_dir / "reporting" / "summary.md").exists()
    assert (run_dir / "reporting" / "final_summary.json").exists()
    assert (run_dir / "reporting" / "stage_cards.json").exists()
    assert (run_dir / "logs" / "events.jsonl").exists()

    events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    event_types = {e["event"] for e in events}
    assert {"node_start", "node_end", "stage_card_emitted", "tool_call", "artifact_saved"}.issubset(event_types)

    assert state.final_summary is not None
    assert len(state.stage_cards) >= 4


def test_runner_has_no_delete_operations_exposed() -> None:
    parser = build_arg_parser()
    option_strings = []
    for action in parser._actions:
        option_strings.extend(action.option_strings)
    forbidden = ["--delete", "--remove", "--clean", "--purge"]
    assert all(flag not in option_strings for flag in forbidden)
