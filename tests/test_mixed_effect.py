import numpy as np
import pandas as pd
from pathlib import Path
import uuid

from src.agent.data_manager import DataManager
from src.skills.modeling.mixed_effect import run_mixed_effect_model


def test_mixed_effect_outputs_required_artifacts(monkeypatch) -> None:
    local_root = Path("test_output") / f"mixed_{uuid.uuid4().hex[:8]}"
    local_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AGENT_STORAGE_ROOT", str(local_root / "runs"))
    rng = np.random.default_rng(42)
    rows = []
    for g in range(12):
        group_bias = rng.normal(0, 1)
        for t in range(8):
            x = rng.normal(0, 1)
            y = 2.0 * x + group_bias + rng.normal(0, 0.3)
            rows.append({"group_id": f"g{g}", "time": t, "x": x, "y": y})
    df = pd.DataFrame(rows)

    dm = DataManager()
    key = dm.register(df, source_name="mixed_case", stage="feat")
    plan = {"target_column": "y", "group_column": "group_id"}
    run_id = "run_test_mixed"
    result = run_mixed_effect_model(dm, key, run_id=run_id, plan=plan)
    assert "icc" in result

    base = local_root / "runs" / run_id / "modeling" / "mixed_effect"
    assert (base / "mixed_model.pkl").exists()
    assert (base / "fe_table.csv").exists()
    assert (base / "re_variance.json").exists()
    assert (base / "ICC.json").exists()
    assert (base / "comparison_with_ols.json").exists()
