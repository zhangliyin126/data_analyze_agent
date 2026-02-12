import json
from pathlib import Path
import uuid

import pandas as pd

from src.agent.data_manager import DataManager
from src.skills.profiling.dataset_profile import profile_dataset


def test_profile_contains_required_fields(monkeypatch) -> None:
    local_root = Path("test_output") / f"profile_{uuid.uuid4().hex[:8]}"
    local_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AGENT_STORAGE_ROOT", str(local_root / "runs"))
    dm = DataManager()
    df = pd.DataFrame(
        {
            "patient_id": ["a", "a", "b", "b"],
            "visit_date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "y": [10.0, 10.5, 12.0, 12.2],
        }
    )
    key = dm.register(df, source_name="profile_case", stage="raw")
    out = profile_dataset(dm, key, run_id="run_test_profile", schema={})

    profile = out["profile"]
    assert profile["n_rows"] == 4
    assert "column_types" in profile
    assert "missing_ratio" in profile
    assert "distribution_stats" in profile
    assert "group_candidates" in profile

    profile_path = local_root / "runs" / "run_test_profile" / "profiling" / "profile.json"
    assert profile_path.exists()
    loaded = json.loads(profile_path.read_text(encoding="utf-8"))
    assert loaded["n_cols"] == 4
