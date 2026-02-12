import pandas as pd
import pytest

from src.agent.data_manager import DataManager
from src.agent.state import AgentState


def test_data_manager_register_get_clone_release() -> None:
    dm = DataManager()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    key = dm.register(df, source_name="unit", stage="raw")
    loaded = dm.get(key)
    assert loaded.equals(df)

    clone_key = dm.clone(key, "clone", stage="clean")
    clone_df = dm.get(clone_key)
    assert clone_df.equals(df)

    dm.release(clone_key)
    with pytest.raises(KeyError):
        dm.get(clone_key)


def test_agent_state_forbids_dataframe_in_state() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError):
        AgentState(
            run_id="run_20260212_000000_abcd1234",
            df_key="df_raw_xxx",
            user_intent="test",
            profile_summary={"bad": df},
        )

