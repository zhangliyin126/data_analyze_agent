from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .agent_runner import run_agent_from_df

try:
    import streamlit as st
except Exception:  # pragma: no cover - optional runtime dependency
    st = None


def _load_schema(uploaded) -> Dict[str, Any]:
    if uploaded is None:
        return {}
    return json.loads(uploaded.read().decode("utf-8"))


def render_app() -> None:
    if st is None:
        raise RuntimeError("streamlit is not installed.")

    st.title("Data Analysis Agent MVP")
    uploaded_data = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
    uploaded_schema = st.file_uploader("Upload column_schema.json", type=["json"])
    intent = st.text_input("Intent", value="请分析数据并给出建模结论")
    use_mock = st.checkbox("Use Mock LLM", value=True)

    if st.button("Run"):
        if uploaded_data is None:
            st.error("Please upload a data file.")
            return
        suffix = Path(uploaded_data.name).suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(uploaded_data)
        else:
            df = pd.read_excel(uploaded_data)
        schema = _load_schema(uploaded_schema)
        state = run_agent_from_df(df=df, intent=intent, schema=schema, use_mock_llm=use_mock)

        st.subheader("Final Summary")
        st.text(state.final_summary or "")
        st.subheader("Stage Cards")
        for card in state.stage_cards:
            st.json(card.model_dump())
        st.subheader("Artifacts")
        st.json(state.artifact_index)

        question = st.text_input("Ask follow-up question", value="哪个模型表现最好？")
        if st.button("Answer"):
            st.info("Use local_llm or mock_llm answer interface in next iteration.")


if __name__ == "__main__":  # pragma: no cover
    render_app()

