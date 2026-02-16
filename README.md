# Data Analysis Agent MVP

## Quick Start

1. Create a virtual environment with Python 3.10+.
2. Install dependencies:
   - `pip install -e .[dev]`
3. Copy env template:
   - `copy .env.example .env` (Windows)
4. Run tests:
   - `pytest -q`
5. Run CLI:
   - `python -m src.app.agent_runner --data path/to/data.csv --intent "预测目标列 y"`
6. Run with the prepared test dataset:
   - `python -m src.app.agent_runner --data sample_data/test_dataset_full.csv --schema sample_data/test_column_schema.json --intent "预测目标列 y" --use-mock-llm`

## Project Structure

- `src/agent`: state/data manager/decision rules/llm_orchestration/graph/observer
- `src/skills`: profiling/eda/cleaning/feature engineering/modeling/reporting
- `src/app`: config loader, llm clients, runner, streamlit app
- `tests`: unit + smoke tests
- `storage/runs/{run_id}`: runtime artifacts

## Runtime Notes

- DataFrame is isolated in `DataManager`; state only carries `df_key` and summaries.
- LLM receives profile/schema/stage summaries only.
- Mixed Effect path is mandatory for hierarchical or repeated-measure data.

## Deletion Approval Rule (Highest Priority)

- Any deletion instruction requires explicit user approval before execution.
- This includes shell commands (`rm`, `del`, `Remove-Item`, `git clean`, `git reset --hard`) and any code path that performs file or directory deletion.
- If approval is not granted, use non-destructive alternatives such as versioned overwrite, archival, or rename.
