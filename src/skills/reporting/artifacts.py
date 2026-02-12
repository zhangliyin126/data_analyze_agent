from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _storage_root() -> Path:
    return Path(os.getenv("AGENT_STORAGE_ROOT", "storage/runs"))


def _run_stage_dir(run_id: str, stage: str) -> Path:
    path = _storage_root() / run_id / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def _reporting_dir(run_id: str) -> Path:
    path = _storage_root() / run_id / "reporting"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_str(path: Path) -> str:
    return str(path.as_posix())


def save_json(run_id: str, stage: str, name: str, payload: Dict[str, Any]) -> str:
    path = _run_stage_dir(run_id, stage) / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return _to_str(path)


def save_csv(run_id: str, stage: str, name: str, df: pd.DataFrame) -> str:
    path = _run_stage_dir(run_id, stage) / name
    df.to_csv(path, index=False)
    return _to_str(path)


def save_plot(run_id: str, stage: str, name: str, fig: Any) -> str:
    path = _run_stage_dir(run_id, stage) / name
    fig.savefig(path, bbox_inches="tight")
    return _to_str(path)


def upsert_artifact_index(run_id: str, key: str, path: str) -> None:
    index_path = _reporting_dir(run_id) / "artifact_index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    else:
        payload = {}
    payload[key] = path
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

