from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv


@dataclass
class LLMConfig:
    api_base: str
    model_name: str
    api_key: str


def load_env(env_path: str | None = None) -> None:
    if env_path:
        load_dotenv(env_path)
    else:
        default_path = Path(".env")
        if default_path.exists():
            load_dotenv(default_path)


def load_llm_config(env_path: str | None = None) -> LLMConfig:
    load_env(env_path)
    api_base = os.getenv("LLM_API_BASE", "").strip()
    model_name = os.getenv("LLM_NAME", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    missing = [k for k, v in {"LLM_API_BASE": api_base, "LLM_NAME": model_name, "LLM_API_KEY": api_key}.items() if not v]
    if missing:
        raise ValueError(f"Missing required env vars: {', '.join(missing)}")
    return LLMConfig(api_base=api_base, model_name=model_name, api_key=api_key)


def as_headers(cfg: LLMConfig) -> Dict[str, str]:
    return {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}

