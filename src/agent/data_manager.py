from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class DataEntry:
    source_name: str
    df: pd.DataFrame


class DataManager:
    def __init__(self) -> None:
        self._store: Dict[str, DataEntry] = {}

    @staticmethod
    def _new_key(stage: str) -> str:
        return f"df_{stage}_{uuid.uuid4().hex[:8]}"

    def register(self, df: pd.DataFrame, source_name: str, stage: str = "raw") -> str:
        df_key = self._new_key(stage)
        self._store[df_key] = DataEntry(source_name=source_name, df=df.copy())
        return df_key

    def get(self, df_key: str) -> pd.DataFrame:
        if df_key not in self._store:
            raise KeyError(f"Unknown df_key: {df_key}")
        return self._store[df_key].df

    def clone(self, df_key: str, new_name: str, stage: str = "clone") -> str:
        source_df = self.get(df_key)
        return self.register(source_df.copy(), source_name=new_name, stage=stage)

    def put(self, df_key: str, df: pd.DataFrame) -> None:
        if df_key not in self._store:
            raise KeyError(f"Unknown df_key: {df_key}")
        entry = self._store[df_key]
        self._store[df_key] = DataEntry(source_name=entry.source_name, df=df.copy())

    def release(self, df_key: str) -> None:
        if df_key not in self._store:
            raise KeyError(f"Unknown df_key: {df_key}")
        self._store.pop(df_key)

    def exists(self, df_key: str) -> bool:
        return df_key in self._store

