from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


class Observer:
    def __init__(self, storage_root: str | None = None) -> None:
        root = storage_root or os.getenv("AGENT_STORAGE_ROOT", "storage/runs")
        self.storage_root = Path(root)

    def _run_dir(self, run_id: str) -> Path:
        path = self.storage_root / run_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "logs").mkdir(parents=True, exist_ok=True)
        (path / "reporting").mkdir(parents=True, exist_ok=True)
        return path

    def emit_stream_event(self, run_id: str, event: Dict[str, Any]) -> None:
        run_dir = self._run_dir(run_id)
        payload = {"ts": _ts(), "run_id": run_id, **event}
        log_path = run_dir / "logs" / "stream.log"
        events_path = run_dir / "logs" / "events.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{payload['ts']}] {payload.get('event', 'event')} | {payload}\n")
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_node_start(self, run_id: str, node: str, df_key: str) -> None:
        self.emit_stream_event(
            run_id,
            {
                "event": "node_start",
                "node": node,
                "df_key": df_key,
                "message": f"start node={node}",
            },
        )

    def on_node_end(self, run_id: str, node: str, outputs: Dict[str, Any]) -> None:
        self.emit_stream_event(
            run_id,
            {
                "event": "node_end",
                "node": node,
                "outputs": outputs,
                "message": f"end node={node}",
            },
        )

    def emit_stage_card(self, run_id: str, card: Dict[str, Any]) -> None:
        run_dir = self._run_dir(run_id)
        cards_path = run_dir / "reporting" / "stage_cards.json"
        if cards_path.exists():
            existing = json.loads(cards_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        else:
            existing = []
        existing.append(card)
        cards_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        self.emit_stream_event(run_id, {"event": "stage_card_emitted", "card": card})
