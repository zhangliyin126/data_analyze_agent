from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List

from .config_loader import as_headers, load_llm_config


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response.")
        return json.loads(match.group(0))


class LocalLLMClient:
    def __init__(self, env_path: str | None = None, timeout: int = 60) -> None:
        self.cfg = load_llm_config(env_path)
        self.timeout = timeout

    def _chat(self, system_prompt: str, user_payload: Dict[str, Any]) -> str:
        body = {
            "model": self.cfg.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": 0.1,
        }
        endpoint = self.cfg.api_base.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=as_headers(self.cfg),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM HTTP error: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM connection error: {exc}") from exc

        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError(f"Invalid LLM response: {payload}")
        return str(choices[0]["message"]["content"])

    def _chat_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(3):
            try:
                text = self._chat(system_prompt, user_payload)
                return _extract_json(text)
            except Exception as exc:  # pragma: no cover - depends on upstream model quality
                last_error = exc
        raise RuntimeError(f"Failed to parse LLM JSON response after retries: {last_error}")

    def plan(
        self,
        profile: Dict[str, Any],
        schema: Dict[str, Any],
        candidates: Dict[str, Any],
        user_intent: str,
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are Planner for a data-analysis agent. "
            "Return strict JSON only with keys: "
            "task_type,data_structure,target_column,only_eda,use_mixed_effect,"
            "group_column,time_column,feature_strategy,modeling_strategy,evaluation_strategy,reasoning."
        )
        payload = {
            "profile": profile,
            "schema": schema,
            "candidates": candidates,
            "user_intent": user_intent,
        }
        return self._chat_json(system_prompt, payload)

    def interpret_stage(self, stage: str, stage_result: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = (
            "You are Interpreter for a data-analysis pipeline. "
            "Return strict JSON with: key_findings(list), risks(list), next_step(str)."
        )
        payload = {"stage": stage, "stage_result": stage_result}
        return self._chat_json(system_prompt, payload)

    def answer(self, question: str, artifact_summaries: List[Dict[str, Any]], final_summary: str) -> str:
        system_prompt = "You are Analyst. Answer based only on provided summaries. Be concise."
        payload = {"question": question, "artifact_summaries": artifact_summaries, "final_summary": final_summary}
        return self._chat(system_prompt, payload)

