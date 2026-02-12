from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.skills.reporting.artifacts import save_json, save_plot, upsert_artifact_index


def run_evaluation(
    run_id: str,
    task_type: str,
    baseline_metrics: Dict[str, Any],
    mixed_effect_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task_type": task_type,
        "baseline": baseline_metrics,
        "mixed_effect": mixed_effect_metrics or {},
    }

    best_section = "baseline"
    if mixed_effect_metrics and mixed_effect_metrics.get("comparison"):
        cmp = mixed_effect_metrics["comparison"]
        mixed_aic = cmp.get("mixed_aic")
        ols_aic = cmp.get("ols_aic")
        if mixed_aic is not None and ols_aic is not None and mixed_aic < ols_aic:
            best_section = "mixed_effect"

    payload["best_section"] = best_section
    eval_path = save_json(run_id, "evaluation", "evaluation.json", payload)
    upsert_artifact_index(run_id, "evaluation.evaluation", eval_path)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title("Evaluation Overview")
    ax.bar(["baseline", "mixed_effect"], [1, 1 if mixed_effect_metrics else 0])
    plot_path = save_plot(run_id, "evaluation", "evaluation_overview.png", fig)
    plt.close(fig)
    upsert_artifact_index(run_id, "evaluation.overview_plot", plot_path)

    stage_result = {
        "stage": "evaluation",
        "run_id": run_id,
        "df_key_in": "",
        "df_key_out": "",
        "actions": ["run_evaluation"],
        "metrics": {"best_section": best_section},
        "artifacts": [eval_path, plot_path],
    }
    stage_path = save_json(run_id, "evaluation", "evaluation_stage_result.json", stage_result)
    upsert_artifact_index(run_id, "evaluation.stage_result", stage_path)
    return {"stage_result": stage_result, "evaluation": payload}

