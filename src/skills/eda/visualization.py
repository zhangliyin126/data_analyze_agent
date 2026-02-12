from __future__ import annotations

from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.agent.data_manager import DataManager
from src.skills.reporting.artifacts import save_json, save_plot, upsert_artifact_index


def run_visualization(data_manager: DataManager, df_key: str, run_id: str) -> Dict[str, Any]:
    df = data_manager.get(df_key)
    artifacts = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        show_cols = numeric_cols[:4]
        fig, axes = plt.subplots(1, len(show_cols), figsize=(4 * len(show_cols), 3))
        if len(show_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, show_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(col)
        dist_path = save_plot(run_id, "profiling", "distribution_overview.png", fig)
        plt.close(fig)
        upsert_artifact_index(run_id, "profiling.distribution_overview", dist_path)
        artifacts.append(dist_path)

        corr = df[numeric_cols].corr(numeric_only=True)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, cmap="viridis", ax=ax2)
        ax2.set_title("Correlation Heatmap")
        heatmap_path = save_plot(run_id, "profiling", "correlation_heatmap.png", fig2)
        plt.close(fig2)
        upsert_artifact_index(run_id, "profiling.correlation_heatmap", heatmap_path)
        artifacts.append(heatmap_path)

    stage_result = {
        "stage": "profiling",
        "run_id": run_id,
        "df_key_in": df_key,
        "df_key_out": df_key,
        "actions": ["run_visualization"],
        "metrics": {"numeric_columns": len(numeric_cols)},
        "artifacts": artifacts,
    }
    stage_path = save_json(run_id, "profiling", "visualization_stage_result.json", stage_result)
    upsert_artifact_index(run_id, "profiling.visualization_stage_result", stage_path)
    return stage_result

