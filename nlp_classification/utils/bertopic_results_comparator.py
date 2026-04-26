"""Utilities to compare BERTopic run metrics across saved model outputs."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt


METRICS_TO_PLOT = [
    "coherence_c_v",
    "coherence_u_mass",
    "coherence_c_npmi",
    "silhouette_score",
    "topic_diversity",
    "outliers_ratio",
    "outliers_count",
    "topics_count",
]

LOWER_IS_BETTER = {"outliers_ratio", "outliers_count"}


def _load_run_metrics(metrics_path: pathlib.Path) -> dict[str, Any]:
    with metrics_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    model_id = payload.get("model_id") or metrics_path.parents[0].name
    row = {"model_id": model_id, "metrics_path": str(metrics_path)}
    for metric in METRICS_TO_PLOT:
        row[metric] = payload.get(metric)
    return row


def _plot_metric(df: pd.DataFrame, metric: str, output_dir: pathlib.Path) -> pathlib.Path:
    metric_df = df.dropna(subset=[metric]).copy()
    if metric_df.empty:
        raise ValueError(f"No data available for metric '{metric}'.")

    ascending = metric in LOWER_IS_BETTER
    metric_df = metric_df.sort_values(metric, ascending=ascending)

    labels = metric_df["model_id"].tolist()
    values = metric_df[metric].tolist()

    # Scale figure height by the number of runs for readability.
    fig_height = max(4.5, min(18.0, 0.45 * len(labels)))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    bars = ax.barh(labels, values)
    ax.set_title(f"BERTopic comparison: {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("model_id")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:.6f}" if isinstance(value, float) else f" {value}",
            va="center",
            fontsize=8,
        )

    output_path = output_dir / f"compare_{metric}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _build_markdown_table(df: pd.DataFrame) -> str:
    """Create a markdown table without external dependencies."""
    if df.empty:
        return "| model_id |\n| --- |\n| (no runs found) |"

    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in df.itertuples(index=False):
        row_values = []
        for value in row:
            if isinstance(value, float):
                row_values.append(f"{value:.6f}")
            elif pd.isna(value):
                row_values.append("")
            else:
                row_values.append(str(value))
        lines.append("| " + " | ".join(row_values) + " |")

    return "\n".join(lines)


def generate_metrics_comparison_graphs(
    base_path: str, output_subdir: str = "metrics_comparison"
) -> dict[str, Any]:
    """Traverse saved runs and generate cross-run metric comparison graphs.

    Parameters
    ----------
    base_path
        Base output path containing BERTopic run folders (typically the same
        ``base_path/folder_name`` used in ``BerTopicModelBuilder``).
    output_subdir
        Subfolder under ``base_path`` where comparison assets are saved.

    Returns
    -------
    dict
        Summary containing discovered run count and generated plot paths.
    """
    base = pathlib.Path(base_path)
    metrics_files = sorted(
        base.rglob("runs/*/evaluation_metrics.json")
    )
    if not metrics_files:
        raise FileNotFoundError(
            f"No run metrics files found under '{base}'. "
            "Expected: runs/*/evaluation_metrics.json"
        )

    rows = [_load_run_metrics(path) for path in metrics_files]
    df = pd.DataFrame(rows).drop_duplicates(subset=["model_id"], keep="last")

    output_dir = base / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_plots = {}
    for metric in METRICS_TO_PLOT:
        try:
            plot_path = _plot_metric(df, metric, output_dir)
            generated_plots[metric] = str(plot_path)
        except ValueError:
            # Skip metrics that are not available across runs.
            continue

    summary_path = output_dir / "comparison_summary.csv"
    df.to_csv(summary_path, index=False)

    sort_metric = "coherence_c_v" if "coherence_c_v" in df.columns else "model_id"
    table_df = df.sort_values(sort_metric, ascending=False).copy()
    table_df = table_df[
        ["model_id", "coherence_c_v", "coherence_c_npmi", "silhouette_score", "topic_diversity", "outliers_ratio", "outliers_count", "topics_count"]
    ]

    table_html_path = output_dir / "comparison_table.html"
    table_df.to_html(table_html_path, index=False)

    table_md_path = output_dir / "comparison_table.md"
    table_md_path.write_text(_build_markdown_table(table_df), encoding="utf-8")

    return {
        "base_path": str(base),
        "runs_found": int(df.shape[0]),
        "summary_csv": str(summary_path),
        "table_html": str(table_html_path),
        "table_markdown": str(table_md_path),
        "plots": generated_plots,
    }
