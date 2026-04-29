from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def generate_twitter_card(
    metrics: dict[str, float],
    upcoming: pd.DataFrame,
    title_odds: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(14, 8), facecolor="#fff8ef")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.2], width_ratios=[1, 1])

    ax0 = fig.add_subplot(grid[0, 0])
    ax0.axis("off")
    ax0.text(0.0, 0.95, "IPL Prediction Research", fontsize=28, fontweight="bold", color="#6a1b09")
    ax0.text(0.0, 0.72, f"Strict Pre-match Accuracy: {_format_pct(metrics['accuracy'])}", fontsize=18, color="#1b4332")
    ax0.text(0.0, 0.56, f"Brier Score: {metrics['brier_score']:.3f}", fontsize=18, color="#1b4332")
    ax0.text(0.0, 0.40, f"Log Loss: {metrics['log_loss']:.3f}", fontsize=18, color="#1b4332")
    ax0.text(0.0, 0.24, f"ROC-AUC: {metrics['roc_auc']:.3f}", fontsize=18, color="#1b4332")
    ax0.text(0.0, 0.06, "Research prototype; use with uncertainty bands and validation context", fontsize=12, color="#6c757d")

    ax1 = fig.add_subplot(grid[0, 1])
    ax1.axis("off")
    ax1.text(0.0, 0.95, "Next Research Calls", fontsize=24, fontweight="bold", color="#0b3954")
    for row_idx, row in enumerate(upcoming.head(4).itertuples(index=False), start=1):
        y = 0.95 - row_idx * 0.2
        prob = max(row.predicted_prob_team_a, row.predicted_prob_team_b)
        ax1.text(
            0.0,
            y,
            f"{row.date.strftime('%d %b')}  {row.team_a} vs {row.team_b}",
            fontsize=15,
            color="#102542",
        )
        ax1.text(
            0.0,
            y - 0.08,
            f"Lean: {row.predicted_winner} ({_format_pct(prob)}) at {row.venue}",
            fontsize=13,
            color="#bc3908",
        )

    ax2 = fig.add_subplot(grid[1, :])
    odds = title_odds.head(5)
    ax2.set_title("Simulation Output", fontsize=22, fontweight="bold", color="#102542")
    ax2.set_xlabel("Champion Probability (model scenario)")
    if odds.empty:
        ax2.axis("off")
        ax2.text(0.0, 0.5, "No simulation output available", fontsize=16, color="#6c757d")
    else:
        ax2.barh(odds["team"][::-1], odds["champion_probability"][::-1], color="#ff7f11")
        ax2.set_xlim(0, max(0.05, float(odds["champion_probability"].max()) * 1.15))
        for idx, value in enumerate(odds["champion_probability"][::-1]):
            ax2.text(float(value) + 0.003, idx, _format_pct(float(value)), va="center", fontsize=12)

    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_thread(
    metrics: dict[str, float],
    upcoming: pd.DataFrame,
    title_odds: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    top_match = upcoming.iloc[0] if not upcoming.empty else None
    lines = [
        "I am building an IPL match prediction research project.",
        "",
        "Current strict pre-match validation:",
        f"- 2025 holdout accuracy: {_format_pct(metrics['accuracy'])}",
        f"Brier score: {metrics['brier_score']:.3f}",
        f"ROC-AUC: {metrics['roc_auc']:.3f}",
        f"ECE: {metrics['expected_calibration_error']:.3f}",
        "",
        "Current model uses only pre-match-safe numeric features:",
        "- team form before the match",
        "- venue priors and home advantage",
        "- team momentum and NRR trend",
        "- head-to-head history",
        "- Elo-style strength",
        "",
        "Confirmed-XI player features are now separated into a toss-confirmed feature set, because using known XIs for historical rows is not a fair match-eve setup.",
        "",
    ]
    if top_match is not None:
        top_prob = max(top_match.predicted_prob_team_a, top_match.predicted_prob_team_b)
        lines.extend(
            [
                f"Next research call: {top_match.team_a} vs {top_match.team_b}",
                f"Model lean: {top_match.predicted_winner} at {_format_pct(top_prob)}",
                f"Venue: {top_match.venue} | Date: {top_match.date.strftime('%d %b %Y')}",
                "",
            ]
        )
    lines.extend(
        [
            "Research artifacts generated:",
            "- validation predictions",
            "- feature availability audit",
            "- walk-forward benchmark report",
            "- data quality summary",
            "- upcoming fixture probabilities with caveats",
        ]
    )
    destination.write_text("\n".join(lines) + "\n")


def write_summary_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=str))
