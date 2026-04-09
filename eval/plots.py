"""Plots for lottery elicitation eval JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def agent_gamma_mse_bars(eval_json_path: str, output_path: str | None = None) -> None:
    with open(eval_json_path, encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary", data)
    agents = list(summary.keys())
    means = [summary[a].get("gamma_mse_mean", 0.0) for a in agents]
    stds = [summary[a].get("gamma_mse_std", 0.0) for a in agents]
    _, ax = plt.subplots()
    x = np.arange(len(agents))
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=25, ha="right")
    ax.set_ylabel("Gamma MSE (mean ± std)")
    ax.set_title("Gamma MSE by agent")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def agent_reward_bars(eval_json_path: str, output_path: str | None = None) -> None:
    with open(eval_json_path, encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary", data)
    agents = list(summary.keys())
    means = [summary[a].get("reward_mean", 0.0) for a in agents]
    stds = [summary[a].get("reward_std", 0.0) for a in agents]
    _, ax = plt.subplots()
    x = np.arange(len(agents))
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=25, ha="right")
    ax.set_ylabel("Episode reward")
    ax.set_title("Mean reward by agent")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def rolling_mean_from_raw(
    eval_json_path: str,
    metric_key: str,
    window: int = 20,
    output_path: str | None = None,
) -> None:
    """Rolling mean of per-episode metric from raw results (first agent if multiple)."""
    with open(eval_json_path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("raw", {})
    if not raw:
        return
    agent = next(iter(raw))
    episodes = raw[agent]
    ys = [float(e[metric_key]) for e in episodes if e.get(metric_key) is not None]
    if len(ys) < window:
        return
    arr = np.array(ys)
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    roll = (cumsum[window:] - cumsum[:-window]) / window
    _, ax = plt.subplots()
    ax.plot(np.arange(window, window + len(roll)), roll)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Rolling mean {metric_key} (w={window})")
    ax.set_title(f"{agent}: {metric_key}")
    plt.tight_layout()
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p)
        plt.close()
    else:
        plt.show()
