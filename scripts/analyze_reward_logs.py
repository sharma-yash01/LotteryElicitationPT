#!/usr/bin/env python3
"""Analyze lottery GRPO ``reward_logs.jsonl`` (episode JSON from training/grpo_train.py).

Usage:
  pip install -r requirements.analysis.txt
  python scripts/analyze_reward_logs.py path/to/reward_logs.jsonl --out-dir ./reward_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skip line {lineno}: {e}", file=sys.stderr)
    return rows


def episodes_to_dataframe(episodes: list[dict]) -> pd.DataFrame:
    records = []
    for i, ep in enumerate(episodes):
        steps = ep.get("steps") or []
        fo = ep.get("final_observation") or {}
        meta = fo.get("metadata") or {}
        rb = meta.get("reward_breakdown") or {}
        valid = sum(1 for s in steps if s.get("was_valid_parse"))
        nst = len(steps)
        records.append({
            "episode_idx": i,
            "episode_id": ep.get("episode_id"),
            "timestamp_utc": ep.get("timestamp_utc"),
            "episode_reward": float(ep.get("episode_reward", 0.0)),
            "num_steps": int(ep.get("num_steps", 0)),
            "parse_valid_rate": (valid / nst) if nst else 0.0,
            "final_gamma_mse": rb.get("gamma_mse"),
            "final_lambda_mse": rb.get("lambda_mse"),
            "final_hl_acc": rb.get("holt_laury_accuracy"),
            "final_obs_reward": fo.get("reward"),
        })
    return pd.DataFrame(records)


def steps_to_dataframe(episodes: list[dict]) -> pd.DataFrame:
    rows = []
    for i, ep in enumerate(episodes):
        eid = ep.get("episode_id")
        for s in ep.get("steps") or []:
            rows.append({
                "episode_idx": i,
                "episode_id": eid,
                "step_index": s.get("step_index"),
                "raw_env_reward": float(s.get("raw_env_reward", 0.0) or 0.0),
                "step_reward": float(s.get("step_reward", 0.0) or 0.0),
                "was_valid_parse": bool(s.get("was_valid_parse", False)),
                "done_after_step": bool(s.get("done_after_step", False)),
                "choice": s.get("choice"),
            })
    return pd.DataFrame(rows)


def print_summary(ep_df: pd.DataFrame, st_df: pd.DataFrame) -> None:
    n_ep = len(ep_df)
    print("=== Lottery reward log summary ===")
    print(f"Episodes: {n_ep}  |  Step rows: {len(st_df)}")
    if n_ep == 0:
        return

    er = ep_df["episode_reward"]
    print("\n--- Episode reward ---")
    print(er.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())
    print(f"Fraction episode_reward <= -2.0: {(er <= -2.0).mean():.3f}")
    print(f"Fraction episode_reward == 0:     {(er == 0).mean():.3f}")

    if "parse_valid_rate" in ep_df.columns:
        print("\n--- Parse validity (per episode) ---")
        print(ep_df["parse_valid_rate"].describe().to_string())

    if len(st_df):
        print("\n--- Per-step was_valid_parse ---")
        print(f"Fraction valid: {st_df['was_valid_parse'].mean():.3f}")
        fr = 1.0 - st_df.groupby("episode_idx")["was_valid_parse"].mean()
        print("\n--- Fallback rate per episode (invalid-parse fraction) ---")
        print(fr.describe().to_string())


def _roll_window(n: int) -> int:
    return max(1, min(25, n // 20)) if n else 1


def _plot_rolling_numeric_series(
    ep_df: pd.DataFrame,
    col: str,
    title: str,
    path: Path,
    window: int,
) -> None:
    y = pd.to_numeric(ep_df[col], errors="coerce")
    if y.notna().sum() < 1:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    roll = y.rolling(window=window, min_periods=1).mean()
    ax.plot(ep_df["episode_idx"], y, alpha=0.25, label=col)
    ax.plot(ep_df["episode_idx"], roll, color="darkred", label=f"rolling mean w={window}")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all(ep_df: pd.DataFrame, st_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(ep_df["episode_reward"], kde=True, ax=ax, bins=min(40, max(10, len(ep_df) // 5)))
    ax.set_title("Episode reward distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "01_episode_reward_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(ep_df["num_steps"], discrete=True, ax=ax)
    ax.set_title("Steps per episode")
    fig.tight_layout()
    fig.savefig(out_dir / "02_num_steps_hist.png", dpi=150)
    plt.close(fig)

    if len(st_df):
        fig, ax = plt.subplots(figsize=(6, 4))
        st_df["was_valid_parse"].astype(float).value_counts(normalize=True).sort_index().plot(
            kind="bar", ax=ax, color=["coral", "steelblue"]
        )
        ax.set_title("Parse validity (all steps)")
        ax.set_xticklabels(["invalid", "valid"], rotation=0)
        fig.tight_layout()
        fig.savefig(out_dir / "03_parse_validity.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    window = max(1, min(25, len(ep_df) // 20))
    roll = ep_df["episode_reward"].rolling(window=window, min_periods=1).mean()
    ax.plot(ep_df["episode_idx"], ep_df["episode_reward"], alpha=0.25, label="episode_reward")
    ax.plot(ep_df["episode_idx"], roll, color="darkred", label=f"rolling mean w={window}")
    ax.legend()
    ax.set_title("Episode reward over log order")
    fig.tight_layout()
    fig.savefig(out_dir / "04_reward_trajectory.png", dpi=150)
    plt.close(fig)

    window = _roll_window(len(ep_df))
    _plot_rolling_numeric_series(
        ep_df,
        "final_gamma_mse",
        "Final gamma MSE (from reward_breakdown)",
        out_dir / "05_gamma_mse_trajectory.png",
        window,
    )
    _plot_rolling_numeric_series(
        ep_df,
        "final_lambda_mse",
        "Final lambda MSE (from reward_breakdown)",
        out_dir / "06_lambda_mse_trajectory.png",
        window,
    )
    _plot_rolling_numeric_series(
        ep_df,
        "final_hl_acc",
        "Holt–Laury accuracy (final)",
        out_dir / "07_hl_accuracy_trajectory.png",
        window,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ep_df["num_steps"], ep_df["episode_reward"], alpha=0.35, s=12)
    ax.set_xlabel("Steps per episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Reward vs steps (efficiency vs return)")
    fig.tight_layout()
    fig.savefig(out_dir / "08_reward_vs_steps_scatter.png", dpi=150)
    plt.close(fig)

    if len(st_df):
        fr_series = 1.0 - st_df.groupby("episode_idx")["was_valid_parse"].mean()
        ep_plot = ep_df.copy()
        ep_plot["fallback_rate"] = ep_plot["episode_idx"].map(fr_series)
        if ep_plot["fallback_rate"].notna().any():
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(
                ep_plot["episode_idx"],
                ep_plot["fallback_rate"],
                alpha=0.45,
                label="fallback_rate",
            )
            roll_fb = ep_plot["fallback_rate"].rolling(window=window, min_periods=1).mean()
            ax.plot(ep_plot["episode_idx"], roll_fb, color="darkred", label=f"rolling mean w={window}")
            ax.set_ylim(-0.05, 1.05)
            ax.legend()
            ax.set_title("Invalid-parse fraction per episode (parser fallback load)")
            fig.tight_layout()
            fig.savefig(out_dir / "09_fallback_rate_trajectory.png", dpi=150)
            plt.close(fig)

    print(f"\nFigures written to: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze lottery reward_logs.jsonl")
    parser.add_argument("jsonl_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("reward_log_analysis"))
    args = parser.parse_args()

    if not args.jsonl_path.is_file():
        print(f"Error: file not found: {args.jsonl_path}", file=sys.stderr)
        sys.exit(1)

    episodes = load_jsonl(args.jsonl_path)
    if not episodes:
        print("No valid rows; exiting.", file=sys.stderr)
        sys.exit(2)

    ep_df = episodes_to_dataframe(episodes)
    st_df = steps_to_dataframe(episodes)
    print_summary(ep_df, st_df)
    plot_all(ep_df, st_df, args.out_dir)


if __name__ == "__main__":
    main()
