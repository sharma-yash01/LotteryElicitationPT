"""Configuration for GRPO + LotteryElicitation OpenEnv training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingRuntimeConfig:
    """Runtime controls for training/grpo_train.py (rollout_func path)."""

    alpha: float = 1.0
    format_weight: float = 0.1
    max_tokens_per_step: int = 512
    curriculum_stage: int = 1

    log_rewards: bool = True
    log_every_n_steps: int = 1
    reward_log_path: str = ""

    def resolved_reward_log_path(self, output_dir: str) -> str:
        if self.reward_log_path:
            return self.reward_log_path
        return str(Path(output_dir) / "reward_logs.jsonl")
