"""Fixed Holt-Laury pairs + grid-search final estimate (dict observations)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from eval.baselines.holt_laury_pairs import HOLT_LAURY_PAIR_DICTS


def _le_env_root() -> Path:
    return Path(__file__).resolve().parents[3] / "LotteryElicitationEnv"


def _ensure_le_env_on_path() -> None:
    root = _le_env_root()
    if root.is_dir():
        s = str(root)
        if s not in sys.path:
            sys.path.insert(0, s)


def _lottery_from_dict(payload: dict):
    _ensure_le_env_on_path()
    from env.models import Lottery, LotteryOutcome

    outcomes = payload.get("outcomes", [])
    return Lottery(
        outcomes=[
            LotteryOutcome(value=float(o["value"]), probability=float(o["probability"]))
            for o in outcomes
        ]
    )


def _respondent_choice(lottery_a, lottery_b, *, gamma: float, lambda_: float) -> str:
    _ensure_le_env_on_path()
    from env.respondent import respondent_choice

    return respondent_choice(
        lottery_a=lottery_a,
        lottery_b=lottery_b,
        gamma=float(gamma),
        lambda_=float(lambda_),
        noise_std=0.0,
    )


class HoltLauryReplayBaseline:
    """Presents standard H-L pairs; grid-search θ on the final step."""

    def __init__(
        self,
        gamma_range: tuple[float, float] = (0.2, 1.5),
        lambda_range: tuple[float, float] = (1.0, 4.5),
        grid_step: float = 0.01,
    ):
        self.gamma_range = gamma_range
        self.lambda_range = lambda_range
        self.grid_step = grid_step

    def select_action(self, obs: dict) -> dict:
        idx = int(obs.get("step_idx", 0))
        pair_idx = min(idx, len(HOLT_LAURY_PAIR_DICTS) - 1)
        lottery_a, lottery_b = HOLT_LAURY_PAIR_DICTS[pair_idx]

        theta_estimate = None
        if int(obs.get("steps_remaining", 0)) <= 1:
            theta_estimate = self._fit_from_choices(obs.get("history") or [])

        out: dict = {"lottery_a": lottery_a, "lottery_b": lottery_b}
        if theta_estimate is not None:
            out["theta_estimate"] = theta_estimate
        return out

    def _fit_from_choices(self, history: list[dict]) -> dict[str, float]:
        if not history:
            return {
                "gamma": 0.5 * (self.gamma_range[0] + self.gamma_range[1]),
                "lambda": 0.5 * (self.lambda_range[0] + self.lambda_range[1]),
            }

        gamma_grid = np.arange(self.gamma_range[0], self.gamma_range[1] + 1e-9, self.grid_step)
        lambda_grid = np.arange(self.lambda_range[0], self.lambda_range[1] + 1e-9, self.grid_step)

        gamma_mid = 0.5 * (self.gamma_range[0] + self.gamma_range[1])
        lambda_mid = 0.5 * (self.lambda_range[0] + self.lambda_range[1])

        best_score = -1
        best_dist = float("inf")
        best_theta = {"gamma": float(gamma_mid), "lambda": float(lambda_mid)}

        for gamma in gamma_grid:
            for lambda_ in lambda_grid:
                score = 0
                for row in history:
                    la = _lottery_from_dict(row["lottery_a"])
                    lb = _lottery_from_dict(row["lottery_b"])
                    predicted = _respondent_choice(la, lb, gamma=float(gamma), lambda_=float(lambda_))
                    score += int(predicted == row["choice"])

                dist = (float(gamma) - gamma_mid) ** 2 + (float(lambda_) - lambda_mid) ** 2
                if score > best_score or (score == best_score and dist < best_dist):
                    best_score = score
                    best_dist = dist
                    best_theta = {"gamma": float(gamma), "lambda": float(lambda_)}

        return best_theta
