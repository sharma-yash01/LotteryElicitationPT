"""Random lottery pairs; prior midpoint estimate on the final step."""

from __future__ import annotations

from typing import Any

import numpy as np


class RandomGuessBaseline:
    """Random valid lotteries; mirrors LotteryElicitationEnv/baselines/random_lottery.py."""

    def __init__(
        self,
        min_outcome_value: float | None = None,
        max_outcome_value: float | None = None,
    ):
        self.min_outcome_value = min_outcome_value
        self.max_outcome_value = max_outcome_value

    def _bounds(self, obs: dict) -> tuple[float, float]:
        lo = self.min_outcome_value
        hi = self.max_outcome_value
        if lo is None:
            lo = float(obs.get("min_outcome_value", -50.0))
        if hi is None:
            hi = float(obs.get("max_outcome_value", 100.0))
        return lo, hi

    def _sample_lottery(self, obs: dict, rng: np.random.Generator) -> dict[str, Any]:
        lo, hi = self._bounds(obs)
        p = float(rng.uniform(0.1, 0.9))
        v1 = float(rng.uniform(lo, hi))
        v2 = float(rng.uniform(lo, hi))
        return {
            "outcomes": [
                {"value": round(v1, 2), "probability": round(p, 4)},
                {"value": round(v2, 2), "probability": round(1.0 - p, 4)},
            ]
        }

    def select_action(self, obs: dict, rng: np.random.Generator) -> dict[str, Any]:
        lottery_a = self._sample_lottery(obs, rng)
        lottery_b = self._sample_lottery(obs, rng)
        out: dict[str, Any] = {"lottery_a": lottery_a, "lottery_b": lottery_b}
        if int(obs.get("steps_remaining", 99)) <= 1:
            gr = obs.get("gamma_range") or [0.2, 1.5]
            lr = obs.get("lambda_range") or [1.0, 4.5]
            out["theta_estimate"] = {
                "gamma": 0.5 * (float(gr[0]) + float(gr[1])),
                "lambda": 0.5 * (float(lr[0]) + float(lr[1])),
            }
        return out
