"""Holt-Laury lottery pairs as plain dicts (matches LotteryElicitationEnv/env/holt_laury.py)."""

from __future__ import annotations


def _hl_pair(p_high: float) -> tuple[dict, dict]:
    p_low = 1.0 - p_high
    safe = {
        "outcomes": [
            {"value": 2.00, "probability": p_high},
            {"value": 1.60, "probability": p_low},
        ]
    }
    risky = {
        "outcomes": [
            {"value": 3.85, "probability": p_high},
            {"value": 0.10, "probability": p_low},
        ]
    }
    return safe, risky


HOLT_LAURY_PAIR_DICTS: list[tuple[dict, dict]] = [
    _hl_pair(0.1),
    _hl_pair(0.2),
    _hl_pair(0.3),
    _hl_pair(0.4),
    _hl_pair(0.5),
    _hl_pair(0.6),
    _hl_pair(0.7),
    _hl_pair(0.8),
    _hl_pair(0.9),
    _hl_pair(1.0),
]
