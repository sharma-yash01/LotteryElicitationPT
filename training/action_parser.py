"""Parse LLM JSON output into env step actions."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

_PROB_SUM_TOL = 1e-3


def _normalize_lottery_probabilities(lot: dict[str, Any]) -> dict[str, Any]:
    """Renormalize outcome probabilities to sum to exactly 1.0 (server Pydantic uses 1e-6)."""
    outcomes = lot["outcomes"]
    total = sum(float(o["probability"]) for o in outcomes)
    if total <= 0.0:
        return {
            "outcomes": [
                {"value": float(o["value"]), "probability": float(o["probability"])}
                for o in outcomes
            ]
        }
    n = len(outcomes)
    new_outcomes: list[dict[str, Any]] = []
    acc = 0.0
    for i, o in enumerate(outcomes):
        if i == n - 1:
            p = 1.0 - acc
        else:
            p = float(o["probability"]) / total
            acc += p
        new_outcomes.append({"value": float(o["value"]), "probability": p})
    return {"outcomes": new_outcomes}


def _normalize_ranges(obs: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    def pair(key: str, default: tuple[float, float]) -> tuple[float, float]:
        v = obs.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return (float(v[0]), float(v[1]))
        return default

    return pair("gamma_range", (0.2, 1.5)), pair("lambda_range", (1.0, 4.5))


def _value_bounds(obs: dict) -> tuple[float, float]:
    return (
        float(obs.get("min_outcome_value", -50.0)),
        float(obs.get("max_outcome_value", 100.0)),
    )


def _random_fallback_action(obs: dict, rng: np.random.Generator | None) -> dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng()
    lo, hi = _value_bounds(obs)

    def _random_lottery() -> dict[str, Any]:
        p = float(rng.uniform(0.1, 0.9))
        v1 = float(rng.uniform(lo, hi))
        v2 = float(rng.uniform(lo, hi))
        return _normalize_lottery_probabilities({
            "outcomes": [
                {"value": round(v1, 2), "probability": round(p, 4)},
                {"value": round(v2, 2), "probability": round(1.0 - p, 4)},
            ]
        })

    action: dict[str, Any] = {
        "lottery_a": _random_lottery(),
        "lottery_b": _random_lottery(),
    }
    if int(obs.get("steps_remaining", 1)) <= 1:
        gamma_range, lambda_range = _normalize_ranges(obs)
        action["theta_estimate"] = {
            "gamma": (gamma_range[0] + gamma_range[1]) / 2,
            "lambda": (lambda_range[0] + lambda_range[1]) / 2,
        }
    return action


def _strip_think_blocks(text: str) -> str:
    """Remove Qwen3-style <think>...</think> blocks (and redacted variants) before JSON extraction."""
    # Qwen3 hybrid thinking delimiters
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL)
    # Log-style redaction wrappers
    text = re.sub(r"<redacted_thinking>[\s\S]*?</redacted_thinking>", "", text, flags=re.DOTALL)
    return text


def _strip_json_fences(text: str) -> str:
    s = text.strip()
    if "```" in s:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return s


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _loads_lenient(blob: str) -> Any:
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", blob)
        return json.loads(fixed)


def _valid_lottery(
    lot: Any,
    *,
    v_lo: float,
    v_hi: float,
) -> bool:
    if not isinstance(lot, dict):
        return False
    outcomes = lot.get("outcomes")
    if not isinstance(outcomes, list):
        return False
    n = len(outcomes)
    if n < 2 or n > 3:
        return False
    s = 0.0
    for o in outcomes:
        if not isinstance(o, dict):
            return False
        if "value" not in o or "probability" not in o:
            return False
        val = float(o["value"])
        pr = float(o["probability"])
        if val < v_lo or val > v_hi:
            return False
        if pr < 0.0 or pr > 1.0:
            return False
        s += pr
    return abs(s - 1.0) <= _PROB_SUM_TOL


def _finalize_action(action: dict[str, Any], obs: dict) -> dict[str, Any]:
    if int(obs.get("steps_remaining", 1)) <= 1 and "theta_estimate" not in action:
        gamma_range, lambda_range = _normalize_ranges(obs)
        action = dict(action)
        action["theta_estimate"] = {
            "gamma": (gamma_range[0] + gamma_range[1]) / 2,
            "lambda": (lambda_range[0] + lambda_range[1]) / 2,
        }
    if action.get("terminate_early") and "theta_estimate" not in action:
        action = dict(action)
        action["terminate_early"] = False
    return action


def parse_llm_output(
    text: str,
    obs: dict,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, Any], bool]:
    """Parse LLM text → action dict for env.step(); second value is parse validity."""
    stripped = _strip_think_blocks(text)
    stripped = _strip_json_fences(stripped)
    blob = _extract_first_json_object(stripped)
    if blob is None:
        return _random_fallback_action(obs, rng), False
    try:
        data = _loads_lenient(blob)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _random_fallback_action(obs, rng), False
    if not isinstance(data, dict):
        return _random_fallback_action(obs, rng), False

    v_lo, v_hi = _value_bounds(obs)
    la = data.get("lottery_a")
    lb = data.get("lottery_b")
    if not _valid_lottery(la, v_lo=v_lo, v_hi=v_hi) or not _valid_lottery(lb, v_lo=v_lo, v_hi=v_hi):
        return _random_fallback_action(obs, rng), False

    la_n = _normalize_lottery_probabilities(la)
    lb_n = _normalize_lottery_probabilities(lb)

    action: dict[str, Any] = {
        "lottery_a": la_n,
        "lottery_b": lb_n,
    }
    if "theta_estimate" in data and isinstance(data["theta_estimate"], dict):
        te = data["theta_estimate"]
        if "gamma" in te and "lambda" in te:
            action["theta_estimate"] = {
                "gamma": float(te["gamma"]),
                "lambda": float(te["lambda"]),
            }
    if data.get("terminate_early") is True:
        action["terminate_early"] = True

    action = _finalize_action(action, obs)
    return action, True
