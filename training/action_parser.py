"""Parse LLM JSON output into env step actions."""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

_PROB_SUM_TOL = 1e-3


def _safe_int(x: Any, default: int = 1) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _safe_float(x: Any) -> float | None:
    """Coerce to float; reject nan/inf and failed coercion."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _normalize_lottery_probabilities(lot: dict[str, Any]) -> dict[str, Any]:
    """Renormalize outcome probabilities to sum to exactly 1.0 (server Pydantic uses 1e-6)."""
    outcomes = lot["outcomes"]
    probs: list[float] = []
    vals: list[float] = []
    for o in outcomes:
        pv = _safe_float(o.get("probability"))
        vv = _safe_float(o.get("value"))
        if pv is None or vv is None:
            raise ValueError("non-finite lottery outcome")
        probs.append(pv)
        vals.append(vv)
    total = sum(probs)
    if total <= 0.0:
        return {"outcomes": [{"value": vals[i], "probability": probs[i]} for i in range(len(outcomes))]}
    n = len(outcomes)
    new_outcomes: list[dict[str, Any]] = []
    acc = 0.0
    for i in range(n):
        if i == n - 1:
            p = 1.0 - acc
        else:
            p = probs[i] / total
            acc += p
        new_outcomes.append({"value": vals[i], "probability": p})
    return {"outcomes": new_outcomes}


def _normalize_ranges(obs: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    def pair(key: str, default: tuple[float, float]) -> tuple[float, float]:
        v = obs.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            a = _safe_float(v[0])
            b = _safe_float(v[1])
            if a is not None and b is not None:
                return (a, b)
        return default

    return pair("gamma_range", (0.2, 1.5)), pair("lambda_range", (1.0, 4.5))


def _value_bounds(obs: dict) -> tuple[float, float]:
    lo = _safe_float(obs.get("min_outcome_value", -50.0))
    hi = _safe_float(obs.get("max_outcome_value", 100.0))
    return (
        lo if lo is not None else -50.0,
        hi if hi is not None else 100.0,
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
    if _safe_int(obs.get("steps_remaining"), 1) <= 1:
        gamma_range, lambda_range = _normalize_ranges(obs)
        action["theta_estimate"] = {
            "gamma": (gamma_range[0] + gamma_range[1]) / 2,
            "lambda": (lambda_range[0] + lambda_range[1]) / 2,
        }
    return action


def fallback_action(obs: dict, rng: np.random.Generator | None = None) -> dict[str, Any]:
    """Public alias for rollout crash recovery (same as internal random fallback)."""
    return _random_fallback_action(obs, rng)


def _strip_think_blocks(text: str) -> str:
    """Remove Qwen3-style <think>...</think> blocks (and redacted variants) before JSON extraction."""
    # Qwen3 hybrid thinking delimiters
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL)
    # Unclosed <think> (model hit token limit before closing) — strip to end of string
    text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.DOTALL)
    # Log-style redaction wrappers
    text = re.sub(r"<redacted_thinking>[\s\S]*?</redacted_thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<redacted_thinking>[\s\S]*$", "", text, flags=re.DOTALL)
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
        val = _safe_float(o.get("value"))
        pr = _safe_float(o.get("probability"))
        if val is None or pr is None:
            return False
        if val < v_lo or val > v_hi:
            return False
        if pr < 0.0 or pr > 1.0:
            return False
        s += pr
    return abs(s - 1.0) <= _PROB_SUM_TOL


def _theta_from_payload(te: dict[str, Any], obs: dict) -> dict[str, float] | None:
    """Parse theta_estimate if both keys present and in range; else None (caller keeps lotteries-only)."""
    if "gamma" not in te or "lambda" not in te:
        return None
    g = _safe_float(te.get("gamma"))
    lam = _safe_float(te.get("lambda"))
    if g is None or lam is None:
        return None
    (g_lo, g_hi), (l_lo, l_hi) = _normalize_ranges(obs)
    if not (g_lo <= g <= g_hi and l_lo <= lam <= l_hi):
        return None
    return {"gamma": g, "lambda": lam}


def _finalize_action(action: dict[str, Any], obs: dict) -> dict[str, Any]:
    if _safe_int(obs.get("steps_remaining"), 1) <= 1 and "theta_estimate" not in action:
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


def _set_parse_failure_reason(holder: list[str] | None, msg: str) -> None:
    if holder is not None:
        holder.clear()
        holder.append(msg)


def parse_llm_output(
    text: str,
    obs: dict,
    *,
    rng: np.random.Generator | None = None,
    failure_reason: list[str] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Parse LLM text → action dict for env.step(); second value is parse validity.

    If *failure_reason* is a list, on failure it is replaced with a single human-readable reason
    (empty list on success).
    """
    if failure_reason is not None:
        failure_reason.clear()

    stripped = _strip_think_blocks(text)
    stripped = _strip_json_fences(stripped)
    blob = _extract_first_json_object(stripped)
    if blob is None:
        _set_parse_failure_reason(failure_reason, "no_json_object")
        return _random_fallback_action(obs, rng), False
    try:
        data = _loads_lenient(blob)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        _set_parse_failure_reason(failure_reason, f"json_decode:{type(e).__name__}")
        return _random_fallback_action(obs, rng), False
    if not isinstance(data, dict):
        _set_parse_failure_reason(failure_reason, "payload_not_object")
        return _random_fallback_action(obs, rng), False

    v_lo, v_hi = _value_bounds(obs)
    la = data.get("lottery_a")
    lb = data.get("lottery_b")
    ok_a, ok_b = _valid_lottery(la, v_lo=v_lo, v_hi=v_hi), _valid_lottery(lb, v_lo=v_lo, v_hi=v_hi)
    if not ok_a or not ok_b:
        which = []
        if not ok_a:
            which.append("lottery_a")
        if not ok_b:
            which.append("lottery_b")
        _set_parse_failure_reason(failure_reason, f"invalid_lottery:{','.join(which)}")
        return _random_fallback_action(obs, rng), False

    try:
        la_n = _normalize_lottery_probabilities(la)
        lb_n = _normalize_lottery_probabilities(lb)
    except Exception as e:
        _set_parse_failure_reason(failure_reason, f"normalize_lottery:{type(e).__name__}")
        return _random_fallback_action(obs, rng), False

    action: dict[str, Any] = {
        "lottery_a": la_n,
        "lottery_b": lb_n,
    }
    if "theta_estimate" in data and isinstance(data["theta_estimate"], dict):
        te = data["theta_estimate"]
        parsed_te = _theta_from_payload(te, obs)
        if parsed_te is not None:
            action["theta_estimate"] = parsed_te
        elif "gamma" in te or "lambda" in te:
            # Partial or non-numeric theta must not crash; strip and let _finalize inject on last turn.
            pass
    if data.get("terminate_early") is True:
        action["terminate_early"] = True

    try:
        action = _finalize_action(action, obs)
    except Exception as e:
        _set_parse_failure_reason(failure_reason, f"finalize_action:{type(e).__name__}")
        return _random_fallback_action(obs, rng), False

    return action, True
