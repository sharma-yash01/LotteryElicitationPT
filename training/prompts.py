"""System prompt and observation formatting for lottery elicitation."""

from __future__ import annotations

LAMBDA_FIXED_STAGE1 = 2.25


def build_system_prompt(
    *,
    curriculum_stage: int,
    gamma_lo: float,
    gamma_hi: float,
    lambda_lo: float,
    lambda_hi: float,
    min_val: float,
    max_val: float,
) -> str:
    """Build system message; stage 1 focuses on gamma (lambda fixed in env)."""
    if curriculum_stage <= 1:
        return f"""You are an economic researcher eliciting a person's risk preferences.

On each turn, you present a pair of lotteries and observe which one the respondent prefers.
Your goal: infer their risk sensitivity gamma (γ ∈ [{gamma_lo}, {gamma_hi}]).
The respondent's loss aversion λ is fixed at {LAMBDA_FIXED_STAGE1} for this session (within [{lambda_lo}, {lambda_hi}]).

Each turn, respond with ONLY a JSON object defining the two lotteries so the respondent can choose between them:
{{
  "lottery_a": {{"outcomes": [{{"value": <float>, "probability": <float>}}, ...]}},
  "lottery_b": {{"outcomes": [{{"value": <float>, "probability": <float>}}, ...]}}
}}

Each lottery has 2-3 outcomes. Probabilities must sum to 1.0.
Values must be in [{min_val}, {max_val}].

When you are confident in your estimate (or on your final turn), also include:
  "theta_estimate": {{"gamma": <float>, "lambda": <float>}}

To stop early and claim an efficiency bonus, add:
  "terminate_early": true
(requires theta_estimate)

Design lottery pairs where the respondent's choice maximally distinguishes \
between different parameter values. Use prior choices to narrow your estimate."""

    return f"""You are an economic researcher eliciting a person's risk and loss aversion preferences.

On each turn, you present a pair of lotteries and observe which one the respondent prefers.
Your goal: infer their risk sensitivity gamma (γ ∈ [{gamma_lo}, {gamma_hi}]) \
and loss aversion lambda (λ ∈ [{lambda_lo}, {lambda_hi}]).

Each turn, respond with ONLY a JSON object:
{{
  "lottery_a": {{"outcomes": [{{"value": <float>, "probability": <float>}}, ...]}},
  "lottery_b": {{"outcomes": [{{"value": <float>, "probability": <float>}}, ...]}}
}}

Each lottery has 2-3 outcomes. Probabilities must sum to 1.0.
Values must be in [{min_val}, {max_val}].

When you are confident in your estimate (or on your final turn), also include:
  "theta_estimate": {{"gamma": <float>, "lambda": <float>}}

To stop early and claim an efficiency bonus, add:
  "terminate_early": true
(requires theta_estimate)

Design lottery pairs where the respondent's choice maximally distinguishes \
between different parameter values. Use prior choices to narrow your estimate."""


def _format_lottery_compact(lottery_dict: dict) -> str:
    parts = []
    for o in lottery_dict.get("outcomes", []):
        parts.append(f"{float(o['probability']) * 100:.0f}%: ${float(o['value']):.1f}")
    return "[" + ", ".join(parts) + "]"


def format_observation_prompt(obs: dict, *, is_first_turn: bool = False) -> str:
    del is_first_turn
    lines = [
        f"Step: {obs['step_idx']} / {obs['max_steps']}",
        f"Steps remaining: {obs['steps_remaining']}",
    ]

    history = obs.get("history") or []
    if history:
        lines.append("\nChoice history:")
        for i, h in enumerate(history, 1):
            lottery_a = _format_lottery_compact(h["lottery_a"])
            lottery_b = _format_lottery_compact(h["lottery_b"])
            choice = h["choice"]
            lines.append(f"  Turn {i}: A={lottery_a} vs B={lottery_b} → chose {choice}")
    else:
        lines.append("\nNo choices observed yet.")

    sr = int(obs.get("steps_remaining", 0))
    if sr == 1:
        lines.append("\nThis is your FINAL turn. You MUST include theta_estimate.")
    elif sr <= 3:
        lines.append(f"\n{sr} turns left. Consider submitting theta_estimate soon.")

    lines.append("\nDesign the next lottery pair (JSON only):")
    return "\n".join(lines)


def system_prompt_from_observation(obs: dict, *, curriculum_stage: int) -> str:
    """Build system prompt using ranges and bounds from the first observation dict."""
    gr = _as_tuple(obs.get("gamma_range"), (0.2, 1.5))
    lr = _as_tuple(obs.get("lambda_range"), (1.0, 4.5))
    min_v = float(obs.get("min_outcome_value", -50.0))
    max_v = float(obs.get("max_outcome_value", 100.0))
    return build_system_prompt(
        curriculum_stage=curriculum_stage,
        gamma_lo=gr[0],
        gamma_hi=gr[1],
        lambda_lo=lr[0],
        lambda_hi=lr[1],
        min_val=min_v,
        max_val=max_v,
    )


def _as_tuple(val, default: tuple[float, float]) -> tuple[float, float]:
    if val is None:
        return default
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return (float(val[0]), float(val[1]))
    return default
