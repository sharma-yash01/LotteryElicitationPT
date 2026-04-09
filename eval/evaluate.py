"""Run baselines against remote LotteryElicitation OpenEnv; aggregate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.baselines import HoltLauryReplayBaseline, RandomGuessBaseline
from eval.baselines.llm.base import BaseLLMLotteryBaseline
from training.openenv_runtime import LotteryElicitationClient, to_openenv_base_url


def _parse_csv_names(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _select_action(
    agent: Any,
    obs: dict,
    rng: np.random.Generator,
    *,
    llm_max_new_tokens: int | None,
) -> dict:
    if isinstance(agent, RandomGuessBaseline):
        return agent.select_action(obs, rng)
    if isinstance(agent, BaseLLMLotteryBaseline):
        return agent.select_action(obs, max_new_tokens=llm_max_new_tokens)
    mod = getattr(agent.__class__, "__module__", "")
    if mod.endswith(".hf_policy") and agent.__class__.__name__ == "HFLotteryPolicy":
        return agent.select_action(obs, rng=rng, max_new_tokens=llm_max_new_tokens)
    if hasattr(agent, "select_action"):
        return agent.select_action(obs)
    raise TypeError(f"Unsupported agent: {type(agent)}")


def evaluate_agent(
    env,
    agent: Any,
    n_episodes: int,
    seed: int,
    *,
    rng: np.random.Generator | None = None,
    llm_max_new_tokens: int | None = None,
    curriculum_stage: int | None = None,
) -> list[dict]:
    rng = rng or np.random.default_rng(seed)
    results: list[dict] = []
    for ep in range(n_episodes):
        reset_kw: dict = {"seed": seed + ep}
        if curriculum_stage is not None:
            reset_kw["curriculum_stage"] = int(curriculum_stage)
        result = env.reset(**reset_kw)
        obs = result.observation
        steps = 0
        while not obs.get("done", False):
            action = _select_action(
                agent,
                obs,
                rng,
                llm_max_new_tokens=llm_max_new_tokens,
            )
            result = env.step(action)
            obs = result.observation
            steps += 1
        state = env.state()
        results.append({
            "gamma_mse": state.get("gamma_mse"),
            "lambda_mse": state.get("lambda_mse"),
            "holt_laury_accuracy": state.get("holt_laury_prediction_accuracy"),
            "total_reward": state.get("total_reward"),
            "steps_taken": steps,
            "estimated_gamma": state.get("estimated_gamma"),
            "estimated_lambda": state.get("estimated_lambda"),
            "true_gamma": state.get("true_gamma"),
            "true_lambda": state.get("true_lambda"),
        })
    return results


def summarize(results: list[dict]) -> dict[str, float]:
    def col(key: str) -> list[float]:
        return [float(x) for x in (r.get(key) for r in results) if x is not None]

    g = col("gamma_mse")
    l = col("lambda_mse")
    h = col("holt_laury_accuracy")
    rw = [float(r["total_reward"]) for r in results if r.get("total_reward") is not None]
    st = [float(r["steps_taken"]) for r in results]

    def stats(xs: list[float]) -> tuple[float, float]:
        if not xs:
            return float("nan"), float("nan")
        return float(np.mean(xs)), float(np.std(xs))

    gm, gs = stats(g)
    lm, ls = stats(l)
    hm, hs = stats(h)
    rm, rs = stats(rw)
    am, astd = stats(st)
    return {
        "gamma_mse_mean": gm,
        "gamma_mse_std": gs,
        "lambda_mse_mean": lm,
        "lambda_mse_std": ls,
        "hl_accuracy_mean": hm,
        "hl_accuracy_std": hs,
        "reward_mean": rm,
        "reward_std": rs,
        "avg_steps": am,
        "avg_steps_std": astd,
    }


def _build_baselines(args: argparse.Namespace) -> dict[str, Any]:
    baselines: dict[str, Any] = {
        "random": RandomGuessBaseline(),
        "holt_laury": HoltLauryReplayBaseline(),
    }

    requested = _parse_csv_names(args.baselines)
    needs_llm = args.include_llm or any(name.startswith("llm_") for name in requested)
    if needs_llm:
        from eval.baselines.llm.api_chat import APIChatBaseline
        from eval.baselines.llm.local_vllm import LocalVLLMBaseline

        baselines["llm_api"] = APIChatBaseline(
            curriculum_stage=args.curriculum_stage,
            timeout_s=args.llm_timeout_s,
            max_retries=args.llm_max_retries,
            temperature=args.llm_temperature,
        )
        baselines["llm_local"] = LocalVLLMBaseline(
            curriculum_stage=args.curriculum_stage,
            timeout_s=args.llm_timeout_s,
            max_retries=args.llm_max_retries,
            temperature=args.llm_temperature,
        )

    if args.policy_model:
        from eval.baselines.hf_policy import HFLotteryPolicy

        baselines["trained_hf"] = HFLotteryPolicy(
            args.policy_model,
            curriculum_stage=args.curriculum_stage,
            max_new_tokens=args.llm_max_new_tokens,
        )

    if not requested:
        return baselines

    if "trained_hf" in requested and not args.policy_model:
        raise ValueError("Baseline trained_hf requires --policy_model (HF hub id or local checkpoint path).")

    unknown = [name for name in requested if name not in baselines]
    if unknown:
        available = ", ".join(sorted(baselines.keys()))
        raise ValueError(f"Unknown baselines: {unknown}. Available: {available}")

    return {name: baselines[name] for name in requested}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lottery elicitation baselines")
    parser.add_argument("--env_base_url", type=str, default=None)
    parser.add_argument("--space_url", type=str, default=None)
    parser.add_argument(
        "--baselines",
        type=str,
        default="",
        help="Comma-separated: random, holt_laury, trained_hf (needs --policy_model), llm_api, llm_local.",
    )
    parser.add_argument(
        "--policy_model",
        type=str,
        default=None,
        help="HF model id or local path for trained_hf baseline (GRPO save dir).",
    )
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--include_llm", action="store_true")
    parser.add_argument("--llm_max_new_tokens", type=int, default=512)
    parser.add_argument("--llm_timeout_s", type=float, default=30.0)
    parser.add_argument("--llm_max_retries", type=int, default=2)
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument(
        "--curriculum_stage",
        type=int,
        default=1,
        help="Passed to env.reset and LLM / trained_hf prompts; align with deployed EnvConfig.",
    )
    args = parser.parse_args()

    env_base_url = to_openenv_base_url(
        env_base_url=args.env_base_url,
        space_url=args.space_url,
    )

    selected = _build_baselines(args)

    all_results: dict[str, list[dict]] = {}
    with LotteryElicitationClient(base_url=env_base_url).sync() as env:
        for name, agent in selected.items():
            rng = np.random.default_rng(args.seed)
            all_results[name] = evaluate_agent(
                env,
                agent,
                args.n_episodes,
                args.seed,
                rng=rng,
                llm_max_new_tokens=args.llm_max_new_tokens,
                curriculum_stage=args.curriculum_stage,
            )

    summary = {name: summarize(runs) for name, runs in all_results.items()}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw": all_results}, f, indent=1)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
