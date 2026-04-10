"""GRPO training with TRL rollout_func and remote LotteryElicitation OpenEnv."""

from __future__ import annotations

import argparse
import copy
import json
import threading
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import torch.distributed as dist

from training.action_parser import parse_llm_output, _strip_think_blocks
from training.config import TrainingRuntimeConfig
from training.openenv_runtime import LotteryElicitationClient, to_openenv_base_url
from training.prompts import format_observation_prompt, system_prompt_from_observation

if TYPE_CHECKING:
    from trl import GRPOTrainer

def _merge_chat_template_kwargs_for_reasoning_mode(
    base: dict,
    *,
    reasoning_mode: str,
) -> dict:
    """Override enable_thinking when reasoning_mode is on/off; auto leaves base unchanged."""
    mode = (reasoning_mode or "auto").strip().lower()
    out = dict(base)
    if mode == "on":
        out["enable_thinking"] = True
    elif mode == "off":
        out["enable_thinking"] = False
    elif mode != "auto":
        raise ValueError(f"reasoning_mode must be auto, on, or off (got {reasoning_mode!r})")
    return out


ENV_BASE_URL: str = ""
RUNTIME_CFG: TrainingRuntimeConfig | None = None
REWARD_LOG_PATH: str = ""
EPISODE_LOG_COUNT: int = 0
LOG_LOCK = threading.Lock()

# Align with LotteryElicitationEnv.env.config.EnvConfig.max_steps (remote hard cap).
LOTTERY_ENV_MAX_STEPS = 10


def _write_episode_log(entry: dict) -> None:
    global EPISODE_LOG_COUNT
    if RUNTIME_CFG is None or not RUNTIME_CFG.log_rewards or not REWARD_LOG_PATH:
        return
    with LOG_LOCK:
        EPISODE_LOG_COUNT += 1
        every_n = max(1, int(RUNTIME_CFG.log_every_n_steps))
        if EPISODE_LOG_COUNT % every_n != 0:
            return
        log_path = Path(REWARD_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")


class EpisodeSession:
    """One remote episode: reset + step with parsed action dict."""

    def __init__(self, base_url: str):
        self.client = LotteryElicitationClient(base_url=base_url)
        self.reward = 0.0
        self.done = False
        self._obs: dict | None = None
        self.episode_id = ""
        self.step_logs: list[dict] = []
        self._env: Any = None
        self._conn_cm: Any = None

    def __enter__(self) -> EpisodeSession:
        sync_maker = getattr(self.client, "sync", None)
        if callable(sync_maker):
            self._conn_cm = sync_maker()
            self._env = self._conn_cm.__enter__()
        else:
            self._conn_cm = None
            self.client.__enter__()
            self._env = self.client
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        try:
            if self._conn_cm is not None:
                return self._conn_cm.__exit__(exc_type, exc_val, exc_tb)
            return self.client.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._env = None
            self._conn_cm = None

    def reset_episode(
        self,
        seed: int | None = None,
        curriculum_stage: int | None = None,
    ) -> dict:
        if self._env is None:
            raise RuntimeError("EpisodeSession must be used as a context manager.")
        self.reward = 0.0
        self.done = False
        self.episode_id = uuid4().hex
        self.step_logs = []
        cs = curriculum_stage
        if cs is None and RUNTIME_CFG is not None:
            cs = RUNTIME_CFG.curriculum_stage
        reset_kw: dict[str, Any] = {}
        if seed is not None:
            reset_kw["seed"] = seed
        if cs is not None:
            reset_kw["curriculum_stage"] = int(cs)
        result = self._env.reset(**reset_kw)
        self._obs = result.observation
        return self._obs

    def apply_action(self, action_dict: dict, *, raw_llm_text: str, was_valid: bool) -> None:
        if self._env is None:
            raise RuntimeError("EpisodeSession must be used as a context manager.")
        if self.done:
            raise ValueError("Episode is over.")
        result = self._env.step(action_dict)
        self._obs = result.observation
        raw_step_reward = float(result.reward or 0.0)
        step_reward = raw_step_reward * (RUNTIME_CFG.alpha if RUNTIME_CFG else 1.0)
        self.reward += step_reward
        self.done = bool(result.done)

        self.step_logs.append({
            "step_index": len(self.step_logs) + 1,
            "raw_llm_text": raw_llm_text,
            "action": action_dict,
            "was_valid_parse": was_valid,
            "choice": (self._obs or {}).get("last_choice"),
            "step_reward": step_reward,
            "raw_env_reward": raw_step_reward,
            "done_after_step": self.done,
        })

        if self.done:
            _write_episode_log({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "episode_id": self.episode_id,
                "episode_reward": self.reward,
                "num_steps": len(self.step_logs),
                "steps": self.step_logs,
                "final_observation": self._obs,
            })


def _tokenize_messages(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    chat_template: str | None,
    chat_template_kwargs: dict,
    tools,
    add_generation_prompt: bool,
) -> list[int]:
    try:
        tokenized = tokenizer.apply_chat_template(
            conversation=[messages],
            tools=tools,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            padding=True,
            **chat_template_kwargs,
        )
    except TypeError:
        # Tokenizer does not support enable_thinking (non-Qwen model); retry without it.
        kwargs_no_thinking = {k: v for k, v in chat_template_kwargs.items() if k != "enable_thinking"}
        tokenized = tokenizer.apply_chat_template(
            conversation=[messages],
            tools=tools,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            padding=True,
            **kwargs_no_thinking,
        )
    row_ids = tokenized["input_ids"][0]
    row_mask = tokenized["attention_mask"][0]
    return [int(t) for t, m in zip(row_ids, row_mask, strict=True) if m]


def _squeeze_vllm_logprobs(logprobs) -> list[float] | None:
    if logprobs is None:
        return None
    out: list[float] = []
    for seq in logprobs:
        for lp in seq:
            out.append(float(lp[0]) if lp and lp[0] is not None else 0.0)
    return out


@contextmanager
def _temporary_vllm_max_tokens(trainer: "GRPOTrainer", max_tokens: int):
    vg = trainer.vllm_generation
    prev = vg.max_completion_length
    vg.max_completion_length = max_tokens
    try:
        yield
    finally:
        vg.max_completion_length = prev


def _per_step_cap(trainer: "GRPOTrainer") -> int:
    global_max = int(trainer.args.max_completion_length)
    if RUNTIME_CFG is None:
        return max(1, global_max)
    return max(1, min(int(RUNTIME_CFG.max_tokens_per_step), global_max))


def _needs_vllm_server_generate_padding(trainer: "GRPOTrainer") -> bool:
    vg = getattr(trainer, "vllm_generation", None)
    if vg is None or getattr(vg, "mode", None) != "server":
        return False
    if not dist.is_available() or not dist.is_initialized():
        return False
    return dist.get_world_size() > 1


def _pad_vllm_server_generates_to_target(
    trainer: "GRPOTrainer", *, before_ids: list[int], num_dummy: int
) -> None:
    """Cheap generate() calls for NCCL lockstep; outputs discarded."""
    for _ in range(max(0, num_dummy)):
        with _temporary_vllm_max_tokens(trainer, 1):
            trainer.vllm_generation.generate(
                prompts=[before_ids],
                images=None,
                num_generations=1,
            )


def _rollout_one_episode(
    seed_messages: list,
    trainer: "GRPOTrainer",
    *,
    tok,
    chat_template,
    chat_template_kwargs: dict,
    tools,
    max_episode_turns: int,
    episode_seed: int | None,
) -> tuple[list[int], list[int], list[float], list[int], float, float]:
    messages = copy.deepcopy(seed_messages)
    rng = np.random.default_rng(episode_seed)

    with EpisodeSession(ENV_BASE_URL) as session:
        obs = session.reset_episode(seed=episode_seed)
        if not isinstance(messages[-1].get("content"), str):
            raise TypeError(
                "rollout_func expects last message content to be a string for observation append."
            )

        if messages and messages[0].get("role") == "system":
            cs = RUNTIME_CFG.curriculum_stage if RUNTIME_CFG else 1
            messages[0]["content"] = system_prompt_from_observation(obs, curriculum_stage=cs)

        messages[-1]["content"] = messages[-1]["content"] + "\n\n" + format_observation_prompt(
            obs, is_first_turn=True
        )

        prompt_ids_fixed = _tokenize_messages(
            tok,
            messages,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
            tools=tools,
            add_generation_prompt=True,
        )

        per_episode_generate_target = min(max_episode_turns, LOTTERY_ENV_MAX_STEPS)
        last_before_ids = prompt_ids_fixed

        completion_ids: list[int] = []
        env_mask: list[int] = []
        logprob_seq: list[float] = []
        format_ok_count = 0
        turns = 0

        while not session.done and turns < per_episode_generate_target:
            turns += 1
            step_cap = _per_step_cap(trainer)
            before_ids = _tokenize_messages(
                tok,
                messages,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
                tools=tools,
                add_generation_prompt=True,
            )
            last_before_ids = before_ids

            with _temporary_vllm_max_tokens(trainer, step_cap):
                _, gen_ids_batch, logprobs_raw, _ = trainer.vllm_generation.generate(
                    prompts=[before_ids],
                    images=None,
                    num_generations=1,
                )
            gen_ids = gen_ids_batch[0]
            gen_lp = _squeeze_vllm_logprobs(logprobs_raw)
            if gen_lp is None or len(gen_lp) != len(gen_ids):
                gen_lp = [0.0] * len(gen_ids)

            text = tok.decode(gen_ids, skip_special_tokens=True)

            # Strip think tokens from training tensor (Fix 9 step 2).
            # Think blocks inflate completion_ids but carry no useful gradient
            # signal — re-encode only the JSON-relevant text.
            stripped_text = _strip_think_blocks(text)
            stripped_ids = tok.encode(stripped_text, add_special_tokens=False)
            stripped_lp = [0.0] * len(stripped_ids)

            completion_ids.extend(stripped_ids)
            env_mask.extend([1] * len(stripped_ids))
            logprob_seq.extend(stripped_lp)

            messages.append({"role": "assistant", "content": text})

            action_dict, was_valid = parse_llm_output(text, session._obs or {}, rng=rng)
            format_ok_count += int(was_valid)
            session.apply_action(action_dict, raw_llm_text=text, was_valid=was_valid)

            if session.done:
                break

            after_asst_ids = _tokenize_messages(
                tok,
                messages,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
                tools=tools,
                add_generation_prompt=True,
            )
            messages.append(
                {"role": "user", "content": format_observation_prompt(session._obs or {})}
            )
            after_user_ids = _tokenize_messages(
                tok,
                messages,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
                tools=tools,
                add_generation_prompt=True,
            )
            suffix = after_user_ids[len(after_asst_ids) :]
            completion_ids.extend(suffix)
            env_mask.extend([0] * len(suffix))
            logprob_seq.extend([0.0] * len(suffix))

        if _needs_vllm_server_generate_padding(trainer):
            n_pad = max(0, per_episode_generate_target - turns)
            _pad_vllm_server_generates_to_target(
                trainer, before_ids=last_before_ids, num_dummy=n_pad
            )

        # Hard-cap total completion length to max_completion_length so TRL
        # training tensors stay within GPU memory budget (OOM fix #1).
        max_total = int(trainer.args.max_completion_length)
        if len(completion_ids) > max_total:
            completion_ids = completion_ids[:max_total]
            env_mask = env_mask[:max_total]
            logprob_seq = logprob_seq[:max_total]

        format_score = format_ok_count / max(1, turns)
        return (
            prompt_ids_fixed,
            completion_ids,
            logprob_seq,
            env_mask,
            float(session.reward),
            float(format_score),
        )


def build_rollout_func(
    max_episode_turns: int = 20,
    *,
    env_seed: int = 0,
):
    def rollout_func(prompts: list, trainer: "GRPOTrainer") -> dict[str, Any]:
        tok = trainer.processing_class
        chat_template = getattr(trainer, "chat_template", None)
        chat_kwargs = getattr(trainer, "chat_template_kwargs", None) or {}
        tools = getattr(trainer, "tools", None) or None

        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        all_env_mask: list[list[int]] = []
        all_env_reward: list[float] = []
        all_format_score: list[float] = []

        for i, seed_messages in enumerate(prompts):
            ep_seed = env_seed + i
            p, c, lp, m, r, fs = _rollout_one_episode(
                seed_messages,
                trainer,
                tok=tok,
                chat_template=chat_template,
                chat_template_kwargs=chat_kwargs,
                tools=tools,
                max_episode_turns=max_episode_turns,
                episode_seed=ep_seed,
            )
            all_prompt_ids.append(p)
            all_completion_ids.append(c)
            all_logprobs.append(lp)
            all_env_mask.append(m)
            all_env_reward.append(r)
            all_format_score.append(fs)

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_mask": all_env_mask,
            "env_reward": all_env_reward,
            "format_score": all_format_score,
        }

    return rollout_func


def reward_from_env(prompts, completions, completion_ids, **kwargs):
    env_reward = kwargs.get("env_reward")
    if env_reward is not None:
        return [float(r) for r in env_reward]
    return [0.0] * len(prompts)


def reward_format(prompts, completions, completion_ids, **kwargs):
    """Auxiliary JSON format reward in [-0.1, 0.1] scaled by format_weight (via caller config)."""
    format_scores = kwargs.get("format_score")
    if format_scores is None:
        return [0.0] * len(prompts)
    w = RUNTIME_CFG.format_weight if RUNTIME_CFG else 0.0
    return [w * (0.2 * float(s) - 0.1) for s in format_scores]


def main():
    global ENV_BASE_URL, RUNTIME_CFG, REWARD_LOG_PATH

    parser = argparse.ArgumentParser(description="GRPO lottery elicitation training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--max_tokens_per_step", type=int, default=512)
    parser.add_argument("--max_episode_turns", type=int, default=5)
    parser.add_argument("--curriculum_stage", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--format_weight", type=float, default=0.1)
    parser.add_argument("--no_format_reward", action="store_true", help="Disable auxiliary format reward.")
    parser.add_argument("--env_seed", type=int, default=0, help="Base seed offset for env.reset per prompt.")
    parser.add_argument("--no_log_rewards", action="store_true")
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--reward_log_path", type=str, default="")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--vllm_mode", type=str, default="colocate", choices=["colocate", "server"])
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--vllm_server_port", type=int, default=8001)
    parser.add_argument("--vllm_group_port", type=int, default=51216)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--output_dir", type=str, default="runs/grpo_train")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--env_base_url", type=str, default=None)
    parser.add_argument("--space_url", type=str, default=None)
    parser.add_argument(
        "--reasoning_mode",
        type=str,
        default="off",
        choices=["auto", "on", "off"],
        help="Chat-template enable_thinking: off disables Qwen3 think blocks (default); on forces them; auto leaves base unchanged.",
    )
    args = parser.parse_args()

    if args.per_device_train_batch_size % args.num_generations != 0:
        raise SystemExit(
            "per_device_train_batch_size must be divisible by num_generations "
            f"({args.per_device_train_batch_size} % {args.num_generations} != 0)."
        )

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    ENV_BASE_URL = to_openenv_base_url(
        env_base_url=args.env_base_url,
        space_url=args.space_url,
    )
    RUNTIME_CFG = TrainingRuntimeConfig(
        alpha=args.alpha,
        format_weight=0.0 if args.no_format_reward else args.format_weight,
        max_tokens_per_step=args.max_tokens_per_step,
        curriculum_stage=args.curriculum_stage,
        log_rewards=not args.no_log_rewards,
        log_every_n_steps=args.log_every_n_steps,
        reward_log_path=args.reward_log_path,
    )
    REWARD_LOG_PATH = RUNTIME_CFG.resolved_reward_log_path(args.output_dir)
    if RUNTIME_CFG.log_rewards:
        print(
            f"Reward episode logs: {REWARD_LOG_PATH} "
            f"(every {RUNTIME_CFG.log_every_n_steps} episodes)"
        )

    placeholder_system = (
        "You are an economic researcher eliciting preferences via lottery choices. "
        "Respond with JSON only as instructed."
    )
    dataset = Dataset.from_dict({
        "prompt": [
            [
                {"role": "system", "content": placeholder_system},
                {"role": "user", "content": "Begin elicitation."},
            ]
        ]
        * 100
    })

    merged_chat_template_kwargs = _merge_chat_template_kwargs_for_reasoning_mode(
        {},
        reasoning_mode=args.reasoning_mode,
    )
    print(f"Reasoning mode: {args.reasoning_mode!r}  chat_template_kwargs={merged_chat_template_kwargs!r}")

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=True,
        chat_template_kwargs=merged_chat_template_kwargs,
        vllm_mode=args.vllm_mode,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_group_port=args.vllm_group_port,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=not args.no_bf16,
        logging_steps=1,
        save_strategy="epoch",
    )

    reward_funcs: list = [reward_from_env]
    if RUNTIME_CFG.format_weight > 0:
        reward_funcs.append(reward_format)

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
        rollout_func=build_rollout_func(
            max_episode_turns=args.max_episode_turns,
            env_seed=args.env_seed,
        ),
        args=grpo_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*attention mask.*", category=UserWarning)
    main()
