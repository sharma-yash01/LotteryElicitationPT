# LotteryElicitationPT

GRPO / TRL post-training for [LotteryElicitationEnv](../LotteryElicitationEnv): the policy steps a remote OpenEnv server over WebSocket, emits JSON lottery pairs, and receives a **sparse terminal** reward.

## Layout

- `training/` — `config`, `openenv_runtime` (`LotteryElicitationClient`), `prompts`, `action_parser`, `grpo_train` (`rollout_func`, `EpisodeSession`, `main`)
- `eval/` — baselines (`random`, `holt_laury`, `trained_hf`, `llm/*`), `evaluate.py`, `plots.py`
- `scripts/` — Lambda / CARC launchers, `analyze_reward_logs.py`

## Setup

```bash
cd LotteryElicitationPT
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # or requirements.lambda.txt on GPU images
```

## Train (local env example)

Point `ENV_BASE_URL` at your OpenEnv base URL (same host/port as the FastAPI app, no trailing path).

```bash
export ENV_BASE_URL=http://127.0.0.1:9000
python -m training.grpo_train \
  --model Qwen/Qwen3-0.6B \
  --env_base_url "$ENV_BASE_URL" \
  --curriculum_stage 1 \
  --max_completion_length 2048 \
  --output_dir runs/grpo_lottery
```

**Curriculum:** `--curriculum_stage` updates the **system prompt** and is sent on each `env.reset` as `curriculum_stage` so latent `theta` sampling matches the prompt. Deploy `EnvConfig.max_steps` / ranges consistent with that stage (see impl plan §14).

**Rewards:** `reward_from_env` uses the rollout `env_reward` field. If `--format_weight > 0` (default `0.1`), a second reward `reward_format` uses the rollout `format_score` list (valid JSON fraction per episode). Disable with `--no_format_reward`.

## Evaluate baselines

From this directory (`PYTHONPATH=.` is set implicitly if you use `python -m` after `cd` here):

```bash
python -m eval.evaluate \
  --env_base_url "$ENV_BASE_URL" \
  --baselines random,holt_laury \
  --n_episodes 200 \
  --seed 42 \
  --output eval_results.json
```

LLM baselines (`llm_api`, `llm_local`): `--include_llm` or `--baselines llm_api`; set `BASELINE_*` env vars as in `eval/baselines/llm/api_chat.py`.

`holt_laury_replay` imports `env.respondent` from sibling `LotteryElicitationEnv` (repo layout).

### Trained checkpoint (`trained_hf`)

Load a Hugging Face hub id or a local GRPO save directory (same layout as `trainer.save_model` output). Requires GPU for reasonable speed unless the model is tiny.

```bash
python -m eval.evaluate \
  --env_base_url "$ENV_BASE_URL" \
  --policy_model runs/grpo_lottery \
  --baselines random,holt_laury,trained_hf \
  --n_episodes 50 \
  --curriculum_stage 1 \
  --output eval_with_policy.json
```

With default empty `--baselines`, every registered agent runs, including `trained_hf` when `--policy_model` is set. To skip the policy, omit `--policy_model` or pass an explicit `--baselines` list that excludes `trained_hf`.

## Lambda / CARC

- `scripts/bootstrap_lambda.sh` — venv + `requirements.lambda.txt`
- `scripts/run_grpo_lambda.sh` — needs `LEPT_ROOT`, `LEPT_VENV`, `ENV_BASE_URL`
- `scripts/submit_grpo_carc.sh` + `run_grpo_carc.sbatch` — Slurm path

## Analyze reward logs

```bash
pip install -r requirements.analysis.txt
python scripts/analyze_reward_logs.py runs/grpo_lottery/reward_logs.jsonl --out-dir reward_analysis
```

## Risks (training)

Sparse terminal reward and JSON parse noise can slow learning; use stage 1 / shorter `max_steps` on the env, optional format reward, and parser fallbacks (see `training/action_parser.py`).
