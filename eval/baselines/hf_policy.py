"""Local Hugging Face causal LM as a lottery elicitation policy (eval)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.action_parser import parse_llm_output
from training.prompts import format_observation_prompt, system_prompt_from_observation


class HFLotteryPolicy:
    """Loads a saved GRPO/HF checkpoint and generates JSON actions like training."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        curriculum_stage: int = 1,
        max_new_tokens: int = 512,
    ):
        self.curriculum_stage = curriculum_stage
        self.default_max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kw: dict[str, Any] = {"trust_remote_code": True}
        if torch.cuda.is_available():
            load_kw["device_map"] = "auto"
            bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            load_kw["torch_dtype"] = torch.bfloat16 if bf16_ok else torch.float16
        else:
            load_kw["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kw)
        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()
        self._device = next(self.model.parameters()).device

    def select_action(
        self,
        obs: dict,
        *,
        rng: np.random.Generator | None = None,
        max_new_tokens: int | None = None,
    ) -> dict:
        rng = rng or np.random.default_rng(0)
        sys_c = system_prompt_from_observation(obs, curriculum_stage=self.curriculum_stage)
        user_c = format_observation_prompt(obs)
        messages = [
            {"role": "system", "content": sys_c},
            {"role": "user", "content": user_c},
        ]
        cap = int(max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens)
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        input_ids = input_ids.to(self._device)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        with torch.inference_mode():
            out = self.model.generate(
                input_ids,
                max_new_tokens=cap,
                do_sample=False,
                pad_token_id=pad_id,
            )
        gen_ids = out[0, input_ids.shape[1] :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        action, _ = parse_llm_output(text, obs, rng=rng)
        return action
