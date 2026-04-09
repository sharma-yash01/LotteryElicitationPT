"""Shared scaffolding for LLM lottery baselines."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from training.action_parser import parse_llm_output
from training.prompts import format_observation_prompt, system_prompt_from_observation


class BaseLLMLotteryBaseline(ABC):
    def __init__(
        self,
        *,
        curriculum_stage: int = 1,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.0,
    ):
        self.curriculum_stage = curriculum_stage
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.temperature = temperature

    @staticmethod
    def get_required_env(name: str, default: str | None = None) -> str:
        value = os.getenv(name, default)
        if value is None or not value.strip():
            raise ValueError(f"Missing required env var: {name}")
        return value

    def select_action(
        self,
        observation: dict,
        *,
        max_new_tokens: int | None = None,
        **_context: Any,
    ) -> dict:
        sys_c = system_prompt_from_observation(
            observation, curriculum_stage=self.curriculum_stage
        )
        user_c = format_observation_prompt(observation)
        prompt = sys_c + "\n\n" + user_c
        retries = max(0, int(self.max_retries))
        last_exc: Exception | None = None
        rng = np.random.default_rng(0)
        for attempt in range(retries + 1):
            try:
                text = self._complete(prompt=prompt, max_new_tokens=max_new_tokens)
                if text and text.strip():
                    action, _ = parse_llm_output(text, observation, rng=rng)
                    return action
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < retries:
                    time.sleep(min(2.0, 0.5 * (attempt + 1)))
        raise RuntimeError(f"LLM baseline failed after retries: {last_exc}") from last_exc

    @abstractmethod
    def _complete(self, *, prompt: str, max_new_tokens: int | None = None) -> str:
        """Return raw model text (JSON)."""
