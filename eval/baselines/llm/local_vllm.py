"""Local vLLM / OpenAI-compatible server for lottery baseline."""

from __future__ import annotations

from eval.baselines.llm.api_chat import APIChatBaseline


class LocalVLLMBaseline(APIChatBaseline):
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        curriculum_stage: int = 1,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.0,
    ):
        super().__init__(
            base_url=base_url
            or self.get_required_env("BASELINE_LOCAL_BASE_URL", "http://127.0.0.1:8001/v1"),
            api_key=self.get_required_env("BASELINE_LOCAL_API_KEY", "local"),
            model=model or self.get_required_env("BASELINE_LOCAL_MODEL"),
            curriculum_stage=curriculum_stage,
            timeout_s=timeout_s,
            max_retries=max_retries,
            temperature=temperature,
        )
