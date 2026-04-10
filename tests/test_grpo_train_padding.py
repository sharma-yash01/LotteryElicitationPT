"""Unit tests for vLLM server + DDP generate() padding (NCCL lockstep)."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _ensure_openenv_stubs() -> None:
    if "openenv.core.env_client" in sys.modules:
        return
    ct = types.ModuleType("openenv.core.client_types")

    class StepResult:
        def __init__(self, observation, reward=None, done=False):
            self.observation = observation
            self.reward = reward
            self.done = done

    ct.StepResult = StepResult

    ec = types.ModuleType("openenv.core.env_client")

    class EnvClient:
        pass

    ec.EnvClient = EnvClient

    core = types.ModuleType("openenv.core")
    core.client_types = ct
    core.env_client = ec
    openenv = types.ModuleType("openenv")
    openenv.core = core
    sys.modules["openenv"] = openenv
    sys.modules["openenv.core"] = core
    sys.modules["openenv.core.client_types"] = ct
    sys.modules["openenv.core.env_client"] = ec


_ensure_openenv_stubs()

from training import grpo_train as gt


class TestVllmServerPadding(unittest.TestCase):
    def _trainer(self, *, mode: str = "server"):
        t = MagicMock()
        t.vllm_generation = MagicMock()
        t.vllm_generation.mode = mode
        t.vllm_generation.max_completion_length = 512
        t.vllm_generation.generate.return_value = ([], [[1]], None, None)
        t.args = MagicMock()
        t.args.max_completion_length = 2048
        return t

    def test_needs_padding_false_colocate(self):
        self.assertFalse(gt._needs_vllm_server_generate_padding(self._trainer(mode="colocate")))

    @patch.object(gt.dist, "get_world_size", return_value=1)
    @patch.object(gt.dist, "is_initialized", return_value=True)
    @patch.object(gt.dist, "is_available", return_value=True)
    def test_needs_padding_false_world_size_1(self, _a, _b, _c):
        self.assertFalse(gt._needs_vllm_server_generate_padding(self._trainer()))

    @patch.object(gt.dist, "get_world_size", return_value=3)
    @patch.object(gt.dist, "is_initialized", return_value=True)
    @patch.object(gt.dist, "is_available", return_value=True)
    def test_needs_padding_true_server_multi(self, _a, _b, _c):
        self.assertTrue(gt._needs_vllm_server_generate_padding(self._trainer()))

    @patch.object(gt.dist, "is_initialized", return_value=False)
    @patch.object(gt.dist, "is_available", return_value=True)
    def test_needs_padding_false_dist_not_init(self, _a, _b):
        self.assertFalse(gt._needs_vllm_server_generate_padding(self._trainer()))

    def test_pad_calls_generate_n_times(self):
        t = self._trainer()
        gt._pad_vllm_server_generates_to_target(t, before_ids=[1, 2, 3], num_dummy=4)
        self.assertEqual(t.vllm_generation.generate.call_count, 4)

    def test_pad_zero_no_calls(self):
        t = self._trainer()
        gt._pad_vllm_server_generates_to_target(t, before_ids=[1], num_dummy=0)
        t.vllm_generation.generate.assert_not_called()

    def test_pad_negative_no_calls(self):
        t = self._trainer()
        gt._pad_vllm_server_generates_to_target(t, before_ids=[1], num_dummy=-1)
        t.vllm_generation.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
