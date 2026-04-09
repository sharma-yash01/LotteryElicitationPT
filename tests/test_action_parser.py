"""Tests for training/action_parser.py."""

from __future__ import annotations

import json
import unittest

import numpy as np

from training.action_parser import parse_llm_output

_BASE_OBS = {
    "step_idx": 0,
    "max_steps": 5,
    "steps_remaining": 3,
    "gamma_range": [0.2, 1.5],
    "lambda_range": [1.0, 4.5],
    "min_outcome_value": -50.0,
    "max_outcome_value": 100.0,
    "history": [],
}

_LOT = {
    "outcomes": [
        {"value": 10.0, "probability": 0.5},
        {"value": 0.0, "probability": 0.5},
    ]
}


def _valid_payload() -> str:
    return json.dumps({"lottery_a": _LOT, "lottery_b": _LOT})


class TestActionParser(unittest.TestCase):
    def test_valid_clean_json(self):
        action, ok = parse_llm_output(_valid_payload(), _BASE_OBS, rng=np.random.default_rng(0))
        self.assertTrue(ok)
        self.assertIn("lottery_a", action)
        self.assertIn("lottery_b", action)

    def test_fenced_json(self):
        text = "```json\n" + _valid_payload() + "\n```"
        action, ok = parse_llm_output(text, _BASE_OBS, rng=np.random.default_rng(0))
        self.assertTrue(ok)
        self.assertEqual(action["lottery_a"]["outcomes"][0]["value"], 10.0)

    def test_trailing_comma(self):
        text = """
        {
          "lottery_a": {"outcomes": [{"value": 1.0, "probability": 0.5}, {"value": 2.0, "probability": 0.5},]},
          "lottery_b": {"outcomes": [{"value": 3.0, "probability": 0.5}, {"value": 4.0, "probability": 0.5},]},
        }
        """
        _, ok = parse_llm_output(text, _BASE_OBS, rng=np.random.default_rng(0))
        self.assertTrue(ok)

    def test_garbage_uses_fallback(self):
        action, ok = parse_llm_output("not json {{{", _BASE_OBS, rng=np.random.default_rng(42))
        self.assertFalse(ok)
        self.assertIn("lottery_a", action)

    def test_final_step_injects_theta(self):
        obs = {**_BASE_OBS, "steps_remaining": 1}
        action, ok = parse_llm_output(_valid_payload(), obs, rng=np.random.default_rng(0))
        self.assertTrue(ok)
        self.assertIn("theta_estimate", action)
        self.assertIn("gamma", action["theta_estimate"])

    def test_terminate_early_without_theta_stripped(self):
        obs = {**_BASE_OBS, "steps_remaining": 2}
        text = json.dumps(
            {
                "lottery_a": {
                    "outcomes": [
                        {"value": 1.0, "probability": 0.5},
                        {"value": 2.0, "probability": 0.5},
                    ]
                },
                "lottery_b": {
                    "outcomes": [
                        {"value": 3.0, "probability": 0.5},
                        {"value": 4.0, "probability": 0.5},
                    ]
                },
                "terminate_early": True,
            }
        )
        action, ok = parse_llm_output(text, obs, rng=np.random.default_rng(0))
        self.assertTrue(ok)
        self.assertFalse(action.get("terminate_early"))

    def test_out_of_bounds_rejected(self):
        bad = {
            "outcomes": [
                {"value": 200.0, "probability": 0.5},
                {"value": 0.0, "probability": 0.5},
            ]
        }
        text = json.dumps({"lottery_a": bad, "lottery_b": _LOT})
        _, ok = parse_llm_output(text, _BASE_OBS, rng=np.random.default_rng(0))
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
