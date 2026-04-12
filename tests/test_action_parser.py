"""Tests for training/action_parser.py."""

from __future__ import annotations

import json
import unittest

import numpy as np

from training.action_parser import _normalize_lottery_probabilities, parse_llm_output

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

    def test_probabilities_sum_exactly_one_after_parse(self):
        """Borderline sums pass _valid_lottery (1e-3) but must normalize for env (1e-6)."""
        lot = {
            "outcomes": [
                {"value": 1.0, "probability": 0.3333},
                {"value": 2.0, "probability": 0.3333},
                {"value": 3.0, "probability": 0.3334},
            ]
        }
        text = json.dumps({"lottery_a": lot, "lottery_b": _LOT})
        action, ok = parse_llm_output(text, _BASE_OBS, rng=np.random.default_rng(0))
        self.assertTrue(ok)
        for key in ("lottery_a", "lottery_b"):
            probs = [o["probability"] for o in action[key]["outcomes"]]
            self.assertAlmostEqual(sum(probs), 1.0, places=12)

    def test_normalize_lottery_two_outcomes_remainder(self):
        n = _normalize_lottery_probabilities({
            "outcomes": [
                {"value": 0.0, "probability": 0.41},
                {"value": 1.0, "probability": 0.59},
            ]
        })
        self.assertEqual(sum(o["probability"] for o in n["outcomes"]), 1.0)

    def test_fallback_lotteries_sum_to_one(self):
        action, ok = parse_llm_output("not json", _BASE_OBS, rng=np.random.default_rng(123))
        self.assertFalse(ok)
        for key in ("lottery_a", "lottery_b"):
            probs = [o["probability"] for o in action[key]["outcomes"]]
            self.assertEqual(sum(probs), 1.0)


    def test_think_block_before_json_stripped(self):
        """Qwen3 think block before valid JSON should parse correctly."""
        think_prefix = "<think>\nLet me reason about this lottery elicitation...\n</think>\n"
        text = think_prefix + _valid_payload()
        action, ok = parse_llm_output(text, _BASE_OBS, rng=np.random.default_rng(0))
        self.assertTrue(ok)
        self.assertIn("lottery_a", action)
        self.assertIn("lottery_b", action)

    def test_redacted_thinking_before_json_stripped(self):
        """<redacted_thinking> wrapper before valid JSON should parse correctly."""
        text = "<redacted_thinking>some reasoning</redacted_thinking>\n" + _valid_payload()
        action, ok = parse_llm_output(text, _BASE_OBS, rng=np.random.default_rng(0))
        self.assertTrue(ok)
        self.assertIn("lottery_a", action)

    def test_null_outcome_value_rejected(self):
        bad = {
            "outcomes": [
                {"value": None, "probability": 0.5},
                {"value": 0.0, "probability": 0.5},
            ]
        }
        text = json.dumps({"lottery_a": bad, "lottery_b": _LOT})
        reasons: list[str] = []
        _, ok = parse_llm_output(text, _BASE_OBS, rng=np.random.default_rng(0), failure_reason=reasons)
        self.assertFalse(ok)
        self.assertTrue(reasons and reasons[0].startswith("invalid_lottery"))

    def test_non_numeric_probability_rejected(self):
        bad = {
            "outcomes": [
                {"value": 1.0, "probability": "not_a_float"},
                {"value": 2.0, "probability": 0.5},
            ]
        }
        text = json.dumps({"lottery_a": bad, "lottery_b": _LOT})
        _, ok = parse_llm_output(text, _BASE_OBS, rng=np.random.default_rng(0))
        self.assertFalse(ok)

    def test_nan_probability_rejected(self):
        bad = {
            "outcomes": [
                {"value": 1.0, "probability": float("nan")},
                {"value": 2.0, "probability": 0.5},
            ]
        }
        text = json.dumps({"lottery_a": bad, "lottery_b": _LOT})
        _, ok = parse_llm_output(text, _BASE_OBS, rng=np.random.default_rng(0))
        self.assertFalse(ok)

    def test_invalid_theta_stripped_valid_lotteries(self):
        """Out-of-range or null theta must not crash; lotteries still valid."""
        text = json.dumps(
            {
                "lottery_a": _LOT,
                "lottery_b": _LOT,
                "theta_estimate": {"gamma": None, "lambda": 2.0},
            }
        )
        action, ok = parse_llm_output(text, _BASE_OBS, rng=np.random.default_rng(0))
        self.assertTrue(ok)
        self.assertNotIn("theta_estimate", action)

    def test_theta_out_of_range_stripped(self):
        text = json.dumps(
            {
                "lottery_a": _LOT,
                "lottery_b": _LOT,
                "theta_estimate": {"gamma": 99.0, "lambda": 2.0},
            }
        )
        action, ok = parse_llm_output(text, _BASE_OBS, rng=np.random.default_rng(0))
        self.assertTrue(ok)
        self.assertNotIn("theta_estimate", action)

    def test_failure_reason_on_no_json(self):
        reasons: list[str] = []
        _, ok = parse_llm_output("no brace here", _BASE_OBS, rng=np.random.default_rng(0), failure_reason=reasons)
        self.assertFalse(ok)
        self.assertEqual(reasons, ["no_json_object"])


if __name__ == "__main__":
    unittest.main()
