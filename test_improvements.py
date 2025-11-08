import json
import unittest
from unittest.mock import patch

import requests

from verl.utils.reward_score.sandbox_fusion import utils as sandbox_utils
from verl.utils.reward_score.livecodebench import code_math


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):  # pragma: no cover - defensive path for non-2xx
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class CheckCorrectnessStatsTests(unittest.TestCase):
    def setUp(self):
        self.sandbox_url = "http://sandbox.test/run_code"

    def test_collect_stats_counts_success_and_compile_error(self):
        success_payload = {
            "status": "Success",
            "compile_result": None,
            "run_result": {
                "status": "Finished",
                "stdout": "1",
                "stderr": "",
                "return_code": 0,
                "execution_time": 0.01,
            },
            "files": {},
        }
        compile_error_payload = {
            "status": "Failed",
            "compile_result": {
                "status": "Error",
                "stderr": "Syntax error",
                "return_code": 1,
                "execution_time": 0.02,
            },
            "run_result": None,
            "files": {},
        }

        def fake_post(*args, **kwargs):
            payload = json.loads(kwargs["data"])
            code = payload["code"]
            if "case0" in code:
                return FakeResponse(success_payload)
            return FakeResponse(compile_error_payload)

        in_outs = {
            "inputs": ["1", "2"],
            "outputs": ["1", "2"],
            "assert_case": ["#case0", "#case1"],
        }

        with patch("verl.utils.reward_score.sandbox_fusion.utils.requests.post", side_effect=fake_post):
            results, metadata, summary = sandbox_utils.check_correctness(
                sandbox_fusion_url=self.sandbox_url,
                in_outs=in_outs,
                generation="print(input())",
                timeout=5,
                collect_stats=True,
            )

        self.assertEqual(summary["total_cases"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["status_breakdown"].get("compile_error"), 1)
        self.assertEqual(summary["status_breakdown"].get("success"), 1)
        self.assertEqual(summary.get("stdin_supplied_cases"), 2)
        self.assertEqual(summary.get("stdin_mismatch_count"), 0)
        self.assertTrue(any(result is True for result in results))
        self.assertTrue(any(result == -4 for result in results))
        self.assertTrue(any(meta.get("status") == "compile_error" for meta in metadata))

    def test_collect_stats_flags_stdin_mismatch_cases(self):
        success_payload = {
            "status": "Success",
            "compile_result": None,
            "run_result": {
                "status": "Finished",
                "stdout": "done",
                "stderr": "",
                "return_code": 0,
                "execution_time": 0.01,
            },
            "files": {},
        }

        def fake_post(*args, **kwargs):
            return FakeResponse(success_payload)

        in_outs = {
            "inputs": ["42"],
            "outputs": ["ok"],
        }

        with patch("verl.utils.reward_score.sandbox_fusion.utils.requests.post", side_effect=fake_post):
            _, _, summary = sandbox_utils.check_correctness(
                sandbox_fusion_url=self.sandbox_url,
                in_outs=in_outs,
                generation="print('hi')",
                timeout=5,
                collect_stats=True,
            )

        self.assertEqual(summary.get("stdin_supplied_cases"), 1)
        self.assertEqual(summary.get("stdin_mismatch_count"), 1)
        samples = summary.get("stdin_mismatch_samples", [])
        self.assertTrue(samples)
        self.assertEqual(samples[0]["case_index"], 0)


class ComputeScoreRobustnessTests(unittest.TestCase):
    def test_compute_score_returns_syntax_error_on_invalid_code(self):
        completion = """```python\ndef answer(:\n    return 42\n```"""
        test_cases = {"inputs": ["1"], "outputs": ["1"]}

        result = code_math.compute_score(completion, test_cases, timeout=5)

        self.assertEqual(result["error"], "syntax_error")
        self.assertFalse(result["acc"])
        self.assertEqual(result["score"], -1.0)

    def test_compute_score_includes_sandbox_stats_when_available(self):
        completion = """```python\nprint('hello')\n```"""
        test_cases = {"inputs": [""], "outputs": [""], "fn_name": "noop"}

        sandbox_summary = {
            "total_cases": 2,
            "success_count": 1,
            "success_rate": 0.5,
            "status_breakdown": {"success": 1, "wrong_answer": 1},
        }
        with patch(
            "verl.utils.reward_score.livecodebench.code_math.check_correctness",
            return_value=([True, False], [{}, {}], sandbox_summary),
        ):
            result = code_math.compute_score(completion, test_cases, timeout=5)
        print(result)
        self.assertIn("sandbox_stats", result)
        self.assertEqual(result["sandbox_stats"], sandbox_summary)


if __name__ == "__main__":
    unittest.main()
