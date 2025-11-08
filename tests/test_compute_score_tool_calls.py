from unittest.mock import patch
import sys
import types

import pytest

if "datasets" not in sys.modules:
    sys.modules["datasets"] = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: None)

from recipe.demystify import reward


def _make_score_dict(score):
    return {"score": score, "pred": "", "other": 0}


def test_compute_score_uses_explicit_tool_calls():
    extra_info = {"tool_parser_detected_tool_calls": 3, "num_turns": 10}
    with patch.object(reward.math_dapo, "compute_score", return_value=_make_score_dict(-1.0)) as mock_score:
        res = reward.compute_score("math", "ans", "gt", extra_info)
    mock_score.assert_called_once()
    assert res["score"] == pytest.approx(-0.7)


def test_compute_score_fall_back_to_num_turns():
    extra_info = {"num_turns": 6}
    with patch.object(reward.math_dapo, "compute_score", return_value=_make_score_dict(-1.0)):
        res = reward.compute_score("math", "ans", "gt", extra_info)
    # num_turns=6 -> tool_calls=(6-2)//2=2 -> reward adds 0.2 but capped at -0.6
    assert res["score"] == pytest.approx(-0.8)

