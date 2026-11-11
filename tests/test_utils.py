from __future__ import annotations

from src.utils import extract_score, is_btts


def test_extract_score_flat_format() -> None:
    m = {"score": {"home": 2, "away": 1}}
    assert extract_score(m) == (2, 1)
    assert is_btts(m) is True


def test_extract_score_final_score_format() -> None:
    m = {"score": {"final_score": {"home": 0, "away": 3}}}
    assert extract_score(m) == (0, 3)
    assert is_btts(m) is False


def test_extract_score_top_level_fields() -> None:
    m = {"home_score": 1, "away_score": 1}
    assert extract_score(m) == (1, 1)


def test_extract_score_missing() -> None:
    assert extract_score({}) == (None, None)
    assert extract_score({"score": {"home": "x"}}) == (None, None)
    assert is_btts({"score": {}}) is False
