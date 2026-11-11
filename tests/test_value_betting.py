from __future__ import annotations

from src.value_betting import (
    calculate_ev,
    find_value_bet,
    kelly_stake,
    remove_vig,
)


def test_remove_vig_sums_to_one():
    p_yes, p_no = remove_vig(1.80, 2.10)
    assert abs(p_yes + p_no - 1.0) < 1e-9


def test_ev_positive_when_model_beats_odds():
    assert calculate_ev(0.60, 2.0) > 0
    assert calculate_ev(0.40, 2.0) < 0


def test_kelly_zero_without_edge():
    assert kelly_stake(0.5, 2.0) == 0.0
    assert kelly_stake(0.6, 2.0) > 0.0


def test_find_value_bet_respects_min_edge():
    vb = find_value_bet(0.60, 2.00, 1.80, min_edge=0.03)
    assert vb is not None and vb["edge"] >= 0.03
    assert find_value_bet(0.55, 2.00, 1.80, min_edge=0.10) is None
