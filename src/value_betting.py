# src/value_betting.py
"""
Módulo de value betting: EV, vig removal, Kelly fraccional.
Convierte probabilidad del modelo en decisión de apuesta rentable.
"""

from __future__ import annotations

from typing import Optional


def calculate_ev(p_model: float, odds: float) -> float:
    """
    Expected Value de una apuesta binaria.
    EV = p*(odds-1) - (1-p)
    EV > 0 → +EV → apostar.
    """
    return p_model * (odds - 1) - (1 - p_model)


def remove_vig(odds_yes: float, odds_no: float) -> tuple[float, float]:
    """
    Elimina el margen (vig) del bookmaker.
    Método: normalización proporcional.
    Devuelve (p_fair_yes, p_fair_no).
    """
    inv_yes = 1.0 / odds_yes
    inv_no = 1.0 / odds_no
    total = inv_yes + inv_no
    if total == 0:
        return 0.5, 0.5
    return inv_yes / total, inv_no / total


def kelly_stake(p: float, odds: float, fraction: float = 0.25) -> float:
    """
    Kelly fraccional.
    f* = (p*b - q) / b, donde b = odds-1, q = 1-p.
    Devuelve fracción del bankroll a apostar (0 si no hay edge).
    """
    b = odds - 1
    if b <= 0:
        return 0.0
    q = 1 - p
    f_star = (p * b - q) / b
    if f_star <= 0:
        return 0.0
    return f_star * fraction


def find_value_bet(
    p_model: float,
    odds_yes: float,
    odds_no: float,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
) -> Optional[dict]:
    """
    Evalúa si hay value bet:
    1. Calcula p_market_fair (sin vig)
    2. edge = p_model - p_market_fair
    3. Si edge >= min_edge y EV > 0 → value bet
    """
    p_market_fair, _ = remove_vig(odds_yes, odds_no)
    edge = p_model - p_market_fair
    ev = calculate_ev(p_model, odds_yes)

    if edge >= min_edge and ev > 0:
        return {
            "edge": round(edge, 4),
            "ev": round(ev, 4),
            "p_model": round(p_model, 4),
            "p_market_fair": round(p_market_fair, 4),
            "odds": odds_yes,
            "kelly_fraction": round(
                kelly_stake(p_model, odds_yes, kelly_fraction), 4
            ),
        }
    return None


def extract_match_btts_odds(odds_data: dict) -> tuple[Optional[float], Optional[float]]:
    """
    Extrae las cuotas BTTS yes/no del primer bookmaker disponible.
    Devuelve (odds_yes, odds_no) con None si no hay datos.
    """
    for bk in odds_data.get("bookmakers", []) or []:
        btts = (bk.get("markets") or {}).get("btts", {})
        yes = btts.get("yes", {})
        no = btts.get("no", {})
        o_yes = yes.get("last_seen") or yes.get("opening")
        o_no = no.get("last_seen") or no.get("opening")
        if o_yes and o_no:
            try:
                return float(o_yes), float(o_no)
            except (ValueError, TypeError):
                continue
    return None, None


def best_odds_btts_yes(odds_data: dict) -> tuple[Optional[float], str]:
    """
    Extrae la MEJOR cuota de BTTS Yes entre todos los bookmakers.
    Devuelve (mejor_odds, bookmaker_name) o (None, "").
    """
    best_odds = 0.0
    best_book = ""
    for bk in odds_data.get("bookmakers", []) or []:
        btts = (bk.get("markets") or {}).get("btts", {})
        yes = btts.get("yes", {})
        val = yes.get("last_seen") or yes.get("opening")
        if val:
            try:
                odds_val = float(val)
                if odds_val > best_odds:
                    best_odds = odds_val
                    best_book = bk.get("bookmaker", "")
            except (ValueError, TypeError):
                pass
    return (best_odds if best_odds > 0 else None), best_book
