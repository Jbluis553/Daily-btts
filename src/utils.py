# src/utils.py
"""Utilidades compartidas."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def extract_score(match: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Soporta los formatos de score reportados por la API.

    Orden de intento:
      1. match["score"]["home"] / ["away"]
      2. match["score"]["final_score"]["home"] / ["away"]
      3. match["home_score"] / match["away_score"]
    Devuelve (None, None) si no hay marcador válido.
    """
    score = match.get("score") or match.get("scores") or {}
    home = score.get("home")
    away = score.get("away")
    if home is None or away is None:
        final = score.get("final_score") or {}
        home = home if home is not None else final.get("home")
        away = away if away is not None else final.get("away")
    if home is None or away is None:
        home = match.get("home_score")
        away = match.get("away_score")
    try:
        return (
            int(home) if home is not None else None,
            int(away) if away is not None else None,
        )
    except (TypeError, ValueError):
        return None, None


def is_btts(match: Dict[str, Any]) -> bool:
    h, a = extract_score(match)
    return h is not None and a is not None and h > 0 and a > 0


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0
