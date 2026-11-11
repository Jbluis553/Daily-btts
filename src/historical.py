# src/models/historical.py
"""Modelo histórico simple: tasa BTTS de los últimos N partidos."""
from __future__ import annotations

from typing import Any, Dict, List

from ..utils import is_btts, safe_div
from .base import BaseBTTSModel


class HistoricalBTTS(BaseBTTSModel):
    name = "historical"

    def __init__(self, n: int = 10) -> None:
        self.n = n

    @staticmethod
    def _btts_rate(matches: List[Dict[str, Any]]) -> float:
        if not matches:
            return 0.0
        hits = sum(1 for m in matches if is_btts(m))
        return 100.0 * safe_div(hits, len(matches))

    def predict(self, home_matches, away_matches) -> float:
        home_rate = self._btts_rate(home_matches[: self.n])
        away_rate = self._btts_rate(away_matches[: self.n])
        if home_rate == 0.0 and away_rate == 0.0:
            return 0.0
        return round((home_rate + away_rate) / 2, 2)
