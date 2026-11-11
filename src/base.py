# src/models/base.py
"""Contrato común de los modelos BTTS."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseBTTSModel(ABC):
    name: str = "base"

    @abstractmethod
    def predict(
        self, home_matches: List[Dict[str, Any]], away_matches: List[Dict[str, Any]]
    ) -> float:
        """Devuelve probabilidad BTTS en porcentaje 0-100."""
        raise NotImplementedError
