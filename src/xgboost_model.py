# src/models/xgboost_model.py
"""XGBoost con features de goles.

Usa un modelo entrenado con datos REALES de TheStatsAPI persistido en
models/trained/xgboost_btts.joblib (generado con TRAIN_MODELS=1).

Si no existe modelo entrenado, usa un fallback determinístico honesto
basado en estadística real — nunca datos sintéticos.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .base import BaseBTTSModel
from .training import FEATURES_PER_TEAM, load_trained, team_features

logger = logging.getLogger(__name__)


class XGBoostBTTS(BaseBTTSModel):
    name = "xgboost"

    def __init__(self, n: int = 15) -> None:
        self.n = n
        self._model = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._model = load_trained("xgboost_btts.joblib")
        if self._model is None:
            logger.info(
                "XGBoost: sin modelo entrenado real (entrena con TRAIN_MODELS=1); "
                "usando fallback determinístico."
            )

    def predict(self, home_matches, away_matches) -> float:
        self._load()
        fh = team_features(home_matches[: self.n])
        fa = team_features(away_matches[: self.n])
        x = np.array(fh + fa).reshape(1, -1)

        if self._model is not None:
            p = float(self._model.predict_proba(x)[0][1])
        else:
            # Fallback honesto y determinístico sobre estadística real
            avg_btts = (fh[0] + fa[0]) / 2
            z = 0.04 * (avg_btts - 55) + 0.25 * (fh[3] + fa[3] - fh[4] - fa[4])
            p = 1 / (1 + np.exp(-z))
        return round(max(1.0, min(99.0, p * 100)), 2)
