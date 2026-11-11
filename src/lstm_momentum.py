# src/models/lstm_momentum.py
"""LSTM de momentum sobre secuencias recientes.

Usa un modelo entrenado con secuencias REALES de TheStatsAPI persistido
en models/trained/lstm_btts.keras (generado con TRAIN_MODELS=1 y
tensorflow instalado).

Si no existe modelo entrenado o tensorflow no está instalado, usa un
fallback determinístico de media móvil BTTS (mismo contrato predict).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from src.utils import extract_score
from .base import BaseBTTSModel
from .training import SEQ_LEN, load_trained

logger = logging.getLogger(__name__)


def _sequence(matches: List[Dict[str, Any]], seq_len: int) -> np.ndarray:
    """Secuencia [gf, ga, btts] por partido (cronológica), padded al inicio.

    Perspectiva correcta: goles del equipo dueño del historial como 'a favor'.
    """
    team_id = matches[0].get("_team_id") if matches else None
    rows = []
    for m in matches[:seq_len]:
        h, a = extract_score(m)
        if h is None or a is None:
            continue
        home_id = (m.get("home_team") or {}).get("id")
        own_home = team_id is None or str(home_id) == str(team_id)
        g_for, g_against = (h, a) if own_home else (a, h)
        rows.append([float(g_for), float(g_against),
                     1.0 if g_for > 0 and g_against > 0 else 0.0])
    rows = rows[::-1]  # cronológico (el input se construye más reciente → más antiguo invertido)
    while len(rows) < seq_len:
        rows.insert(0, [0.0, 0.0, 0.0])
    return np.array(rows[-seq_len:], dtype=np.float32)


class LSTMMomentum(BaseBTTSModel):
    name = "lstm_momentum"

    def __init__(self, seq_len: int = SEQ_LEN, epochs: int = 30) -> None:
        self.seq_len = seq_len
        self.epochs = epochs
        self._model = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._model = load_trained("lstm_btts.keras")
        if self._model is None:
            logger.info(
                "LSTM: sin modelo entrenado real (TRAIN_MODELS=1 + tensorflow); "
                "usando fallback de media móvil."
            )

    def predict(self, home_matches, away_matches) -> float:
        self._load()
        sh = _sequence(home_matches, self.seq_len)
        sa = _sequence(away_matches, self.seq_len)

        if self._model is not None:
            # FIX: input de 6 canales (home+away concatenados), no reshape inválido
            x = np.concatenate([sh, sa], axis=1).reshape(1, self.seq_len, 6)
            p = float(self._model.predict(x, verbose=0)[0][0])
        else:
            # Fallback: media BTTS reciente combinada home/away
            p = 0.5 * (sh[:, 2].mean() + sa[:, 2].mean())
            p = float(np.clip(p, 0.05, 0.95))
        return round(p * 100, 2)
