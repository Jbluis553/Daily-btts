# src/models/training.py
"""Entrenamiento real de los modelos ML con datos reales de TheStatsAPI.

Dataset: historiales reales de partidos finished (ya descargados por
DataFetcher.get_teams_histories). Construcción SIN data leakage: para
cada partido real del pasado, las features solo usan partidos anteriores
a su fecha (cutoff temporal).

Si no hay muestras suficientes (< MIN_SAMPLES), NO se entrena nada: los
modelos caen a un fallback determinístico honesto y lo dicen en los logs.
Nunca se usan datos sintéticos.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils import extract_score, is_btts

logger = logging.getLogger(__name__)

MIN_SAMPLES = 200
MODELS_DIR = Path(__file__).resolve().parent / "trained"
FEATURES_PER_TEAM = 10
SEQ_LEN = 8


def team_features(matches: List[Dict[str, Any]], cutoff: Optional[str] = None) -> List[float]:
    """Features reales de un equipo: [btts_rate, gf_avg, ga_avg, gf_last5,
    ga_last5, clean_sheets_pct, failed_score_pct, over25_pct, n_matches,
    form_trend].

    Perspectiva correcta: los goles del equipo dueño del historial se
    toman siempre como 'a favor', tanto si jugó de local como de visitante.

    `cutoff` (opcional, ISO date): solo usa partidos con utc_date < cutoff.
    Esencial para entrenar sin leakage.
    """
    gf, ga = [], []
    btts_hits = cs = fs = o25 = 0
    team_id = None
    for m in matches:
        if cutoff and (m.get("utc_date") or "") >= cutoff:
            continue
        h, a = extract_score(m)
        if h is None or a is None:
            continue
        if team_id is None:
            team_id = m.get("_team_id")
        home_id = (m.get("home_team") or {}).get("id")
        is_own_home = team_id is None or str(home_id) == str(team_id)
        g_for, g_against = (h, a) if is_own_home else (a, h)
        gf.append(g_for)
        ga.append(g_against)
        if g_for > 0 and g_against > 0:
            btts_hits += 1
        if g_against == 0:
            cs += 1
        if g_for == 0:
            fs += 1
        if g_for + g_against > 2.5:
            o25 += 1
    n = len(gf)
    if n == 0:
        return [0.0] * FEATURES_PER_TEAM
    k5 = min(5, n)
    return [
        100.0 * btts_hits / n,
        sum(gf) / n,
        sum(ga) / n,
        sum(gf[:k5]) / k5,
        sum(ga[:k5]) / k5,
        100.0 * cs / n,
        100.0 * fs / n,
        100.0 * o25 / n,
        float(min(n, 30)),
        (sum(gf[:k5]) - sum(gf[k5:2 * k5])) / k5,
    ]


def build_dataset(
    histories: Dict[str, List[Dict[str, Any]]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera (X, y) reales y sin leakage.

    Para cada equipo y cada partido real con índice i (del más reciente
    al más antiguo): features del equipo = partidos más antiguos que i
    (cutoff = fecha del partido i); features del rival = SU historial
    completo filtrado por el mismo cutoff; etiqueta = BTTS real del
    partido i. Muestras sin rival disponible se omiten.
    """
    X: List[List[float]] = []
    y: List[int] = []
    for team_id, matches in histories.items():
        if len(matches) < 10:
            continue
        for i in range(len(matches) - 1, 8, -1):
            match = matches[i]
            h, a = extract_score(match)
            if h is None or a is None:
                continue
            cutoff = match.get("utc_date")
            if not cutoff:
                continue
            opp_id = str(
                ((match.get("away_team") or {}).get("id")
                 if str((match.get("home_team") or {}).get("id")) == str(team_id)
                 else (match.get("home_team") or {}).get("id")) or ""
            )
            opp_hist = histories.get(opp_id)
            if not opp_hist:
                continue
            feat_own = team_features(matches[i + 1:], cutoff)
            feat_opp = team_features(opp_hist, cutoff)
            X.append(feat_own + feat_opp)
            y.append(1 if is_btts(match) else 0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def build_sequences(
    histories: Dict[str, List[Dict[str, Any]]], seq_len: int = SEQ_LEN
) -> Tuple[np.ndarray, np.ndarray]:
    """Secuencias reales [gf, ga, btts] para el LSTM.

    Para cada partido real con índice i: input = los `seq_len` partidos
    anteriores (cronológicos), etiqueta = BTTS real del partido i.
    """
    X: List[np.ndarray] = []
    y: List[int] = []
    for team_id, matches in histories.items():
        if len(matches) < seq_len + 1:
            continue
        for i in range(len(matches) - 1, seq_len - 1, -1):
            match = matches[i]
            if not is_btts(match) and extract_score(match)[0] is None:
                continue
            seq = []
            for m in matches[i - seq_len:i][::-1]:  # cronológico
                h, a = extract_score(m)
                if h is None or a is None:
                    break
                team_id_m = m.get("_team_id")
                home_id = (m.get("home_team") or {}).get("id")
                own_home = team_id_m is None or str(home_id) == str(team_id_m)
                g_for, g_against = (h, a) if own_home else (a, h)
                seq.append([float(g_for), float(g_against),
                            1.0 if g_for > 0 and g_against > 0 else 0.0])
            if len(seq) != seq_len:
                continue
            X.append(np.array(seq, dtype=np.float32))
            y.append(1 if is_btts(match) else 0)
    if not X:
        return np.zeros((0, seq_len, 3), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return np.stack(X), np.array(y, dtype=np.int64)


def train_and_save(histories: Dict[str, List[Dict[str, Any]]]) -> None:
    """Entrena XGBoost y CatBoost con datos reales y los persiste (joblib).

    Con TRAIN_MODELS=1 el analyzer invoca esto automáticamente después
    de descargar los historiales del día.
    """
    X, y = build_dataset(histories)
    logger.info("Dataset real: %d muestras (positivas: %d)", len(y), int(y.sum()))
    if len(y) < MIN_SAMPLES:
        logger.warning(
            "Solo %d muestras reales (< %d): se OMITE el entrenamiento ML. "
            "Los modelos usarán su fallback determinístico honesto.",
            len(y), MIN_SAMPLES,
        )
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42,
        )
        model.fit(X, y)
        joblib.dump(model, MODELS_DIR / "xgboost_btts.joblib")
        logger.info("XGBoost entrenado con %d muestras reales → %s",
                    len(y), MODELS_DIR / "xgboost_btts.joblib")
    except ImportError as exc:
        logger.warning("xgboost/joblib no disponible, se omite: %s", exc)

    try:
        import joblib
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            loss_function="Logloss", random_seed=42, verbose=0,
        )
        model.fit(X, y)
        joblib.dump(model, MODELS_DIR / "catboost_btts.joblib")
        logger.info("CatBoost entrenado con %d muestras reales → %s",
                    len(y), MODELS_DIR / "catboost_btts.joblib")
    except ImportError as exc:
        logger.warning("catboost/joblib no disponible, se omite: %s", exc)


def train_lstm_and_save(
    histories: Dict[str, List[Dict[str, Any]]], seq_len: int = SEQ_LEN,
    epochs: int = 30,
) -> None:
    """Entrena el LSTM con secuencias reales (requiere tensorflow)."""
    X, y = build_sequences(histories, seq_len)
    logger.info("Secuencias reales: %d", len(y))
    if len(y) < MIN_SAMPLES:
        logger.warning(
            "Solo %d secuencias reales (< %d): se omite el entrenamiento LSTM.",
            len(y), MIN_SAMPLES,
        )
        return
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras import layers, models

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model = models.Sequential([
            layers.Input(shape=(seq_len, 3)),
            layers.LSTM(16),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
        model.save(MODELS_DIR / "lstm_btts.keras")
        logger.info("LSTM entrenado con %d secuencias reales → %s",
                    len(y), MODELS_DIR / "lstm_btts.keras")
    except ImportError:
        logger.warning("tensorflow no instalado: se omite el entrenamiento LSTM.")


def load_trained(name: str) -> Any:
    """Carga un modelo persistido, o None si no existe."""
    path = MODELS_DIR / name
    if not path.exists():
        return None
    try:
        if name.endswith(".keras"):
            import tensorflow as tf  # noqa: F401
            return tf.keras.models.load_model(path)
        import joblib
        return joblib.load(path)
    except Exception as exc:
        logger.warning("No se pudo cargar %s: %s", path, exc)
        return None
