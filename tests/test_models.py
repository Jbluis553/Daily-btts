from __future__ import annotations

import os

os.environ.setdefault("CACHE_DIR", "./cache_test")

from src.models import (  # noqa: E402
    BivariatePoisson,
    CatBoostBTTS,
    HistoricalBTTS,
    LSTMMomentum,
    XGBoostBTTS,
)
from src.models.training import build_dataset, team_features  # noqa: E402


def _mk(home_id, away_id, h, a, date, team_id=None):
    m = {
        "home_team": {"id": home_id, "name": "H"},
        "away_team": {"id": away_id, "name": "A"},
        "score": {"home": h, "away": a},
        "utc_date": date,
    }
    if team_id:
        m["_team_id"] = str(team_id)
    return m


def test_team_features_away_perspective():
    # Equipo 2 siempre de visitante: mete 3 (away=3), recibe 1 (home=1)
    hist = [_mk(1, 2, 1, 3, f"2026-01-{i+10:02d}", team_id=2) for i in range(5)]
    f = team_features(hist)
    assert f[1] == 3.0   # gf_avg
    assert f[2] == 1.0   # ga_avg


def test_bivariate_poisson_away_lambda():
    hist_home = [_mk(1, 2, 2, 0, f"2026-01-{i+10:02d}") for i in range(10)]
    hist_away = [_mk(1, 2, 0, 3, f"2026-01-{i+10:02d}", team_id=2) for i in range(10)]
    att, def_ = BivariatePoisson._team_lambda(hist_away, 2)
    assert att == 3.0 and def_ == 0.0


def test_historical_model():
    hist = [_mk(1, 2, 1, 1, "2026-01-01"), _mk(1, 2, 2, 0, "2026-01-02"),
            _mk(1, 2, 0, 0, "2026-01-03"), _mk(1, 2, 3, 2, "2026-01-04")]
    assert HistoricalBTTS(n=10).predict(hist, hist) == 50.0
    assert HistoricalBTTS(n=10).predict([], []) == 0.0


def test_ml_models_fallback_is_deterministic_and_real_stats_based():
    """Sin modelo entrenado: fallback honesto, determinista, en rango válido."""
    hist_home = [_mk(1, 2, 2, 1, f"2026-01-{i+10:02d}", team_id=1) for i in range(15)]
    hist_away = [_mk(1, 2, 1, 2, f"2026-01-{i+10:02d}", team_id=2) for i in range(15)]
    for cls in (XGBoostBTTS, CatBoostBTTS, LSTMMomentum):
        m1 = cls().predict(hist_home, hist_away)
        m2 = cls().predict(hist_home, hist_away)
        assert m1 == m2, f"{cls.__name__} no es determinista"
        assert 1.0 <= m1 <= 100.0


def test_build_dataset_no_leakage():
    """Solo genera muestras cuando el rival también tiene historial."""
    hist_a = [_mk(1, 2, 1, 1, f"2026-01-{i+10:02d}", team_id=1) for i in range(12)]
    hist_b = [_mk(1, 2, 2, 2, f"2026-01-{i+10:02d}", team_id=2) for i in range(12)]
    X, y = build_dataset({"1": hist_a, "2": hist_b})
    assert len(y) > 0
    assert set(y.tolist()) <= {0, 1}
    X2, y2 = build_dataset({"1": hist_a})  # sin rival: sin muestras
    assert len(y2) == 0
