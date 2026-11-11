# src/analyzer.py
"""Orquestador principal de análisis multi-modelo."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .data_fetcher import DataFetcher
from .injuries import InjuryImpactAnalyzer
from .models import (
    BivariatePoisson,
    CatBoostBTTS,
    HistoricalBTTS,
    LSTMMomentum,
    XGBoostBTTS,
)
from .value_betting import extract_match_btts_odds, find_value_bet

logger = logging.getLogger(__name__)


def _weights() -> Dict[str, float]:
    """
    Pesos del ensemble desde env, ej:
    MODEL_WEIGHTS=historical:0.15,bivariate_poisson:0.25,lstm_momentum:0.15,
                  xgboost:0.25,catboost:0.20
    """
    raw = os.environ.get("MODEL_WEIGHTS", "")
    w: Dict[str, float] = {}
    if raw:
        for part in raw.split(","):
            try:
                k, v = part.split(":")
                w[k.strip()] = float(v)
            except ValueError:
                logger.warning("Peso inválido en MODEL_WEIGHTS: %s", part)
    return w


class BTTSAnalyzer:
    def __init__(self, fetcher: DataFetcher, client=None) -> None:
        self.fetcher = fetcher
        self.client = client or getattr(fetcher, "client", None)
        self.injuries = InjuryImpactAnalyzer()
        self.weights = _weights()
        self.models = [
            HistoricalBTTS(n=10),
            BivariatePoisson(n=20),
            LSTMMomentum(seq_len=8),
            XGBoostBTTS(n=15),
            CatBoostBTTS(n=15),
        ]

    def _weighted_ensemble(self, scores: Dict[str, float]) -> float:
        valid = {k: v for k, v in scores.items() if v > 0}
        if not valid:
            return 0.0
        if not self.weights:
            return round(sum(valid.values()) / len(valid), 2)
        num = sum(valid[k] * self.weights.get(k, 0.0) for k in valid)
        den = sum(self.weights.get(k, 0.0) for k in valid)
        if den <= 0:
            return round(sum(valid.values()) / len(valid), 2)
        return round(num / den, 2)

    def _fetch_btts_odds(self, match: Dict[str, Any]) -> tuple:
        """Devuelve (odds_yes, odds_no) o (None, None).

        Solo consulta si el partido declara odds_available (recomendación
        de la guía de odds de la API). El path es configurable por si la
        API lo cambia (ODDS_ENDPOINT).
        """
        if self.client is None:
            return None, None
        match_id = match.get("id")
        if not match_id or not match.get("odds_available"):
            return None, None
        try:
            import os as _os
            template = _os.environ.get(
                "ODDS_ENDPOINT", "/football/matches/{match_id}/odds"
            )
            odds = self.client.get(template.format(match_id=match_id))
            return extract_match_btts_odds(odds)
        except Exception as exc:
            logger.debug("Sin cuotas para match %s: %s", match_id, exc)
            return None, None

    def analyze(
        self,
        upcoming: List[Dict[str, Any]],
        min_pct: float = 80.0,
        top_n: int = 20,
        min_edge: Optional[float] = None,
        fetch_odds: bool = True,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        # ── Precarga masiva de historiales en paralelo ──
        team_ids = []
        for match in upcoming:
            home = match.get("home_team") or {}
            away = match.get("away_team") or {}
            if home.get("id"):
                team_ids.append(home["id"])
            if away.get("id"):
                team_ids.append(away["id"])
        team_cache = self.fetcher.get_teams_histories(team_ids, n=30)

        # ── Entrenamiento opcional con datos reales (TRAIN_MODELS=1) ──
        if os.environ.get("TRAIN_MODELS", "") == "1":
            from .models.training import train_and_save, train_lstm_and_save
            logger.info("TRAIN_MODELS=1: entrenando con datos reales descargados...")
            train_and_save(team_cache)
            train_lstm_and_save(team_cache)
            # Recargar modelos recién entrenados
            for model in self.models:
                if hasattr(model, "_load"):
                    model._loaded = False
                    model._model = None
                    model._load()

        for idx, match in enumerate(upcoming, 1):
            home = match.get("home_team") or {}
            away = match.get("away_team") or {}
            home_id = home.get("id")
            away_id = away.get("id")
            match_id = match.get("id")
            if not home_id or not away_id:
                continue

            logger.info(
                "[%d/%d] %s vs %s",
                idx, len(upcoming), home.get("name"), away.get("name")
            )

            home_m = team_cache.get(str(home_id), [])
            away_m = team_cache.get(str(away_id), [])

            scores = {}
            for model in self.models:
                try:
                    scores[model.name] = model.predict(home_m, away_m)
                except Exception as exc:
                    logger.warning("Modelo %s falló: %s", model.name, exc)
                    scores[model.name] = 0.0

            ensemble = self._weighted_ensemble(scores)

            # ── Bajas reales (injuries API + lineups confirmados) ──
            inj = self.injuries.analyze_injuries(
                home_id, away_id, self.client, match_id=match_id,
                home_name=home.get("name"), away_name=away.get("name"),
            )
            adjusted = self.injuries.adjust_ensemble_by_injuries(
                ensemble, inj["home_injury_impact"], inj["away_injury_impact"]
            )

            # FIX: el umbral mínimo ahora SÍ filtra
            if adjusted < min_pct:
                continue

            row = {
                "match_id": match_id,
                "utc_date": match.get("utc_date"),
                "competition_id": match.get("competition_id"),
                "home_team": home.get("name"),
                "away_team": away.get("name"),
                "home_team_id": home_id,
                "away_team_id": away_id,
                **{f"prob_{k}": v for k, v in scores.items()},
                "ensemble_pct": ensemble,
                "adjusted_ensemble_pct": adjusted,
                "home_injury_impact": inj["home_injury_impact"],
                "away_injury_impact": inj["away_injury_impact"],
                "injuries_home": len(inj["home_injuries"]),
                "injuries_away": len(inj["away_injuries"]),
                "min_edge_mode": min_edge is not None,
            }

            # ── Value betting (solo con cuotas reales) ──
            if fetch_odds and self.client is not None:
                o_yes, o_no = self._fetch_btts_odds(match)
                if o_yes and o_no:
                    row["odds_yes"], row["odds_no"] = o_yes, o_no
                    vb = find_value_bet(
                        adjusted / 100, o_yes, o_no,
                        min_edge=min_edge if min_edge is not None else 0.03,
                    )
                    if vb:
                        row.update({
                            "edge": vb["edge"],
                            "ev": vb["ev"],
                            "p_market_fair": vb["p_market_fair"],
                            "kelly_stake": vb["kelly_fraction"],
                        })
                if min_edge is not None and row.get("edge", 0) < min_edge:
                    continue

            results.append(row)

        if min_edge is not None:
            results.sort(key=lambda r: r.get("edge", 0), reverse=True)
        else:
            results.sort(
                key=lambda r: r["adjusted_ensemble_pct"], reverse=True
            )
        return results[:top_n]
