# src/injuries.py
"""Análisis de bajas y su impacto en las predicciones BTTS.

Fuente primaria: endpoint de injuries de TheStatsAPI
(configurable con INJURIES_ENDPOINT, por defecto
/football/teams/{team_id}/injuries). Fuente complementaria:
ausencias detectadas en lineups confirmados. Si ambas fallan,
impacto = 0 (se reporta, no se finge).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .lineups import LineupAnalyzer

logger = logging.getLogger(__name__)

# Ruta configurable por si la API cambia el endpoint de injuries.
INJURIES_ENDPOINT = os.environ.get(
    "INJURIES_ENDPOINT", "/football/teams/{team_id}/injuries"
)

# Estados que cuentan como baja confirmada
ABSENT_STATUSES = {"out", "injured", "injury", "suspended", "doubtful",
                   "not_available", "unavailable"}


def _impact_from_position(position: Optional[str]) -> float:
    """Impacto por posición (misma regla que LineupAnalyzer)."""
    from .lineups import LineupAnalyzer as _LA
    return _LA._position_impact(position)


class InjuryImpactAnalyzer:
    """Analiza el impacto de bajas con datos reales de la API."""

    def _fetch_team_injuries(self, client: Any, team_id: Any) -> list:
        """Bajas reales reportadas por la API para un equipo."""
        try:
            data = client.get(INJURIES_ENDPOINT.format(team_id=team_id))
        except Exception as exc:
            logger.debug("Sin endpoint de injuries para %s: %s", team_id, exc)
            return []
        items = data.get("data") or data.get("injuries") or []
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            status = str(it.get("status") or it.get("availability") or "").lower()
            if status and status not in ABSENT_STATUSES:
                continue  # solo bajas/dudas confirmadas
            out.append({
                "team": it.get("team") or it.get("team_name") or "",
                "player": it.get("player") or it.get("name") or it.get("full_name") or "",
                "position": it.get("position") or it.get("role") or "",
                "status": status or "unknown",
                "reason": it.get("reason") or it.get("type") or status or "",
                "source": "api_injuries",
            })
        return [i for i in out if i["player"]]

    def analyze_injuries(
        self,
        home_id: str,
        away_id: str,
        client: Any,
        match_id: Optional[int | str] = None,
        home_name: Optional[str] = None,
        away_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Combina bajas reales de la API con ausencias de lineups confirmados.

        Hace como máximo: 2 calls de injuries (home/away) + 1 de lineups
        (compartida entre ambos). Cualquier fallo parcial degrada a 0
        ese componente, nunca inventa datos.
        """
        empty = {
            "home_injuries": [],
            "away_injuries": [],
            "home_injury_impact": 0.0,
            "away_injury_impact": 0.0,
            "combined_injury_severity": 0.0,
        }
        if client is None:
            return empty

        home_inj = self._fetch_team_injuries(client, home_id)
        away_inj = self._fetch_team_injuries(client, away_id)

        # Lineups confirmados como señal complementaria (1 solo fetch)
        if match_id is not None:
            try:
                raw = LineupAnalyzer.fetch_lineups(client, match_id)
                if raw:
                    home_l = LineupAnalyzer.analyze_match_lineups(
                        client, match_id, home_name, raw_payload=raw
                    )
                    away_l = LineupAnalyzer.analyze_match_lineups(
                        client, match_id, away_name, raw_payload=raw
                    )
                    # Unir con dedup por jugador
                    home_seen = {i["player"].lower() for i in home_inj}
                    away_seen = {i["player"].lower() for i in away_inj}
                    for a in home_l["absences"]:
                        if a.get("player", "").lower() not in home_seen:
                            a["source"] = "lineup"
                            home_inj.append(a)
                    for a in away_l["absences"]:
                        if a.get("player", "").lower() not in away_seen:
                            a["source"] = "lineup"
                            away_inj.append(a)
            except Exception as exc:
                logger.debug("Lineups no disponibles para %s: %s", match_id, exc)

        home_impact = self._aggregate_impact(home_inj)
        away_impact = self._aggregate_impact(away_inj)

        return {
            "home_injuries": home_inj,
            "away_injuries": away_inj,
            "home_injury_impact": home_impact,
            "away_injury_impact": away_impact,
            "combined_injury_severity": round(min(home_impact + away_impact, 0.90), 2),
        }

    @staticmethod
    def _aggregate_impact(injuries: list) -> float:
        if not injuries:
            return 0.0
        impacts = [_impact_from_position(i.get("position")) for i in injuries]
        impacts = [i for i in impacts if i > 0]
        if not impacts:
            return 0.0
        return round(min(sum(impacts) / len(impacts), 0.90), 2)

    @staticmethod
    def adjust_ensemble_by_injuries(
        ensemble_score: float,
        home_injury_impact: float,
        away_injury_impact: float,
    ) -> float:
        """Ajusta el score del ensemble según bajas confirmadas."""
        if home_injury_impact == 0.0 and away_injury_impact == 0.0:
            return ensemble_score

        if home_injury_impact > 0.5:
            ensemble_score *= 1.0 - (home_injury_impact * 0.20)
        if away_injury_impact > 0.5:
            ensemble_score *= 1.0 - (away_injury_impact * 0.20)

        return round(max(0.0, min(100.0, ensemble_score)), 2)
