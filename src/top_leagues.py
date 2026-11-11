# src/top_leagues.py
"""
Competiciones soportadas.

⚠️ Los competition_id NUNCA se hardcodean: son datos de la API y pueden
cambiar. Se resuelven en runtime desde GET /football/competitions
(LeagueResolver), que es la única fuente de verdad.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Solo metadatos — SIN ids inventados.
LEAGUES: Dict[str, Dict[str, Any]] = {
    # ─── LIGAS ───────────────────────────────────────────────────────────────
    "premier_league": {"name": "Premier League",       "country": "England",       "type": "league"},
    "bundesliga":     {"name": "Bundesliga",           "country": "Germany",       "type": "league"},
    "la_liga":        {"name": "La Liga",              "country": "Spain",         "type": "league"},
    "serie_a_ita":    {"name": "Serie A",              "country": "Italy",         "type": "league"},
    "ligue_1":        {"name": "Ligue 1",              "country": "France",        "type": "league"},
    "brasileirao":    {"name": "Brasileirão Série A",  "country": "Brazil",        "type": "league"},
    "jupiler_pro":    {"name": "Jupiler Pro League",   "country": "Belgium",       "type": "league"},
    "primeira_liga":  {"name": "Primeira Liga",        "country": "Portugal",      "type": "league"},
    "championship":   {"name": "Championship",         "country": "England",       "type": "league"},
    "liga_mx":        {"name": "Liga MX",              "country": "Mexico",        "type": "league"},
    "mls":            {"name": "Major League Soccer",  "country": "USA",           "type": "league"},
    "eredivisie":     {"name": "Eredivisie",           "country": "Netherlands",   "type": "league"},
    # ─── TORNEOS INTERNACIONALES ─────────────────────────────────────────────
    "uefa_champions_league": {"name": "UEFA Champions League", "country": "Europe",        "type": "tournament", "cross_league": True},
    "uefa_europa_league":    {"name": "UEFA Europa League",    "country": "Europe",        "type": "tournament", "cross_league": True},
    "copa_libertadores":     {"name": "CONMEBOL Libertadores","country": "South America", "type": "tournament", "cross_league": True},
    "copa_sudamericana":     {"name": "CONMEBOL Sudamericana","country": "South America", "type": "tournament", "cross_league": True},
}


class LeagueResolver:
    """Resuelve competition_id desde la API y filtra partidos. Thread-safe."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._ids: Optional[set] = None
        self._lock = threading.Lock()

    def _resolve_all(self) -> set:
        with self._lock:
            if self._ids is not None:
                return self._ids
            logger.info(
                "Resolviendo competition_id de %d competiciones desde la API...",
                len(LEAGUES),
            )
            resolved = set()
            try:
                data = self._client.get(
                    "/football/competitions", params={"per_page": 100}
                )
                comps = data.get("data", [])
                by_name = {
                    (c.get("name") or "").strip().lower(): c.get("id")
                    for c in comps if isinstance(c, dict)
                }
                for key, league in LEAGUES.items():
                    comp_id = by_name.get(league["name"].lower())
                    if comp_id:
                        league["competition_id"] = comp_id
                        resolved.add(comp_id)
                    else:
                        logger.warning(
                            "No se encontró '%s' en /football/competitions",
                            league["name"],
                        )
            except Exception as exc:
                logger.error("No se pudieron resolver competiciones: %s", exc)
            self._ids = resolved
            logger.info("Competiciones resueltas: %d/%d", len(resolved), len(LEAGUES))
            return resolved

    def filter_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        allowed = self._resolve_all()
        if not allowed:
            logger.warning(
                "Sin competition_id resueltos: se devuelven TODOS los partidos "
                "(desactiva el filtro para no perder datos)."
            )
            return list(matches)
        return [m for m in matches if m.get("competition_id") in allowed]


def get_competition_id(key: str, client: Any = None) -> str:
    """Resuelve el competition_id de una competición (desde la API si hace falta)."""
    league = LEAGUES.get(key)
    if league is None:
        raise KeyError(
            f"Competition '{key}' not found. Available: {list(LEAGUES.keys())}"
        )
    if league.get("competition_id"):
        return league["competition_id"]
    if client is None:
        raise RuntimeError(
            f"'{key}' requiere lookup: pasa un StatsAPIClient para resolverlo"
        )
    LeagueResolver(client)._resolve_all()  # resuelve y cachea en LEAGUES
    comp_id = league.get("competition_id")
    if not comp_id:
        raise RuntimeError(f"Could not resolve competition_id for '{league['name']}'")
    return comp_id


def get_all_keys(competition_type: str = None) -> list:
    """Devuelve todas las keys, filtradas por tipo si se especifica."""
    if competition_type is None:
        return list(LEAGUES.keys())
    return [k for k, v in LEAGUES.items() if v["type"] == competition_type]


def get_cross_league_keys() -> list:
    """Competiciones con equipos de múltiples ligas (torneos internacionales)."""
    return [k for k, v in LEAGUES.items() if v.get("cross_league", False)]
