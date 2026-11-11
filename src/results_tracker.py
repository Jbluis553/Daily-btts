# src/results_tracker.py
"""Tracking de predicciones vs resultados reales: ROI, yield, hit-rate."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .utils import is_btts

logger = logging.getLogger(__name__)

DB = Path("predictions_log.jsonl")


def log_predictions(rows: List[Dict[str, Any]]) -> None:
    """Registra predicciones del día para liquidarlas después."""
    if not rows:
        return
    with DB.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "match_id": r["match_id"],
                "utc_date": r["utc_date"],
                "home": r["home_team"],
                "away": r["away_team"],
                "p": r["adjusted_ensemble_pct"] / 100,
                "odds": r.get("odds_yes"),
                "kelly": r.get("kelly_stake"),
                "edge": r.get("edge"),
                "min_edge_mode": r.get("min_edge_mode", False),
                "settled": False,
            }) + "\n")
    logger.info("Predicciones registradas: %d", len(rows))


def settle(fetcher) -> None:
    """Liquida predicciones pendientes contra resultados reales."""
    if not DB.exists():
        logger.info("No hay log de predicciones aún.")
        return

    lines = [json.loads(l) for l in DB.read_text().splitlines() if l.strip()]
    pend = [
        r for r in lines
        if not r["settled"] and r.get("odds") and r.get("match_id")
    ]
    if not pend:
        logger.info("No hay predicciones pendientes de liquidar.")
        return

    for r in pend:
        try:
            m = fetcher.get_match(r["match_id"])
            if m.get("status") != "finished":
                continue
            r["settled"] = True
            r["hit"] = is_btts(m)
            stake = r.get("kelly") if r.get("kelly") else 1.0
            r["pnl"] = stake * (r["odds"] - 1) if r["hit"] else -stake
        except Exception as exc:
            logger.debug("No se pudo liquidar %s: %s", r.get("match_id"), exc)

    DB.write_text("\n".join(json.dumps(r) for r in lines) + "\n")

    done = [r for r in lines if r.get("settled") and r.get("pnl") is not None]
    if done:
        pnl = sum(r["pnl"] for r in done)
        staked = sum(
            (r.get("kelly") if r.get("kelly") else 1.0) for r in done
        )
        hits = sum(1 for r in done if r["hit"])
        logger.info(
            "=== TRACKER === picks:%d | hit-rate:%.1f%% | P/L:%+.2fu | "
            "yield:%+.1f%%",
            len(done),
            100 * hits / len(done),
            pnl,
            100 * pnl / staked if staked else 0.0,
        )
