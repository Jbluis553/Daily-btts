# src/exporter.py
from __future__ import annotations

import csv
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def print_table(results: List[Dict[str, Any]], min_pct: float) -> None:
    """Imprime la tabla final de resultados con impacto de bajas."""
    if not results:
        print(f"\nNo hay partidos que cumplan el criterio (min {min_pct}%).\n")
        return

    headers = [
        "#", "Fecha UTC", "Local", "Visitante", "Hist%", "Pois%", "LSTM%",
        "XGB%", "Cat%", "Ens%", "Ajust%", "Chg", "Bajas L/V",
        "Odds", "Edge%", "EV", "Kelly",
    ]

    table = []
    for idx, r in enumerate(results, 1):
        ensemble = r.get("ensemble_pct", 0)
        adjusted = r.get("adjusted_ensemble_pct", ensemble)
        adjustment = "↓" if adjusted < ensemble else "="
        table.append([
            idx,
            (r.get("utc_date") or "")[:16].replace("T", " "),
            r.get("home_team", ""),
            r.get("away_team", ""),
            f"{r.get('prob_historical', 0):.1f}",
            f"{r.get('prob_bivariate_poisson', 0):.1f}",
            f"{r.get('prob_lstm_momentum', 0):.1f}",
            f"{r.get('prob_xgboost', 0):.1f}",
            f"{r.get('prob_catboost', 0):.1f}",
            f"{ensemble:.1f}",
            f"{adjusted:.1f}",
            adjustment,
            f"{r.get('injuries_home', 0)}/{r.get('injuries_away', 0)}",
            r.get("odds_yes", ""),
            f"{r.get('edge', 0) * 100:.1f}" if r.get("edge") else "",
            r.get("ev", ""),
            r.get("kelly_stake", ""),
        ])

    widths = [
        max(len(str(x)) for x in [h] + [row[i] for row in table])
        for i, h in enumerate(headers)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print()
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in table:
        print(fmt.format(*[str(c) for c in row]))
    print()


def export_csv(
    results: List[Dict[str, Any]], date_from: str, date_to: str
) -> str:
    """Exporta resultados al CSV incluyendo datos de bajas y value."""
    if not results:
        logger.warning("No hay resultados para exportar a CSV.")
        return ""

    filename = f"btts_{date_from}_to_{date_to}.csv"
    fieldnames = list(results[0].keys())

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    logger.info("CSV exportado → %s", filename)
    return filename
