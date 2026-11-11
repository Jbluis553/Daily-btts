# src/cli.py
from __future__ import annotations

import argparse
import logging
import os
from typing import Sequence

from .analyzer import BTTSAnalyzer
from .api_client import StatsAPIClient
from .config import API_KEY_ENV, Settings
from .data_fetcher import DataFetcher
from .exporter import export_csv, print_table

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parsea los argumentos de la CLI."""
    parser = argparse.ArgumentParser(
        description="BTTS Multi-Model Predictor – TheStatsAPI"
    )
    parser.add_argument("--date-from", default=os.environ.get("DATE_FROM", ""))
    parser.add_argument("--date-to", default=os.environ.get("DATE_TO", ""))
    parser.add_argument(
        "--min-pct", type=float,
        default=float(os.environ.get("MIN_PCT", "80")),
    )
    parser.add_argument(
        "--top-n", type=int,
        default=int(os.environ.get("TOP_N", "20")),
    )
    parser.add_argument(
        "--min-edge", type=float,
        default=float(os.environ.get("MIN_EDGE", "0")),
        help="Edge mínimo vs mercado (0 = desactivado, modo umbral clásico)",
    )
    parser.add_argument(
        "--bankroll", type=float,
        default=float(os.environ.get("BANKROLL", "100")),
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get(API_KEY_ENV, ""),
        help=f"API key de TheStatsAPI (o env {API_KEY_ENV})",
    )
    parser.add_argument(
        "--settle", action="store_true",
        help="Liquida predicciones pendientes y reporta ROI",
    )
    parser.add_argument(
        "--no-odds", action="store_true",
        help="No consultar cuotas",
    )
    return parser.parse_args(argv)


def cli(argv: Sequence[str] | None = None) -> int:
    """Ejecuta el flujo principal de predicción con análisis de bajas."""
    args = parse_args(argv)

    settings = Settings.from_dict(
        {
            "date_from": args.date_from,
            "date_to": args.date_to,
            "min_pct": args.min_pct,
            "top_n": args.top_n,
            "log_level": args.log_level,
            "min_edge": args.min_edge,
            "bankroll": args.bankroll,
            "api_key": args.api_key,
        }
    )

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== BTTS Multi-Model Predictor (TheStatsAPI) ===")
    logger.info("Rango de fechas : %s → %s", settings.date_from, settings.date_to)
    logger.info("Umbral mínimo   : %.1f%%", settings.min_pct)
    logger.info("Top N           : %d", settings.top_n)
    if settings.min_edge > 0:
        logger.info("Modo value bet  : edge >= %.2f", settings.min_edge)
    if os.environ.get("TRAIN_MODELS", "") == "1":
        logger.info("Entrenamiento   : ACTIVO (datos reales, sin leakage)")

    try:
        with StatsAPIClient(settings) as client:
            fetcher = DataFetcher(client)

            if args.settle:
                from .results_tracker import settle

                settle(fetcher)
                return 0

            analyzer = BTTSAnalyzer(fetcher, client=client)

            upcoming = fetcher.get_upcoming_matches(
                date_from=settings.date_from,
                date_to=settings.date_to,
            )

            if not upcoming:
                logger.warning("No se encontraron partidos en el rango indicado.")
                return 0

            top_results = analyzer.analyze(
                upcoming,
                min_pct=settings.min_pct,
                top_n=settings.top_n,
                min_edge=settings.min_edge if settings.min_edge > 0 else None,
                fetch_odds=not args.no_odds,
            )

        print_table(top_results, settings.min_pct)
        export_csv(top_results, settings.date_from, settings.date_to)

        from .results_tracker import log_predictions

        log_predictions(top_results)
        logger.info("Proceso terminado correctamente.")
        return 0
    except Exception as exc:  # pragma: no cover
        logger.exception("Error fatal: %s", exc)
        return 1
