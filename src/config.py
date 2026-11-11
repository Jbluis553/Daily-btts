# src/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

# Carga .env si existe (python-dotenv). Sin fallar si no está instalado.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

API_KEY_ENV = "THESTATSAPI_KEY"  # nombre oficial documentado


@dataclass(frozen=True)
class Settings:
    """Configuración central del proyecto."""

    date_from: str
    date_to: str
    min_pct: float = 80.0
    top_n: int = 20
    log_level: str = "INFO"
    min_edge: float = 0.0
    bankroll: float = 100.0
    api_key: str = ""
    api_base_url: str = "https://api.thestatsapi.com"

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]] = None) -> "Settings":
        payload = values or {}
        today = datetime.now(timezone.utc).date().isoformat()

        date_from = _normalize_date(payload.get("date_from"), today)
        date_to = _normalize_date(payload.get("date_to"), date_from)
        if date_from > date_to:
            raise ValueError(
                f"date_from ({date_from}) > date_to ({date_to})"
            )

        min_pct = _safe_float(
            payload.get("min_pct", os.environ.get("MIN_PCT", "80")),
            "min_pct",
        )
        if not 0 < min_pct <= 100:
            raise ValueError(f"min_pct debe estar en (0, 100], recibido: {min_pct}")

        top_n = _safe_int(
            payload.get("top_n", os.environ.get("TOP_N", "20")),
            "top_n",
        )
        if top_n < 1:
            raise ValueError(f"top_n debe ser >= 1, recibido: {top_n}")

        log_level = str(
            payload.get("log_level", os.environ.get("LOG_LEVEL", "INFO"))
        ).upper()
        if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"log_level inválido: {log_level}")

        min_edge = _safe_float(
            payload.get("min_edge", os.environ.get("MIN_EDGE", "0")),
            "min_edge",
        )

        bankroll = _safe_float(
            payload.get("bankroll", os.environ.get("BANKROLL", "100")),
            "bankroll",
        )
        if bankroll <= 0:
            raise ValueError(f"bankroll debe ser > 0, recibido: {bankroll}")

        api_key = str(
            payload.get("api_key", os.environ.get(API_KEY_ENV, ""))
        )
        api_base_url = str(
            payload.get(
                "api_base_url",
                os.environ.get("THESTATS_BASE_URL", "https://api.thestatsapi.com"),
            )
        )

        return cls(
            date_from=date_from,
            date_to=date_to,
            min_pct=min_pct,
            top_n=top_n,
            log_level=log_level,
            min_edge=min_edge,
            bankroll=bankroll,
            api_key=api_key,
            api_base_url=api_base_url,
        )


def _normalize_date(value: Optional[str], default: str) -> str:
    clean = (value or "").strip()
    if not clean:
        return default
    date.fromisoformat(clean)  # ValueError si no es YYYY-MM-DD válido
    return clean


def _safe_float(raw: Any, name: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} no es un número válido: {raw!r}") from exc


def _safe_int(raw: Any, name: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} no es un entero válido: {raw!r}") from exc
