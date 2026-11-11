# src/api_client.py
"""
Cliente HTTP robusto para TheStatsAPI.
- Rate limiting proactivo (token bucket) — plan 1000 req/min, THREAD-SAFE
- Reintento con backoff exponencial + jitter (sin doble retry)
- Respeta Retry-After del servidor
- Caché local con TTL por tipo de endpoint
- Tracking de requests
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter

from src.config import API_KEY_ENV, Settings

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
BACKOFF_BASE = 2.0       # segundos
BACKOFF_MAX = 60.0       # tope de espera
JITTER = 0.2             # ±20%

# Plan actual: 1000 requests/minuto
DEFAULT_MAX_RPM = int(os.environ.get("MAX_RPM", "1000"))

# TTL de caché en segundos, por prefijo de endpoint (orden: específico → general).
CACHE_TTL: dict[str, int] = {
    "/football/teams/": 1800,          # injuries por equipo: datos cambian
    "/football/players/": 1800,
    "/football/matches/": 3600,        # detalle de partido / lineups / odds
    "/football/matches": 6 * 3600,     # listados de partidos
    "/football/teams": 24 * 3600,
    "/football/competitions": 7 * 24 * 3600,
}
DEFAULT_TTL = 3600


# ─── Rate Limiter (token bucket, thread-safe) ────────────────────────────────

class RateLimiter:
    """
    Token bucket: garantiza no superar `max_rpm` requests por minuto.
    Seguro bajo concurrencia (ThreadPoolExecutor de data_fetcher).
    """

    def __init__(self, max_rpm: int = DEFAULT_MAX_RPM):
        self.max_rpm = max(1, int(max_rpm * 0.9))
        self.min_interval = 60.0 / self.max_rpm
        self._last_request: float = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_request = time.monotonic()


# ─── Cliente ─────────────────────────────────────────────────────────────────

class StatsAPIClient:
    """
    Cliente de TheStatsAPI con:
    - Rate limiting proactivo (1000 req/min con margen de seguridad)
    - Backoff exponencial con jitter
    - Retry-After
    - Caché local con TTL (opcional)
    - Connection pool de 20 conexiones
    """

    def __init__(
        self,
        settings: Settings,
        max_rpm: int = DEFAULT_MAX_RPM,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ) -> None:
        self.api_key = settings.api_key
        if not self.api_key:
            raise EnvironmentError(
                "api_key no está definida. "
                f"Exporta {API_KEY_ENV} o pásala con --api-key / Settings"
            )
        self.base_url = settings.api_base_url.rstrip("/")

        self.rate_limiter = RateLimiter(max_rpm=max_rpm)
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir or os.environ.get("CACHE_DIR", "./cache"))
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = self._create_session()
        self._requests_made = 0

    def _create_session(self) -> requests.Session:
        """Sesión con pool de 20 conexiones y sin retry de urllib3."""
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=0,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "BTTS-Predictor/3.0 (production)",
        })
        return session

    # ─── Caché ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(endpoint: str, params: Optional[Dict]) -> str:
        raw = endpoint + json.dumps(params or {}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    @staticmethod
    def _ttl_for(endpoint: str) -> int:
        for prefix, ttl in CACHE_TTL.items():
            if endpoint.startswith(prefix):
                return ttl
        return DEFAULT_TTL

    def _cache_get(self, key: str, ttl: int) -> Optional[Dict]:
        if not self.use_cache:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > ttl:
            logger.debug("CACHE EXPIRED: %s", key)
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _cache_set(self, key: str, data: Dict) -> None:
        if not self.use_cache:
            return
        try:
            with open(self._cache_path(key), "w") as f:
                json.dump(data, f)
        except OSError as exc:
            logger.debug("No se pudo escribir caché %s: %s", key, exc)

    # ─── Normalización ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        if not endpoint:
            return "/"
        endpoint = endpoint.strip()
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        if endpoint.startswith("/api"):
            endpoint = endpoint[len("/api"):]
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return endpoint

    # ─── Core GET ────────────────────────────────────────────────────────────

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """GET con rate-limiting proactivo, backoff+jitter, Retry-After y caché."""
        normalized = self._normalize_endpoint(endpoint)
        url = f"{self.base_url}{normalized}"
        cache_enabled = self.use_cache if use_cache is None else use_cache
        ck = self._cache_key(normalized, params)
        ttl = self._ttl_for(normalized)

        # 1. Caché
        if cache_enabled:
            cached = self._cache_get(ck, ttl)
            if cached is not None:
                logger.debug("CACHE HIT: %s", normalized)
                return cached

        # 2. Bucle de reintentos (único mecanismo)
        last_exc: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            self.rate_limiter.wait()

            try:
                resp = self.session.get(
                    url, params=params or {}, timeout=REQUEST_TIMEOUT
                )
                self._requests_made += 1

                # ── 429: Rate limit ──
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        wait = float(retry_after)
                    else:
                        wait = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
                    wait *= (1 + random.uniform(-JITTER, JITTER))
                    logger.warning(
                        "429 Rate limit → esperando %.1fs (attempt %d/%d)",
                        wait, attempt, MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

                # ── 5xx: Error transitorio ──
                if resp.status_code in (500, 502, 503, 504):
                    wait = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
                    wait *= (1 + random.uniform(-JITTER, JITTER))
                    logger.warning(
                        "HTTP %d → retry en %.1fs (attempt %d/%d)",
                        resp.status_code, wait, attempt, MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

                # ── 401: Auth error → NO retry, fail fast ──
                if resp.status_code == 401:
                    logger.error(
                        "401 Unauthorized. Verifica %s. URL: %s",
                        API_KEY_ENV, url,
                    )
                    resp.raise_for_status()

                # ── Otros 4xx: No retry ──
                if 400 <= resp.status_code < 500:
                    logger.error(
                        "HTTP %d (no retryable): %s",
                        resp.status_code, resp.text[:200],
                    )
                    resp.raise_for_status()

                # ── 200: Éxito ──
                data = resp.json()

                remaining = resp.headers.get("X-RateLimit-Remaining")
                if remaining is not None:
                    logger.debug("Rate limit remaining: %s", remaining)

                if cache_enabled:
                    self._cache_set(ck, data)

                return data

            except requests.Timeout:
                last_exc = requests.Timeout(
                    f"Timeout after {REQUEST_TIMEOUT}s: {url}"
                )
                wait = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
                wait *= (1 + random.uniform(-JITTER, JITTER))
                logger.warning(
                    "Timeout → retry en %.1fs (attempt %d/%d)",
                    wait, attempt, MAX_RETRIES,
                )
                time.sleep(wait)

            except requests.ConnectionError as exc:
                last_exc = exc
                wait = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
                wait *= (1 + random.uniform(-JITTER, JITTER))
                logger.warning(
                    "ConnectionError → retry en %.1fs (attempt %d/%d)",
                    wait, attempt, MAX_RETRIES,
                )
                time.sleep(wait)

            except requests.HTTPError:
                raise

            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "RequestException: %s (attempt %d/%d)",
                    exc, attempt, MAX_RETRIES,
                )
                if attempt == MAX_RETRIES:
                    break
                time.sleep(BACKOFF_BASE * attempt)

        raise RuntimeError(
            f"Fallo definitivo en {normalized} after {MAX_RETRIES} attempts. "
            f"Last error: {last_exc}"
        )

    # ─── Paginación ──────────────────────────────────────────────────────────

    def get_all(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        per_page: int = 100,
    ) -> list[Dict[str, Any]]:
        """GET paginado: descarga todas las páginas automáticamente."""
        params = dict(params or {})
        params["per_page"] = per_page
        all_items: list[Dict] = []
        page = 1

        while True:
            params["page"] = page
            data = self.get(endpoint, params)
            items = data.get("data", [])
            all_items.extend(items)

            meta = data.get("meta", {})
            total_pages = meta.get("total_pages", 1)
            logger.debug(
                "Page %d/%d → %d items (total: %d)",
                page, total_pages, len(items), len(all_items),
            )

            if page >= total_pages:
                break
            page += 1

        return all_items

    # ─── Context manager ─────────────────────────────────────────────────────

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "StatsAPIClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def requests_made(self) -> int:
        return self._requests_made

    def __repr__(self) -> str:
        return (
            f"StatsAPIClient(requests_made={self._requests_made}, "
            f"rpm_limit={self.rate_limiter.max_rpm})"
        )
