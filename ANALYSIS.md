# Auditoría y mejoras del repositorio Probabilidad_btts

Fecha de auditoría: 2026-10-22

## Hallazgos corregidos

### Datos ficticios eliminados
1. **`top_leagues.py`**: todos los `competition_id` estaban hardcodeados con
   valores inventados (`comp_3039`, `comp_0256`, `comp_298265`...). Ahora se
   resuelven en runtime desde `GET /football/competitions` (`LeagueResolver`).
2. **`models/xgboost_model.py`, `catboost_model.py`, `lstm_momentum.py`**:
   el "entrenamiento" generaba datos aleatorios sintéticos
   (`rng.normal(...)`, `base_rate=0.55`). Los modelos no aprendían nada real.
   Ahora solo predicen si existe un modelo entrenado con datos reales
   (`src/models/trained/`, generado con `TRAIN_MODELS=1`). El dataset de
   entrenamiento se construye sin data leakage (cutoff temporal por partido).
   Sin modelo entrenado, usan un fallback determinístico honesto y lo reportan.
3. **`ANALYSIS.md` anterior / README**: el ejemplo numérico ("Barcelona 82.5%")
   era salida ilustrativa presentada como real. Ahora se marca como ilustrativo.

### Errores de lógica corregidos
4. **Poisson bivariada**: `_team_lambda` no distinguía local/visitante → el λ
   del visitante se calculaba con perspectiva de local. Corregido pasando el
   `team_id` real.
5. **`analyzer.analyze`**: el parámetro `min_pct` existía pero nunca filtraba.
   Ahora descarta partidos con `adjusted < min_pct`.
6. **LSTM**: `np.stack([sh, sa]).reshape(1, seq_len, 3)` lanzaba ValueError con
   TF instalado (dimensiones incompatibles). Input correcto de 6 canales.
7. **Rate limiter**: `_last_request` sin lock → inseguro con 20 workers.
   Ahora usa `threading.Lock`.
8. **`backtest.py`**: importaba `fetch_all_matches`, `fetch_match_odds` y
   `evaluate_ensemble`, que no existen en el proyecto (ImportError seguro).
   Eliminado; su funcionalidad puede reconstruirse sobre componentes reales.

### Inconsistencias corregidas
9. Variable de API key: el código leía `THESTATS_API_KEY` pero README/workflow
   usaban `THESTATSAPI_KEY`. Unificado en `THESTATSAPI_KEY` (nombre documentado).
10. Defaults de `MIN_PCT`: código decía 60, README/env decían 80. Unificado en 80.
11. Dos archivos de entorno contradictorios (`.env.example` / `env.example`).
    Unificado en `.env.example` único.
12. `.env` nunca se cargaba pese a listar `python-dotenv` en requirements.
    Ahora `config.py` hace `load_dotenv()`.
13. Endpoint de odds inventado (`/odds/matches/{id}/btts`) → path configurable
    bajo `/football/` con chequeo de `odds_available` antes de consultar.
14. "Bajas" solo por heurística sobre lineups → ahora fuente primaria es el
    endpoint real de injuries (configurable) y lineups son complemento.
15. Caché: injuries compartían TTL de 24h con datos de equipo → TTL específico
    de 30 min para rutas `/football/teams/` y `/football/players/`.

## Arquitectura resultante

```
Usuario → cli.py → config.py (.env validado)
  → StatsAPIClient (rate limit thread-safe, backoff+jitter, Retry-After, caché TTL)
  → DataFetcher (paginación + historiales paralelos)
  → LeagueResolver (competition_id reales desde la API)
  → BTTSAnalyzer (5 modelos → ensemble ponderado → ajuste por bajas → filtro)
  → exporter / results_tracker
```

## Cobertura de tests

`tests/` (sin red, sin API key):
- config: defaults coherentes, validación de rangos
- utils: formatos de score soportados
- models: perspectiva local/visitante en features y Poisson, determinismo
  del fallback, dataset sin leakage
- top_leagues: ids resueltos desde API (mock), fallback ante fallo de red
- value_betting: vig removal, EV, Kelly
- data_fetcher: contrato de fechas, marca `_team_id` en historiales
