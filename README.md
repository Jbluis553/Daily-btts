# BTTS Multi-Model Predictor – TheStatsAPI

Predictor de Both Teams To Score (BTTS) basado en 5 modelos, ensemble
ponderado y análisis de bajas con **datos reales de TheStatsAPI**.

## Qué hace

- Consulta partidos futuros con `date_from` y `date_to` desde TheStatsAPI
- **Resuelve los `competition_id` en runtime desde `/football/competitions`** (nunca hardcodeados)
- Obtiene historial reciente por equipo (últimos 30 partidos, en paralelo)
- Obtiene **bajas reales** desde el endpoint de injuries (configurable, por defecto `/football/teams/{id}/injuries`), complementadas con ausencias en lineups confirmados
- Evalúa 5 modelos de BTTS:
  - **Historical**: tasa BTTS de últimos 10 partidos
  - **Bivariate Poisson**: λ de ataque/defensa con perspectiva local/visitante correcta
  - **LSTM Momentum**: solo si existe un modelo entrenado con datos reales (`TRAIN_MODELS=1` + tensorflow); si no, media móvil BTTS
  - **XGBoost / CatBoost**: solo si existen modelos entrenados con datos reales; si no, fallback determinístico basado en estadística real
- Calcula ensemble (ponderado con `MODEL_WEIGHTS` o promedio simple)
- **Ajusta el ensemble según bajas** (delanteros/extremos con impacto > 0.5)
- **Filtra por `MIN_PCT`** y ordena por score ajustado
- Exporta a CSV e imprime tabla

> ⚠️ **Sin datos ficticios**: los modelos ML nunca se entrenan con datos
> sintéticos. Sin modelo entrenado real, usan un fallback determinístico
> y lo reportan en los logs.

## Requisitos

- Python 3.11+
- Clave de TheStatsAPI en la variable **`THESTATSAPI_KEY`**

## Configuración

```bash
cp .env.example .env
# editar .env con tu THESTATSAPI_KEY
```

Variables soportadas (ver `.env.example`): `THESTATSAPI_KEY`, `DATE_FROM`,
`DATE_TO`, `MIN_PCT` (default 80), `TOP_N` (default 20), `MIN_EDGE`,
`BANKROLL`, `MODEL_WEIGHTS`, `MAX_RPM`, `HISTORY_WORKERS`, `TRAIN_MODELS`,
`INJURIES_ENDPOINT`, `ODDS_ENDPOINT`, `LOG_LEVEL`.

## Ejecución

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --date-from 2026-10-22 --date-to 2026-10-22 --min-pct 80 --top-n 20
```

### Entrenar los modelos ML con datos reales

```bash
TRAIN_MODELS=1 python main.py --date-from hoy --date-to hoy
```

Se construye un dataset **sin data leakage** (features solo con partidos
anteriores a cada partido etiquetado) y se persisten en
`src/models/trained/`. Si hay menos de 200 muestras reales, se omite el
entrenamiento y los modelos usan su fallback honesto.

### Liquidar predicciones y ver ROI

```bash
python main.py --settle
```

## Salida (ejemplo ilustrativo — los números dependen de datos reales del día)

```
#  Fecha UTC         Local    Visitante  Hist%  Pois%  LSTM%  XGB%  Cat%  Ens%  Ajust%  Chg  Bajas (L/V)
1  2026-10-22 15:00  EquipoA  EquipoB    80.0   78.4   75.0   79.1  77.8  78.1  78.1    =    0/0
```

- `Ens%`: ensemble puro · `Ajust%`: ensemble ajustado por bajas
- `Chg`: `↓` (reducido por bajas) / `=` (sin cambio) · `Bajas (L/V)`: bajas relevantes

## GitHub Actions

`.github/workflows/daily_btts.yml` corre diario a las 06:00 UTC (o manual
con parámetros). Requiere el secret **`THESTATSAPI_KEY`**.

## Tests

```bash
pytest tests/
```

## Estructura

```
src/
  cli.py / main.py    Entrada y flujo principal
  config.py           Configuración (lee .env, valida rangos)
  api_client.py       HTTP con reintentos, rate limiter thread-safe y caché TTL
  data_fetcher.py     Partidos + historiales en paralelo
  top_leagues.py      Competiciones (ids resueltos desde la API en runtime)
  analyzer.py         Orquestador de modelos + bajas + value betting
  injuries.py         Bajas reales (API) + lineups, ajuste del ensemble
  lineups.py          Extracción de ausencias en alineaciones confirmadas
  exporter.py         CSV + tabla de consola
  models/             5 modelos BTTS + training.py (dataset real sin leakage)
  results_tracker.py  Log de predicciones y liquidación (ROI/yield)
  value_betting.py    EV, vig removal, Kelly fraccional
tests/                Tests unitarios (sin red)
```

## Notas de esta versión (mejoras sobre la anterior)

- Eliminados todos los `competition_id` ficticios → resolución dinámica.
- Eliminado el entrenamiento sintético de XGBoost/CatBoost/LSTM.
- Corregido el λ de Poisson del visitante (perspectiva local/visitante).
- Corregido el `reshape` inválido del LSTM y el rate limiter no thread-safe.
- El umbral `MIN_PCT` ahora filtra de verdad.
- Variable de entorno unificada: `THESTATSAPI_KEY`.
- Un solo `.env.example` (eliminado el duplicado contradictorio).
- Eliminado `src/backtest.py` (importaba funciones inexistentes → ImportError).
  Un backtest correcto puede reconstruirse sobre `DataFetcher` + `training.build_dataset`.
