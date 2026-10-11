# agente_churn — Predicción de bajas BAJA+2 (DMEyF 2026, UBA)

Pipeline en Python para la competencia de churn bancario de **Data Mining en Economía y Finanzas**,
alineado con el repositorio de la cátedra (`dmecoyfin/dmeyf2026`, en particular `z494_TareaHogar_04`).
Desarrollado con asistencia de IA (Claude) como experimento: ¿puede la IA hacer lo que pide la materia?

## Versión `verscla02`

Referencia cargada en el bot de entregas. Corresponde al experimento **7001**.

| Decisión | Valor |
|---|---|
| Dataset | `competencia_01_crudo.csv` (202103 a 202108), clase calculada como en z101/z494 |
| Meses | train 202104 → test 202106 · train_final 202106 → future 202108 (un mes de hueco, como z494) |
| Target | clase01 = BAJA+1 + BAJA+2; corte por cantidad de envíos |
| Variables | 154 originales + 29 bancarias + lags/deltas 1-3 + media y pendiente de 3 meses → 1.256 columnas |
| Drift (z701) | Excluidas: `Visa/Master_fultimo_cierre`, `Visa/Master_Finiciomora`, deltas de `cpayroll_trx`. Sueldo (`mpayroll`, `mpayroll2`, `payroll_total`) como posición dentro del mes para neutralizar el aguinaldo de junio |
| Optuna | 50 trials, CV 5 folds sobre AUC, undersampling 10% de CONTINUA. Mejor: lr 0,0164 · feature_fraction 0,107 · num_leaves 13 · min_data_in_leaf 53 · 659 árboles (AUC CV 0,931) |
| Semillerío | 20 semillas primas de 6 dígitos |

**Resultado en el mes de test (202106):** ganancia del semillerío **$404 M con 14.500 envíos**
(meseta de ~$400 M entre 13.500 y 16.000 envíos; una semilla suelta: $395 M ± $8 M).
Simulación público/privado 30/70: $444 M / $387 M.

**Entrega:** 14.500 `numero_de_cliente` de 202108, una columna sin encabezado
(`data/exp/exp7001/entrega_exp7001_k14500.csv`, no se sube al repo).

## Cómo reproducirlo

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # en Mac además: brew install libomp
bash tests/smoke_test.sh                 # prueba con datos sintéticos (~1 min)
# poner la ruta del crudo en config/experimento.yaml (datos.crudo)
nohup caffeinate -i bash run_pipeline.sh > data/pipeline.log 2>&1 &
```

Todos los parámetros y decisiones están en `config/experimento.yaml`, comentados.
El detalle del método y las reglas está en `CLAUDE.md`.

## Estructura

```
config/experimento.yaml   parámetros y decisiones del experimento
src/01 … 06               ingesta+clase · EDA · variables+drift · Optuna · semillerío+entrega · SHAP+MLflow
src/config.py             ganancia, semillas primas, utilidades
src/experimental/         ensamble LGBM+XGB+RF (fuera del flujo principal)
tests/                    dataset sintético y test de humo
```
