# CLAUDE.md — Pipeline de churn bancario (DMEyF 2026, UBA)

Pipeline de predicción de bajas (BAJA+2) alineado con el repositorio de la cátedra
`dmecoyfin/dmeyf2026`, en particular `ensembles/z494_TareaHogar_04.ipynb` (flujo completo),
`monday/z101` (clase), `z402` (FE en SQL), `z601` (clases y cortes), `z301` (incertidumbre,
público/privado) y `z701` (drift). Ante una duda de método, manda lo que hace la cátedra.

## 1. Métrica
Ganancia = Σ sobre los enviados de (+$1.072.500 si es BAJA+2, −$27.500 si no).
El modelo aprende clase01 = BAJA+1 + BAJA+2 (z494), por eso **el corte es por cantidad de envíos (k)**,
no por probabilidad 1/40. k* se elige en el mes de test sobre la curva de ganancia del semillerío.

## 2. Meses (config/experimento.yaml → `meses`)
| Rol | Default | Regla |
|---|---|---|
| train | 202102–202104 | Optuna (CV) y modelo de testeo |
| test | 202106 | Se mira una vez: curva de ganancia, público/privado simulado y k* |
| train_final | 202104–202106 | Misma ventana corrida dos meses |
| future | 202108 | Mes de la entrega |
Entre el último mes de entrenamiento y el mes predicho queda **un mes sin usar** (z494: 202104 → 202106),
igual que en la entrega (junio → agosto; julio no tiene clase completa).

## 3. Reglas que no se rompen
- **Nada de información del futuro**: en FE solo `LAG` y ventanas hacia atrás. `LEAD` solo para armar la clase (01).
- **El mes de test no se usa para elegir hiperparámetros.** Optuna optimiza con CV dentro de `meses.train`.
- `numero_de_cliente`, `foto_mes` y las columnas de clase nunca son variables (`src/config.py: NO_FEATURES`).
- Ningún modelo final con una sola semilla: semillerío de semillas primas de 6 dígitos.
- El drift **no se purga en automático**: 03 genera `drift_psi.csv`; las variables a sacar se ponen a mano en `fe.excluir`.
- Todos los parámetros viven en `config/experimento.yaml`. Nada de meses ni rutas hardcodeados en `src/`.
- Cambiar `experimento` en cada corrida nueva: todo se escribe en `data/exp/exp<N>/`.

## 4. Feature engineering (paso 3)
1. **Variables de dominio bancario** (`VARIABLES_BANCARIAS` en `src/03_feature_engineering_drift.py`), cada una con su
   explicación: tarjetas Visa+Master consolidadas (deuda, consumo, límite, uso del límite, pago sobre saldo, peor estado,
   mora, tarjetas activas, sin Visa), liquidez y deuda (inversiones, préstamos, deuda de tarjeta sobre saldos, pasivos/activos),
   sueldo (total, cobra sueldo, saldo sobre sueldo, transferencias netas), canales (digital vs atención, débitos automáticos,
   pago de servicios) y vínculo (seguros, rentabilidad y comisiones por producto, edad al alta).
   Para agregar una: sumar una tupla (nombre, SQL, columnas requeridas, descripción) a la lista.
2. **Historia por cliente** (z402) sobre todas las variables, incluidas las bancarias: `fe.lags`, `fe.deltas`, y media y
   pendiente de 3 meses para `fe.ventana_variables`. Sin historia previa, el valor queda vacío (LightGBM lo maneja).
3. **Drift** (z701): PSI de `train_final` vs `future`. No se borra nada solo: las decisiones van a `fe.excluir`.
Los nulos NO se imputan: LightGBM los maneja y muchas veces son información (ej. sin tarjeta Master).
`numero_de_cliente` se fuerza a entero en el paso 1 (el CSV trae algún id en notación científica).

## 5. Pipeline
| Paso | Script | Qué hace | Salida |
|---|---|---|---|
| 1 | `src/01_data_ingestion.py` | Calcula `clase_ternaria` desde el crudo (z101/z494) en DuckDB | `data/dataset_con_clase.parquet` |
| 2 | `src/02_eda_financial_diagnosis.py` | EDA: transacciones, Visa_status, saldos, nulos y ceros por mes | `data/eda_financial_summary.json` |
| 3 | `src/03_feature_engineering_drift.py` | Variables bancarias (29), luego lags, deltas, media y pendiente de 3 meses sobre todas; rank por mes opcional; reporte PSI train_final vs future | `fe.parquet`, `drift_psi.csv` |
| 4 | `src/04_optuna_tuning.py` | Optuna con undersampling de CONTINUA y CV de AUC (z494) | `optuna_mejores.json`, `db/optuna.db` |
| 5 | `src/05_semillerio_corte.py` | Semillerío; fase test (curva, público/privado, k*) y fase final (entrega: ids de los k* elegidos, una columna sin encabezado) | `ganancia_test.csv`, `resumen_05.json`, `entrega_exp<N>_k<k>.csv` |
| 6 | `src/06_business_shap_mlops.py` | TreeSHAP sobre un modelo final, variables con dirección del efecto, MLflow | `reports/`, `db/mlflow.db` |

Fuera del flujo principal: `src/experimental/05_b_multi_model_ensemble.py` (LGBM+XGB+RF).
**No respeta el protocolo de meses** (predice y optimiza sobre el mismo mes). No usar para entregar sin adaptarlo.
z501 (clusters con RF) es una herramienta de perfilado de BAJA+2: se corre aparte con el script de la cátedra.

## 6. Cómo correr y cómo validar un cambio
- Entorno: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Pipeline real: `bash run_pipeline.sh` (corridas largas con `nohup` o `tmux`, nunca desde una pestaña del navegador).
- **Test de humo, obligatorio después de cualquier cambio en `src/`**: `bash tests/smoke_test.sh`.
  Corre 01→06 sobre un crudo sintético (`tests/make_synthetic_data.py`, config `tests/smoke_config.yaml`) en ~30 s.
  Un cambio no está terminado hasta que termina con `SMOKE TEST OK`. Los datos sintéticos no sirven para evaluar modelos.

## 7. Pendiente / a confirmar
- Confirmar nombre del crudo y meses disponibles del dataset de la competencia 2 (`datos.crudo`, `meses`).
- Revisar `drift_psi.csv` con datos reales y decidir `fe.excluir` y `fe.rank_por_mes` (z701, z402).
- Lag 6 y features adicionales: agregar en `fe.lags` según memoria disponible (lags de todas las variables multiplican columnas).
- Comparar target clase01 vs BAJA+2 puro (`target.positivos`) con dos experimentos.

## 8. Estructura
```
.
├── CLAUDE.md
├── config/experimento.yaml       # único lugar de parámetros
├── run_pipeline.sh
├── requirements.txt
├── data/                         # datos y experimentos (ignorado en Git)
├── db/                           # Optuna y MLflow en SQLite (ignorado en Git)
├── src/
│   ├── config.py                 # carga de config, ganancia, semillas primas
│   ├── 01_data_ingestion.py … 06_business_shap_mlops.py
│   └── experimental/05_b_multi_model_ensemble.py
└── tests/
    ├── make_synthetic_data.py
    ├── smoke_config.yaml
    └── smoke_test.sh
```
