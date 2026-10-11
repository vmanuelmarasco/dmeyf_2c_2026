# Notebooks del pipeline paso a paso

## `competencia_v5.ipynb` — versión `verscla03`

Pipeline armado celda por celda a partir de los notebooks de la cátedra (z101, z402, z601), con las
salidas de la corrida que generó la entrega `verscla03` (ganancia pública **94,820**, superior a
`verscla02` con 94,325).

| Paso | Celdas | Qué hace |
|---|---|---|
| Clase | 1–4 | Carga del crudo (ids forzados a entero) y clasificador propio de z101 (BAJA+1 / BAJA+2 / CONTINUA) |
| Variables | 5–7 | Historia por cliente (lags, deltas, promedios de 3 meses) y señales de desvinculación respaldadas por el EDA: caída de actividad, caída de consumo, baja de productos, dejó de cobrar el sueldo (reciente), pocos productos, préstamo terminado o por terminar, índice de desvinculación |
| Datos | 8–9 | Meses: entrena 202104, mide 202106, entrega 202108. Fuera del modelo: ids, monto de sueldo (aguinaldo) y fechas con drift |
| Optuna | 10–13 | Código de z601: CV con StratifiedShuffleSplit y ganancia como métrica, 30 trials |
| Sobreajuste | 14 | Abril visto $994 M · CV $514 M · junio $342 M (umbral 0,025, una semilla) |
| Semillerío | 15 | 5 semillas propias, curva de ganancia por envíos en junio |
| Targets | 16 | BAJA+2 vs resto: $380,6 M · **BAJA+1+2 vs CONTINUA: $412,0 M (14.500 envíos)** · ensamble por ranking: $403,2 M |
| Entrega | 17 | Modelo final entrenado en 202106 con BAJA+1+2, semillerío de 5, top 14.500 de 202108, una columna sin encabezado |

Con 175 variables elegidas por razones de negocio supera en junio a `verscla02` ($404 M con ~1.250 variables).

## Versión `verscla04` — ensamble con un modelo con undersampling

Celdas 18–21 agregadas a `competencia_v5.ipynb` sobre la base de `verscla03`.

| Paso | Celdas | Qué hace |
|---|---|---|
| Optuna con undersampling | 18 | 10% de CONTINUA + todas las bajas, target BAJA+1+2, CV estratificada de 5 folds con la ganancia como métrica (pesos de CONTINUA ajustados por el undersampling), 100 trials. Mejor CV: $569 M (lr 0,069, 35 hojas, min_data_in_leaf 53, feature_fraction 0,48, 76 árboles). Es una cifra optimista: elige el mejor corte en cada fold |
| Semillerío | 19 | 5 semillas; cada una sortea otro 10% de CONTINUA |
| Comparación en junio | 20 | verscla03 **$412,0 M** (14.500) · undersampling $389,4 M (15.000) · ensamble por ranking $405,4 M (14.500) |
| Entrega | 21 | Ensamble por ranking de verscla03 + undersampling, entrenado en 202106, top 14.500 de 202108 → `entregas/mi_pipeline_ens_us_k14500.csv` |

**Bot (público, ~25% de agosto):** verscla04 95,1225 contra verscla03 94,8200.

**Lectura:** la diferencia en el público (+0,3 M sobre el 25% del mes) está dentro del ruido, mientras que en
junio, un mes completo que no se usó para elegir nada, verscla03 le gana al ensamble por $6,6 M. Por eso se
recomienda como entrega final volver a enviar `verscla03` (`mi_pipeline_b2_k14500.csv`). Optimizar con
undersampling acelera Optuna, pero en este caso el modelo resultante es más débil y no aporta al ensamble.

## Versión `verscla05` — variables estilo American Express (entrega final)

Experimentos sobre verscla03, todos medidos en junio con semillerío de 5 (meseta = promedio de la ganancia
entre 12.500 y 15.000 envíos). El código está en `verscla05_celdas.py`.

| Experimento | Meseta junio | Conclusión |
|---|---|---|
| verscla03 (referencia) | 408,2 | — |
| Undersampling 25% / 50% con params de verscla03 | 408,9 / 405,3 | Empate: el undersampling no cuesta ganancia; lo que fallaba en verscla04 era el Optuna optimista |
| Marzo + abril → junio | 381,8 | Peor: marzo es el primer mes y no tiene lags |
| Sin fechas de tarjeta | 407,1 | Empate |
| Rank por mes de 7 montos / de todos | 399,6 / 380,3 | Peor: el percentil destruye información absoluta (ceros, negativos) |
| Sin conteo de sueldo | 405,6 | Empate |
| **Variables Amex (+28)** | **426,9** | **+18,7 M, gana en todos los cortes** |

**Validación adversarial.** Abril vs junio AUC 1,000: no es drift sino falta de historia (lags 2 y 3 vacíos en
abril, así que el modelo de prueba nunca los usa y el final sí). Junio vs agosto AUC 0,993: fechas de tarjeta,
montos inflados y conteo de sueldo (aguinaldo). El drift es detectable pero no perjudica a LightGBM.

**Variables Amex.** Cambio relativo contra el mes anterior para 20 variables (`_var1`: −0,5 = cayó a la mitad,
independiente de la inflación) y manejo de la tarjeta (deuda tras el pago, pago sobre saldo, pago sobre mínimo,
uso del límite). El mes anterior existe para abril, junio y agosto, así que el modelo de prueba y el final ven lo
mismo. 10 de las 25 variables más importantes son nuevas.

**Entrega.** Entrenada en 202106 con 203 variables, semillerío de 5, top 13.500 de 202108.

| Envío al bot | Archivos | Ganancia pública |
|---|---|---|
| `verscla05semillas` | `entregas/verscla05_seed*_k13500.csv` (5) | media 96,316 · desvío 1,684 |
| `verscla05` / `verscla05final` | `entregas/verscla05_amex_k13500.csv` | **98,285** (verscla03: 94,820) |

El semillerío supera en ~2 puntos al promedio de las semillas sueltas, y hasta una semilla promedio supera a verscla03.

## Versión `verscla06` — Optuna nuevo sobre las variables de verscla05 (experimento, no supera a verscla05)

Celdas 29–32 del notebook. Los modelos entrenados quedan en `modelos/modelo_verscla06_seed*.txt`
(se cargan con `lgb.Booster(model_file=...)`).

| Paso | Qué hace | Resultado |
|---|---|---|
| Más variables (celda 29) | Contadores de "desarme" (`n_cayeron_mitad`, `n_apagados`, `n_subieron_doble`) y `_var1` de las 140 variables | Meseta 426,9 / 430,6 contra 426,9: empate. `n_apagados` no separa (0,4%–1,1% de BAJA+2): las señales elegidas a mano le ganan al conteo masivo |
| Optuna (celda 30) | 25% de CONTINUA, CV de 5 folds, métrica = ganancia promedio en la franja de 12.500 a 15.000 envíos (no el mejor corte de cada fold), MedianPruner, 100 trials | Mejor CV $529 M (trial 97): lr 0,026, num_leaves 274, min_data_in_leaf 983, feature_fraction 0,24, 514 árboles. Con 983 clientes por hoja sobre ~42.000 filas, el árbol no puede pasar de ~40 hojas: la regularización la hace min_data_in_leaf |
| Junio (celda 31) | Semillerío de 5, 25% de CONTINUA por semilla | Meseta **431,8** contra 426,9 (+4,9, gana en los 11 cortes); ensamble 429,3 |
| Entrega (celda 32) | Entrenada en 202106, top 13.500 de 202108 (93,2% de coincidencia con verscla05) | `entregas/verscla06_k13500.csv` y 5 archivos por semilla |

**Bot:** la media de las 5 semillas no superó a verscla05, así que la entrega final sigue siendo `verscla05`
(98,285). La mejora de +4,9 en junio estaba en el límite del ruido y el público no la confirmó.

## Versión `verscla07` — ensamble LightGBM + LightGBM Optuna + XGBoost (ENTREGA FINAL)

Celdas 33–34 del notebook. XGBoost 2.1.4 con las 203 variables de verscla05, sin optimizar
(`hist`, max_bin 31, eta 0,03, max_depth 6, min_child_weight 20, subsample 0,8, colsample_bytree 0,4,
600 árboles), semillerío de 5. Modelos en `modelos/modelo_xgb_verscla07_seed*.json`.

| Modelo (junio) | Meseta 12.500–15.000 | Contra verscla05 |
|---|---|---|
| LightGBM verscla05 | 426,9 | — |
| XGBoost solo | 429,1 | +2,2 |
| LightGBM + XGBoost | 430,4 | +3,5 |
| **LightGBM + LightGBM Optuna (verscla06) + XGBoost** | **432,8** | **+5,9 (gana en 9 cortes, empata en 2, no pierde en ninguno)** |

Ensamble por promedio de rankings de los tres semillerios, entrenados en 202106; top 13.500 de 202108
(`entregas/verscla07_ens3_k13500.csv`, 96,3% de coincidencia con verscla05).

**Bot:** 97,955 (verscla05: 98,285). El criterio se fijó antes del envío: quedarse con el ensamble salvo que el
público cayera más de 1 punto, porque 0,33 está muy por debajo del desvío entre semillas (1,68) y un ensamble
de tres modelos distintos reduce la varianza en el privado. **verscla07 es la entrega final.**

## Resumen de versiones

| Rama | Bot (público) | Qué es |
|---|---|---|
| `verscla02` | 94,325 | agente_churn, pipeline de scripts |
| `verscla03` | 94,820 | Notebook paso a paso, target BAJA+1+2 |
| `verscla04` | 95,123 | Ensamble con undersampling (descartado) |
| `verscla05` | 98,285 | Variables estilo American Express |
| `verscla06` | — | Optuna con métrica de franja (no superó a verscla05 en el bot) |
| **`verscla07`** | **97,955** | **Ensamble LightGBM + LightGBM Optuna + XGBoost — entrega final** |
