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
