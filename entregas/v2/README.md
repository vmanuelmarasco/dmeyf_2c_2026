# Entrega v2 — Primera Competencia, predicción de 202108

**Fecha:** 2026-10-11 · **Autor:** Manuel Marasco  
**Modelo:** ensamble LightGBM `OPTUNA_ENSEMBLE_TRIAL_3`  
**Clientes seleccionados:** 8.750 de 164.647

Esta carpeta es una copia congelada de la segunda entrega. Los archivos de la raíz
del repositorio pueden seguir cambiando; los de esta carpeta documentan exactamente
la versión entregada.

## Archivo para entregar

`entrega_v2_ids.csv` contiene únicamente los 8.750 `numero_de_cliente` seleccionados,
sin encabezado y uno por línea. Es el formato corto esperado por el bot.

También se conservan:

| Archivo | Contenido |
|---|---|
| `entrega_competencia_v4_final.csv` | Cartera completa de agosto: 164.647 IDs y columna binaria `Predicted` |
| `ids_baja_mas_2_con_encabezado.csv` | Los 8.750 IDs seleccionados, con encabezado |
| `competencia_v4.ipynb` | Desarrollo completo: features, validación temporal, Optuna, ensamble y entrenamiento final |
| `modelos/` | Los cinco modelos LightGBM entrenados |
| `decision_modelo_final_v4.csv` | Comparación final entre candidatos |
| `confirmacion_modelo_final_v4.csv` | Resultados por mes de los ensambles finalistas |
| `ranking_optuna_ensemble_robusto_v3.csv` | Búsqueda de hiperparámetros del ensamble |
| `resumen_features_finales_general.csv` | Última evaluación controlada de bloques de features |

El dataset crudo no se incluye por tamaño.

## Configuración final

| Concepto | Valor |
|---|---|
| Algoritmo | LightGBM `gbdt`, objetivo binario |
| Target de entrenamiento | `BAJA+1` o `BAJA+2` = 1; `CONTINUA` = 0 |
| Evento que determina la ganancia | `BAJA+2` |
| Features | 295 |
| Meses de entrenamiento final | 202103, 202104, 202105 y 202106 |
| Filas de entrenamiento | 654.066 |
| Positivos de entrenamiento | 8.067 |
| Mes predicho | 202108 |
| Semillas | 100129, 100183, 100151, 500041 y 900007 |
| Ensamble | Promedio de las cinco probabilidades |
| Regla de selección | Top 8.750 por probabilidad del ensamble |

Iteraciones finales por semilla:

| Semilla | Iteraciones |
|---:|---:|
| 100129 | 175 |
| 100183 | 156 |
| 100151 | 155 |
| 500041 | 126 |
| 900007 | 183 |

Hiperparámetros del trial ganador:

```text
max_depth=9
num_leaves=43
learning_rate=0.054932
min_data_in_leaf=1468
feature_fraction=0.273826
bagging_fraction=0.770894
lambda_l1=0.000002
lambda_l2=0.000093
min_gain_to_split=2.039981
max_bin=255
```

## Evidencia de selección

En la confirmación con cinco semillas, el trial ganador alcanzó:

| Validación | Ganancia del ensamble | Diferencia contra parámetros base |
|---:|---:|---:|
| 202105 | 285,175 M | +4,400 M |
| 202106 | 368,775 M | +1,100 M |
| Promedio | 326,975 M | +2,750 M |

Ganó contra la referencia en los dos meses. La mejora media del ensamble frente al
promedio de sus modelos individuales fue de 5,280 M.

## Interpretación del archivo

`Predicted = 1` no significa certeza de baja. Identifica los 8.750 clientes con mayor
riesgo estimado y, por lo tanto, priorizados para el estímulo. Aunque el modelo se
entrena con `BAJA+1 + BAJA+2`, toda la selección y la ganancia se evaluaron sobre el
evento de negocio `BAJA+2`; este esquema superó al entrenamiento exclusivo con
`BAJA+2` en la validación temporal.

## Reproducción

Abrir `competencia_v4.ipynb`, ajustar `dataset_path` a la ubicación local del CSV
crudo y ejecutar las celdas en orden. La última celda reconstruye los temporales B6
sobre los seis meses, entrena los cinco modelos, promedia las probabilidades y genera
la selección top 8.750.
