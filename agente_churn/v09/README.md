# verscla09 — entrega final (público 101,09)

Pipeline en un script (`pipeline_v8.py`) que arma todo desde el crudo en DuckDB y entrena con LightGBM,
más un módulo de experimentos (`lab.py`) y su notebook (`experimentos_v7.ipynb`).

## El cambio que movió la aguja
Hasta verscla07 el modelo de la entrega se entrenaba **solo con 202106**. verscla09 lo entrena con
**202103–202106** (cuatro veces más bajas para aprender), con `min_data_in_leaf` escalado por la cantidad
de filas (`param_normalizado`, z494), y suma **lag 1 y delta 1 de todas las variables** (z402) a las 203 de verscla05.

| Versión | Variables | Entrena la entrega con | Público: media por semilla | Público: semillerío |
|---|---|---|---|---|
| verscla05 | 203 (señales + Amex) | 202106 | 96,32 ± 1,68 (5 archivos) | 98,285 |
| verscla07 | ensamble de 3 modelos | 202106 | — | 97,955 |
| ens4 (`verscla08_ens4_k13500.csv`) | verscla07 + verscla02 | 202106 | — | 98,70 |
| verscla08 | 203 (señales + Amex) | 202103–202106 | — | 96,69 (enviado como `verscla09b`) |
| **verscla09** | **484 (+ lag 1 y delta 1 de todas)** | **202103–202106** | **100,31 ± 1,38 (10 archivos)** | **101,09** |

Lectura: con las 203 variables, entrenar con más meses no mejora; el salto aparece al combinar los 4 meses
con lag 1 y delta 1 de todas las variables, que describen a cada cliente "en movimiento" de la misma forma
en todos los meses (las señales de verscla05 dependen de lags 2 y 3, vacíos en marzo y abril).

## Cómo reproducir
**`verscla09.py`** es el script único y autocontenido de la entrega final: lee el crudo, arma la clase y las
variables, entrena con 202103–202106 y escribe el CSV. No depende de ningún otro archivo.
```bash
python verscla09.py                 # entrega: entregas_verscla09/verscla09_k13000.csv + 10 archivos por semilla
python verscla09.py --validar       # además: ganancia en 202106 entrenando con 202103-202104
```
Los scripts de trabajo con los que se llegó (dependen de `resultados/base/` de competencia_v5):
```bash
python pipeline_v8.py --solo-final --features todas --semillas-final 10 --k 13000 --nombre verscla09
python pipeline_v8.py --solo-final --semillas-final 10 --k 13000                     # verscla08
python pipeline_v8.py --solo-validacion --features todas                           # validación en 202106
```
Hiperparámetros: los de verscla05 (`resultados/base/config.json`), 655 árboles.

## Experimentos descartados en el camino (`registro.csv`, medidos en 202106)
Variables bancarias del agente, ranking (`rank_xendcg`, `lambdarank`), null importance, hojas de
Random Forest (fuga de información), pesos para BAJA+2, ensamble con verscla02: ninguno superó
a verscla07 por más de 10 M. Todos los enfoques quedaron entre 426 y 436 M en junio: el techo estaba
en los datos de entrenamiento de la entrega, no en el modelo.

## Archivos
- `entregas/verscla09_k13000.csv`: **entrega final** (13.000 ids, sin encabezado).
- `entregas/verscla09_semillas/`: 10 archivos, uno por semilla (envío `verscla09.semillas`).
- `entregas/verscla08_*`: la variante con 203 variables.
