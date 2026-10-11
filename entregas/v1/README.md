# Entrega v1 — Primera Competencia, predicción de 202108

**Fecha:** 2026-10-10 · **Autor:** Manuel Marasco
**Ganancia en validación (junio 2021, 5 semillas declaradas):** 414,04 M
**Clientes estimulados en agosto:** 9.520 de 164.647 (5,78 %)

Esta carpeta es una **copia congelada**. Los archivos de la raíz del repo van a
seguir cambiando; estos no.

## Contenido

| archivo | qué es |
|---|---|
| `entrega_agosto_estimular.csv` | **el archivo subido**: 9.520 clientes a estimular, una columna |
| `entrega_agosto_estimular_con_flag.csv` | los mismos 9.520 con `Predicted` (todos 1) |
| `entrega_agosto.csv` | la cartera completa de agosto: 164.647 filas con 0/1 |
| `competencia_02.ipynb` | el pipeline: carga, target, controles de fuga, fábrica de features |
| `entrega_agosto.py` | entrenamiento final y predicción de agosto |
| `BITACORA_EXP02.md` | el experimento que decidió el juego de features |

## Cómo reproducirla desde el dataset crudo

```bash
# 1. El pipeline: deja competencia_01.duckdb con la base y las 796 features.
#    ~25 min. Verifica granularidad, conteos, target y ausencia de fuga.
jupyter nbconvert --execute --to notebook --inplace competencia_02.ipynb

# 2. Entrenamiento final y predicción. ~5 min.
python3 entrega_agosto.py
```

Salida: `entrega_agosto.csv` (cartera completa con 0/1), idéntico bit a bit al de esta
carpeta. El listado corto de clientes a estimular son las filas con `Predicted = 1`.

**Qué significa `Predicted`:** `1` = el modelo estima `PROB(BAJA+2) >= 0,025` y por lo
tanto conviene gastar el estímulo de $27.500 en retenerlo. `0` = no estimular. En
agosto: 9.520 unos y 155.127 ceros.

El CSV crudo (`competencia_01_crudo.csv`) no está en el repo por tamaño. Se baja de
`https://storage.googleapis.com/open-courses/dmeyf2026-9c6f/competencia_01_crudo.csv`
y se deja en `data/` o en la ruta de Drive que resuelve la celda 3 del notebook.

## Configuración exacta

| | |
|---|---|
| Algoritmo | LightGBM, `objective: binary` |
| Target | `BAJA+2` = 1 · `BAJA+1` y `CONTINUA` = 0 |
| Features | 940 = 144 crudas + 796 temporales |
| Meses de entrenamiento | 202103, 202104, 202105, 202106 |
| Mes predicho | 202108 |
| Iteraciones | 150, fijas, sin early stopping |
| Semillas | 100129, 100183, 100151, 500041, 900007 |
| Ensamble | promedio de las 5 probabilidades |
| Corte | `PROB(BAJA+2) >= 0,025` |

Hiperparámetros (trial 25 de la optimización previa, ver commit `188da6e`):

```
max_depth 12 · num_leaves 57 · learning_rate 0,0144493962
min_data_in_leaf 1614 · feature_fraction 0,2613620757
bagging_fraction 0,5276823218 · bagging_freq 1
lambda_l1 6,957248086e-05 · lambda_l2 4,968141077e-06
min_gain_to_split 1,5074952598 · max_bin 31
```

## Las decisiones, y la evidencia de cada una

**Por qué 940 features y no las 238 anteriores.** El Experimento 02 las comparó contra
las 144 crudas con 20 semillas, apareado por semilla y con Wilcoxon de rangos signados:
mayo +6,30 M (p = 0,0008, 15/20 semillas) y junio +49,18 M (p = 0,0000019, 19/20). La
regla de aceptación —ganar en los dos folds, con significancia y con mejora ≥ 1 %— se
fijó antes de medir. Además bajan la varianza entre semillas en junio, de 16,44 M a
11,14 M.

**Por qué el corte es 0,025 y no uno calibrado.** El Experimento 01 probó cinco reglas
alternativas (k absoluto, fracción de población, múltiplo de la masa de riesgo, umbral
empírico, recalibrado isotónico), eligiendo el parámetro en un mes y midiéndolo en el
otro. Ninguna gana en los dos sentidos del cruce. El k óptimo se mueve +71 % entre mayo
y junio mientras la tasa de bajas se mueve +26 %, así que no es prevalencia
normalizable. La virtud de 0,025 no es ser óptimo: es **no tener parámetro que deba
transferir**.

**Por qué 150 iteraciones fijas y sin early stopping.** Elegir la iteración mirando el
mes de validación contamina la estimación de ese mes. Y no hay un criterio que
transfiera: mayo mejora monótonamente hasta 300 iteraciones (20/20 semillas, p < 0,0001)
mientras junio tiene su máximo en 150 y empeora en 200 y 300 (p = 0,0014 y p < 0,0001).
Con dos folds que se contradicen, se fija y se deja constante entre brazos.

**Por qué promedio y no mediana.** Medido en los dos folds con las 5 semillas: en mayo
la mediana ganaba 0,14 M (ruido), en junio perdía 11,28 M. La regla era ganar en los
dos.

**Por qué 5 semillas y no 20.** Ensamblar satura en 5: 1 semilla da 392,02 M en junio,
5 dan 397,15 M y 20 dan 396,96 M. Las semillas adicionales sirven para **medir** (bajan
el error estándar de la comparación), no para ensamblar.

**Por qué `PROFUNDIDAD_LAG = 2` y no 3.** `lag3` es 100 % nulo en el train de los dos
folds: los meses de entrenamiento son los del arranque de la ventana, donde la historia
todavía no existe. LightGBM nunca la usa para un corte y no hay forma de medir si
aporta.

## Controles de fuga de información

Seis fugas distintas, cada una atajada en un lugar propio. Están en el bloque 4 del
notebook:

| | fuga | control |
|---|---|---|
| F1 | ventanas de resultado solapadas | `assert` de gap ≥ 2 en los dos folds |
| F2 | etiqueta en el mes de scoring | 202108 verificado 100 % sin clase |
| F3 | target propagado entre meses | 4.070 clientes cambian de clase |
| F4 | `numero_de_cliente` / `foto_mes` como predictores | lista de prohibidas |
| F5 | feature que lee el futuro | test de truncamiento |
| F6 | fuga de selección | parámetro elegido en un mes, medido en el otro |

**F5 está probado, no asumido.** Un gate que siempre pasa es indistinguible de un gate
roto, así que se le plantan dos fugas a propósito y se verifica que las detecte:

| columna | filas que cambian al truncar | veredicto |
|---|---|---|
| `lead()` directo | 163.253 | detectada |
| `percent_rank() over ()` sin partición por mes | 654.065 | detectada |
| la versión correcta, particionada por mes | 0 | limpia |

Y las 796 features de producción dan **idénticas** al reconstruirlas sobre la tabla
truncada en 202106. Eso es prueba, no indicio.

## Qué queda pendiente

1. **Optuna sobre las 940 features.** Los hiperparámetros son del trial 25, optimizado
   sobre las 238 anteriores: `feature_fraction = 0,26` y `num_leaves = 57` son
   conservadores para 940 columnas. Es la mejora más grande que queda.
2. **Canaritos.** De las 796 temporales no se sabe cuáles aportan. Cortar las que no le
   ganan a columnas de ruido puro debería mejorar y acelerar.
3. **Separar rango intra-mes de historia.** En 202103 solo 60 de las 796 features tienen
   valor, y son todas de `percent_rank` intra-mes: el +6,30 M de mayo viene enteramente
   de ahí, sin usar historia. Falta medir cuánto del +49,18 M de junio es rango y cuánto
   es historia.

## Limitaciones conocidas

- Solo dos folds de validación con gap 2, y el de mayo es ciego a la historia (entrena
  con 202103, donde todos los lags son nulos). La historia quedó sostenida por un único
  fold.
- Mayo entrena con un mes (960 positivos) y junio con dos (2.099): parte de la
  diferencia entre folds puede venir del tamaño del train, no del mes.
- La ganancia de junio con estas 5 semillas (414,04 M) está unos 10 M por encima del
  promedio de 20 semillas. Es suerte de semilla, no selección —se declararon en el
  CheckIn antes de medir— pero conviene descontarla al proyectar agosto.
- Las 940 features se midieron en bloque; no se sabe el aporte individual de cada una.
