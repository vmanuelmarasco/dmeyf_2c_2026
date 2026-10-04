# Bitácora de experimentos — DMEyF 2026

---

## 2026-08-23 — z201_Sobre_Árboles (sklearn, mes 202104, sin partición)

**Hipótesis:** un árbol de decisión clasifica bien BAJA+2 y la clase predicha sirve para decidir la campaña.

**Qué cambié:** nada, es el notebook base + 2 variantes de la tarea.
- M1: `gini`, `max_depth=5`, `min_samples_split=80`, `min_samples_leaf=1`
- M2: idem con `criterion='entropy'`
- M3: idem M1 con `ccp_alpha=0.0005`

**Semilla:** `random_state=17` (única, provisoria — no es de mis 5 primas)

**Resultados (TRAIN, 163.284 filas, 0,70% BAJA+2):**

| Modelo | Hojas | Ganancia máx | Corte | Estimulados | AUC |
|---|---|---|---|---|---|
| M1 gini | 23 | $431.475.000 | 0,0275 | 12.111 | 0,880 |
| M2 entropy | 30 | $461.477.500 | 0,0268 | 11.060 | pendiente |
| M3 ccp_alpha | 2 | $356.702.500 | 0,0520 | 12.029 | pendiente |

**Conclusión en una frase:** el árbol no sirve como clasificador sino como ordenador de clientes — la ganancia sale del ranking por `prob_baja_2` cortado en 0,025 (umbral derivado de 27.500/1.072.500), no de la clase predicha, y gini/entropy son ciegos a esa economía.

**Sobre las métricas clásicas:** no hay que elegir entre modelos por AUC, accuracy ni F1. Esas métricas responden "¿ordena/clasifica bien?" y el banco pregunta "¿cuánta plata deja?". Con costos asimétricos 39:1 y 0,7% de positivos, las dos preguntas dan respuestas distintas. Ir "al llano" no es ir a la intuición: es reemplazar una métrica proxy por la métrica objetivo, que es más exigente, no menos.

**Sobre el punto de corte:** el óptimo empírico (0,0275) cae donde cae porque una hoja aporta ganancia positiva sii `p > 27.500/1.100.000 = 0,025`; ordenando por p descendente, la suma acumulada necesariamente peaquea en la última hoja por encima de ese valor. No es un hallazgo, es un chequeo de consistencia. Se usa el 0,025 teórico y no el empírico porque el empírico es la probabilidad accidental de una hoja de ESTE árbol con ESTA semilla en ESTE mes — se mueve con el modelo; el teórico sale de la economía y no se mueve nunca.

**Hallazgo secundario (M3):** podó a 2 hojas y estimuló casi los mismos clientes que M1 (12.029 vs 12.111), ganando $75M menos. La profundidad no compra volumen, compra **resolución en el ranking**. Ahí está el chiste de la cátedra sobre `ccp_alpha`.

**Mecanismo gini vs entropy:** entropía usa log, muy sensible cuando una clase ocupa una fracción chica del nodo; gini es cuadrática y plana en esa zona. Con BAJA+2 al 0,7%, entropía valora cortes que gini descarta → más hojas (30 vs 23) y hojas rentables más grandes (top: 117 clientes con p=0,23, contra hojas de 1-2 clientes en gini). No es que entropía persiga BAJA+2: es agnóstica a qué clase sea la rara. Ambos criterios son ciegos al tamaño absoluto del nodo, y por eso ninguno optimiza ganancia, que es `N × (1.100.000p − 27.500)` con N multiplicando.

**Corolario:** pureza sin volumen no sirve. La hoja de gini con p=1,0 y n=1 es inútil por dos razones: estadísticamente no es una probabilidad sino un caso (sobreajuste puro), y comercialmente aporta $1.072.500 contra los $26M de una hoja de 117 clientes con p=0,23.

**Qué aprendí sobre árboles en sí:**
1. Parte el espacio en cajas, una variable por vez → cada hoja es un subgrupo describible en castellano (material para el video a Miranda).
2. La hoja devuelve una distribución, no una clase; `Predicted Class` es solo el argmax calculado después.
3. Elegir el corte es greedy y minimiza impureza local, sin mirar la fórmula de ganancia.
4. `max_depth`, `min_samples_split`, `min_samples_leaf`, `ccp_alpha`, `max_leaf_nodes` son cinco perillas para lo mismo: **cuántas hojas**.
5. Con la clase al 0,7%, apretar esa perilla borra la señal (los grupos con BAJA+2 son chicos por definición).
6. Hace selección de variables solo y es brutal: la mayoría de las 155 columnas quedaron en importancia 0,000. Un árbol de profundidad 5 no puede mirar más que un puñado. → esa limitación es la que justifica ensambles (RF, LightGBM).

**Dos cosas que no dice el notebook:**
- No escalé nada y funcionó: los árboles son invariantes a transformaciones monótonas, solo les importa el orden. El FE que mueve la aguja es **crear** variables (ratios, deltas mes a mes), no transformar las existentes.
- `random_state` casi no hizo nada: un árbol es cuasi-determinístico dados datos e hiperparámetros. La variabilidad que se mide con Monte Carlo en z0201 **viene de la partición, no del modelo**.

**Invalidaciones / deuda técnica:**
- BUG: en el bloque de matriz de confusión del M2, `no_evento` usa `leaf_df["BAJA+1"]` (modelo 1) en vez de `leaf_df2`. Pandas alinea por índice → filas 0-22 con valores de otro árbol, filas 23-29 en NaN. Corrompe TP/TN/FP/FN, ROC y los 4 cortes óptimos del M2. **No afecta ganancia ni gan_acumulada** (se calculan directo de las 3 clases).
- BUG: la ROC del M3 grafica `data=leaf2_df` (modelo 1). Corregir y re-ejecutar.
- Falta AUC de M2 y M3 → pregunta de la tarea sin responder.
- `numero_de_cliente` sigue dentro de X (ID; el árbol puede memorizar).
- `get_leaf_info` duplicada 3x → refactorizar a función única que reciba el modelo y devuelva la tabla completa, **antes** del Grid Search. Con 3 modelos copiados a mano metí 2 bugs; con 54 combinaciones no los encuentro nunca.

**Validez: BAJA.** Una semilla, sin partición train/test, un solo mes, medido sobre train. La diferencia M2−M1 (+$30M, +7%) NO es concluyente: un árbol más granular ajusta mejor el train por construcción. Requiere Monte Carlo + Wilcoxon (z0201) antes de tomarla como decisión de modelado.

**Para el video a Michelina:** material de "experimento subóptimo" de manual — medí en train, festejé 7% de mejora, y el 7% puede ser ruido.


He leído el notebook completo. El EDA arranca en la celda 66 y va hasta el final. Te lo desgloso bloque por bloque.

---

## Contexto: por qué el EDA está *al final*

Es deliberado y es lo más interesante del diseño del notebook. Primero entrenaste tres árboles a ciegas, calculaste ganancias y ROC, y recién ahí mirás los datos — pero ya no los mirás al azar: los mirás **guiado por lo que el modelo eligió**. El EDA no es exploración libre, es una auditoría de las decisiones del árbol.

Un detalle a tener presente todo el tiempo: `data` está filtrado a `foto_mes == 202104` desde la celda 8. Todo el EDA describe **un solo mes**, no el dataset completo.

---

## 1. Tabla de sumarización (celda 67)

```python
desc_stats = data.describe(include='all')
missing_values = data.isnull().sum()
unique_values = data.nunique()
```

Arma una tabla de 155 filas (una por variable) donde a los estadísticos clásicos de `describe()` (count, mean, std, min, cuartiles, max) le pega dos columnas que pandas no trae y que en este dataset valen oro:

**`Unique Values` — para cazar categóricas disfrazadas de numéricas.**
Todas las columnas del dataset son numéricas, pero muchas no son cantidades. En la salida se ve claro:
- `foto_mes` → 1 valor único. Es constante (filtraste un solo mes) → **no aporta absolutamente nada al modelo**, es ruido puro que ocupa lugar.
- `cliente_vip` → 2 valores. Es un booleano.
- `internet` → 5 valores. Es una categórica ordinal o un código.
- `Visa_cconsumos` → 132 valores, con media 8.16 y max 180. Esa sí es una cantidad real.

Esto importa porque un árbol trata a `internet` como si "3 > 2 > 1" tuviera sentido cuantitativo. Si en el diccionario de datos resulta que son códigos sin orden, el corte que hace el árbol es geométricamente válido pero conceptualmente basura.

**`Missing Values` — para ver el mapa de nulos.**
`Visa_cconsumos` tiene 21.790 faltantes de 163.284 (13,3%). No son errores de carga: son clientes **sin tarjeta Visa**. El NaN acá *significa algo* ("este cliente no tiene el producto"), y no es lo mismo que un NaN por dato perdido. Cómo los tratás (dejarlos, imputar con 0, imputar con la media, crear una flag) cambia el modelo, y es justamente lo que la cátedra manda a discutir en los experimentos colectivos.

Dato fino que sale de cruzar la tabla con lo de más abajo: `Visa_status` tiene 7.987 nulos, pero `Visa_cconsumos` tiene 21.790. **No todas las variables de Visa tienen el mismo patrón de nulos** — o sea que "no tener Visa" no es una sola condición binaria limpia. Vale la pena investigarlo.

**Qué te da:** el mapa de ruta. Cuáles variables son constantes (tirar), cuáles son categóricas mal tipadas (tratar aparte), dónde están los agujeros, y qué variables tienen colas raras (comparando `max` contra el percentil 75).

---

## 2. Feature importances (celda 69)

```python
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
```

El top 5 del modelo original (gini, depth 5):

| variable | importancia |
|---|---|
| `ctrx_quarter` | 0.328 |
| `Master_status` | 0.172 |
| `mcuentas_saldo` | 0.155 |
| `mprestamos_personales` | 0.105 |
| `Master_Fvencimiento` | 0.095 |

Cinco variables explican el **85%** de la reducción de impureza. Las otras 150 se reparten el 15%, y muchísimas están en 0.000 exacto — el árbol **nunca las usó**. Con `max_depth=5` tenés como máximo 31 nodos internos, así que a lo sumo 31 variables pueden aparecer; el resto queda afuera por construcción.

**Advertencia importante sobre esta métrica:** la importancia de sklearn es *MDI* (mean decrease in impurity), y hereda exactamente el problema que discutiste en la celda 24 — **está calculada en unidades de gini, no en pesos**. Una variable puede ser "importante" para separar CONTINUA de todo lo demás y ser inútil para encontrar BAJA+2. Además, MDI tiene sesgo conocido hacia variables de alta cardinalidad (más puntos de corte posibles = más chances de encontrar uno que reduzca gini). `ctrx_quarter` es de alta cardinalidad; parte de ese 0.328 puede ser artefacto.

**Qué te da:** una lista corta y ordenada de dónde vale la pena gastar tiempo mirando. No una verdad sobre qué causa la baja.

---

## 3. `ctrx_quarter` — densidad por clase (celda 71)

```python
g = sns.FacetGrid(data, row="clase_ternaria", height=3, aspect=2)
g.map(sns.histplot, "ctrx_quarter", stat='density')
```

Tres paneles apilados (uno por clase), mismo eje X, histograma de `ctrx_quarter` (cantidad de transacciones en el trimestre).

**El truco clave acá es `stat='density'`.** Sin eso el gráfico sería inservible: CONTINUA tiene ~161.000 filas y BAJA+2 tiene 1.139. En conteos absolutos, la barra de BAJA+2 sería una línea invisible al lado de la de CONTINUA. `density` normaliza cada panel para que el área sume 1, así comparás **formas de distribución** en vez de volúmenes. Es la respuesta directa al problema de desbalance que venís arrastrando desde la celda 11.

**Qué buscar (que es la pregunta que el notebook te deja abierta):** si la masa de BAJA+2 está corrida hacia la izquierda respecto de CONTINUA — es decir, si los que se van ya venían transaccionando menos. Ese es el patrón esperable y el que hace que `ctrx_quarter` sea la variable #1: la baja no es un evento súbito, es un **apagamiento gradual** que deja rastro en los datos meses antes. Si además ves un pico marcado en 0 o cerca de 0 en BAJA+2 que no existe en CONTINUA, ahí tenés la señal.

---

## 4. `Visa_status` — crosstab (celda 73)

```python
pd.crosstab(data['Visa_status'], data['clase_ternaria'])
```

Acá cambia el método **porque cambia la variable**: `Visa_status` es numérica pero tiene 4 valores distintos. Una densidad sobre 4 valores discretos no dice nada; una tabla de contingencia sí.

| Visa_status | BAJA+1 | BAJA+2 | CONTINUA |
|---|---|---|---|
| 0.0 | 641 | 787 | 153.265 |
| 6.0 | 6 | 2 | 6 |
| 7.0 | 19 | 20 | 307 |
| 9.0 | 75 | 23 | 146 |

Leído en proporciones, que es como hay que leerlo:
- `status = 0`: 787 / 154.693 = **0,51%** de BAJA+2
- `status = 6`: 2 / 14 = **14,3%** (pero con n=14, estadísticamente no vale nada)
- `status = 7`: 20 / 346 = **5,8%**
- `status = 9`: 23 / 244 = **9,4%**

El patrón es nítido: cualquier `status > 0` multiplica por 10-20 la tasa de baja respecto de `status = 0`. Y todos esos valores están muy por encima del umbral de 0,025.

**Lo que el crosstab te oculta:** `pd.crosstab` descarta los NaN silenciosamente. La tabla suma 155.297 clientes, pero el dataset tiene 163.284 → **faltan 7.987 clientes** que no aparecen en ningún lado. Y de los 1.139 BAJA+2 totales, la tabla solo muestra 832 → **307 BAJA+2 (el 27%) tienen `Visa_status` nulo**. Ese grupo es invisible en esta tabla y no es chico. Es exactamente el tipo de cosa que la sumarización de la celda 67 te tenía que hacer sospechar de antemano.

---

## 5. La regla de negocio y el lift (celda 75)

Acá el notebook da el salto conceptual: en vez de un modelo, prueba **una sola condición** — `Visa_status > 0` — y la evalúa como si fuera un clasificador.

```
Total de clientes:              163.284
P(BAJA+2 | Visa_status > 0):      0,0745
P(BAJA+2):                        0,00698
Lift:                            10,68
```

**El lift es la métrica central de este bloque.** Dice: entre los clientes con `Visa_status > 0`, la tasa de BAJA+2 es **10,68 veces** la tasa de la población general. Es la traducción a números de "esta variable discrimina".

Y conecta directo con toda la primera mitad del notebook: 0,0745 es casi **3× el umbral de rentabilidad de 0,025**. Una sola condición, sin entrenar nada, ya te pone en zona rentable.

> Un detalle de nomenclatura: la etiqueta dice `P(Visa_status > 0 y BAJA+2)` pero la fórmula divide por `cantidad_visa_gt_0`, no por el total — o sea que es la **condicional** `P(BAJA+2 | Visa_status > 0)`, no la conjunta. El cálculo está bien (el lift necesita la condicional); el nombre de la fila está mal. Vale corregirlo si vas a mostrar esto.

---

## 6. Matriz de confusión de la regla + ganancia (celda 76)

```
TP:        45
FN:     1.094
FP:       559
Recall:    0,0395
Ganancia: 32.890.000 $
```

Este bloque existe para mostrarte el **precio del lift**. Un lift de 10,68 suena espectacular hasta que ves el recall: la regla captura el **3,95%** de las bajas. Se le escapan 1.094 de 1.139.

Y sin embargo: `45 × 1.072.500 − 559 × 27.500 = 32.890.000 $`. Contra los **43.147.500 $** del árbol completo, una regla de una sola línea te da el **76% de la ganancia**.

Esa comparación es la lección del bloque, y son dos cosas a la vez:

1. **A favor de la regla:** con la asimetría de costos de este problema, no necesitás cobertura, necesitás densidad. Estimular 604 clientes bien elegidos te deja casi tanta plata como todo el árbol.
2. **En contra:** el 96% de la ganancia potencial sigue tirada en la mesa. El árbol no está tan lejos de la regla simple, lo cual dice menos sobre lo buena que es la regla que sobre **lo malo que es el árbol** con `max_depth=5`.

Notá también que `FP = cantidad_visa_gt_0 - TP` mete a BAJA+1 y CONTINUA en la misma bolsa — consistente con la fórmula de ganancia de la celda 21, donde ambos cuestan `costo_estimulo`.

---

## 7. `mpasivos_margen` — boxplots (celdas 78 y 80)

**Primer intento, crudo:**
```python
sns.boxplot(x='clase_ternaria', y='mpasivos_margen', data=data)
```
Sale ilegible. Las tres cajas quedan aplastadas contra el eje y todo lo que ves son puntos dispersos hacia arriba.

**Los "circulitos" son outliers** según la regla estándar del boxplot: puntos a más de 1,5 × IQR del cuartil correspondiente. Con una variable de plata como esta, la distribución tiene cola larguísima a la derecha (unos pocos clientes con márgenes enormes), y esos extremos estiran el eje Y tanto que la caja — donde vive el 50% central de los datos, que es lo que querés comparar — se comprime hasta desaparecer.

**Segundo intento, podado:**
```python
lower_bound = data['mpasivos_margen'].quantile(0.05)
upper_bound = data['mpasivos_margen'].quantile(0.95)
filtered_data = data[(data['mpasivos_margen'] >= lower_bound) & (data['mpasivos_margen'] <= upper_bound)]
```
Recorta el 5% inferior y el 5% superior, se queda con el 90% central y vuelve a graficar. Ahora sí las cajas tienen altura y podés comparar medianas y dispersión entre clases.

**Dos advertencias sobre este recorte:**

- Los cuantiles se calculan sobre **toda** la variable, no por clase. Si BAJA+2 se concentra en un extremo de la distribución, el filtro le saca proporcionalmente más filas que a CONTINUA, y estás comparando poblaciones recortadas de forma desigual. Vale la pena chequear cuántos BAJA+2 sobreviven al filtro antes de sacar conclusiones.
- **Esto es solo para visualizar.** No es una decisión de preprocesamiento. Los outliers de `mpasivos_margen` bien podrían ser justamente los clientes más valiosos y más propensos a irse — tirarlos del modelo sería un error caro. El recorte acá es un truco de escala del gráfico, nada más.

---

## Resumen: qué te dio el EDA en total

| Bloque | Herramienta | Para qué sirve |
|---|---|---|
| Sumarización | `describe` + nunique + isnull | Detectar constantes, categóricas disfrazadas y el mapa de nulos |
| Importancias | `feature_importances_` | Priorizar qué mirar (en unidades de gini, no de plata) |
| `ctrx_quarter` | Densidad por clase | Variable continua alta cardinalidad → comparar **formas** con `density` |
| `Visa_status` | Crosstab | Variable discreta baja cardinalidad → tabla de contingencia |
| Regla de negocio | Lift + recall + ganancia | Cuantificar una hipótesis simple en pesos, sin modelo |
| `mpasivos_margen` | Boxplot (crudo y podado) | Variable con cola larga → boxplot, y por qué hay que recortar para verlo |

El hilo conductor: **cada tipo de variable pide un gráfico distinto**, y toda conclusión termina traducida a pesos. Es la misma tesis de la celda 24 aplicada a la exploración.

---

## Sobre la tarea que queda abierta

Las dos consignas finales apuntan a lo mismo:

- **"Sume al análisis los periodos descartados"** — todo el EDA es de 202104. Volver a los otros meses te permite ver *evolución*: `ctrx_quarter` cayendo mes a mes antes de la baja es una señal muchísimo más fuerte que su valor absoluto en un mes.
- **"¿Se pueden construir features?"** — esa es la respuesta directa. Deltas (`ctrx_quarter` de este mes menos el de hace 3), rankings dentro del mes, flags de "tiene Visa / no tiene", conteos de productos activos. Con un solo `foto_mes` no podés construir ninguna de las temporales, y son las que más rinden en este problema.

Si querés, puedo escribir el código del EDA multi-período y las features de delta.