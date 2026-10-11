# Experimento 02 — ¿Sirven las features temporales?

**Fecha:** 2026-10-08 · **Archivo:** `competencia_02.ipynb` (corre de punta a punta, 0 fallos, ~25 min)
**Semillas:** 20 primos en [100003, 999983] · **Iteraciones:** 150 fijas, sin early stopping
**Folds:** `[202103] → 202105` y `[202103,202104] → 202106` (gap 2)

## Hipótesis

Las features temporales (lags, deltas, ventanas móviles, pendientes, rangos intra-mes)
le ganan al set de 144 columnas crudas, medido con poder suficiente y en los dos folds.

**Regla de decisión, fijada antes de medir:** gana solo si en **los dos** folds mejora con
Wilcoxon apareado p < 0,05 **y** la mejora supera el 1 % de la ganancia del incumbente.

## Resultado: **ACEPTADO**

| | crudas (144) | con tiempo (940) | delta | p (Wilcoxon) | semillas a favor |
|---|---|---|---|---|---|
| **mayo** | 235,00 M | 241,31 M | **+6,30 M** (+2,68 %) | 8,4 × 10⁻⁴ | 15/20 |
| **junio** | 351,33 M | 400,51 M | **+49,18 M** (+14,00 %) | 1,9 × 10⁻⁶ | 19/20 |

Error estándar del delta: 1,63 M en mayo, 4,83 M en junio. IC95 de junio: [+39,72, +58,65].

**Efecto secundario que no se buscaba:** las features temporales también **bajan la
varianza entre semillas** en junio, de 16,44 M a 11,14 M. Miden mejor y además miden más
estable.

**Costo:** el entrenamiento pasa de 5,6 s a 14,0 s por semilla en junio (2,5×). Es el
argumento concreto para el undersampling de CONTINUA como palanca de cómputo.

## El matiz que hay que decir en voz alta

Los dos folds **no están midiendo lo mismo**.

El fold mayo entrena con 202103, que es el primer mes de la ventana. Ahí, de las 796
features temporales, **solo 60 tienen valor**: las de `percent_rank` intra-mes. Las otras
736 son 100 % nulas, porque no hay historia que laguear.

| familia | features | % con valor en 202103 |
|---|---|---|
| `_rango_mes` | 60 | **100 %** |
| `_lag1`, `_lag2`, `_delta1`, `_delta2`, `_media3`, `_min3`, `_max3`, `_volatilidad3`, `_pendiente3`, `_apago`, `_prendio`, `_delta_rel1`, `_vs_media3` | 736 | **0 %** |

O sea:

- El **+6,30 M de mayo** viene **enteramente de los 60 rangos intra-mes**, que no usan
  historia y son inmunes al drift por construcción. Es la defensa contra data drifting
  que enseña z402, y sola ya paga.
- El **+49,18 M de junio** es rango intra-mes **más** historia real.
- Por lo tanto **la historia propiamente dicha quedó validada por un solo fold.** La
  regla de "gana en los dos" se cumplió formalmente, pero no con el mismo contenido.

Eso no invalida la aceptación: ningún brazo pierde en ningún fold y los dos efectos son
del signo correcto. Sí obliga a dos experimentos de seguimiento.

## Bugs encontrados y corregidos en el camino

1. **Ventanas móviles de profundidad variable.** `media3`, `min3` y `max3` promediaban
   **1 mes en 202103 y 4 en 202106**: la misma columna con significado distinto según el
   mes, o sea un proxy de `foto_mes` servido en bandeja. Corregido con una guarda
   `case when count(x) over v3 >= 4 then ... end`: sin ventana completa, NULL.
2. **`lag3` inútil en esta ventana.** 100 % nulo en el train de los dos folds, así que
   LightGBM no la usa nunca para un corte y no hay forma de medir si aporta.
   `PROFUNDIDAD_LAG` bajó a 2, parametrizado para subirlo en la Última Competencia.
   **Esto explica el experimento viejo de LAG1** (+5,43 M, p=0,17): la mitad del train
   no tenía la columna.
3. **`nbformat` sin `\n`** al final de cada línea de `source`: las celdas no parsean.
4. **`%sql duckdb:///archivo` choca con `duckdb.connect()`**: DuckDB no permite el mismo
   archivo con dos configuraciones en un proceso. Se abre **una** conexión y se le pasa a
   jupysql con `%sql con`.

## El gate F5 está probado, no asumido

Un gate que siempre pasa es indistinguible de un gate roto. Se le plantaron dos fugas a
propósito y las dos fueron detectadas:

| columna | filas que cambian al truncar | veredicto |
|---|---|---|
| `trampa_lead` (`lead()` directo) | 163.253 | FUGA DETECTADA |
| `trampa_rank_global` (`percent_rank() over ()` sin partición) | 654.065 | FUGA DETECTADA |
| `control_rank_por_mes` (la versión correcta, particionada) | 0 | limpia |

Y las **796 features de producción pasan**: son idénticas al truncar la tabla en 202106,
lo cual es prueba —no indicio— de que ninguna lee el futuro.

## Conclusión en una frase

Las features temporales aportan de verdad (+14 % en junio, p = 2 × 10⁻⁶) y además
estabilizan la medición, pero en esta ventana de 6 fotos el fold de mayo solo puede
validar los rangos intra-mes, así que la historia quedó sostenida por un único fold.

## Recomendación concreta

1. **Quedarse con el set de 940 features.** Reemplaza al de 238 como incumbente.
2. **Aislar los rangos intra-mes en su propio experimento.** Si 60 columnas sin historia
   explican el +6,30 M de mayo, hay que medir cuánto del +49,18 M de junio es rango y
   cuánto es historia. Son dos mecanismos distintos y conviene saber el peso de cada uno.
3. **Undersampling de CONTINUA como palanca de cómputo**, no como técnica de balanceo. El
   entrenamiento se volvió 2,5× más lento; si se recupera esa velocidad sin perder
   ganancia, se habilitan 100 semillas, que es lo único que permite resolver efectos de
   5 M.
4. **Optuna recién después**, sobre este set ya fijado, y con una pasada de re-chequeo de
   las features con los parámetros nuevos.

## Limitaciones

- Dos folds, y uno de ellos ciego a la historia. No se puede separar "el efecto no
  transfiere" de "mayo es un mes atípico". Es la restricción dura de la ventana de 6 fotos.
- Mayo entrena con un mes (960 positivos) y junio con dos (2.099): parte de la diferencia
  entre folds puede venir del tamaño del train, no del mes.
- Las 940 features se midieron en bloque. No se sabe cuáles de las 796 aportan: para eso
  van los canaritos.

## Reproducción

Abrir `competencia_02.ipynb` y correr de arriba a abajo. El banco de medición cachea en
`banco_medicion/` con firma por contenido: relanzar no reentrena lo ya medido.
