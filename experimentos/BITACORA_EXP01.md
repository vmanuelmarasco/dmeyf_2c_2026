# Experimento 01 — La regla de corte

**Fecha:** 2026-10-08 · **Semillas:** 20 primos en [100003, 999983] (las 5 del CheckIn + 15 de medición)
**Modelo:** trial 25 congelado, 238 features, 150 iteraciones fijas (sin early stopping)
**Folds:** `[202103] → 202105` y `[202103,202104] → 202106` (gap de 2 meses, verificado por assert)

## Hipótesis

**H1:** el umbral teórico 0,025 no maximiza la ganancia, porque las probabilidades de
LightGBM no están calibradas en la cola superior. Existe una regla de corte, con su
parámetro elegido **fuera** del mes que se mide, que gana significativamente más.

**H0:** 0,025 es insuperable; cualquier ganancia extra viene de haber elegido el corte
en el mismo mes donde se lo mide.

## Diseño

Ceteris paribus estricto: un único modelo, las mismas semillas, los mismos meses. Lo
único que cambia entre brazos es la regla de decisión. Selección cruzada entre meses
(ajustar en mayo → medir en junio, y al revés) para no elegir el corte donde se mide.

Regla de decisión fijada **antes** de ver resultados: una regla reemplaza a 0,025 solo si
(1) gana en los dos sentidos del cruce, (2) Wilcoxon apareado p < 0,05, y (3) mejora ≥ 1%.

## Resultado: **H1 rechazada**

| Regla | mayo→junio | junio→mayo |
|---|---|---|
| R0 umbral teórico 0,025 | 390,93 M (ref) | 239,89 M (ref) |
| R1 k absoluto | −10,14 M (p=0,024) | **−42,91 M** (p<0,0001) |
| R2 fracción de población | −9,96 M (p=0,024) | **−42,59 M** (p<0,0001) |
| R3 múltiplo de la masa de riesgo | +4,19 M (p=0,083) | **−28,86 M** (p<0,0001) |
| R4 umbral empírico | +2,45 M (p=0,330) | **−40,11 M** (p<0,0001) |
| R5 recalibrado isotónico | +2,36 M (p=0,330) | **−40,07 M** (p<0,0001) |

Ninguna regla pasa el criterio 1. La mejor (R3) ni siquiera pasa el 2 en el sentido
favorable. **0,025 se queda.**

## Por qué: el óptimo no viaja entre meses

| | mayo | junio |
|---|---|---|
| Tasa de BAJA+2 | 0,531 % | 0,669 % |
| k óptimo (in-sample) | 8.217 | 14.087 |
| k que da 0,025 | 7.911 | 8.578 |
| Sesgo de elegir el corte in-sample | +4,67 M | **+35,88 M** |

El k óptimo se mueve **+71 %** entre meses mientras la tasa de bajas se mueve +26 %: no es
un efecto de prevalencia que se pueda normalizar. En mayo 0,025 cae casi encima del óptimo;
en junio queda muy corto. Llevar el k de junio a mayo sobre-estimula y cuesta 43 M.

**La virtud de 0,025 no es ser óptimo: es no tener parámetro que deba transferir.**

## Confirmación independiente: la cantidad de iteraciones tampoco viaja

| Iteraciones | mayo | junio |
|---|---|---|
| 100 | 226,00 M | 377,54 M |
| **150** | 239,89 M | **390,93 M** |
| 200 | 247,82 M (20/20, p<0,0001) | 385,70 M (5/20, p=0,0014) |
| 300 | **251,61 M** (20/20, p<0,0001) | 379,20 M (1/20, p<0,0001) |

Mayo pide entrenar más largo, junio pide cortar en 150, **las dos con p < 0,01**. Mismo
modelo, mismos parámetros, meses contiguos. Con solo dos meses válidos que se contradicen,
no hay forma honesta de tunear nada que dependa del mes.

## Presupuesto de medición (lo que obliga a cambiar el protocolo)

- Desvío entre semillas: **mayo 3,78 M · junio 13,18 M** (rango de 47,93 M en junio).
- Semillas necesarias (80 % de poder, apareado) según el desvío de la diferencia:

| desvío de la diferencia | 2,5 M | 5 M | 10 M | 20 M |
|---|---|---|---|---|
| 9,2 M | 109 | 29 | 9 | 4 |
| 13,2 M | 221 | 57 | 16 | 6 |
| 18,2 M | 418 | 106 | 28 | 9 |

El protocolo de 5 semillas solo podía resolver efectos de ~20 M. De las seis decisiones de
bloques ya tomadas, únicamente CARGA (+16,55 M) estaba cerca de esa escala.

- **Ensamblar semillas satura en 5:** 1 semilla 392,02 M → 5 semillas 397,15 M → 20 semillas
  396,96 M, con el desvío entre ensembles cayendo de 13,18 M a 4,33 M. Las 5 del CheckIn
  alcanzan para la entrega. Las semillas extra sirven para **medir**, no para ensamblar.
- **Las 5 semillas declaradas son ~+9,7 M afortunadas en junio** (400,67 M contra 390,93 M del
  promedio de 20, a 150 iteraciones fijas). Es ~1,6 errores estándar: suerte, no selección
  —se declararon antes de medir— pero hay que descontarlo al proyectar agosto.
- **El early stopping no era el problema:** aporta +3,07 M de optimismo en semillas
  individuales y ≈0 en el ensemble.
- **Incertidumbre del leaderboard** (Montecarlo 30 %, método z301): IC95 de ancho ~96 M en los
  dos meses. Dos entregas que difieran menos de ~50 M son indistinguibles en el público.

## Conclusión en una frase

La regla de corte 0,025 se confirma, no por óptima sino por no tener parámetro que transferir;
y el experimento dejó al descubierto que el protocolo de 5 semillas sobre un mes no tiene poder
para sostener ninguna de las seis decisiones de features ya tomadas.

## Recomendación concreta

1. **No tocar el corte.** Queda `PROB(BAJA+2) >= 0,025`.
2. **No tunear iteraciones por mes.** Fijar 150 y dejarlo constante entre brazos.
3. **Protocolo nuevo, obligatorio para toda comparación:** 20 semillas, los dos folds, apareado
   por semilla, Wilcoxon, y el brazo gana solo si gana en los dos meses.
4. **Re-auditar las seis decisiones de bloques** con ese protocolo antes de agregar features nuevas.
5. **Undersampling de CONTINUA, reencuadrado:** no como técnica de desbalance sino como palanca
   de cómputo. Si entrena 5-10× más rápido con la misma ganancia, habilita 100 semillas, que es
   lo único que permite resolver efectos de 5 M.

## Limitaciones

- Solo dos meses de validación con gap 2. No hay forma de distinguir "el efecto no transfiere"
  de "mayo o junio es un mes atípico". Es la restricción dura de la ventana de 6 fotos.
- Mayo entrena con un solo mes (960 positivos) y junio con dos (2.099): parte de la diferencia
  entre folds puede venir del tamaño del train, no del mes.
- El k óptimo in-sample es un estimador sesgado; se reporta solo para cuantificar el sesgo.

## Reproducción

```bash
python3 experimentos/exp01_harness.py    # ~4 min, 20 semillas x 2 folds, persiste probabilidades
python3 experimentos/exp01_analisis.py   # lee de disco, no reentrena
```
