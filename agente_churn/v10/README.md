# verscla10 — verscla09 + variables de dominio bancario + feature_fraction 0,35 + 20 semillas

`verscla10.py` es un script único y autocontenido: lee el crudo, arma la clase, las variables y entrena.

```bash
python verscla10.py                    # entrega: entregas_verscla10/verscla10_k13000.csv + 20 archivos por semilla
python verscla10.py --validar          # además mide en 202106 entrenando con 202103-202104 (5 semillas)
python verscla10.py --db fe.duckdb --memoria 2GB   # en máquinas con poca RAM (< 16 GB)
```

## Qué cambia respecto de verscla09 y por qué

1. **Variables de dominio bancario (73 nuevas).** Salen del catálogo de features del EDA
   (`INFORME_FEATURES_CHURN.md`) y del perfilado de bajas (`INFORME_PERFILES.md`):
   - vinculación: productos activos recalculados, canales usados, cantidad de tarjetas;
   - compromisos domiciliados (débitos automáticos + pago de servicios) y su caída;
   - flujo de fondos: ingresos, egresos, flujo neto, fuga por transferencias, uso de cajeros ajenos;
   - tarjeta: utilización del límite, intensidad de consumo, adelantos de efectivo;
   - precio y rentabilidad: precio efectivo (mantenimiento / patrimonio), desaceleración contra el
     run-rate anual, peso de las comisiones, perfil deudor;
   - dinámica: cada una contra la **media de sus 3 meses anteriores** (cociente y diferencia);
   - eventos: apagó débitos, apagó homebanking, se quedó sin movimientos, bajó tarjetas, meses
     inactivo, racha de caídas de `ctrx_quarter`, empeoró la mora, llamó al call center por primera vez.

   Algunas tasas de BAJA+2 en 202104-202106 (promedio general 0,63 %): bajó la cantidad de tarjetas 6,4 %,
   se quedó sin movimientos 5,2 %, 3 meses inactivo 6,1 %. `d_productos_activos` quedó como la
   segunda variable más importante del modelo, detrás de `ctrx_quarter`.

2. **`feature_fraction` 0,35** (antes 0,81). Con ~560 variables, que cada árbol vea un subconjunto
   distinto reduce la dependencia de unas pocas variables y mejora el semillerío.

3. **20 semillas** en el modelo final (antes 10).

Todo lo demás es igual a verscla09: entrenamiento con 202103-202106, target BAJA+1 + BAJA+2,
`min_data_in_leaf` escalado por filas, hiperparámetros de verscla05, k = 13.000.

## Cómo se decidió (validación pareada)

Protocolo: entrenar con **202103-202104** y medir en **202106** (mismo hueco de dos meses que la entrega
y, como la entrega, con varios meses de entrenamiento), **5 semillas pareadas** por variante.
Métrica: ganancia promedio entre 12.000 y 15.000 envíos. **Regla fijada antes de medir:** se adopta un
cambio si mejora al menos 5 M en promedio y gana en al menos 4 de las 5 semillas.

| Paso | Variante | Meseta | Contra el paso anterior | Semillas que gana | ¿Se adopta? |
|---|---|---|---|---|---|
| 0 | verscla09 (484 variables) | 398,3 | — | — | base |
| 1 | + dominio bancario | 405,6 | +7,3 | 4/5 | sí |
| 1' | verscla09 + feature_fraction 0,35 | 409,5 | +11,2 (vs 0) | 5/5 | sí |
| 2 | **dominio + feature_fraction 0,35** | **413,8** | **+15,5 (vs 0)** | **5/5** | **sí → verscla10** |
| — | sacar deltas de `Visa_mpagado` y `ccajas_depositos` | 401,7 | +3,4 (vs 0) | 3/5 | no |
| — | + pendiente, mínimo y máximo de 3 meses (18 variables) | 409,8 | −4,0 (vs 2) | 1/5 | no |
| — | feature_fraction 0,20 | 418,1 | +4,3 (vs 2) | 5/5 | no (debajo de 5 M) |
| — | num_leaves 150 | 416,2 | +2,5 (vs 2) | 4/5 | no |

Notas:
- `feature_fraction` 0,20 mejoró en las 5 semillas pero por menos de 5 M: queda como candidato
  para medir en el bot con 20 archivos por semilla.
- Las pendientes y mínimos/máximos de 3 meses empeoran: en 202103-202104 se calculan con 1 o 2 meses
  de historia, así que significan otra cosa que en 202106-202108.
- En la curva de validación, verscla10 tiene su zona alta entre 12.000 y 13.500 envíos (máximo 420,8
  en 12.500); se mantiene k = 13.000, que es el que ya rindió en el público.

## Resultado de la corrida

Generado con `python verscla10.py --db fe.duckdb --memoria 2GB` (20 semillas, 202103-202106 → 202108):
- `entregas/verscla10_k13000.csv`: 13.000 ids enteros, sin encabezado, todos de 202108.
- `entregas/semillas/`: 20 archivos, uno por semilla (coinciden en promedio 96,8 % con el semillerío).
- Coincide en un **92,5 %** de los clientes con verscla09.

## Cómo probarlo en el bot (regla fijada antes de mirar)

1. `submit verscla10semillas` con los 20 archivos de `entregas/semillas/`.
2. Se adopta si la media supera la de verscla09 con 20 semillas (99,97) por **más de 1 punto**
   (el error de una media de 20 archivos ronda ±0,35).
3. Último envío, un solo archivo: `submit verscla10` con `verscla10_k13000.csv` si se adopta;
   si no, volver a mandar `verscla09_k13000.csv`.

## Experimentos (`experimentos/`)

`exp.py` corre una variante (5 semillas, 202103-202104 → 202106) y guarda las probabilidades;
`comparar.py` las compara semilla por semilla contra una referencia y aplica la regla.
`dominio.py` arma las variables de dominio y `din.py` las de pendiente y mínimo/máximo (descartadas).
