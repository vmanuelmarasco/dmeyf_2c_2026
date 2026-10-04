# Guía de EDA — DMEyF 2026, Primera Competencia
### v2 — reescrita sobre los scripts de clase (`z402`, `z501`, `z601`)

**Dataset:** `competencia_01` — 154 columnas · 983.061 filas · **202103 a 202108** (6 meses, ~163.8 k clientes/mes)
**Hoy:** 14-sep · **Video Miranda:** 27-sep (**13 días**) · **Cierre competencia:** 04-oct (**20 días**)
**Estado:** clase ternaria construida · población premium resuelta como supuesto declarado

---

## Qué cambió respecto de la v1

La v1 decía *qué* mirar. Esta versión dice *qué mirar y con qué línea de código*, porque aparecieron tres scripts de la cátedra que cubren buena parte del trabajo:

| Script | Qué te da | Dónde entra |
|---|---|---|
| `z402_Feature_Engineering_en_SQL.ipynb` | DuckDB + window functions: `lag`, deltas, medias móviles, `regr_slope`, rankings intra-mes, macros, generador programático de features | Partes **II.2**, **III.1**, **IV**, **V** |
| `z601_clases.ipynb` | LightGBM + Optuna, las dos clases binarias, ganancia acumulada, punto de corte, KS, lift, captura, multi-semilla | Partes **I**, **III.3** |
| `z501_cluster_rf.py` | Clusters de `BAJA+2` por perfil de hojas de un Random Forest + tendencias mensuales en PDF | Parte **VI** (el video a Miranda) |

Y aparece una **Parte VIII** nueva: los bugs y trampas concretas que tienen esos scripts. No son adorno — dos de ellos te hacen entregar basura sin que nada falle.

También cambió el stack de los entregables: la v1 los listaba en `.R` y todo el material de clase es Python + DuckDB. Los entregables ahora son Python.

---

## Cómo usar este documento

Dos objetivos que comparten insumo pero no producto:

- **Ganancia** (competencia, 04-oct): que el modelo no se rompa contra el mes del futuro, y que el feature engineering apunte a lo que separa `BAJA+2`.
- **Miranda** (video, 27-sep): perfiles de clientes que ya se dieron de baja, con acciones comerciales concretas.

Partes I a III → el primero. Parte IV → los dos. Partes V y VI → el segundo.

Todo lo que figura como **hipótesis** está sin verificar: son cosas para ir a buscar, no hallazgos. Las respuestas las escribís vos, porque en diciembre las vas a defender sin ayuda.

**Sigue faltando:** el diccionario de datos (`DMEyF_2026.odt`). Este documento trabaja sobre nombres de columna. `tcallcenter`, `cinversion1`, `cinversion2`, `active_quarter` y `cliente_vip` quedan opacas hasta que lo cargues — y las cinco importan.

---

# PARTE 0 — Punto de partida

## 0.1 Clase ternaria: hecha, con una verificación pendiente

Ya la construiste. Queda un cabo suelto que trajo Denicolay y que hay que cerrar:

**Los intermitentes.** Clientes que desaparecen en *t+1* y **reaparecen** en *t+2*. Denicolay cuenta ~15 y da tres explicaciones posibles: baja involuntaria (botón de baja en Home Banking), error del transaccional que alimenta el DW, o cliente que se dio de baja y el ejecutivo de cuenta lo convenció de volver mostrándole el beneficio que perdía al dejar de ser Premium.

**Qué verificar en tu script:** con la definición naive, un intermitente queda etiquetado como `BAJA+1` en el mes *t* — desapareció en *t+1*. Pero no se fue. ¿Es correcto?

- Contá cuántos son en tu dataset y cómo quedaron clasificados.
- Decidí el tratamiento: dejarlos como están, reetiquetarlos, o excluirlos.
- **El impacto en ganancia es cero** (15 casos sobre 163.284), así que no gastes tiempo optimizando esto. Lo que importa es poder defender la decisión en un oral.

**Y el otro uso, que sí vale:** Denicolay marca que va a ser importante detectar *a quién se parecen* los intermitentes. Un cliente que apretó el botón de baja y volvió es alguien que **estuvo a punto de irse y fue recuperado**. Si el perfil del intermitente se parece al del `BAJA+2`, tenés evidencia de que el perfil es recuperable — y eso es exactamente lo que Miranda quiere escuchar. Guardalo para la Parte VI.

**Chequeo de sanidad general:** prevalencia de `BAJA+2` por mes. Debe rondar 0,5–1 % en todos los meses etiquetables (tu 202104 dio 0,70 %).

**Y ahora con los meses puestos:** el dataset va de **202103 a 202108**. `BAJA+2` en el mes *t* requiere ver *t+2*, así que los meses etiquetables para `BAJA+2` son **202103 a 202106**. Por eso `z501_cluster_rf.py` hace `meses = meses_total[:-1]` y por eso `z601` entrena en 202103 y testea en 202105: los dos son meses con etiqueta completa. **Cuatro meses etiquetables es poco**, y tiene una consecuencia directa: casi todo lo temporal que construyas (lags de 3, medias móviles de 3, pendientes) va a tener NA en los primeros meses. No es un bug, es el largo de la serie.

## 0.2 La población: todos premium, `cliente_vip` es la elite adentro

**Supuesto de trabajo, declarado:**

> El dataset contiene únicamente clientes con **Paquete Premium**. Darse de baja del paquete *es* desaparecer del dataset. `cliente_vip` marca un **subsegmento de alto valor dentro** de ese universo: famosos, grandes gastadores, clientes que el banco no quiere perder.

**En qué se apoya:**

- El programa usa "darse de baja" y "dejar de tener Paquete Premium" como el mismo objeto, sin distinguirlos.
- No hay ninguna columna que marque el paquete en las 154. Si la base fuera mixta y te pidieran perfilar sólo a los premium, tendría que haberla.
- Denicolay describe al cliente recuperado como alguien que perdió un beneficio *"por ya no ser Paquete Premium"* — o sea, salir del dataset y perder el paquete son lo mismo.
- Criterio de dominio: en banca y en emisoras de tarjetas, "premium" es el paquete y "VIP" es la elite adentro del paquete. Son dos niveles, no sinónimos.

**Lo que falta para respaldarlo — cinco minutos, y produce las dos frases que sostienen todo el resto:**

| Test | Qué mirar | Qué prueba |
|---|---|---|
| Proporción vip | `%` con `cliente_vip = 1`, por mes | si diera 100 %, vip = premium y el debate se cierra solo |
| **Piso de productos** | mínimo, P5 y P25 de `cproductos` | un P5 alto = base filtrada, no cartera completa. **El test más fuerte.** |
| Tarjeta de crédito | `%` con `ctarjeta_visa + ctarjeta_master > 0` | ~90 %+ = segmento alto; una cartera masiva argentina no da eso |
| Payroll | `%` con `cpayroll_trx > 0` | los paquetes premium suelen exigir acreditación de haberes |
| Estabilidad de vip | ¿transiciona 1↔0 por cliente entre meses? | si transiciona, es estado comercial dinámico; si es fijo, marca de asignación |

Los cinco son una sola query de DuckDB sobre la tabla que ya cargás en `z402`. No arranques nada más hasta tenerlos.

**Consecuencias, que son las que te desbloquean el trabajo:**

1. **La población de Miranda son tus `BAJA+1` y `BAJA+2`, sin filtro.** No hay que recortar nada.
2. **`cliente_vip` es una variable de segmento, no de target** (Parte IV.9 y Parte VI).
3. Si Denicolay contesta en Zulip, actualizá esto. Si no contesta, **el supuesto declarado alcanza**: en la bitácora y en el video, escrito con todas las letras. Un supuesto explícito y falsable es defendible; uno tácito no.

## 0.3 La nomenclatura es sistemática — y eso ahorra la mitad del trabajo

Patrón de prefijos (**a confirmar con el diccionario**, pero consistente en las 154 columnas):

| Prefijo | Significa | Ejemplo |
|---|---|---|
| `c` | **cantidad** (operaciones o productos) | `ctarjeta_visa_transacciones` |
| `m` | **monto** en pesos | `mtarjeta_visa_consumo` |
| `t` | **flag / tipo** | `tmobile_app`, `tcallcenter` |
| `F` / `f` | **fecha** | `Visa_Fvencimiento`, `Master_fechaalta` |

Esto da gratis la distinción más importante para perfilar: **tenencia vs. intensidad**. Casi toda variable de monto tiene su par de cantidad. `ctarjeta_visa` (¿tiene?) y `mtarjeta_visa_consumo` (¿cuánto gasta?) cuentan historias distintas, y *tiene pero no usa* es de las señales más interesantes que vas a poder construir.

**Y ahora tiene consecuencia operativa:** el patrón es lo que hace posible el generador programático de `z402` (celda 29). Si los nombres son sistemáticos, una lista de campos + un `for` te escribe 600 features sin tipear ninguna. Ver Parte V.

**Cuidado al tipear:** `mttarjeta_visa_debitos_automaticos` y `mttarjeta_master_debitos_automaticos` tienen doble `t`. Es del dataset original.

## 0.4 Alertas de estructura, ya visibles en el header

1. **Granularidad mixta.** `active_quarter` y `ctrx_quarter` son **trimestrales** en un dataset mensual. Van a mostrar escalones en el análisis de consistencia longitudinal. No los marques como rotos por eso. *(Y sin embargo `ctrx_quarter` es la variable que la cátedra usa de ejemplo en los tres notebooks: mirá la salida de `z402` celda 17 — para el cliente 12162107 va 58 → 40 → 32 → 18 → 14 → 14. Esa caída monótona es exactamente la señal que estamos buscando en toda la Parte V.)*

2. **Fechas como enteros.** `Master_Fvencimiento`, `Master_Finiciomora`, `Master_fultimo_cierre`, `Master_fechaalta` y sus cinco equivalentes de Visa son `BIGINT`. **Ya sabemos algo:** la salida de `z402` celda 5 muestra valores como `-182`, `-2220`, `-943`. No son fechas absolutas tipo `20250131`: **ya vienen como días relativos** (negativos = pasado). Eso cambia el tratamiento — no hay que convertir nada, pero sí hay que entender el signo antes de construir derivadas. Confirmalo con el diccionario.

3. **Dolarización.** `mcaja_ahorro_dolares`, `mplazo_fijo_dolares`, `minversion1_dolares`, `Master_msaldodolares`, `Visa_msaldodolares` y las cuatro de forex **driftean por tipo de cambio**, no por comportamiento. Tratarlas junto con las de pesos confunde el diagnóstico.

4. **Los sospechosos de fuga tienen nombre.** `Master_status`, `Visa_status`, `Master_delinquency`, `Visa_delinquency`. Ver Parte III.3 — y ojo que `z501` los mete al Random Forest sin filtrar (Parte VI).

## 0.5 El toolkit que ya tenés — mapa de scripts a tareas

Antes de escribir una línea nueva, mirá qué ya está resuelto:

| Necesitás | Ya está en | Celda / función |
|---|---|---|
| Cargar el CSV a SQL sin pandas | `z402` | celda 4, `read_csv_auto` |
| Sumar Visa + Master sin que el NA arruine todo | `z402` | celda 12, macro `suma_sin_null` |
| Ratio seguro (NA + división por cero) | `z402` | **es la TAREA de la celda 14 — no está hecho** |
| Rankings intra-mes contra data drifting | `z402` | celda 15, `percent_rank` / `cume_dist` / `ntile` |
| Valor del mes anterior | `z402` | celda 17, `lag` |
| Deltas contra 1 y 2 meses | `z402` | celda 19 |
| Media/max/min móvil de 3 meses | `z402` | celdas 21 y 23, `window ventana_3` |
| Pendiente de una variable en el tiempo | `z402` | celda 25, `regr_slope` |
| Generar 600 features sin tipear | `z402` | celda 29, loop de Python |
| Función de ganancia y corte | `z601` | celdas 8 y 28, `ganancia_prob_lgm` / `gan_prob` |
| Curva de ganancia acumulada + corte óptimo | `z601` | celda 43 |
| KS, lift, captura | `z601` | celdas 41, 45, 47 |
| Ganancia por corte, multi-semilla | `z601` | celdas 56–58 |
| Importancia de variables (para cazar leaks) | `z601` | celda 20 |
| Clusters de `BAJA+2` + tendencias mensuales | `z501` | script entero |

Lo que **no** está en ningún lado y tenés que escribir vos: la tabla de consistencia `variable × mes` (Parte II.1), la auditoría de fugas (III.3), y las derivadas de negocio de la Parte IV.

---

# PARTE I — Fundaciones

## I.1 La función de ganancia

Implementá `ganancia(prediccion, clase_real)` en un archivo propio. Es la moneda de la materia: todo hallazgo que no se traduzca a esta función es decoración.

**`z601` ya te da dos versiones, y son distintas:**

```python
# celda 28 — evaluación sobre un set de test
def gan_prob(y_pred, y_true, prop=1):
    ganancia = np.where(y_true == 1, ganancia_acierto, 0) - np.where(y_true == 0, costo_estimulo, 0)
    return ganancia[y_pred >= 0.025].sum() / prop
```

```python
# celda 8 — métrica custom adentro de la CV de LightGBM
def ganancia_prob_lgm(y_pred, data, prop=0.3):
    y_true = data.get_label()
    ganancia = np.where(y_true == 1, ganancia_acierto, 0) - np.where(y_true == 0, costo_estimulo, 0)
    return 'gan_prob', ganancia[y_pred >= 0.025].sum() / prop, True
```

La única diferencia real es el `prop`: adentro de la CV cada fold es el 30 % de los datos (`test_size=0.3`), así que dividir por 0,3 **re-escala la ganancia del fold a la escala del dataset completo**. Sin eso, los números de la optimización y los de test viven en escalas distintas y no se pueden comparar. Entendé esto antes de tocar nada: es el tipo de detalle que en un oral separa a quien escribió el código de quien lo copió.

**Y fijate la convención que usa la cátedra:** `np.where(y_true == 1, ganancia_acierto, 0) - np.where(y_true == 0, costo_estimulo, 0)`. O sea: el acierto paga 1.072.500 **limpio**, y el costo de 27.500 se cobra **sólo a los falsos positivos**. Eso resuelve la duda que la v1 dejaba abierta. Anotalo con fecha en la bitácora, porque el umbral óptimo es el mismo con cualquier convención pero **la ganancia absoluta que reportás no**.

**Derivá el corte vos, en papel, una vez:** si un cliente tiene probabilidad *p* de ser `BAJA+2`, la esperanza de estimularlo es lineal en *p*; igualás a cero y despejás. Con esta convención te da 27.500 / 1.100.000 = 0,025. Foto al repo.

## I.2 Umbral vs. cantidad de envíos — y por qué `z601` te lo demuestra

**Umbral de probabilidad y cantidad de envíos son dos formas de cortar el mismo ranking.** Si el modelo está mal calibrado —y el boosting suele estarlo— cortar en 0,025 no es lo mismo que cortar en los top-N óptimos.

`z601` no lo dice, lo **muestra**, y es el hallazgo más importante del notebook:

1. Celda 49 entrena con la clase binaria alternativa (`BAJA+1` + `BAJA+2` vs `CONTINUA`), corta en 0,025, y la ganancia **cae**. Conclusión aparente: la clase alternativa es peor.
2. Celda 51 dibuja la ganancia acumulada del mismo modelo y encuentra el corte óptimo. La ganancia **sube mucho**. Conclusión real: la clase alternativa es mejor, lo que estaba mal era el umbral.

Qué pasó: al meter `BAJA+1` adentro del positivo, la prevalencia de la clase 1 se multiplica por ~2, las probabilidades predichas se corren hacia arriba en bloque, y 0,025 deja de ser el corte correcto. **El 0,025 es óptimo para la clase original y para un modelo calibrado. Fuera de ahí, no significa nada.**

La consecuencia operativa es dura: **la curva de ganancia contra cantidad de envíos es el gráfico que más veces vas a mirar en la materia**, y el umbral fijo es una trampa que te hace descartar ideas buenas. Cada vez que cambies el target, el feature set o el algoritmo, el corte se recalcula.

*(Ya trabajaste esto con los árboles: el corte empírico 0,0275 no le ganaba al teórico 0,025 porque era artefacto de ese árbol, esa semilla y ese mes. Es el mismo principio visto del otro lado.)*

**Y acá viene la pregunta que la cátedra deja como tarea en la celda 59:** *"¿Qué métrica optimiza cuando tiene las dos bajas juntas?"*. La respuesta está en la tensión de arriba: `ganancia_prob_lgm` optimiza **ganancia con umbral fijo en 0,025**, que para el target combinado es el umbral equivocado. Optuna está buscando el mejor modelo según una métrica que mide mal. **Arreglar esa función —que corte por top-N en vez de por umbral— es probablemente la mejora de mayor retorno por línea de código de toda la competencia.** Hacelo antes de gastar horas de VM optimizando.

## I.3 Composición del target en el tiempo

Tabla `foto_mes × clase_ternaria`, conteos y porcentajes, más un gráfico de tres líneas. Con 202103–202108 son seis filas: se hace en una query.

¿La prevalencia de `BAJA+2` es estable? ¿Hay meses con shock? ¿La cartera crece o se achica? ¿Numerador y denominador se mueven juntos?

Cualquier mes que se aparte más de ~30 % de la mediana es sospechoso, y qué hacés con él (usarlo, excluirlo, ponderarlo) es una decisión que tenés que poder justificar. Con sólo cuatro meses etiquetables, excluir uno es caro: pensalo dos veces.

**Agregá una línea más al gráfico:** prevalencia de `BAJA+2` entre los vip. Si se mueve distinto del resto, ya tenés material para la Parte VI.

## I.4 La batería de diagnóstico — qué mide cada cosa

`z601` te da cuatro instrumentos. No son intercambiables y conviene tener clarísimo qué contesta cada uno, porque en el video a Michelina vas a tener que elegir cuál mostrar:

| Instrumento | Celda | Qué contesta | Unidad |
|---|---|---|---|
| **Ganancia acumulada** | 43 | ¿hasta dónde conviene estimular? | pesos |
| **KS** | 41 | ¿separa las dos poblaciones? | 0 a 1 |
| **Lift** | 45 | ¿cuánto mejor que elegir al azar? | múltiplo |
| **Captura** | 47 | ¿qué proporción de los `BAJA+2` agarré? | 0 a 1 |

**El único que está en unidades de negocio es la ganancia acumulada**, y por eso es el que manda. Los otros tres son diagnósticos: te dicen *por qué* la curva de ganancia es como es.

Dos lecturas que valen para el EDA, no sólo para el modelo:

- **KS entre 0,3 y 0,5 es bueno; arriba de 0,5 sospechá leakage.** Eso convierte al KS en un **detector de fugas** (Parte III.3): si al agregar una variable el KS salta de 0,42 a 0,68, no encontraste una feature mágica, encontraste una variable que se actualiza durante el proceso de baja.
- **La distribución de probabilidades por clase** (celda 39) muestra que `BAJA+1` y `BAJA+2` tienen distribuciones parecidas y `CONTINUA` una distinta. Ese gráfico es, en sí mismo, la justificación empírica para probar el target combinado — y es material de la Parte V.

**Y la regla que atraviesa todo:** una sola corrida no prueba nada. `z601` celdas 32 y 56 corren 20 semillas y grafican la distribución de ganancias. Un modelo que gana 3 % más pero tiene el doble de desvío no es mejor. En la bitácora, toda ganancia va con su desvío y su cantidad de semillas.

---

# PARTE II — Integridad temporal *(programa 13.1 a 13.3)*

**Esta es la parte que protege la ganancia, y es independiente de todo lo que discutimos sobre premium. Los datos son de un banco argentino real: cambios de sistema, productos discontinuados, redefiniciones e inflación. Una variable que en un mes mide una cosa y en otro mide otra es veneno silencioso, y ninguna métrica de performance te avisa.**

## II.1 Consistencia longitudinal *(13.1)*

**Producto:** tabla larga `variable × foto_mes` con `%NA`, `%cero`, mediana, p99 y media por celda. De ahí sale el entregable real: **lista corta de variables sospechosas con veredicto de una línea**.

Esto no está en ningún notebook de la cátedra, pero con DuckDB cargado (`z402` celda 4) sale con `UNPIVOT` y un `group by`: no es un loop de 154 iteraciones en pandas. Es la primera pieza de código propio que vale la pena escribir, y es defendible en el oral porque es tuya.

**Los cuatro patrones a cazar:**

1. **Variable que nace o muere.** `%NA` salta de 0 a 100 (o al revés) en un mes puntual. Candidatas naturales: `cinversion2`/`minversion2`, `mpayroll2`/`cpayroll2_trx`, `catm_trx_other`/`matm_other`, `cmobile_app_trx`.
2. **Cambio de escala.** La mediana se multiplica o divide de golpe. **Distinguir de inflación:** el crecimiento suave y sostenido de *todas* las variables de monto a la vez no es rotura, es el país. Una sola variable que salta ×10 en un mes sí lo es.
3. **Cero que se vuelve NA.** `%cero` y `%NA` se intercambian entre meses consecutivos. Clásico de migración de sistema.
4. **Truncamiento.** El p99 se aplana en un valor fijo: alguien puso un tope.

**Trampa:** mirar sólo la media. Se mueve por outliers y da falso positivo en las ~60 variables de monto. Mirá mediana + `%NA` + `%cero` juntos.

## II.2 Drifting hacia el mes de aplicación *(13.2)* — y el arma que te da `z402`

El bloque anterior mira toda la serie; este mira el salto al futuro, y decide si el modelo funciona en la entrega.

**Tres formas de medirlo**, de menos a más potente:

1. **Diferencia de deciles** entre el mes de aplicación y los de entrenamiento. Simple y transparente.
2. **PSI o KS.** Un número por variable, comparable entre variables. El KS ya lo tenés implementado en `z601` celda 41 — es el mismo `ks_2samp`, cambiando "clase 1 vs clase 0" por "mes de aplicación vs meses de entrenamiento".
3. **Adversarial validation.** Entrenás un clasificador para distinguir "mes de aplicación" vs "meses de entrenamiento". Si logra AUC alto, hay drift; y su *feature importance* (`z601` celda 20) ranquea exactamente las culpables. La más informativa, y el tipo de idea que te diferencia de la cohorte.

**Y ahora la parte que la v1 no tenía: cómo se arregla.** La celda 15 de `z402` no es un ejemplo decorativo de window functions — es **la solución canónica al data drifting**, y está puesta ahí exactamente para eso. La idea:

```sql
select
    cliente_antiguedad
  , percent_rank() over (partition by foto_mes order by cliente_antiguedad) as antig_pct
  , cume_dist()   over (partition by foto_mes order by cliente_antiguedad) as antig_cume
  , ntile(10)     over (partition by foto_mes order by cliente_antiguedad) as antig_decil
from competencia_01
```

El `partition by foto_mes` es todo. **Reemplazás el valor absoluto de la variable por su posición relativa dentro de su propio mes.** Si la inflación multiplicó todos los saldos por 1,4 entre marzo y agosto, el saldo absoluto driftea salvajemente pero *el decil del cliente dentro de su mes no se mueve*. El modelo aprende "este cliente está en el decil 8 de saldo" en vez de "este cliente tiene $ 340.000", y eso sí transfiere al mes del futuro.

Tres sabores, y elegir cuál usar es una decisión tuya que hay que poder defender:

| Función | Qué devuelve | Cuándo |
|---|---|---|
| `percent_rank()` | 0 a 1, continuo | default para variables de monto |
| `cume_dist()` | 0 a 1, proporción acumulada | casi igual; difiere con empates |
| `ntile(10)` | entero 1 a 10 | cuando querés discretizar y que el árbol no gaste splits |

**El experimento obligatorio:** `ntile` con muchos empates es traicionero. Si el 60 % de los clientes tiene `mprestamos_personales = 0`, `ntile(10)` los reparte arbitrariamente entre los deciles 1 a 6 — clientes idénticos en bins distintos. `percent_rank` les da a todos el mismo valor. Probá los dos, medí ganancia con ≥ 5 semillas, y anotá cuál ganó.

**Criterio general:** drift alto **no implica eliminar**. Una variable que driftea pero predice mucho puede seguir conviniendo, y ahora tenés una tercera opción entre "dejarla" y "tirarla": **rankearla**. Se decide contra la ganancia, no contra el PSI.

**Separá tres fuentes antes de concluir:** inflacionaria (las `m` en pesos → el ranking las arregla), cambiaria (las `_dolares` y forex → el ranking también, pero verificá), estructural (el resto → el ranking **no** las arregla; una variable que cambió de definición sigue rota aunque la rankees).

## II.3 Variables rotas *(13.3)*

Para cada sospechosa: ¿la dejás, le asignás NA en los meses malos, la interpolás, **la rankeás**, o la eliminás?

El programa plantea la opción contraintuitiva —*asignarle NA*— como pregunta, no como receta. Es una invitación al experimento.

**Producto:** tabla de variable / diagnóstico / tratamiento / **ganancia con y sin**, más de una semilla. Un tratamiento por vez, ceteris paribus. Y "más de una semilla" ahora tiene código: `z601` celda 56 es el molde exacto.

---

# PARTE III — Semántica de los datos

## III.1 Ceros, nulos y ausencia de producto *(13.4)* — y la tensión con `suma_sin_null`

**La pregunta central:** ¿`0`, `NA` y "no tiene el producto" son lo mismo?

Casi seguro que no, y la estructura de pares `c`/`m` da el test perfecto:

> Cuando `cprestamos_personales == 0`, ¿`mprestamos_personales` es `0` o es `NA`?

Corré ese cruce para cada par (préstamos, plazo fijo, inversiones, tarjetas, cheques, forex, payroll). La respuesta te da **la semántica real del NA en esta base**.

**Y `z402` ya te muestra el caso concreto.** Celda 7:

```
   Master_msaldototal  Visa_msaldototal  tc_saldo_total
0                 NaN              0.00             NaN
6           113086.11               NaN             NaN
```

Cliente 0: **tiene Visa con saldo cero y no tiene Master**. Cliente 6: **tiene Master con $113.086 y no tiene Visa**. El `NaN` acá significa *no tiene el producto*, no *falta el dato* — y por eso la suma da NaN.

La cátedra lo resuelve con una macro (celda 12):

```sql
CREATE OR REPLACE MACRO suma_sin_null(a, b) AS ifnull(a, 0) + ifnull(b, 0);
```

**Y acá está la trampa que el notebook no te dice.** `ifnull(x, 0)` colapsa exactamente la distinción que acabás de descubrir: después de aplicarla, *"tiene tarjeta con saldo cero"* y *"no tiene tarjeta"* son el mismo número. La macro es correcta **para construir el agregado** (`tc_saldo_total` sí debe ser 113.086 y no NaN) e **incorrecta si es tu única representación** del par.

La regla: **agregá con `suma_sin_null`, pero conservá los flags de tenencia por separado.** `tiene_visa`, `tiene_master`, `cantidad_tarjetas` — tres columnas baratas que recuperan lo que la macro tira. Es una decisión de modelado tuya, defendible, y de las que Michelina valora.

**La tarea pendiente de la celda 14:** la macro de ratio seguro. Te va a hacer falta sí o sí en toda la Parte IV, porque casi todas las derivadas buenas son ratios y los denominadores se van a cero. La pista de la cátedra es la doc de funciones numéricas de DuckDB; el candidato obvio es una combinación de `ifnull` con una división protegida. **Pero antes de escribirla, decidí qué querés que devuelva cuando el denominador es cero**, y esa decisión no es técnica:

- `mpagado / mpagominimo` con `mpagominimo = 0` → el cliente no tiene pago mínimo exigible. ¿`NULL`? ¿`0`? ¿Un centinela?
- `msaldototal / mlimitecompra` con `mlimitecompra = 0` → probablemente no tiene tarjeta. Ahí `NULL` es lo honesto.

**Devolver `NULL` es casi siempre la respuesta correcta acá**, porque LightGBM maneja NA nativamente y le da su propio tratamiento en el split — o sea, el modelo puede aprender "denominador cero" como categoría. Devolver `0` le miente. Escribí la macro, posteala en Zulip (es tarea entregable), y anotá el razonamiento.

**Missingness como señal:** tasa de `BAJA+2` condicional a ser NA, por variable. Si difiere marcadamente de la general, el missing informa y merece un `isNA_X`. No generes 154 indicadores por las dudas.

**No imputes por reflejo.** LightGBM maneja NA nativamente.

## III.2 Outliers *(13.5)*

Menos importante de lo que enseña la estadística clásica: un árbol corta por umbrales y es casi indiferente a la magnitud del extremo. Pero **LightGBM discretiza en bins**, y un outlier gigante puede degradar la resolución donde vive el 99 % de la cartera.

**Y en `z601` ese parámetro está a la vista y fijo:** `'max_bin': 31` en todos los params. Treinta y un bins es **poco** — el default de LightGBM es 255. Con `max_bin` bajo, la sensibilidad a outliers sube: menos bins para repartir, y si uno se lleva la cola, el resto de la cartera queda apretado en menos cortes posibles.

Dos consecuencias:

1. **`max_bin` es un hiperparámetro candidato** que Optuna no está optimizando (la celda 8 sólo busca `num_leaves`, `learning_rate`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`). Subirlo cuesta memoria y tiempo pero puede ganar. Es un experimento de una línea.
2. **Es el argumento a favor de rankear** (II.2): `percent_rank` distribuye uniforme en [0,1], que es la entrada ideal para un binning de ancho fijo. Rankear no sólo arregla drift — también le da a LightGBM los 31 bins bien aprovechados.

Mirá p99, p99.9 y máximo de las variables de monto. No toques nada por defecto.

**Trampa, y acá es doble:** winsorizar "por prolijidad" destruye la señal de los clientes de altísimo valor — que en esta base **son los vip**, o sea exactamente la población que menos podés permitirte aplastar.

## III.3 Auditoría de fugas

El target se construye por desaparición del cliente. Cualquier campo que se actualice durante el proceso de baja da performance espectacular en validación y desastre en el mes del futuro.

| Sospechoso | Por qué |
|---|---|
| `Master_status`, `Visa_status` | estado administrativo: puede cambiar al iniciarse la baja |
| `Master_delinquency`, `Visa_delinquency` | ídem, y además puede ser señal legítima de riesgo |
| `Master_fultimo_cierre`, `Visa_fultimo_cierre` | si deja de actualizarse, el valor "envejece" y delata |
| `Master_Finiciomora`, `Visa_Finiciomora` | fecha que sólo existe si pasó algo |
| `active_quarter` | el nombre sugiere flag de actividad; verificá cómo se calcula |

**Tres tests, ahora con código:**

1. **Árbol de una sola variable.** Si separa `BAJA+2` con pureza sospechosa, alarma.
2. **Importancia en LightGBM** (`z601` celda 20). Si una de las sospechosas aparece en el top-5 de `feature_importance()`, no la celebres: audítala. Una variable administrativa que le gana en importancia a `ctrx_quarter` es una bandera roja, no un hallazgo.
3. **Salto de KS** (`z601` celda 41). Corré el modelo con y sin la sospechosa. Si el KS salta por encima de 0,5, la cátedra misma te avisa en el texto de la celda 40 que eso "es raro salvo que haya algo de *leakage*".

**El contraste clave:** mirá el mismo estadístico en `BAJA+1`. Si la variable distingue muchísimo `BAJA+1` y poco `BAJA+2`, **no es leak: es señal tardía**. Legítimo, y material para la Parte V.

**Y el riesgo aguas abajo, que la v1 no podía ver:** `z501_cluster_rf.py` mete **todas** las columnas numéricas al Random Forest (`columnas_features` sólo excluye id, mes, target y grupo). Si `Visa_status` es un leak, va a dominar los splits, y entonces **tus clusters para Miranda van a estar definidos por un artefacto administrativo en vez de por comportamiento de cliente**. Ver Parte VI: hay que correrlo dos veces.

No elimines por sospecha sin medir.

---

# PARTE IV — Las familias de productos

**El contenido bancario. Sirve a los dos objetivos: hipótesis de feature engineering para el modelo, vocabulario de perfiles para Miranda.**

Método común: **tenencia** (¿lo tiene?), **intensidad** (¿cuánto lo usa?) y **la combinación**. Las derivadas son candidatas: cada una vale lo que valga su efecto medido sobre la ganancia.

**Cómo implementarlas:** todas las derivadas de esta parte son SQL de una línea sobre la tabla de `z402`. Los agregados Visa+Master van con `suma_sin_null`; los ratios, con la macro segura de III.1 **que todavía no escribiste**. Escribí esa macro primero o vas a estar debuggeando NaNs toda la tarde.

## IV.1 Valor y rentabilidad

`mrentabilidad`, `mrentabilidad_annual`, `mcomisiones`, `mactivos_margen`, `mpasivos_margen`, `ccomisiones_mantenimiento`, `mcomisiones_mantenimiento`, `ccomisiones_otras`, `mcomisiones_otras`

Probablemente **no predice bien la salida, pero determina cuánto te importa acertar**. Un `BAJA+2` de alta rentabilidad y uno de rentabilidad negativa valen igual en la fórmula de la competencia, pero no valen igual para Miranda.

- **Derivadas:** `mrentabilidad / mrentabilidad_annual` (¿el mes fue mejor o peor que su promedio?), composición margen activo vs. pasivo, comisiones sobre rentabilidad.
- **Cruce obligatorio:** rentabilidad de vip vs. no-vip. Es la validación numérica de que los vip son efectivamente los más valiosos — y el número que le ponés a Miranda cuando decís "no todas las bajas cuestan lo mismo".
- **Hipótesis para Miranda:** hay un perfil que se va **porque le cobran**. Si `mcomisiones_mantenimiento` alto correlaciona con baja en clientes de bajo uso, es un perfil con acción obvia y barata.
- **Hipótesis de deterioro:** rentabilidad mensual cayendo contra su anual = candidata a señal temprana. Y ahora se mide con una línea: `regr_slope(mrentabilidad, cliente_antiguedad) over ventana_3`.

*(`mpasivos_margen` ya la trabajaste en la bitácora, con el boxplot podado al 5–95 %. Recordá la advertencia que escribiste ahí: el recorte era para ver el gráfico, no una decisión de preprocesamiento.)*

## IV.2 Relación transaccional núcleo — "¿soy tu banco principal?"

`tcuentas`, `ccuenta_corriente`, `mcuenta_corriente`, `mcuenta_corriente_adicional`, `ccaja_ahorro`, `mcaja_ahorro`, `mcaja_ahorro_adicional`, `mcaja_ahorro_dolares`, `mcuentas_saldo`, `cdescubierto_preacordado`, `ctarjeta_debito`, `ctarjeta_debito_transacciones`, `mautoservicio`, `ctransferencias_recibidas`, `mtransferencias_recibidas`, `ctransferencias_emitidas`, `mtransferencias_emitidas`, `cextraccion_autoservicio`, `mextraccion_autoservicio`, `ctrx_quarter`

**La pregunta de negocio más importante del EDA:** ¿el futuro `BAJA+2` ya dejó de usar al banco como banco principal, aunque el saldo siga ahí?

- **Derivadas:** ratio emitidas/recibidas (quien emite mucho más de lo que recibe está **sacando plata del banco**); transferencias emitidas sobre saldo; extracciones sobre ingresos; transacciones de débito por mes.
- **Hipótesis central:** la fuga de fondos precede a la baja formal. El dinero se va antes que el cliente. Si se confirma, `mtransferencias_emitidas` y sus deltas son de tus mejores features.
- **Distinción crítica:** *"tiene poco saldo"* y *"tenía saldo y lo bajó 70 %"* son clientes completamente distintos, y sólo el segundo es señal. **Esto es exactamente lo que resuelve `z402`:** el primero es `mcuentas_saldo`, el segundo es `mcuentas_saldo / avg(mcuentas_saldo) over ventana_3`. Es el argumento más fuerte a favor de lags y delta-lags, y ahora es una línea de SQL.

## IV.3 Tarjetas — la familia más rica del dataset

Visa y Master, 22 columnas cada una, espejadas. Tratalas simétricamente y además agregadas (`suma_sin_null`).

**Ratios que casi siempre superan a los montos crudos** — todos necesitan la macro segura:

| Derivada | Qué captura |
|---|---|
| `Visa_msaldototal / Visa_mlimitecompra` | **utilización** — el indicador de estrés financiero por excelencia |
| `Visa_mpagado / Visa_msaldototal` | ¿paga todo o financia? |
| `Visa_mpagado / Visa_mpagominimo` | ¿paga el mínimo? (≈ 1 = cliente al límite) |
| `Visa_mconsumototal / Visa_mlimitecompra` | intensidad de uso relativa al límite otorgado |
| `Visa_madelantopesos / Visa_mlimitecompra` | adelanto de efectivo = urgencia de liquidez |
| suma Visa + Master | límite, saldo, consumo y utilización consolidados |
| `Visa_mconsumosdolares / Visa_mconsumototal` | gasto en dólares → perfil de viajes y compras externas |

**Estados a construir:** no tiene tarjeta / tiene y no usa / uso normal / altamente financiado / en mora.

**Dos hipótesis opuestas, las dos plausibles:**
- El que **se está yendo** deja de consumir: baja `mconsumototal`, baja `cconsumos`, tarjeta dormida.
- El que se va **por estrés financiero** hace lo contrario: utilización que sube, paga el mínimo, aparecen adelantos de efectivo.

Dos perfiles de baja distintos, con acciones comerciales opuestas. Si los dos aparecen, ya tenés dos perfiles para Miranda — y `z501` debería separarlos solo, en clusters distintos. Es la mejor prueba de sanidad del clustering: si los clusters que salen no se parecen a ninguna hipótesis que hayas escrito antes, revisá el script antes de creerle.

**Y con el vip encima:** si los vip son grandes gastadores, el `mconsumototal` y el límite de compra deberían separarlos nítidamente. Un vip cuyo consumo se derrumba es probablemente la señal más cara del dataset.

**Cuidado:** `Visa_status`, `Visa_delinquency` y las fechas son candidatas a fuga (III.3).

## IV.4 Crédito

`cprestamos_personales`, `mprestamos_personales`, `cprestamos_prendarios`, `mprestamos_prendarios`, `cprestamos_hipotecarios`, `mprestamos_hipotecarios`

Pocas variables, conceptualmente potentes: **el crédito ata al cliente**. Un hipotecario es un ancla de años.

- **Hipótesis:** tasa de `BAJA+2` sensiblemente menor entre quienes tienen préstamos, y mínima entre los hipotecarios. Si se confirma, es un hallazgo interpretable y muy narrable.
- **Derivadas:** `tiene_credito`, `cantidad_tipos_credito`.
- **Contra-hipótesis:** el que **termina** de pagar queda súbitamente libre. Detectarlo es literalmente `mprestamos_personales > 0 and lead(mprestamos_personales) = 0`, o el delta negativo que lleva el saldo a cero. Ese es un momento de riesgo identificable y accionable, y se construye con `lag` de `z402`.

## IV.5 Ahorro e inversión

`cplazo_fijo`, `mplazo_fijo_pesos`, `mplazo_fijo_dolares`, `cinversion1`, `minversion1_pesos`, `minversion1_dolares`, `cinversion2`, `minversion2`, `cforex`, `cforex_buy`, `mforex_buy`, `cforex_sell`, `mforex_sell`

- **Hipótesis:** la desinversión precede a la baja. Plazo fijo que no se renueva, inversiones que caen a cero. Otra vez: `delta` y `regr_slope`, no el valor absoluto.
- **Forex aparte:** comprar dólares es normal en Argentina; **vender** puede indicar necesidad de liquidez. El ratio buy/sell puede ser más informativo que cualquiera de los dos.
- `cinversion1` vs `cinversion2` son opacas sin diccionario.

## IV.6 Vinculación — lo que hace costoso irse

`cpayroll_trx`, `mpayroll`, `cpayroll2_trx`, `mpayroll2`, `ccuenta_debitos_automaticos`, `mcuenta_debitos_automaticos`, `ctarjeta_visa_debitos_automaticos`, `mttarjeta_visa_debitos_automaticos`, `ctarjeta_master_debitos_automaticos`, `mttarjeta_master_debitos_automaticos`, `cpagodeservicios`, `mpagodeservicios`, `cpagomiscuentas`, `mpagomiscuentas`, `cseguro_vida`, `cseguro_auto`, `cseguro_vivienda`, `cseguro_accidentes_personales`, `ccaja_seguridad`

**Probablemente la familia con más señal del dataset**, porque mide *costo de cambio*. Sueldo acreditado + tres débitos automáticos + dos seguros = mucha fricción para irse.

- **Derivadas:** `cantidad_debitos_automaticos_total` (cuenta + Visa + Master, con `suma_sin_null`), `cantidad_seguros` (0 a 4), `tiene_payroll`, `tiene_payroll_doble`.
- **Hipótesis fuerte y muy narrable:** *la pérdida del payroll es el evento que precede a la baja*. Si el sueldo deja de acreditarse, el cliente cambió de trabajo o de banco, y la baja viene después. Si se confirma, es probablemente **el hallazgo más accionable del proyecto**: evento detectable, ventana de reacción, acción comercial evidente.
- **Cómo se detecta, concretamente:** `cpayroll_trx > 0` y `lag(cpayroll_trx, 1) = 0` (o al revés) construye el flag `perdio_payroll` en una línea. Después cruzás ese flag contra `clase_ternaria` y tenés el número.
- **Y conecta con premium:** si el paquete se sostiene con acreditación de haberes o saldo mínimo, perder el payroll puede ser *literalmente* lo que hace perder el paquete. Vale la pena mirar si los que pierden payroll salen del dataset poco después.

## IV.7 Canales

`internet`, `thomebanking`, `chomebanking_transacciones`, `tmobile_app`, `cmobile_app_trx`, `tcallcenter`, `ccallcenter_transacciones`, `catm_trx`, `matm`, `catm_trx_other`, `matm_other`, `ccajas_transacciones`, `ccajas_consultas`, `ccajas_depositos`, `ccajas_extracciones`, `ccajas_otras`, `ccajeros_propios_descuentos`, `mcajeros_propios_descuentos`

- **La derivada más interesante:** `catm_trx_other / (catm_trx + catm_trx_other)`. Quien usa cada vez más **cajeros de otras redes** probablemente ya está operando desde otro banco. Señal de migración difícil de ver de otra forma. (Denominador que se va a cero: macro segura.)
- **Digital vs. presencial:** `ccajas_*` alto con `tmobile_app` en cero es un segmento con riesgo y acción distintos que los digitales puros.
- **`tcallcenter`:** el impulso es leerlo como reclamos. **No lo hagas sin diccionario.** Pero si resulta ser contacto con soporte, un pico de llamadas antes de la baja es el cliente avisando que se va — y para los vip, que tienen ejecutivo de cuenta, es doblemente relevante.
- **Abandono silencioso:** caída de `chomebanking_transacciones` y `cmobile_app_trx` con el saldo intacto. El cliente ya se fue mentalmente. Es el perfil "el que ya se mudó" de la Parte VI, y se mide con pendientes.

## IV.8 Fricción y conflicto

`ccheques_depositados_rechazados`, `mcheques_depositados_rechazados`, `ccheques_emitidos_rechazados`, `mcheques_emitidos_rechazados`, `ccheques_depositados`, `mcheques_depositados`, `ccheques_emitidos`, `mcheques_emitidos`, `cdescubierto_preacordado`, más las comisiones de IV.1

Familia chica pero cualitativamente distinta: son **eventos negativos concretos**, no niveles.

- **Derivada:** tasa de rechazo (`rechazados / total`), separando emitidos de depositados. Un cheque propio rechazado es un problema del cliente; uno depositado rechazado es un problema que el cliente sufrió.
- **Hipótesis:** los eventos de fricción son gatillos de baja con timing identificable — candidatos a señal temprana con ventana de acción.
- **Ojo con el denominador:** la enorme mayoría de los clientes tiene cero cheques. La tasa de rechazo va a ser NULL para casi todos. Eso está bien — es el comportamiento correcto de la macro segura — pero significa que la variable sólo informa sobre un subconjunto chico. No la descartes por tener 95 % de NA; medila sobre quienes sí operan con cheques.

## IV.9 Segmento y profundidad de relación (transversal)

**`cliente_vip` entra acá.** Es una variable de segmento de alto valor, y tiene tres usos distintos.

1. **Como corte descriptivo.** Todo lo que calcules en las familias anteriores, calculalo también partido por vip. Las diferencias son el material del video.
2. **Como feature del modelo.** Si el comportamiento de baja de los vip difiere del resto, `cliente_vip` y sus interacciones aportan al modelo.
3. **Como evento, si transiciona.** Si `cliente_vip` pasa de 1 a 0 para un mismo cliente entre meses, eso es una **pérdida de estatus dentro del paquete** y es candidata fuerte a señal temprana de baja. Se detecta con `lag(cliente_vip)` en una línea. Verificalo (test de la sección 0.2).

**El cálculo que hay que correr sí o sí:** tasa de `BAJA+2` de vip contra no-vip. Los dos resultados son buenos:
- Si los vip se van **menos**, confirma que el banco los retiene mejor, y el foco del video pasa a la masa premium.
- Si se van **más**, tenés el titular: *el banco está perdiendo justo a los que no puede perder.*

**Profundidad de relación.** `cproductos` es la variable resumen. Además construí:

- cantidad de **familias** de producto activas (0 a 8, según las secciones de arriba)
- flags: `tiene_tarjeta`, `tiene_credito`, `tiene_inversion`, `tiene_payroll`, `tiene_seguro` — los mismos que III.1 te obliga a conservar
- y la tabla que probablemente sea **el gráfico más claro del video**:

| cantidad de productos | clientes | BAJA+2 | tasa |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4+ | | | |

Si la relación resulta monótona decreciente —hipótesis razonable— tenés un mensaje en una línea: *la profundidad de relación es la mejor póliza contra la fuga*, con acción directa.

**Ojo:** el mínimo de esta tabla también es tu test de composición de la sección 0.2. Si no hay casi nadie con 1 producto, la base está filtrada.

`cliente_edad` y `cliente_antiguedad` cierran el perfil estructural. **No asumas linealidad:** bineá por cuantiles (`ntile` de `z402`, sin `partition by` esta vez — o con, si querés controlar drift) y calculá tasa de `BAJA+2` por bin. La antigüedad suele tener forma de U o de J.

*(Nota sobre `cliente_antiguedad`: `z402` celda 15 la usa de ejemplo y aclara que **no sufre data drifting** — está sólo para mostrar la sintaxis. Y en la celda 25 la usa como eje X de `regr_slope`, lo cual es un truco elegante: como avanza de a 1 por mes, hace de reloj. Entendé eso antes de copiar la línea, porque si el cliente tiene meses faltantes el eje se desalinea.)*

---

# PARTE V — Señales tempranas vs. tardías, y el feature engineering temporal

Esta parte cambió por completo respecto de la v1. Antes era "escribí una lista de hipótesis de features históricos". Ahora es "escribí la lista **y generala**, porque `z402` te da toda la maquinaria".

## V.1 El diagnóstico: tres comparaciones

`BAJA+1` es un negativo en tu modelo, pero conceptualmente es el mismo cliente un mes más tarde. Explotalo.

**Sobre 15–20 variables candidatas (no 154):**

1. **`BAJA+2` vs `CONTINUA`** — ¿qué distingue al que todavía está pero se va en dos meses? La más cercana a señal accionable.
2. **`BAJA+1` vs `CONTINUA`** — ¿cómo luce alguien que ya está saliendo?
3. **`BAJA+1` vs `BAJA+2`** — la más valiosa conceptualmente: ¿qué cambia entre estar a dos meses y estar a uno?

**Cómo leerlo:** si una variable se ordena monótonamente `CONTINUA → BAJA+2 → BAJA+1`, estás viendo **deterioro progresivo**, y es candidata prioritaria a delta-lags y tendencias. Si sólo se mueve en `BAJA+1`, es señal tardía: predice churn pero llega tarde para esta campaña.

Medianas y deciles por clase. No histogramas.

**Y tenés un atajo:** `z601` celda 39 ya grafica la distribución de la probabilidad predicha para las tres clases por separado. Si `BAJA+1` y `BAJA+2` se ven casi iguales ahí, ya sabés que el modelo las trata como la misma población — que es la evidencia a favor del target combinado y, de paso, el argumento de por qué la comparación 3 es interesante.

**Y una cuarta población, cortesía de Denicolay: los intermitentes.** Son clientes que se fueron y volvieron. Compará su perfil contra los `BAJA+2`. Si se parecen, tenés evidencia de que ese perfil **es recuperable** — que es literalmente el argumento comercial de toda la campaña. Con ~15 casos no vas a tener significancia estadística, pero como pieza narrativa vale mucho.

## V.2 La maquinaria: qué construir, con qué línea

`z402` da cinco herramientas. Esta tabla es el núcleo del feature engineering de la competencia:

| Herramienta | Sintaxis | Qué captura | Celda |
|---|---|---|---|
| **Lag** | `lag(x, 1) over (partition by numero_de_cliente order by foto_mes)` | el valor del mes pasado | 17 |
| **Delta** | `x - lag(x, k) over (...)` | cuánto cambió en k meses | 19 |
| **Media móvil** | `avg(x) over ventana_3` | el nivel típico reciente | 21 |
| **Max/min móvil** | `max(x) over ventana_3`, `min(x) over ventana_3` | el techo y el piso recientes | 23 |
| **Pendiente** | `regr_slope(x, cliente_antiguedad) over ventana_3` | **la tendencia**: ¿sube o baja, y qué tan rápido?  | 25 |

donde

```sql
window ventana_3 as (
    partition by numero_de_cliente
    order by foto_mes
    rows between 3 preceding and current row
)
```

**Las cinco familias de derivadas que valen la pena**, en orden de mi apuesta sobre cuál rinde más:

1. **Ratio contra el propio pasado:** `x / avg(x) over ventana_3`. Es la forma canónica de "este mes está por debajo de lo normal *para este cliente*". Normaliza por cliente, no por población — y por eso es robusta a drift y a heterogeneidad de cartera a la vez.
2. **Pendiente:** `regr_slope`. Una sola columna que resume si la variable viene cayendo. Para `ctrx_quarter`, `mcuentas_saldo`, `mtarjeta_visa_consumo` y `mrentabilidad` es probablemente la mejor feature que vas a construir.
3. **Delta simple:** `x - lag(x, 1)`. Barata, interpretable, y el punto de comparación obligatorio contra las dos de arriba.
4. **Rankings intra-mes:** `percent_rank() over (partition by foto_mes ...)`. Es la de la Parte II.2 — anti-drift, no temporal. Ortogonal a las otras tres y por eso se combina bien.
5. **Flags de evento:** `perdio_payroll`, `cancelo_credito`, `perdio_vip`, `plazo_fijo_no_renovado`. Construidas con `lag` + un `case when`. Son las menos "estadísticas" y las más narrables — y para el video valen más que las otras cuatro juntas.

**El generador.** No tipees 600 columnas. La celda 29 de `z402` es el patrón:

```python
campos = ['ctrx_quarter', 'mcuentas_saldo', 'mrentabilidad', ...]
nuevos_features = ""
for campo in campos:
    nuevos_features += f"\n, regr_slope({campo}, cliente_antiguedad) over ventana_3 as {campo}_slope_3"
print(nuevos_features)
```

Pegás la salida en la query y listo. La cátedra remata con que "esto es como funcionan sistemas tipo **dbt**" — y tiene razón, pero para vos lo importante es otra cosa: **es código que podés reescribir de memoria mañana**, que es la regla de autoría que te pusiste. Un loop de cuatro líneas lo defendés en un oral; un notebook de 600 columnas copiadas, no.

**Advertencia de recursos, que la cátedra también marca:** la creación de features es rápida pero intensa. 154 columnas × 5 derivadas = 770 columnas × 983.061 filas. Eso puede no entrar en RAM. Dos salidas: (a) generá por familia y andá guardando, (b) VM de GCP. **Y acordate de la regla no negociable: si tarda más de 15 minutos, va a `tmux` con log a archivo, no a una pestaña de Chrome.**

**Salida obligatoria:** lista escrita de hipótesis de features priorizada por lo que encontraste en V.1, con la derivada elegida para cada una y **el resultado en ganancia**. Insumo directo de la Última Competencia y material del video a Michelina.

## V.3 El experimento que hay que correr, y en qué orden

No agregues las cinco familias juntas: no vas a saber cuál funcionó. Ceteris paribus, una por vez, ≥ 5 semillas, contra el mismo baseline:

| # | Feature set | Qué contesta |
|---|---|---|
| 0 | crudo (154 columnas) | baseline |
| 1 | crudo + deltas 1 y 2 | ¿el cambio aporta sobre el nivel? |
| 2 | crudo + medias móviles y ratios contra media | ¿la normalización por cliente aporta? |
| 3 | crudo + pendientes | ¿una columna resume mejor que tres lags? |
| 4 | rankings intra-mes **en vez de** los valores crudos | ¿el anti-drift compensa lo que se pierde? |
| 5 | crudo + flags de evento | ¿los eventos discretos aportan sobre lo continuo? |

El 4 es el único que **reemplaza** en vez de agregar, y es el más interesante de los seis: si gana, tenés un resultado fuerte y publicable en Zulip. Si pierde, también — y es candidato a Experimento Colaborativo.

---

# PARTE VI — El video a Miranda (27-sep) y `z501_cluster_rf.py`

**Es otro problema.** El modelo separa, dos meses antes, a los que van a vivir de los que van a morir. Miranda pide lo contrario: **describir a los que ya murieron** y encontrar subgrupos con acción comercial. Es segmentación, no predicción. Y la métrica no es la ganancia: es si el perfil sugiere algo que el banco pueda hacer el lunes.

**Población:** tus `BAJA+1` y `BAJA+2`. Sin filtro, porque la base es toda premium (supuesto de 0.2, declaralo en el video).

## VI.1 El eje del video: no todas las bajas cuestan lo mismo

Este es el ángulo, y sale directo de `cliente_vip`.

La fórmula de la competencia trata todas las bajas por igual: 1.072.500 por cada acierto, sea quien sea el cliente. **Miranda no las trata igual.** Un vip que se va —famoso, gran gastador, cliente que el banco no quiere perder— no cuesta lo mismo que una baja promedio.

Ese contraste hace el video, y te da la estructura:

1. **Cuánto se va y cuánto vale.** Volumen de bajas y su valor económico (rentabilidad, saldos, consumo), partido vip / no-vip.
2. **Los perfiles**, tres o cuatro, cada uno con tamaño, dos o tres rasgos distintivos y **una acción concreta**.
3. **El perfil vip por separado**, si resulta distinto del resto. Probablemente sea la parte que más le interesa.
4. **Un cierre accionable:** qué haría el banco el lunes.

Un perfil sin acción asociada no entra al video.

## VI.2 Qué hace `z501_cluster_rf.py`, en cinco pasos

Este script **es** la Parte VI. Entenderlo es más importante que correrlo, porque en diciembre te van a preguntar por qué elegiste este método.

1. **Arma la muestra** (query `_QUERY_MUESTRA`, DuckDB). Toma **todos los clientes que alguna vez fueron `BAJA+2`, con toda su historia**, y les suma **la misma cantidad de clientes fieles** — definidos como presentes en los 6 meses y nunca `BAJA+1`/`BAJA+2`. El muestreo de fieles es `ORDER BY hash(id + seed) LIMIT n`: un shuffle determinístico, reproducible por semilla. Elegante, y es la clase de detalle que conviene poder explicar.

2. **Entrena un Random Forest** sobre filas cliente-mes, con target `grupo` (1 = el cliente es BAJA+2, 0 = fiel). Tres decisiones de diseño:
   - `min_samples_leaf=50` alto **a propósito**: quiere hojas que sean *poblaciones de clientes*, no filas memorizadas.
   - `class_weight='balanced'` porque un fiel aporta 6 filas y un BAJA+2 entre 2 y 5.
   - `oob_score=True` para tener una medida honesta sin separar test.

3. **Le pregunta a cada hoja "cuál fue el último atributo que te definió"** (`atributo_de_hoja`). Después cuenta, para cada cliente, en cuántos pares (mes, árbol) cayó en una hoja definida por cada atributo. Resultado: una matriz `cliente × atributo` de proporciones, que suma 1 por fila.

4. **Clusteriza esa matriz con KMeans** (`k=5` por default, máximo 5), **sólo sobre los `BAJA+2`**. Los clusters se renumeran para que `cluster_1` sea el más grande.

5. **Caracteriza y grafica.** Por cluster, los `top-n` atributos por **lift** (`share_cluster / share_global`). Y después un PDF apaisado con una página por atributo: la evolución mensual de la variable por cluster, con banda de incertidumbre y una tabla de cuántos clientes de cada cluster sobreviven cada mes.

**La idea de fondo, que es lo que hay que poder defender:** la distancia entre dos clientes no es la distancia euclídea entre sus 154 variables. Es **qué tan parecido los trató el bosque**. Y como el bosque fue entrenado para separar bajas de fieles, los ejes de esa distancia son, por construcción, los ejes que importan para el churn. KMeans sólo corta ese espacio en k pedazos.

**Por qué eso es exactamente lo que Miranda pide:** "en qué se diferencian entre sí los que ya murieron" — no *en qué se diferencian de los vivos*, sino *entre sí*. El script clusteriza únicamente los `BAJA+2` (paso 4), pero en un espacio definido por lo que los separa de los fieles. O sea: **distintas razones por las que el bosque los marcó**. Eso es la definición operativa de "perfiles de baja".

## VI.3 Cómo correrlo

```bash
python z501_cluster_rf.py --csv competencia_01_ct_julia.csv --k 5 --top-n 3 --seed <tu_semilla>
```

Necesita el CSV **con la clase ternaria ya construida** — el default apunta a `competencia_01_ct_julia.csv`, así que usá el tuyo. Corre en la laptop: son ~5 k clientes BAJA+2 + 5 k fieles, unas 50–60 k filas. No hace falta VM.

Opciones que importan:

| Flag | Default | Cuándo tocarlo |
|---|---|---|
| `--k` | 5 | **máximo 5** (sólo hay 5 colores definidos). Probá 3 y 4: tres perfiles claros valen más que cinco borrosos |
| `--top-n` | 3 | atributos definitorios por cluster |
| `--banda` | `ic95` | `ic95` para mostrar si la diferencia es real; `iqr` (mediana + Q25–Q75) para variables de monto con cola larga |
| `--min-samples-leaf` | 50 | subirlo = clusters más gruesos y estables; bajarlo = más finos y más ruidosos |
| `--seed` | 214363 | **cambialo por una de tus cinco semillas declaradas** |

**Cómo leer el PDF.** Son ~155 páginas: la de resumen más una por atributo. No las mires todas. `orden_atributos` pone **primero las definitorias** (las del `top-n` de cada cluster, en orden de cluster) y después el resto alfabético. **Las primeras 15 páginas después del resumen son el 90 % del valor.** El resto es para chequear una hipótesis puntual.

En cada página mirá tres cosas:
- **¿La línea del cluster se despega de las otras?** Si un atributo "define" a un cluster pero las cinco líneas están pegadas, el lift es alto por un artefacto del bosque, no por una diferencia real de comportamiento. Descartalo.
- **¿Las bandas se solapan?** Con `ic95`, bandas que no se tocan = diferencia que podés afirmar en cámara.
- **La tabla de `n` al pie.** Los clusters se achican mes a mes porque los clientes se van dando de baja. Si un cluster llega a 202106 con 12 clientes, cualquier cosa que diga esa línea en el último mes es ruido.

## VI.4 Las cuatro cosas que hay que hacerle antes de creerle

Acá es donde tu trabajo se separa del de quien corrió el script y pegó el PDF.

1. **Correlo dos veces: con y sin las variables sospechosas de fuga.** Como dice III.3, `columnas_features` mete *todo* lo numérico, incluidos `Visa_status`, `Master_delinquency` y las fechas de mora. Si esas dominan el bosque, tus "perfiles de cliente" son en realidad **estados administrativos del proceso de baja** — y le vas a contar a la directora comercial que el perfil que se va es "el que tiene el status en 9". Es el error más caro que podés cometer en este video. Filtralas y compará: si los clusters cambian mucho, la versión limpia es la que va al video.

2. **Chequeá estabilidad con ≥ 3 semillas.** KMeans con `n_init=20` es razonablemente estable, pero el muestreo de fieles también depende de la semilla. Si con tres semillas los tamaños de cluster y los atributos definitorios se mantienen, tenés un resultado. Si bailan, bajá `k`. **Esto no es opcional:** es la refutación que la cátedra espera y es media diapositiva del video.

3. **Probá `k` = 3, 4 y 5.** El default de 5 no tiene justificación teórica. Tres perfiles nítidos con acción clara le sirven más a Miranda que cinco que se pisan. Y "elegí 3 porque con 5 dos clusters compartían los mismos atributos definitorios" es una respuesta de oral excelente.

4. **Traducí cada cluster a una frase de negocio antes de aceptarlo.** Si mirás los tres atributos definitorios de un cluster y no podés escribir "estos son los que ___", el cluster no existe para el video aunque exista matemáticamente. Contrastalo contra las hipótesis de perfiles de abajo: si uno de los clusters se parece a "el desvinculado" o "el estresado", ya tenés el guion escrito.

**Y una limitación que conviene declarar, no esconder:** el `grupo=1` incluye *toda la historia* del cliente BAJA+2, incluyendo meses en los que estaba perfectamente sano. Eso es deliberado —el script perfila **clientes**, no **meses**— pero significa que el OOB accuracy que reporta no es comparable con el de un modelo predictivo, y que las etiquetas tienen ruido por diseño. Si Miranda no lo va a preguntar, Michelina sí, y el jurado del torneo también.

## VI.5 Hipótesis de perfiles — conjeturas a confirmar, algunas van a resultar falsas

Escribí estas **antes** de correr `z501`. Si el script te devuelve algo parecido, tenés confirmación; si te devuelve algo distinto, tenés un hallazgo. Si lo corrés primero y después racionalizás, no tenés nada.

| Perfil tentativo | Firma en los datos | Acción posible |
|---|---|---|
| **El desvinculado** | sin payroll, sin débitos automáticos, sin seguros, pocos productos | migrar débitos, ofrecer acreditación de haberes |
| **El estresado** | utilización de tarjeta alta, paga el mínimo, adelantos de efectivo, mora | refinanciación, no retención comercial |
| **El que ya se mudó** | saldo estable pero transaccionalidad en caída, ATM de otras redes, transferencias emitidas altas | detección temprana + contacto proactivo |
| **El caro** | comisiones de mantenimiento altas contra rentabilidad y uso bajos | revisión del esquema de comisiones |
| **El que terminó de pagar** | crédito que llega a cero, sin otros anclajes | oferta de renovación al momento de la cancelación |
| **El vip que se apaga** | vip con consumo y transaccionalidad en caída pero saldo intacto | ejecutivo de cuenta, contacto personal |

## VI.6 El as en la manga: los intermitentes

Denicolay cuenta ~15 clientes que se fueron y volvieron, y menciona que el ejecutivo de cuenta convenció a algunos mostrándoles el beneficio que perdían.

Eso es, en la propia base, **una prueba de concepto de la campaña de retención**. Si podés mostrar que los intermitentes caen mayoritariamente en un cluster identificable, el argumento se cierra solo: *este perfil es recuperable, y ya lo recuperamos quince veces.*

**Y es fácil de medir con lo que ya tenés:** `z501` te devuelve `labels` (cliente → cluster). Cruzá esa serie contra tu lista de intermitentes y contá. Con quince casos no hay estadística. Como cierre de video, es potente. Usalo con honestidad sobre el n.

## VI.7 Método y advertencias

**El clustering jerárquico basado en distancia de Random Forest** está en el programa (capítulo 28); `z501` implementa una variante (perfil de hojas + KMeans en vez de distancia de proximidad + jerárquico). Si te preguntan la diferencia: la distancia de proximidad clásica es "¿cuántas veces estos dos clientes cayeron en la misma hoja?", que es O(n²) y no escala; el perfil de hojas es "¿qué tipo de hojas visitó cada cliente?", que es O(n) y se puede clusterizar con KMeans. Se pierde algo de resolución y se gana poder correrlo en una laptop.

**Trampa:** contarle a Miranda el modelo predictivo en palabras de negocio. Ese es el otro video, en diciembre, para Michelina. Miranda quiere perfiles y acciones. **No menciones Random Forest, KMeans ni lift en el video.** El PDF es tu insumo, no tu entregable.

**Quién es:** llega a gerencia general en dos años, busca ideas que muevan la aguja, y "buen trabajo" de su boca es una sentencia. Tres perfiles con acción concreta y un número de impacto le sirven más que doce hallazgos.

---

# PARTE VII — Lo que este EDA deliberadamente no hace

Cada ítem consume días y no mueve ni la ganancia ni la nota:

- Histogramas de las 154 variables.
- Tabla de 16 estadísticos por columna partida por clase, sobre todo el dataset.
- Matriz de correlación completa para "detectar redundancia": LightGBM convive con variables correlacionadas sin drama.
- Correlación con el target como criterio de selección: target al 0,7 %, relaciones no lineales, categóricas disfrazadas de enteros. No mide lo que creés.
- Imputación de nulos por defecto.
- Normalización o estandarización: a un árbol le da igual cualquier transformación monótona. **(Rankear con `percent_rank` no es esto: no es una transformación monótona global, es una por mes. Ahí está toda la diferencia, y es la respuesta si alguien te dice "pero a un árbol le da igual".)**
- Tests de normalidad.
- **Leer las 155 páginas del PDF de `z501`.**
- **Optimizar hiperparámetros antes de arreglar la métrica de la Parte I.2.** Horas de VM buscando el mejor modelo según una métrica rota.

Si te encontrás haciendo alguna de estas, probablemente estés evitando la parte difícil.

---

# PARTE VIII — Bugs y trampas en los scripts de clase

No son detalles de estilo. Los dos primeros te hacen entregar resultados equivocados **sin que nada falle ni tire error**, que es la peor clase de bug.

## VIII.1 `z402`: el `COPY` final no guarda el feature engineering

Celda 32:

```sql
COPY competencia_01 TO '{dataset_path}competencia_01_fe.csv' (FORMAT CSV, HEADER TRUE);
```

**Dos problemas apilados:**

1. **La interpolación está rota.** `jupysql` usa Jinja con llaves dobles — la celda 4 lo hace bien (`"{{dataset_path + dataset_file}}"`). Acá hay llaves simples, así que el archivo se va a llamar literalmente `{dataset_path}competencia_01_fe.csv`.
2. **Y lo grave: `competencia_01` es la tabla cruda.** Todas las features de las celdas 15 a 29 se calcularon en `select` sueltos que nunca se materializaron. El `COPY` exporta las 154 columnas originales, sin una sola derivada.

**El arreglo:** `CREATE OR REPLACE TABLE competencia_01_fe AS SELECT ... <tus features> ... FROM competencia_01 WINDOW ventana_3 AS (...)`, y recién ahí el `COPY` sobre `competencia_01_fe`, con llaves dobles.

Si no lo ves, corrés todo el pipeline, exportás, entrenás — y el modelo con "feature engineering" da exactamente lo mismo que el baseline. Vas a perder medio día buscando el error en LightGBM.

## VIII.2 `z601`: `best_iter` se calcula y después se ignora

La celda 8 guarda con cuidado la mejor cantidad de iteraciones de cada trial:

```python
trial.set_user_attr("best_iter", best_iter)
```

Y después, en las celdas 18, 30, 32, 36 y 56, siempre lo mismo:

```python
model = lgb.train(params, train_data_lgb, num_boost_round=250) #best_iter)
```

**El 250 está hardcodeado y el `best_iter` está comentado.** Consecuencia directa: Optuna optimiza `learning_rate` sabiendo que un lr bajo necesita más árboles (la celda 13 lo dice explícitamente), pero después entrenás todos los modelos con 250 rondas fijas. **El trial ganador puede ser el que necesitaba 800 árboles, y lo estás entrenando con 250.** Estás comparando modelos en condiciones que no son las que los hicieron ganar.

Probablemente la cátedra lo dejó así para que el notebook corra rápido en clase. Para tu competencia, destapalo. Es un `#` de diferencia y puede ser varios puntos de ganancia.

## VIII.3 `z601`: la métrica de la CV no es la métrica de la competencia

Ya está desarrollado en la Parte I.2, pero va acá también porque es un bug de método, no de código: `ganancia_prob_lgm` corta en `0.025` fijo, y el resultado de la competencia se mide en el corte óptimo por cantidad de envíos. Con el target combinado (`BAJA+1`+`BAJA+2`) el 0,025 queda muy corrido y la métrica miente. **Es, además, la respuesta a la tarea de la celda 59.**

## VIII.4 `z501`: mete las variables de fuga al bosque

Ya desarrollado en VI.4.1. `columnas_features` filtra id, mes, target y grupo — nada más. Todo lo demás entra, incluidos los candidatos a leak de III.3.

## VIII.5 Cosas menores, pero anotalas

- **`z601` celda 56** entrena sobre `train_data_lgb2` (target combinado) y predice sobre `X_test`, no `X_test2`. Las columnas terminan coincidiendo por cómo se hicieron los slices, así que **no** rompe — pero es frágil: si alguna vez recreás `test_data` después de la celda 49, se rompe silenciosamente. Usá `X_test2`.
- **`z501`** tiene `--k` limitado a 5 por la lista `COLORES_CLUSTER`. Si querés más, agregá colores.
- **`z402` celda 15** usa `cliente_antiguedad` de ejemplo y la propia cátedra aclara que esa variable *no* sufre drift. No saques conclusiones de esa salida: es sintaxis, no hallazgo.
- **`z402` celda 25** usa `cliente_antiguedad` como eje X de `regr_slope` porque avanza de a 1 por mes. Si un cliente tiene meses faltantes, el eje se desalinea. Con clientes que entran y salen, verificá.

---

## Entregables

Python + DuckDB, en el repo, numerados para que el orden de ejecución sea obvio:

| Archivo | Qué hace | Insumo de |
|---|---|---|
| `00_verificaciones.py` | intermitentes, prevalencia por mes, los 5 tests de composición premium (0.2) | todo |
| `01_ganancia.py` | función de ganancia **corregida** (corte por top-N), curva ganancia vs. envíos, KS/lift/captura | Partes I, III.3 |
| `02_consistencia.py` | tabla `variable × mes` con %NA, %cero, mediana, p99 | Parte II.1 |
| `03_drifting.py` | ranking de drift al mes de aplicación + adversarial validation | Parte II.2 |
| `04_features.sql` / `.py` | macros (`suma_sin_null` + **ratio seguro**), derivadas de la Parte IV, las 5 familias temporales de la Parte V, generador programático, y el `CREATE TABLE` que **sí** materializa | Partes IV, V |
| `05_experimentos.py` | los 6 feature sets de V.3, ≥ 5 semillas, append a CSV por corrida | Parte V |
| `z501_cluster_rf.py` | ya existe — correr con tu semilla, con y sin variables de fuga, k ∈ {3,4,5} | Parte VI |
| `variables_sospechosas.md` | diagnóstico / tratamiento / ganancia con y sin, una línea por variable | Parte II.3 |
| `hipotesis_features.md` | salida de la Parte V, priorizada, con resultado | Última Competencia |
| `perfiles_premium.md` | los 3–4 clusters traducidos a frases de negocio + acción | video Miranda |
| `bitacora.md` | ya existe — seguir: fecha / hipótesis / qué cambié / semillas / ganancia ± desvío / conclusión en una frase | todo |

**Y en la bitácora, con fecha y con todas las letras:** el supuesto premium de 0.2, la convención de la función de ganancia de I.1, y la decisión sobre los intermitentes de 0.1.

## Reparto de tiempo — 20 días al cierre

Esto ya no es una estimación abstracta: son 13 días al video y 20 al cierre, con ~8 h semanales de dedicación esperada (vas a necesitar más).

| Bloque | Horas | Cuándo | Por qué ahí |
|---|---|---|---|
| **0 + I.1/I.2** (verificaciones + arreglar la métrica) | 3 h | **ya** | La métrica rota invalida todo lo que optimices después. Es lo primero. |
| **V.2 macros + 5 familias temporales** | 4 h | días 1–3 | Es el insumo de todo lo demás, incluido el video. Y es lo que más rinde. |
| **VI** (`z501` × semillas × con/sin leaks × k) | 5 h | días 3–7 | **Deadline duro 27-sep.** Correr temprano: si los clusters no cierran, necesitás margen. |
| **Video Miranda** (guion, grabación, edición) | 6 h | días 8–13 | 5 minutos de video son muchas horas. No lo empieces el 26. |
| **II** (consistencia + drifting) | 5 h | días 8–14, en paralelo | El núcleo defensivo del modelo. Corre mientras editás el video. |
| **III** (semántica + fugas) | 3 h | días 14–16 | |
| **IV** (derivadas de negocio) | 3 h | días 14–17 | Buena parte ya salió como subproducto de V y VI. |
| **V.3** (los 6 experimentos) | 6 h | días 16–20 | Corridas largas: `tmux`, log a archivo, append a CSV. |

**En paralelo desde hoy, no después:** la infraestructura de corridas largas en GCP. El Grid Search de 12+ horas no espera a que termines el EDA, y la regla no negociable es que nada largo corre en una pestaña de Chrome.

**Si tenés que recortar:** sacá la Parte IV (las derivadas de negocio son lindas y las tres cuartas partes salen gratis de V). No saques II ni la corrección de la métrica de I.2.

---

## La pregunta del examen

Por cada bloque que cierres, respondé en 60 segundos y sin mirar:

- **Parte I:** ¿por qué el corte es 0,025 y por qué deja de servir cuando junto las dos bajas?
- **Parte II:** ¿qué arregla `percent_rank() over (partition by foto_mes ...)` que no arregla estandarizar?
- **Parte III:** ¿por qué `suma_sin_null` es correcta y peligrosa al mismo tiempo?
- **Parte V:** ¿por qué `x / avg(x) over ventana_3` puede ganarle a `x` y a `x - lag(x,1)` a la vez?
- **Parte VI:** ¿por qué la distancia entre dos clientes es la distancia entre sus perfiles de hojas, y no la euclídea entre sus 154 variables?

Si alguna no sale, ese bloque no está cerrado.

---

*Todo lo que este documento afirma sobre el dataset es hipótesis hasta que lo verifiques contra el archivo. Todo lo que afirma sobre la materia, contra Zulip y la clase. La composición premium de la base (0.2) es un supuesto razonado, no un hecho confirmado: declaralo como tal hasta que Denicolay responda. El significado preciso de `cliente_vip`, `tcallcenter`, `cinversion1`, `cinversion2` y `active_quarter` requiere el diccionario de datos, que sigue sin cargarse. Los bugs de la Parte VIII son lectura de código, no ejecución: verificalos corriendo los notebooks antes de citarlos en Zulip.*
