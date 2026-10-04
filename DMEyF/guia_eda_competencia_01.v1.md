# Guía de EDA — DMEyF 2026, Primera Competencia

**Dataset:** `competencia_01_crudo.csv` — 154 columnas · **Cierre competencia:** 04-oct · **Video Miranda:** 27-sep
**Estado:** clase ternaria ya construida · población premium resuelta como supuesto declarado

---

## Cómo usar este documento

Dos objetivos que comparten insumo pero no producto:

- **Ganancia** (competencia): que el modelo no se rompa contra el mes del futuro, y que el feature engineering apunte a lo que separa `BAJA+2`.
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

**Chequeo de sanidad general:** prevalencia de `BAJA+2` por mes. Debe rondar 0,5–1 % en todos los meses etiquetables (tu 202104 dio 0,70 %). Y recordá que **los dos últimos meses del dataset no se pueden etiquetar**.

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

**Cuidado al tipear:** `mttarjeta_visa_debitos_automaticos` y `mttarjeta_master_debitos_automaticos` tienen doble `t`. Es del dataset original.

## 0.4 Alertas de estructura, ya visibles en el header

1. **Granularidad mixta.** `active_quarter` y `ctrx_quarter` son **trimestrales** en un dataset mensual. Van a mostrar escalones en el análisis de consistencia longitudinal. No los marques como rotos por eso.

2. **Fechas como enteros.** `Master_Fvencimiento`, `Master_Finiciomora`, `Master_fultimo_cierre`, `Master_fechaalta` y sus cinco equivalentes de Visa son `BIGINT`. Mirá los valores: si son del orden de `20250131`, son fechas absolutas y **hay que convertirlas a antigüedad relativa al `foto_mes`**, o el modelo aprende calendario en vez de comportamiento. Verificación de dos minutos que cambia el tratamiento de nueve columnas.

3. **Dolarización.** `mcaja_ahorro_dolares`, `mplazo_fijo_dolares`, `minversion1_dolares`, `Master_msaldodolares`, `Visa_msaldodolares` y las cuatro de forex **driftean por tipo de cambio**, no por comportamiento. Tratarlas junto con las de pesos confunde el diagnóstico.

4. **Los sospechosos de fuga tienen nombre.** `Master_status`, `Visa_status`, `Master_delinquency`, `Visa_delinquency`. Ver Parte III.3.

---

# PARTE I — Fundaciones

## I.1 La función de ganancia

Implementá `ganancia(prediccion, clase_real)` en un archivo propio. Es la moneda de la materia: todo hallazgo que no se traduzca a esta función es decoración.

Ya venís trabajando con 1.072.500 por acierto y 27.500 por estímulo errado, y con el corte 0,025. **Derivá el corte vos, en papel, una vez**: si un cliente tiene probabilidad *p* de ser `BAJA+2`, la esperanza de estimularlo es lineal en *p*; igualás a cero y despejás. Foto al repo.

**Confirmá la convención:** ¿el costo del estímulo se cobra sólo a los falsos positivos, o a *todos* los estimulados con un bruto mayor por acierto? El umbral óptimo es el mismo; la ganancia absoluta que reportás, no. Preguntalo y anotalo con fecha.

**Y tenelo presente:** umbral de probabilidad y cantidad de envíos son dos formas de cortar el mismo ranking. Si el modelo está mal calibrado —y el boosting suele estarlo— cortar en 0,025 no es lo mismo que cortar en los top-N óptimos. La curva de ganancia contra cantidad de envíos es el gráfico que más veces vas a mirar en la materia.

*(Ya trabajaste esto con los árboles: el corte empírico 0,0275 no le gana al teórico 0,025 porque es artefacto de ese árbol, esa semilla y ese mes. Mismo principio.)*

## I.2 Composición del target en el tiempo

Tabla `foto_mes × clase_ternaria`, conteos y porcentajes, más un gráfico de tres líneas.

¿La prevalencia de `BAJA+2` es estable? ¿Hay meses con shock? ¿La cartera crece o se achica? ¿Numerador y denominador se mueven juntos?

Cualquier mes que se aparte más de ~30 % de la mediana es sospechoso, y qué hacés con él (usarlo, excluirlo, ponderarlo) es una decisión que tenés que poder justificar.

**Agregá una línea más al gráfico:** prevalencia de `BAJA+2` entre los vip. Si se mueve distinto del resto, ya tenés material para la Parte VI.

---

# PARTE II — Integridad temporal *(programa 13.1 a 13.3)*

**Esta es la parte que protege la ganancia, y es independiente de todo lo que discutimos sobre premium. Los datos son de un banco argentino real a lo largo de más de un año: cambios de sistema, productos discontinuados, redefiniciones e inflación. Una variable que en un mes mide una cosa y en otro mide otra es veneno silencioso, y ninguna métrica de performance te avisa.**

## II.1 Consistencia longitudinal *(13.1)*

**Producto:** tabla larga `variable × foto_mes` con `%NA`, `%cero`, mediana, p99 y media por celda. Se arma con un `melt` + agregación por grupo en `data.table`; no es un loop de 154 iteraciones. De ahí sale el entregable real: **lista corta de variables sospechosas con veredicto de una línea**.

**Los cuatro patrones a cazar:**

1. **Variable que nace o muere.** `%NA` salta de 0 a 100 (o al revés) en un mes puntual. Candidatas naturales: `cinversion2`/`minversion2`, `mpayroll2`/`cpayroll2_trx`, `catm_trx_other`/`matm_other`, `cmobile_app_trx`.
2. **Cambio de escala.** La mediana se multiplica o divide de golpe. **Distinguir de inflación:** el crecimiento suave y sostenido de *todas* las variables de monto a la vez no es rotura, es el país. Una sola variable que salta ×10 en un mes sí lo es.
3. **Cero que se vuelve NA.** `%cero` y `%NA` se intercambian entre meses consecutivos. Clásico de migración de sistema.
4. **Truncamiento.** El p99 se aplana en un valor fijo: alguien puso un tope.

**Trampa:** mirar sólo la media. Se mueve por outliers y da falso positivo en las ~60 variables de monto. Mirá mediana + `%NA` + `%cero` juntos.

## II.2 Drifting hacia el mes de aplicación *(13.2)*

El bloque anterior mira toda la serie; este mira el salto al futuro, y decide si el modelo funciona en la entrega.

Tres formas, de menos a más potente:

1. **Diferencia de deciles** entre el mes de aplicación y los de entrenamiento. Simple y transparente.
2. **PSI o KS.** Un número por variable, comparable entre variables.
3. **Adversarial validation.** Entrenás un clasificador para distinguir "mes de aplicación" vs "meses de entrenamiento". Si logra AUC alto, hay drift; y su *feature importance* ranquea exactamente las culpables. La más informativa, y el tipo de idea que te diferencia de la cohorte.

**Criterio:** drift alto **no implica eliminar**. Una variable que driftea pero predice mucho puede seguir conviniendo. Se decide contra la ganancia, no contra el PSI.

**Separá tres fuentes antes de concluir:** inflacionaria (las `m` en pesos), cambiaria (las `_dolares` y forex), estructural (el resto). Sólo la tercera es problema de datos.

## II.3 Variables rotas *(13.3)*

Para cada sospechosa: ¿la dejás, le asignás NA en los meses malos, la interpolás, la rankeás, o la eliminás?

El programa plantea la opción contraintuitiva —*asignarle NA*— como pregunta, no como receta. Es una invitación al experimento.

**Producto:** tabla de variable / diagnóstico / tratamiento / **ganancia con y sin**, más de una semilla. Un tratamiento por vez, ceteris paribus.

---

# PARTE III — Semántica de los datos

## III.1 Ceros, nulos y ausencia de producto *(13.4)*

**La pregunta central:** ¿`0`, `NA` y "no tiene el producto" son lo mismo?

Casi seguro que no, y la estructura de pares `c`/`m` da el test perfecto:

> Cuando `cprestamos_personales == 0`, ¿`mprestamos_personales` es `0` o es `NA`?

Corré ese cruce para cada par (préstamos, plazo fijo, inversiones, tarjetas, cheques, forex, payroll). La respuesta te da **la semántica real del NA en esta base**.

Importa porque "tiene tarjeta con consumo cero" y "no tiene tarjeta" son clientes distintos con riesgos distintos.

**Missingness como señal:** tasa de `BAJA+2` condicional a ser NA, por variable. Si difiere marcadamente de la general, el missing informa y merece un `isNA_X`. No generes 154 indicadores por las dudas.

**No imputes por reflejo.** LightGBM maneja NA nativamente y les da tratamiento propio en el split.

## III.2 Outliers *(13.5)*

Menos importante de lo que enseña la estadística clásica: un árbol corta por umbrales y es casi indiferente a la magnitud del extremo. Pero **LightGBM discretiza en bins**, y un outlier gigante puede degradar la resolución donde vive el 99 % de la cartera.

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

**Test operativo:** árbol de una sola variable. Si separa `BAJA+2` con pureza sospechosa, alarma.

**El contraste clave:** mirá el mismo estadístico en `BAJA+1`. Si la variable distingue muchísimo `BAJA+1` y poco `BAJA+2`, **no es leak: es señal tardía**. Legítimo, y material para la Parte V.

No elimines por sospecha sin medir.

---

# PARTE IV — Las familias de productos

**El contenido bancario. Sirve a los dos objetivos: hipótesis de feature engineering para el modelo, vocabulario de perfiles para Miranda.**

Método común: **tenencia** (¿lo tiene?), **intensidad** (¿cuánto lo usa?) y **la combinación**. Las derivadas son candidatas: cada una vale lo que valga su efecto medido sobre la ganancia.

## IV.1 Valor y rentabilidad

`mrentabilidad`, `mrentabilidad_annual`, `mcomisiones`, `mactivos_margen`, `mpasivos_margen`, `ccomisiones_mantenimiento`, `mcomisiones_mantenimiento`, `ccomisiones_otras`, `mcomisiones_otras`

Probablemente **no predice bien la salida, pero determina cuánto te importa acertar**. Un `BAJA+2` de alta rentabilidad y uno de rentabilidad negativa valen igual en la fórmula de la competencia, pero no valen igual para Miranda.

- **Derivadas:** `mrentabilidad / mrentabilidad_annual` (¿el mes fue mejor o peor que su promedio?), composición margen activo vs. pasivo, comisiones sobre rentabilidad.
- **Cruce obligatorio:** rentabilidad de vip vs. no-vip. Es la validación numérica de que los vip son efectivamente los más valiosos — y el número que le ponés a Miranda cuando decís "no todas las bajas cuestan lo mismo".
- **Hipótesis para Miranda:** hay un perfil que se va **porque le cobran**. Si `mcomisiones_mantenimiento` alto correlaciona con baja en clientes de bajo uso, es un perfil con acción obvia y barata.
- **Hipótesis de deterioro:** rentabilidad mensual cayendo contra su anual = candidata a señal temprana.

## IV.2 Relación transaccional núcleo — "¿soy tu banco principal?"

`tcuentas`, `ccuenta_corriente`, `mcuenta_corriente`, `mcuenta_corriente_adicional`, `ccaja_ahorro`, `mcaja_ahorro`, `mcaja_ahorro_adicional`, `mcaja_ahorro_dolares`, `mcuentas_saldo`, `cdescubierto_preacordado`, `ctarjeta_debito`, `ctarjeta_debito_transacciones`, `mautoservicio`, `ctransferencias_recibidas`, `mtransferencias_recibidas`, `ctransferencias_emitidas`, `mtransferencias_emitidas`, `cextraccion_autoservicio`, `mextraccion_autoservicio`, `ctrx_quarter`

**La pregunta de negocio más importante del EDA:** ¿el futuro `BAJA+2` ya dejó de usar al banco como banco principal, aunque el saldo siga ahí?

- **Derivadas:** ratio emitidas/recibidas (quien emite mucho más de lo que recibe está **sacando plata del banco**); transferencias emitidas sobre saldo; extracciones sobre ingresos; transacciones de débito por mes.
- **Hipótesis central:** la fuga de fondos precede a la baja formal. El dinero se va antes que el cliente. Si se confirma, `mtransferencias_emitidas` y sus deltas son de tus mejores features.
- **Distinción crítica:** *"tiene poco saldo"* y *"tenía saldo y lo bajó 70 %"* son clientes completamente distintos, y sólo el segundo es señal. Sólo se ve con historia — el argumento más fuerte a favor de lags y delta-lags.

## IV.3 Tarjetas — la familia más rica del dataset

Visa y Master, 22 columnas cada una, espejadas. Tratalas simétricamente y además agregadas.

**Ratios que casi siempre superan a los montos crudos:**

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

Dos perfiles de baja distintos, con acciones comerciales opuestas. Si los dos aparecen, ya tenés dos perfiles para Miranda.

**Y con el vip encima:** si los vip son grandes gastadores, el `mconsumototal` y el límite de compra deberían separarlos nítidamente. Un vip cuyo consumo se derrumba es probablemente la señal más cara del dataset.

**Cuidado:** `Visa_status`, `Visa_delinquency` y las fechas son candidatas a fuga (III.3).

## IV.4 Crédito

`cprestamos_personales`, `mprestamos_personales`, `cprestamos_prendarios`, `mprestamos_prendarios`, `cprestamos_hipotecarios`, `mprestamos_hipotecarios`

Pocas variables, conceptualmente potentes: **el crédito ata al cliente**. Un hipotecario es un ancla de años.

- **Hipótesis:** tasa de `BAJA+2` sensiblemente menor entre quienes tienen préstamos, y mínima entre los hipotecarios. Si se confirma, es un hallazgo interpretable y muy narrable.
- **Derivadas:** `tiene_credito`, `cantidad_tipos_credito`.
- **Contra-hipótesis:** el que **termina** de pagar queda súbitamente libre. Si detectás la caída del saldo a cero, ese es un momento de riesgo identificable y accionable.

## IV.5 Ahorro e inversión

`cplazo_fijo`, `mplazo_fijo_pesos`, `mplazo_fijo_dolares`, `cinversion1`, `minversion1_pesos`, `minversion1_dolares`, `cinversion2`, `minversion2`, `cforex`, `cforex_buy`, `mforex_buy`, `cforex_sell`, `mforex_sell`

- **Hipótesis:** la desinversión precede a la baja. Plazo fijo que no se renueva, inversiones que caen a cero.
- **Forex aparte:** comprar dólares es normal en Argentina; **vender** puede indicar necesidad de liquidez. El ratio buy/sell puede ser más informativo que cualquiera de los dos.
- `cinversion1` vs `cinversion2` son opacas sin diccionario.

## IV.6 Vinculación — lo que hace costoso irse

`cpayroll_trx`, `mpayroll`, `cpayroll2_trx`, `mpayroll2`, `ccuenta_debitos_automaticos`, `mcuenta_debitos_automaticos`, `ctarjeta_visa_debitos_automaticos`, `mttarjeta_visa_debitos_automaticos`, `ctarjeta_master_debitos_automaticos`, `mttarjeta_master_debitos_automaticos`, `cpagodeservicios`, `mpagodeservicios`, `cpagomiscuentas`, `mpagomiscuentas`, `cseguro_vida`, `cseguro_auto`, `cseguro_vivienda`, `cseguro_accidentes_personales`, `ccaja_seguridad`

**Probablemente la familia con más señal del dataset**, porque mide *costo de cambio*. Sueldo acreditado + tres débitos automáticos + dos seguros = mucha fricción para irse.

- **Derivadas:** `cantidad_debitos_automaticos_total` (cuenta + Visa + Master), `cantidad_seguros` (0 a 4), `tiene_payroll`, `tiene_payroll_doble`.
- **Hipótesis fuerte y muy narrable:** *la pérdida del payroll es el evento que precede a la baja*. Si el sueldo deja de acreditarse, el cliente cambió de trabajo o de banco, y la baja viene después. Si se confirma, es probablemente **el hallazgo más accionable del proyecto**: evento detectable, ventana de reacción, acción comercial evidente.
- **Y conecta con premium:** si el paquete se sostiene con acreditación de haberes o saldo mínimo, perder el payroll puede ser *literalmente* lo que hace perder el paquete. Vale la pena mirar si los que pierden payroll salen del dataset poco después.

## IV.7 Canales

`internet`, `thomebanking`, `chomebanking_transacciones`, `tmobile_app`, `cmobile_app_trx`, `tcallcenter`, `ccallcenter_transacciones`, `catm_trx`, `matm`, `catm_trx_other`, `matm_other`, `ccajas_transacciones`, `ccajas_consultas`, `ccajas_depositos`, `ccajas_extracciones`, `ccajas_otras`, `ccajeros_propios_descuentos`, `mcajeros_propios_descuentos`

- **La derivada más interesante:** `catm_trx_other / (catm_trx + catm_trx_other)`. Quien usa cada vez más **cajeros de otras redes** probablemente ya está operando desde otro banco. Señal de migración difícil de ver de otra forma.
- **Digital vs. presencial:** `ccajas_*` alto con `tmobile_app` en cero es un segmento con riesgo y acción distintos que los digitales puros.
- **`tcallcenter`:** el impulso es leerlo como reclamos. **No lo hagas sin diccionario.** Pero si resulta ser contacto con soporte, un pico de llamadas antes de la baja es el cliente avisando que se va — y para los vip, que tienen ejecutivo de cuenta, es doblemente relevante.
- **Abandono silencioso:** caída de `chomebanking_transacciones` y `cmobile_app_trx` con el saldo intacto. El cliente ya se fue mentalmente.

## IV.8 Fricción y conflicto

`ccheques_depositados_rechazados`, `mcheques_depositados_rechazados`, `ccheques_emitidos_rechazados`, `mcheques_emitidos_rechazados`, `ccheques_depositados`, `mcheques_depositados`, `ccheques_emitidos`, `mcheques_emitidos`, `cdescubierto_preacordado`, más las comisiones de IV.1

Familia chica pero cualitativamente distinta: son **eventos negativos concretos**, no niveles.

- **Derivada:** tasa de rechazo (`rechazados / total`), separando emitidos de depositados. Un cheque propio rechazado es un problema del cliente; uno depositado rechazado es un problema que el cliente sufrió.
- **Hipótesis:** los eventos de fricción son gatillos de baja con timing identificable — candidatos a señal temprana con ventana de acción.

## IV.9 Segmento y profundidad de relación (transversal)

**`cliente_vip` entra acá.** Ya no es una incógnita del enunciado: es una variable de segmento de alto valor, y tiene tres usos distintos.

1. **Como corte descriptivo.** Todo lo que calcules en las familias anteriores, calculalo también partido por vip. Las diferencias son el material del video.
2. **Como feature del modelo.** Si el comportamiento de baja de los vip difiere del resto, `cliente_vip` y sus interacciones aportan al modelo.
3. **Como evento, si transiciona.** Si `cliente_vip` pasa de 1 a 0 para un mismo cliente entre meses, eso es una **pérdida de estatus dentro del paquete** y es candidata fuerte a señal temprana de baja. Verificalo (test de la sección 0.2).

**El cálculo que hay que correr sí o sí:** tasa de `BAJA+2` de vip contra no-vip. Los dos resultados son buenos:
- Si los vip se van **menos**, confirma que el banco los retiene mejor, y el foco del video pasa a la masa premium.
- Si se van **más**, tenés el titular: *el banco está perdiendo justo a los que no puede perder.*

**Profundidad de relación.** `cproductos` es la variable resumen. Además construí:

- cantidad de **familias** de producto activas (0 a 8, según las secciones de arriba)
- flags: `tiene_tarjeta`, `tiene_credito`, `tiene_inversion`, `tiene_payroll`, `tiene_seguro`
- y la tabla que probablemente sea **el gráfico más claro del video**:

| cantidad de productos | clientes | BAJA+2 | tasa |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4+ | | | |

Si la relación resulta monótona decreciente —hipótesis razonable— tenés un mensaje en una línea: *la profundidad de relación es la mejor póliza contra la fuga*, con acción directa.

**Ojo:** el mínimo de esta tabla también es tu test de composición de la sección 0.2. Si no hay casi nadie con 1 producto, la base está filtrada.

`cliente_edad` y `cliente_antiguedad` cierran el perfil estructural. **No asumas linealidad:** bineá por cuantiles y calculá tasa de `BAJA+2` por bin. La antigüedad suele tener forma de U o de J.

---

# PARTE V — Señales tempranas vs. tardías

`BAJA+1` es un negativo en tu modelo, pero conceptualmente es el mismo cliente un mes más tarde. Explotalo.

**Tres comparaciones, sobre 15–20 variables candidatas (no 154):**

1. **`BAJA+2` vs `CONTINUA`** — ¿qué distingue al que todavía está pero se va en dos meses? La más cercana a señal accionable.
2. **`BAJA+1` vs `CONTINUA`** — ¿cómo luce alguien que ya está saliendo?
3. **`BAJA+1` vs `BAJA+2`** — la más valiosa conceptualmente: ¿qué cambia entre estar a dos meses y estar a uno?

**Cómo leerlo:** si una variable se ordena monótonamente `CONTINUA → BAJA+2 → BAJA+1`, estás viendo **deterioro progresivo**, y es candidata prioritaria a delta-lags y tendencias. Si sólo se mueve en `BAJA+1`, es señal tardía: predice churn pero llega tarde para esta campaña.

**Y una cuarta población, cortesía de Denicolay: los intermitentes.** Son clientes que se fueron y volvieron. Compará su perfil contra los `BAJA+2`. Si se parecen, tenés evidencia de que ese perfil **es recuperable** — que es literalmente el argumento comercial de toda la campaña. Con ~15 casos no vas a tener significancia estadística, pero como pieza narrativa vale mucho.

**Salida obligatoria:** lista escrita de hipótesis de features históricos —diferencias contra el mes anterior, ratios contra el promedio de los últimos meses, pendientes— priorizada por lo que encontraste. Insumo directo de la Última Competencia y material del video a Michelina.

Medianas y deciles por clase. No histogramas.

---

# PARTE VI — El video a Miranda (27-sep)

**Es otro problema.** El modelo separa, dos meses antes, a los que van a vivir de los que van a morir. Miranda pide lo contrario: **describir a los que ya murieron** y encontrar subgrupos con acción comercial. Es segmentación, no predicción. Y la métrica no es la ganancia: es si el perfil sugiere algo que el banco pueda hacer el lunes.

**Población:** tus `BAJA+1` y `BAJA+2`. Sin filtro, porque la base es toda premium (supuesto de 0.2, declaralo en el video).

## El eje del video: no todas las bajas cuestan lo mismo

Este es el ángulo, y sale directo de `cliente_vip`.

La fórmula de la competencia trata todas las bajas por igual: 1.072.500 por cada acierto, sea quien sea el cliente. **Miranda no las trata igual.** Un vip que se va —famoso, gran gastador, cliente que el banco no quiere perder— no cuesta lo mismo que una baja promedio.

Ese contraste hace el video, y te da la estructura:

1. **Cuánto se va y cuánto vale.** Volumen de bajas y su valor económico (rentabilidad, saldos, consumo), partido vip / no-vip.
2. **Los perfiles**, tres o cuatro, cada uno con tamaño, dos o tres rasgos distintivos y **una acción concreta**.
3. **El perfil vip por separado**, si resulta distinto del resto. Probablemente sea la parte que más le interesa.
4. **Un cierre accionable:** qué haría el banco el lunes.

Un perfil sin acción asociada no entra al video.

## Hipótesis de perfiles — conjeturas a confirmar, algunas van a resultar falsas

| Perfil tentativo | Firma en los datos | Acción posible |
|---|---|---|
| **El desvinculado** | sin payroll, sin débitos automáticos, sin seguros, pocos productos | migrar débitos, ofrecer acreditación de haberes |
| **El estresado** | utilización de tarjeta alta, paga el mínimo, adelantos de efectivo, mora | refinanciación, no retención comercial |
| **El que ya se mudó** | saldo estable pero transaccionalidad en caída, ATM de otras redes, transferencias emitidas altas | detección temprana + contacto proactivo |
| **El caro** | comisiones de mantenimiento altas contra rentabilidad y uso bajos | revisión del esquema de comisiones |
| **El que terminó de pagar** | crédito que llega a cero, sin otros anclajes | oferta de renovación al momento de la cancelación |
| **El vip que se apaga** | vip con consumo y transaccionalidad en caída pero saldo intacto | ejecutivo de cuenta, contacto personal |

## El as en la manga: los intermitentes

Denicolay cuenta ~15 clientes que se fueron y volvieron, y menciona que el ejecutivo de cuenta convenció a algunos mostrándoles el beneficio que perdían.

Eso es, en la propia base, **una prueba de concepto de la campaña de retención**. Si podés mostrar que los intermitentes se parecen a un perfil identificable, el argumento se cierra solo: *este perfil es recuperable, y ya lo recuperamos quince veces.*

Con quince casos no hay estadística. Como cierre de video, es potente. Usalo con honestidad sobre el n.

## Método y advertencias

**El clustering jerárquico basado en distancia de Random Forest** está en el programa (capítulo 28) y es el camino natural. Pero un agrupamiento más simple, bien argumentado y con acciones claras funciona igual: Miranda no sabe ni le interesa la técnica.

**Trampa:** contarle a Miranda el modelo predictivo en palabras de negocio. Ese es el otro video, en diciembre, para Michelina. Miranda quiere perfiles y acciones.

**Quién es:** llega a gerencia general en dos años, busca ideas que muevan la aguja, y "buen trabajo" de su boca es una sentencia. Tres perfiles con acción concreta y un número de impacto le sirven más que doce hallazgos.

---

# PARTE VII — Lo que este EDA deliberadamente no hace

Cada ítem consume días y no mueve ni la ganancia ni la nota:

- Histogramas de las 154 variables.
- Tabla de 16 estadísticos por columna partida por clase, sobre todo el dataset.
- Matriz de correlación completa para "detectar redundancia": LightGBM convive con variables correlacionadas sin drama.
- Correlación con el target como criterio de selección: target al 0,7 %, relaciones no lineales, categóricas disfrazadas de enteros. No mide lo que creés.
- Imputación de nulos por defecto.
- Normalización o estandarización: a un árbol le da igual cualquier transformación monótona.
- Tests de normalidad.

Si te encontrás haciendo alguna de estas, probablemente estés evitando la parte difícil.

---

## Entregables

1. `00_verificaciones.R` — intermitentes, prevalencia por mes, tests de composición premium.
2. `01_ganancia.R` — función de ganancia y curva ganancia vs. cantidad de envíos.
3. `02_consistencia.R` — tabla `variable × mes`.
4. `03_drifting.R` — ranking de drift al mes de aplicación.
5. `04_familias.R` — derivadas de la Parte IV y tasa de `BAJA+2` por segmento (incluyendo vip).
6. `variables_sospechosas.md` — diagnóstico, tratamiento y resultado, una línea por variable.
7. `hipotesis_features.md` — salida de la Parte V, insumo de la Última Competencia.
8. `perfiles_premium.md` — insumo del video a Miranda.
9. **Bitácora**: por cada decisión no obvia, fecha / hipótesis / qué cambié / semillas / ganancia / conclusión en una frase. **Y el supuesto de 0.2, declarado con fecha.**

## Reparto de tiempo

| Partes | Horas | Cuándo |
|---|---|---|
| 0, I | 2 h | primero — la clase ternaria ya está, esto son verificaciones |
| II | 5 h | el núcleo defensivo del modelo |
| III | 3 h | semántica + auditoría de fugas |
| IV | 4 h | insumo del modelo **y** de Miranda |
| V | 2 h | hipótesis de feature engineering |
| VI | ventana propia | antes del 27-sep |

**En paralelo, no después:** la infraestructura de corridas largas. El Grid Search de 12+ horas no espera a que termines el EDA.

---

*Todo lo que este documento afirma sobre el dataset es hipótesis hasta que lo verifiques contra el archivo. Todo lo que afirma sobre la materia, contra Zulip y la clase. La composición premium de la base (0.2) es un supuesto razonado, no un hecho confirmado: declaralo como tal hasta que Denicolay responda. El significado preciso de `cliente_vip`, `tcallcenter`, `cinversion1`, `cinversion2` y `active_quarter` requiere el diccionario de datos, que sigue sin cargarse.*
