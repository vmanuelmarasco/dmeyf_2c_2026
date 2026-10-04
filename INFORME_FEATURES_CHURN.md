# Catálogo de features para churn — competencia_01

**Mirada de negocio bancario sobre las 154 columnas.** Qué conviene laguear/deltear, qué combinar,
y qué NO tocar. Ventana: 202103→202108 (6 fotos). Train 202103-202106, score 202108.

---

## 0. Restricción que ordena todo

| | |
|---|---|
| Fotos disponibles | 6 (202103 … 202108) |
| Lags limpios en el mes de scoring (202108) | 202107, 202106, 202105 → **lag 1 a 3** |
| Lags en el borde del train (202103) | 0 → NA |
| Positivos (`BAJA+2`) | ~0,6 % |

**Consecuencias prácticas:**

1. **Profundidad máxima útil: lag1, lag2, lag3.** Un lag4/lag5 sólo existe para 202107-202108 y
   genera un patrón "NA ⇒ mes temprano" que el modelo puede usar como proxy de `foto_mes`. Eso es
   fuga de calendario, no señal de churn.
2. Con 0,6 % de positivos, **no explotes a 1.000+ columnas**. Presupuesto razonable:
   ~40 variables × (lag1 + delta1 + delta3/slope) + ~30 derivadas de negocio ≈ **180-200 columnas nuevas**.
3. Para variables de monto, el **delta absoluto arrastra inflación**. Preferí, en este orden:
   `delta_rel = (x_t − x_{t−1}) / |x_{t−1}|` (NULL si el denominador es 0) ·
   `x_t / media(x_{t−1..t−3})` (auto-normalización, mata inflación y escala de golpe) ·
   `percent_rank() OVER (PARTITION BY foto_mes)` (lo que ya pide la guía contra drift).

---

## 1. TIER 1 — lag + delta sí, sin discusión

Cobertura alta, continuas, y la caída monótona es exactamente la firma del cliente que se va.
Para cada una: `lag1`, `lag2`, `lag3`, `delta1`, `delta_rel1`, `ratio_vs_media_3m`, `slope_3m`,
`min/max_3m`, `racha_caidas`.

### 1.1 Intensidad de uso (el núcleo)
| Variable | Por qué | Nota |
|---|---|---|
| `ctrx_quarter` | **La variable canónica de churn.** 58→40→32→18→14 es el patrón que buscás | Es trimestral: va a mostrar escalones, no la marques como rota |
| `active_quarter` | Flag de actividad; el delta 1→0 es evento, no número | Trimestral también |
| `ctarjeta_debito_transacciones` / `mautoservicio` | Uso cotidiano, 28 % ceros | Par c/m perfectamente alineado |
| `ctarjeta_visa_transacciones` / `mtarjeta_visa_consumo` | Consumo discrecional: lo primero que se apaga | 14 % ceros |
| `ctarjeta_master_transacciones` / `mtarjeta_master_consumo` | Ídem, menor penetración | 61 % ceros → más valor como evento |
| `chomebanking_transacciones` | Canal principal; frecuencia = vinculación | 15 % ceros |
| `cmobile_app_trx` | **Tendencia estructural creciente** (28,8 %→21,0 % de ceros en 6 meses): el que cae a contramano de la cartera es señal fuerte | Normalizar intra-mes |
| `catm_trx` / `matm` · `catm_trx_other` / `matm_other` | Extracción | Ver §3: `other` es señal de competencia |
| `cextraccion_autoservicio` / `mextraccion_autoservicio` | Ídem | |

### 1.2 Saldos y fondeo (la plata se va antes que el cliente)
`mcuentas_saldo` · `mcaja_ahorro` · `mcuenta_corriente` · `mcaja_ahorro_dolares` ·
`mplazo_fijo_dolares` · `minversion1_pesos` · `minversion2`

El **delta de saldos es el leading indicator más temprano**: el cliente vacía 60-90 días antes de
irse. Priorizá `delta_rel` y `ratio_vs_media_3m`, nunca el delta absoluto crudo.

### 1.3 Rentabilidad y precio
`mrentabilidad` · `mrentabilidad_annual` · `mcomisiones` · `mactivos_margen` · `mpasivos_margen` ·
`mcomisiones_mantenimiento` / `ccomisiones_mantenimiento` · `mcomisiones_otras`

Cobertura casi total (0,2 % ceros en `mrentabilidad`). La caída de margen es el churn ya en curso.
⚠️ `mcomisiones_mantenimiento` tiene **PSI 0,208 (MEDIO)** en 202107 — usala, pero rankeada intra-mes.

### 1.4 El ancla (lo que hace costoso irse)
| Variable | Lectura de negocio |
|---|---|
| `cpayroll_trx` / `mpayroll` | **Acreditación de sueldo.** Si se apaga, el cliente ya cambió de banco; sólo falta el trámite. 44 % de la cartera lo tiene |
| `ccuenta_debitos_automaticos` / `mcuenta_debitos_automaticos` | **Señal N°1 de churn en banca retail.** Cada domiciliación es un costo de salida. Que bajen = está migrando los compromisos |
| `ctarjeta_visa_debitos_automaticos` / `mttarjeta_visa_debitos_automaticos` | Ídem sobre tarjeta. Ojo: ceros 42,9 %→35,0 %, hay tendencia de cartera |
| `ctarjeta_master_debitos_automaticos` / `mttarjeta_master_debitos_automaticos` | Ídem |
| `cpagomiscuentas` / `mpagomiscuentas` · `cpagodeservicios` / `mpagodeservicios` | Pago de servicios = hábito |

⚠️ `mpayroll` y `cpayroll_trx` tienen **PSI 0,249 / 0,178 (MEDIO)** en 202106. El nivel está movido
por inflación salarial; el *delta relativo* y el *rank intra-mes* sobreviven, el delta absoluto no.

### 1.5 Flujos de entrada/salida
`ctransferencias_recibidas` / `mtransferencias_recibidas` · `ctransferencias_emitidas` / `mtransferencias_emitidas` ·
`ccheques_depositados` / `mcheques_depositados` · `ccheques_emitidos` / `mcheques_emitidos`

### 1.6 Tarjeta de crédito — nivel de cuenta
`Visa_msaldototal` · `Visa_mlimitecompra` · `Visa_mconsumototal` · `Visa_mpagominimo` ·
`Master_msaldototal` · `Master_mlimitecompra` · `Master_mconsumototal` · `Master_mpagominimo`

El **delta del límite de compra** merece atención aparte: una baja de límite es una decisión *del
banco*, y suele preceder la baja del cliente (causalidad invertida pero predictiva).
`Visa_mlimitecompra` y `Master_mlimitecompra` tienen **0 % de ceros** → cobertura perfecta.

---

## 2. TIER 2 — lag sí, delta sólo como EVENTO binario

Variables lumpy (78-99 % ceros). El delta numérico es ruido; lo que predice es el **cambio de estado**.
Construí dos banderas por variable, no un delta:

```
baja_producto_X  = (X_{t-1} > 0 AND X_t = 0)
alta_producto_X  = (X_{t-1} = 0 AND X_t > 0)
```

| Variable | % ceros | Lectura |
|---|---|---|
| `cplazo_fijo` | 86,6 % | **Plazo fijo que no se renueva = plata que se va.** La bandera de baja es muy predictiva |
| `cinversion1` / `cinversion2` | 95,5 % / 96,8 % | Rescate de inversión, ídem |
| `cprestamos_personales` / `mprestamos_personales` | 79,3 % | **Préstamo vigente = no puede irse.** La baja del préstamo libera al cliente → el fin del crédito es un disparador clásico |
| `cprestamos_prendarios` / `cprestamos_hipotecarios` | 99,1 % / 99,0 % | Ancla máxima. Como flag de tenencia, no como delta |
| `cseguro_vida` (88,5 %) · `cseguro_auto` · `cseguro_vivienda` · `cseguro_accidentes_personales` | 88-99 % | Cross-sell: la baja de seguros suele anticipar la baja total |
| `ccaja_seguridad` | 93,2 % | Tenencia pegajosa |
| `cdescubierto_preacordado` | 4,5 % | Acá sí el nivel importa: es línea otorgada |
| `ccallcenter_transacciones` / `tcallcenter` | 92,7 % | **Fricción.** "Llamó este mes y no en los 3 anteriores" es la feature, no el conteo |
| `ccajas_transacciones` / `_consultas` / `_extracciones` / `_otras` | 90-99 % | Uso de sucursal: alto en perfil tradicional |
| `cforex_sell` / `mforex_sell` | ~96 % | Dolarización, ver §3.3 |
| `ccheques_depositados_rechazados` · `ccheques_emitidos_rechazados` | 99,8 % | Fricción severa, como flag |
| `Visa_delinquency` / `Visa_status` · `Master_delinquency` / `Master_status` | 97-99,7 % | Mora: flag + "empeoró vs mes anterior" |

---

## 3. COMBINATORIAS DE NEGOCIO

Acá está el valor real. Seis ejes, con la lógica bancaria detrás de cada uno.

### 3.1 Consolidación de tarjetas (Visa + Master → "tarjeta total")
Usá `suma_sin_null(a,b)` para agregar, **pero conservá los flags de tenencia aparte** —
`ifnull(x,0)` colapsa "tiene tarjeta con saldo 0" contra "no tiene tarjeta", que son dos clientes
completamente distintos.

```
tiene_visa, tiene_master, cant_tarjetas
tc_saldo_total      = suma_sin_null(Visa_msaldototal, Master_msaldototal)
tc_limite_total     = suma_sin_null(Visa_mlimitecompra, Master_mlimitecompra)
tc_consumo_total    = suma_sin_null(Visa_mconsumototal, Master_mconsumototal)
tc_pagominimo_total = suma_sin_null(Visa_mpagominimo, Master_mpagominimo)
tc_cconsumos        = suma_sin_null(Visa_cconsumos, Master_cconsumos)
```

Y sobre eso, los ratios que usa cualquier área de riesgo:

| Feature | Fórmula | Lectura |
|---|---|---|
| **Utilización** | `tc_saldo_total / tc_limite_total` | El ratio clásico. Alta utilización + caída de consumo = estrés |
| **Δ Utilización** | delta de lo anterior | Más predictivo que el nivel |
| **Holgura** | `tc_limite_total − tc_saldo_total` | Capacidad disponible sin usar |
| **Esfuerzo de pago** | `tc_mpagado / tc_saldo_total` | ¿Paga todo o revolvea? |
| **Flag revolver** | `tc_mpagado ≈ tc_pagominimo_total` | Paga el mínimo: cliente caro y frágil |
| **Intensidad de consumo** | `tc_consumo_total / tc_limite_total` | Apetito vs capacidad |
| **Ticket promedio** | `tc_consumo_total / tc_cconsumos` | Cambio de patrón de gasto |
| **Dolarización del consumo** | `mconsumosdolares / mconsumototal` | Fuga a dólar |
| **Estrés de liquidez** | `madelantosefectivo / mconsumototal` | Adelanto de efectivo = señal roja |
| **Vida de la tarjeta** | `foto_mes − Visa_fechaalta` | §3.6 |

⚠️ **`Visa_mpagado` y `Master_mpagado`: NO deltear sin tratar primero.** Ver §4.

### 3.2 Profundidad de la relación (vinculación)
```
productos_activos  = suma de todos los flags c*>0  (recalculado; no confíes sólo en cproductos)
canales_usados     = cuántos de {homebanking, mobile, atm, cajas, autoservicio, callcenter} con trx>0
trx_total          = suma de todas las c*_transacciones
share_digital      = (chomebanking_transacciones + cmobile_app_trx) / trx_total
share_sucursal     = ccajas_transacciones / trx_total
compromisos_domiciliados = ccuenta_debitos_automaticos + ctarjeta_visa_debitos_automaticos
                           + ctarjeta_master_debitos_automaticos + cpagomiscuentas + cpagodeservicios
cliente_anclado    = (cpayroll_trx>0) AND (compromisos_domiciliados>0) AND (cant_tarjetas>0)
```

**`compromisos_domiciliados` y su delta son, por experiencia, la mejor feature construida de todo
el set.** Nadie cancela cinco domiciliaciones por casualidad: las migra. Y las migra *antes* de
cerrar la cuenta, que es justo la ventana de `BAJA+2`.

`share_digital` + su delta separan dos churns distintos: el digital que se va a un neobanco, y el
tradicional que simplemente deja de operar.

### 3.3 Flujo de fondos — ¿el dinero entra o se escapa?
```
ingresos  = mpayroll + mtransferencias_recibidas + mcheques_depositados
egresos   = mtransferencias_emitidas + mextraccion_autoservicio + matm + matm_other
            + mcheques_emitidos + mpagomiscuentas + mpagodeservicios
flujo_neto           = ingresos − egresos
retencion_de_sueldo  = mcuentas_saldo / mpayroll          -- cuánto del sueldo queda adentro
velocidad_de_vaciado = egresos / mpayroll                 -- >1 sostenido = se desfonda
fuga_a_otro_banco    = mtransferencias_emitidas / ingresos
uso_atm_ajeno        = matm_other / (matm + matm_other)   -- YA opera con la competencia
dolarizacion         = (mcaja_ahorro_dolares + mplazo_fijo_dolares + minversion1_dolares
                        + mforex_buy) / aum_total
aum_total            = mcuentas_saldo + mplazo_fijo_pesos + mplazo_fijo_dolares
                        + minversion1_pesos + minversion2
```

`uso_atm_ajeno` merece énfasis: `catm_trx_other` pasó de 75,5 % a 78,0 % de ceros, pero el cliente
que sube su uso de cajeros ajenos mientras baja el propio **ya tiene la cuenta nueva abierta**.
Esa combinación (`Δcatm_trx < 0` AND `Δcatm_trx_other > 0`) es una interacción explícita que vale
la pena construir a mano en vez de esperar que el árbol la encuentre.

### 3.4 Rentabilidad y precio efectivo
```
rentab_por_producto  = mrentabilidad / productos_activos
roa_cliente          = mrentabilidad / aum_total
peso_comisiones      = mcomisiones / mrentabilidad          -- relación sostenida sólo por cargos
precio_efectivo      = mcomisiones_mantenimiento / aum_total -- lo que le cuesta estar
desaceleracion       = mrentabilidad − (mrentabilidad_annual / 12)
perfil_deudor        = mactivos_margen / (mactivos_margen + mpasivos_margen)
```

`desaceleracion` es casi gratis: el dataset ya trae el acumulado anual, así que el mes contra el
run-rate te da un detector de quiebre sin necesidad de ventana.

`precio_efectivo` alto + `aum_total` bajo + `trx_total` bajo = **cliente que paga mantenimiento por
una cuenta que no usa**. En retail ese cluster churnea casi por definición: con 70 % de ceros en
`mcomisiones_mantenimiento`, el 30 % que sí paga está autoseleccionado.

### 3.5 Fricción y deterioro
```
llamo_sin_antecedente = ccallcenter_transacciones>0 AND lag1..3 = 0
rechazos_total        = ccheques_depositados_rechazados + ccheques_emitidos_rechazados
tasa_rechazo          = ccheques_emitidos_rechazados / ccheques_emitidos
usa_descubierto       = cdescubierto_preacordado>0 AND mcuenta_corriente<0
mora_empeora          = (Visa_delinquency > lag1) OR (Master_delinquency > lag1)
dias_en_mora          = foto_mes − Visa_Finiciomora        -- sólo 0,6 % lo tiene: flag, no número
```

### 3.6 Fechas → duraciones (el oro que queda sin usar)
Hay 9 columnas de fecha (`Fvencimiento`, `fechaalta`, `fultimo_cierre`, `Finiciomora` × Visa/Master).
Laguearlas no significa nada; **convertirlas a distancia temporal sí**:

```
dias_a_vencimiento_tc   = Visa_Fvencimiento − foto_mes
vida_tarjeta            = foto_mes − Visa_fechaalta
dias_desde_ultimo_cierre= foto_mes − Visa_fultimo_cierre
tarjeta_vs_relacion     = vida_tarjeta / cliente_antiguedad
```

**`dias_a_vencimiento_tc` es un disparador administrativo de baja**: la tarjeta que vence y no se
renueva arrastra el cierre del paquete. Y `Fvencimiento` tiene 0 % de ceros, cobertura total.

`tarjeta_vs_relacion` separa al cliente de origen del que sumó la tarjeta después — dos curvas de
supervivencia distintas.

### 3.7 Dinámica: la forma de la caída, no sólo su tamaño
Para las ~15 variables más fuertes del Tier 1:
```
caida_sostenida = (x_t − avg(x_{t-1..t-3})) / avg(x_{t-1..t-3})
racha_bajas     = cuántos meses consecutivos con delta < 0
meses_inactivo  = cuántos de los últimos 3 con trx_total = 0
apago_producto  = x_t = 0 AND x_{t-1} > 0
slope_3m        = regr_slope(x, cliente_antiguedad)   -- antigüedad avanza de a 1: hace de reloj
```

⚠️ `regr_slope` contra `cliente_antiguedad` se desalinea si el cliente tiene meses faltantes. En
esta ventana hay clientes con menos de 6 fotos — verificá `COUNT(*) por cliente` antes de confiar
en la pendiente.

Las banderas `apago_producto` sobre **payroll, débitos automáticos, tarjeta, homebanking y plazo
fijo** son cinco columnas binarias que valen más que cien deltas numéricos.

---

## 4. TRAMPAS DETECTADAS EN ESTOS DATOS (no las descubras en producción)

| Variable | Evidencia | Qué hacer |
|---|---|---|
| **`Visa_mpagado`** | % ceros: 91,1 / 91,4 / **73,3** / 89,8 / 90,4 / **72,7** — oscila con el ciclo de cierre, y **PSI 0,251 ALTO justo en 202108, el mes de scoring** | **No deltear crudo.** Un `Δ202107→202108` es artefacto de captura, no comportamiento. Normalizar contra el ciclo o usar sólo como "pagó / no pagó" |
| **`Master_mpagado`** | 92-95 % de ceros; mismo patrón atenuado | Ídem |
| **`ccajas_depositos`** | **100 % ceros en 202105** (PSI 0,358 ALTO) | Mes roto. NA en 202105, y los deltas de 202105-202106 quedan contaminados |
| `mpayroll` / `cpayroll_trx` | PSI 0,249 / 0,178 MEDIO en 202106 | Usar delta *relativo* y rank intra-mes; evitar nivel crudo |
| `Visa_mconsumosdolares` | PSI 0,216 MEDIO en 202108 | Drift cambiario → rankear |
| `mcomisiones_mantenimiento` | PSI 0,208 MEDIO en 202107 | Rankear |
| `Master_*consumo/pago/adelanto` | **58-60 % de nulos** (sin cierre en el mes) | El nulo significa "no hubo resumen", no "no gastó". `isNA` es la feature; no imputes 0 |
| `Visa_*consumo/pago/adelanto` | 12,7 % de nulos | Ídem |
| `Master_Finiciomora` / `Visa_Finiciomora` | 99,7 % / 99,4 % nulos | Sólo flag binario |
| Outliers en 202108 | 94 % de los extremos de Visa caen en 202108 | Con `max_bin: 31` un outlier se come la resolución. Clipear a p99.5 o rankear antes de entrenar |

**Variables a dejar afuera de lag/delta directamente** (casi constantes: no hay varianza para que el
delta signifique algo): `mpayroll2` / `cpayroll2_trx` (99,9 % ceros) · `mcuenta_corriente_adicional`
(99,95 %) · `cliente_vip` (99,7 %) · `tmobile_app` (97,4 % — es flag, usala como tenencia) ·
`mplazo_fijo_pesos` (99,5 %) · `minversion1_dolares` (99,7 %) · `cforex_buy` / `mforex_buy` (99,8 %) ·
`Master_madelantodolares` / `Visa_madelantodolares` (99,99 %) · `cheques_*_rechazados` (99,8 %) ·
`mprestamos_prendarios` / `mprestamos_hipotecarios` (99,0-99,2 %).
Todas valen como **flag de tenencia**, ninguna como serie.

`cliente_edad` y `cliente_antiguedad` son relojes monótonos: el delta es constante = 1. No las
laguees; usalas como **denominador y como eje temporal**.

---

## 5. Las 10 apuestas, en orden

Si tuvieras que entregar mañana y sólo pudieras construir diez cosas:

1. `Δ compromisos_domiciliados` (3m) — migración de domiciliaciones
2. `apago_payroll` — se cortó la acreditación de sueldo
3. `slope_3m(ctrx_quarter)` + `racha_bajas(ctrx_quarter)` — la curva canónica
4. `Δ_rel(mrentabilidad)` y `desaceleracion` vs run-rate anual
5. `utilizacion_tc` + su delta
6. `velocidad_de_vaciado` y `retencion_de_sueldo` + deltas
7. `share_digital` + delta — y la interacción `Δcatm_trx<0 AND Δcatm_trx_other>0`
8. `apago_plazo_fijo` / `apago_inversion` — rescate de fondos
9. `precio_efectivo` (comisiones de mantenimiento / AUM) × `trx_total` bajo
10. `dias_a_vencimiento_tc` — el disparador administrativo

---

## 6. Notas de implementación

- Window base: `WINDOW w AS (PARTITION BY numero_de_cliente ORDER BY foto_mes)`. Ya hay 21
  `PARTITION BY` en `competencia.ipynb`, el patrón está armado.
- **Macro de ratio seguro: que devuelva `NULL` con denominador 0**, no 0. LightGBM maneja NA
  nativamente y puede aprender "denominador cero" como categoría propia; un 0 le miente.
- Agregá con `suma_sin_null`, pero **sumá siempre los flags de tenencia al lado**.
- Si entrenás con lag3, 202103 y 202104 quedan con NA estructural. Dos opciones defendibles:
  entrenar sobre 202105-202106 solamente, o dejar los NA y verificar que el modelo no esté usando
  el patrón de nulos como proxy de `foto_mes` (mirá si `foto_mes` o algún `lag3_*` aparece alto en
  importancia — es la señal de que se filtró el calendario).
- Un tratamiento por vez, ceteris paribus, y medí ganancia con varias semillas. Las features de
  §3.2 y §3.3 son las que más deberían mover la aguja; si no mueven nada, revisá el ratio seguro
  antes de descartar la idea.
