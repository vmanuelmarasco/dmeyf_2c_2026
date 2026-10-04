# Informe descriptivo de los perfiles de clientes que dejan el paquete Premium

## Fuente, alcance y criterio de lectura

Este informe utiliza como única fuente `/Users/manuelmarasco/Desktop/dmeyf402/EDA_Miranda.ipynb`. No se utilizaron `EDA.ipynb`, `EDA_limpio.ipynb` ni versiones anteriores. Tampoco se reejecutó, modificó, reconstruyó o cuestionó el clustering: se adoptaron sin cambios sus etiquetas, integrantes y tamaños finales.

La población analizada está formada exclusivamente por **1.991 bajas estrictas con tres meses consecutivos de historia**: **Transaccionales multicanal, 979 clientes; Vinculados por nómina, 135; Vinculados por crédito, 92; y Desvinculación avanzada, 785**. La suma auditada es **979 + 135 + 92 + 785 = 1.991**.[^F1][^F2]

Una “baja estricta” es una ausencia sostenida durante el seguimiento observable, no una cancelación contractual demostrada. El universo general del notebook contiene 3.977 bajas estrictas; de ellas, 1.991 poseen los tres meses consecutivos exigidos para la segmentación y 1.986 tienen historia parcial. Los 1.103 casos censurados y los 23 intermitentes no integran los perfiles aquí descriptos.[^F3][^F4]

Se emplean tres momentos relativos:

- **t−2:** dos observaciones antes del último mes activo;
- **t−1:** observación inmediatamente anterior;
- **t0:** último mes activo antes de la ausencia.

En las trayectorias, el denominador permanece constante en cada perfil: 979, 135, 92 y 785 clientes, respectivamente. Las transacciones corresponden a la **mediana de `ctrx_quarter` (transacciones trimestrales registradas en cada fotografía mensual)**; productos y familias activas también se resumen con la mediana. Los porcentajes de presencia o uso se calculan sobre el total de clientes del perfil. Los abandonos se calculan sobre ese mismo denominador y requieren valor igual a cero en t0 después de un valor positivo en t−1 o t−2.[^F5][^F6][^F7]

Los importes se informan como **$ nominales del campo**, tal como los presenta el notebook. El símbolo no supone una conversión a dólares estadounidenses. No se agregan campos identificados en pesos con campos identificados en dólares, no se ajusta por inflación y no se suman rentabilidad, comisiones y márgenes.

## Síntesis ejecutiva

La segmentación describe cuatro formas distintas de llegar a la salida, no cuatro escalones de una misma gravedad.

- **Transaccionales multicanal** reúne 979 clientes, equivalentes al 49,17% de las 1.991 bajas segmentadas. Conserva una relación amplia con tarjetas, canales digitales y débitos, pero durante la ventana descienden las transacciones, el uso de tarjeta y el saldo. Es el grupo con mayor oportunidad de detección temprana basada en contracción de uso, porque todavía presenta actividad material en t0.[^F2][^F5]

- **Vinculados por nómina** comprende 135 clientes, el 6,78%. Es el perfil de mayor actividad, mayor amplitud de familias y mejor saldo terminal. Su señal distintiva es la nómina: el 66,67% mantiene payroll en t0 y el 33,33% lo abandonó durante la ventana. La interrupción de la acreditación constituye una señal comercial concreta, aunque no demuestra por sí sola intención de salida.[^F2][^F6]

- **Vinculados por crédito** contiene 92 clientes, el 4,62%. El 100% mantiene algún crédito en t0, pero el segmento combina baja actividad, escaso uso de tarjetas, saldo mediano negativo y 83,70% de saldos negativos. Su mayor rentabilidad registrada no debe leerse aisladamente como una relación saludable: puede asociarse con financiación, cargos u otros conceptos que el notebook no permite descomponer como identidad contable.[^F2][^F8]

- **Desvinculación avanzada** agrupa 785 clientes, el 39,43%. Presenta la menor cantidad mediana de productos y familias activas, la menor actividad, el saldo mediano más negativo y la mayor proporción de saldos negativos. También registra las mayores pérdidas de tenencia de tarjeta y canales digitales. Es el estado terminal más deteriorado, aunque no equivale a una relación nula: 43,95% todavía utiliza canales digitales en t0.[^F2][^F5][^F6]

El orden por actividad y amplitud de vinculación es nítido: **Vinculados por nómina > Transaccionales multicanal > Vinculados por crédito > Desvinculación avanzada**. El mismo orden aparece en las familias activas medianas —4, 3, 2 y 1— y en las transacciones trimestrales medianas de t0 —72, 37, 12 y 4—.[^F2]

El orden de saldo también es claro: **Vinculados por nómina ($ 7.178,78) > Transaccionales multicanal ($ 117,30) > Vinculados por crédito (−$ 2.767,68) > Desvinculación avanzada (−$ 7.420,04)**. La proporción de saldos negativos confirma la misma secuencia: 22,22%, 43,51%, 83,70% y 88,54%.[^F2]

En cambio, la rentabilidad registrada rompe esa escala: el perfil de crédito muestra la mayor mediana mensual ($ 2.067,44) y anual registrada ($ 21.948,20), mientras que nómina —el grupo más activo y con mejor saldo— presenta la menor mediana mensual ($ 1.153,53). Por eso actividad, saldo, dependencia de producto y rentabilidad no deben fundirse en un único puntaje de “calidad”.[^F8]

## Tabla comparativa integrada

Todas las medianas y porcentajes de esta tabla corresponden a t0, salvo las trayectorias expresamente indicadas. El denominador de cada porcentaje es el n del perfil; el denominador del peso relativo es 1.991.[^F2][^F5][^F8]

| Dimensión | Transaccionales multicanal | Vinculados por nómina | Vinculados por crédito | Desvinculación avanzada |
|---|---:|---:|---:|---:|
| Clientes | 979 | 135 | 92 | 785 |
| Peso en las 1.991 bajas | 49,17% | 6,78% | 4,62% | 39,43% |
| Edad mediana, años | 47 | 45 | 45 | 45 |
| Antigüedad mediana registrada* | 115 | 69 | 61,5 | 67 |
| Productos, mediana | 7 | 7 | 6 | 5 |
| Familias activas, mediana | 3 | 4 | 2 | 1 |
| Transacciones trimestrales, mediana | 37 | 72 | 12 | 4 |
| Transacciones t−2 → t−1 → t0 | 42 → 40 → 37 | 85 → 78 → 72 | 12,5 → 12 → 12 | 6 → 5 → 4 |
| Payroll activo | 1,02% | 66,67% | 3,26% | 1,02% |
| Uso de tarjeta | 69,87% | 56,30% | 10,87% | 9,94% |
| Uso digital | 86,52% | 93,33% | 64,13% | 43,95% |
| Emite transferencias | 32,18% | 61,48% | 5,43% | 3,18% |
| Tiene crédito | 4,29% | 9,63% | 100,00% | 0,51% |
| Débitos automáticos | 61,59% | 52,59% | 20,65% | 14,39% |
| Pago de servicios | 19,71% | 40,00% | 0,00% | 0,76% |
| Saldo mediano, $ nominales | 117,30 | 7.178,78 | −2.767,68 | −7.420,04 |
| Clientes con saldo negativo | 43,51% | 22,22% | 83,70% | 88,54% |
| Rentabilidad mensual mediana, $ nominales | 1.703,68 | 1.153,53 | 2.067,44 | 1.902,38 |
| Rentabilidad anual registrada mediana, $ nominales | 9.670,50 | 9.086,31 | 21.948,20 | 12.759,96 |
| Señal distintiva | Ecosistema transaccional multicanal todavía activo | Nómina, máxima actividad y mejor saldo | Crédito presente en todo el segmento | Mínima vinculación y mayor deterioro terminal |

* El notebook muestra el campo `cliente_antiguedad`, pero no explicita su unidad en una salida o nota. Por ello se conserva como “antigüedad registrada” y no se la convierte a meses o años.

## Ficha 1. Transaccionales multicanal

### Quiénes integran el perfil

El perfil contiene **979 clientes (49,17% del total segmentado)**. Presenta una edad mediana de **47 años**, una antigüedad mediana registrada de **115**, **7 productos** medianos y **3 familias activas** medianas.[^F2] Es el segmento más numeroso y el de mayor antigüedad registrada.

### Productos, servicios y forma de vinculación

La relación se apoya en varios canales y comportamientos, sin depender de una única familia: **69,87% usa tarjeta, 86,52% utiliza canales digitales, 61,59% registra débitos automáticos, 32,18% emite transferencias y 19,71% paga servicios**. Además, **21,55% tiene seguros, 7,66% inversiones y 4,29% créditos**; sólo **1,02%** mantiene payroll activo.[^F2]

La combinación de uso de tarjeta, digitalización y débitos lo distingue de los perfiles de crédito y desvinculación avanzada. Frente a nómina, comparte una cantidad mediana de 7 productos, pero tiene menos familias activas, menor actividad y una vinculación mucho menos asociada con payroll.[^F2]

### Nivel de actividad y trayectoria previa

La mediana de transacciones trimestrales desciende de **42 en t−2 a 40 en t−1 y 37 en t0**. La mediana de productos se mantiene en **7** durante los tres momentos. El uso de tarjeta baja de **78,75% a 75,89% y 69,87%**, mientras que el uso digital pasa de **87,84% a 85,19% y 86,52%**.[^F5]

El saldo mediano cae de **$ 2.119,47 en t−2 a $ 1.902,23 en t−1 y $ 117,30 en t0**. La trayectoria no implica que cada individuo haya seguido exactamente ese recorrido: se trata de medianas de corte para el mismo conjunto de 979 clientes.[^F5]

### Abandonos observados

La contracción es distribuida, no concentrada en una única familia: **10,32% abandona el uso de tarjeta, 9,30% el pago de servicios, 7,15% inversiones, 7,05% débitos automáticos, 6,84% la tenencia de tarjeta, 6,84% canales digitales, 6,03% créditos, 3,98% seguros y 3,17% payroll**.[^F6]

Estos porcentajes miden el paso desde un valor positivo en t−2 o t−1 a cero en t0. No identifican el motivo del cambio ni prueban que el abandono de una familia haya provocado la salida.

### Saldos y rentabilidad

El saldo terminal es casi neutro en mediana (**$ 117,30**), pero **43,51%** del perfil presenta saldo negativo. La rentabilidad mensual mediana es **$ 1.703,68**, con intervalo intercuartílico de **$ 397,50 a $ 2.578,16**; **85,50%** tiene rentabilidad mensual positiva. La rentabilidad anual registrada mediana es **$ 9.670,50**.[^F8]

Las comisiones, mantenimiento y márgenes se presentan por separado en la sección económica. No corresponde deducir de estas cifras una rentabilidad “neta” construida por suma o resta.

### Qué lo distingue y qué hipótesis permite plantear

Su rasgo central es que la salida ocurre mientras todavía existe un ecosistema transaccional relevante. No es el grupo de mayor actividad, pero está claramente por encima de crédito y desvinculación avanzada. La señal comercial no es una ruptura única, sino una disminución conjunta de transacciones, tarjeta y saldo.

Como hipótesis, podría evaluarse un contacto temprano ante caídas simultáneas de actividad y uso de productos todavía vigentes. El resultado debería medirse mediante retención incremental, recuperación de actividad y costo por cliente retenido. No puede afirmarse, con esta evidencia, que el contacto evite la salida.

## Ficha 2. Vinculados por nómina

### Quiénes integran el perfil

El perfil contiene **135 clientes (6,78%)**. Su edad mediana es **45 años**, la antigüedad mediana registrada es **69**, la mediana de productos es **7** y la de familias activas es **4**, la mayor de los cuatro grupos.[^F2]

### Productos, servicios y forma de vinculación

En t0, **66,67% mantiene payroll activo**. Al mismo tiempo, **33,33% abandonó payroll durante la ventana**, por lo que la vinculación reciente por nómina caracteriza al conjunto aunque no siempre permanezca activa al final.[^F2][^F6]

Es el perfil con mayor uso digital (**93,33%**), transferencias (**61,48%**) y pago de servicios (**40,00%**). También registra **56,30% de uso de tarjeta, 52,59% de débitos automáticos, 21,48% de seguros, 12,59% de inversiones y 9,63% de créditos**.[^F2]

### Nivel de actividad y trayectoria previa

La mediana de transacciones trimestrales disminuye de **85 en t−2 a 78 en t−1 y 72 en t0**, pero continúa siendo la más alta. Los productos se mantienen en una mediana de **7**. El uso de tarjeta evoluciona de **60,00% a 62,22% y 56,30%**; el digital se mantiene prácticamente estable, de **92,59% a 92,59% y 93,33%**.[^F5]

El saldo mediano pasa de **$ 19.791,00 en t−2 a $ 23.033,22 en t−1 y $ 7.178,78 en t0**. El último descenso es marcado, aunque el nivel terminal sigue siendo el más favorable de los cuatro perfiles.[^F5]

### Abandonos observados

La pérdida de payroll, **33,33%**, es por amplio margen la señal más específica. Los demás abandonos son menores: **8,15% uso de tarjeta, 7,41% débitos automáticos, 6,67% pago de servicios, 5,93% inversiones, 5,19% tenencia de tarjeta, 3,70% créditos, 2,96% canales digitales y 1,48% seguros**.[^F6]

No se observa una desvinculación digital general: el uso digital terminal permanece por encima del 93%. La desaparición de payroll puede responder a cambios laborales, administrativos, estacionales o de elección bancaria; el notebook no permite distinguir esas causas.

### Saldos y rentabilidad

El saldo mediano terminal es **$ 7.178,78** y sólo **22,22%** presenta saldo negativo, ambos resultados más favorables que en los otros perfiles. La rentabilidad mensual mediana es **$ 1.153,53**, la menor de los cuatro, con intervalo intercuartílico de **$ 477,88 a $ 3.074,62**; **86,67%** registra rentabilidad mensual positiva. La rentabilidad anual registrada mediana es **$ 9.086,31**.[^F8]

La aparente combinación de alta actividad, buen saldo y menor rentabilidad mensual refuerza la necesidad de no ordenar los perfiles con un único indicador económico.

### Qué lo distingue y qué hipótesis permite plantear

El perfil se distingue por un anclaje identificable —la acreditación de haberes— unido a alta actividad y amplia vinculación. La señal más accionable es la interrupción reciente del payroll antes de que desaparezca el resto de la relación.

Como hipótesis, podría investigarse la causa de esa interrupción y probarse una intervención condicionada a la continuidad o recuperación de nómina. Las métricas deberían incluir recuperación de payroll, permanencia posterior y margen incremental. No corresponde asumir que la interrupción sea voluntaria ni que un beneficio garantice permanencia.

## Ficha 3. Vinculados por crédito

### Quiénes integran el perfil

El perfil contiene **92 clientes (4,62%)**. Registra una edad mediana de **45 años**, antigüedad mediana registrada de **61,5**, **6 productos** medianos y **2 familias activas** medianas.[^F2]

### Productos, servicios y forma de vinculación

El **100,00% tiene algún crédito en t0**, rasgo que define al grupo. Fuera de ese anclaje, la vinculación es estrecha: **64,13% usa canales digitales, 20,65% débitos automáticos, 16,30% seguros, 10,87% tarjeta, 5,43% transferencias, 3,26% payroll y 1,09% inversiones**. El pago de servicios es **0,00%**.[^F2]

El 0,00% en pago de servicios es un cero observado en el indicador construido por el notebook; no autoriza a transformar en cero los valores nulos de variables crudas de tarjeta u otras familias.

### Nivel de actividad y trayectoria previa

La mediana de transacciones trimestrales se mantiene casi estable: **12,5 en t−2, 12 en t−1 y 12 en t0**. Los productos permanecen en una mediana de **6**. El uso de tarjeta cae de **19,57% a 14,13% y 10,87%**, mientras el uso digital pasa de **61,96% a 61,96% y 64,13%**.[^F5]

El saldo mediano se mantiene negativo: **−$ 2.167,04 en t−2, −$ 3.069,22 en t−1 y −$ 2.767,68 en t0**. Se observa una mejora entre t−1 y t0, pero no una reversión del signo.[^F5]

### Abandonos observados

No se registra abandono de crédito (**0,00%**), consistente con su presencia en todo el segmento en t0. Los abandonos de otras familias son **8,70% en uso de tarjeta, 6,52% en canales digitales, 5,43% en tenencia de tarjeta, 5,43% en débitos automáticos, 5,43% en seguros, 1,09% en payroll y 0,00% en pago de servicios e inversiones**.[^F6]

Los ceros de abandono no equivalen a retención exitosa: también pueden reflejar una exposición previa inexistente o muy reducida.

### Saldos y rentabilidad

El saldo terminal mediano es **−$ 2.767,68** y **83,70%** presenta saldo negativo. La rentabilidad mensual mediana es **$ 2.067,44**, la mayor de los cuatro perfiles; su intervalo intercuartílico es **$ 414,30 a $ 3.001,23** y **88,04%** registra rentabilidad positiva. La rentabilidad anual registrada mediana también es la más alta, **$ 21.948,20**.[^F8]

Esta rentabilidad no prueba una relación saludable o sostenible. El notebook no permite identificar de forma aditiva cuánto proviene de crédito, cargos, comisiones, margen u otros conceptos, ni incorpora costo de riesgo.

### Qué lo distingue y qué hipótesis permite plantear

El perfil no es simplemente una versión “más desvinculada” del multicanal: mantiene un producto ancla universal, exhibe actividad baja pero estable y combina rentabilidad registrada alta con saldos predominantemente negativos.

Como hipótesis, cualquier intervención debería comenzar con una revisión conjunta de vinculación y riesgo antes de evaluar refinanciación, beneficios o asistencia. Las métricas deberían incluir regularización, riesgo incremental, permanencia y resultado económico neto. No corresponde ampliar exposición crediticia basándose sólo en la rentabilidad registrada.

## Ficha 4. Desvinculación avanzada

### Quiénes integran el perfil

El perfil contiene **785 clientes (39,43%)**. Su edad mediana es **45 años**, la antigüedad mediana registrada es **67**, la mediana de productos es **5** y la de familias activas es **1**, los valores más bajos de la segmentación.[^F2]

### Productos, servicios y forma de vinculación

En t0, **43,95% usa canales digitales**, pero las demás señales son reducidas: **14,39% registra débitos automáticos, 10,32% seguros, 9,94% uso de tarjeta, 3,18% transferencias, 2,04% inversiones, 1,02% payroll, 0,76% pago de servicios y 0,51% crédito**.[^F2]

El uso digital residual evita describir al segmento como totalmente inactivo. Sin embargo, su amplitud de relación es la menor: una sola familia activa en mediana.

### Nivel de actividad y trayectoria previa

La mediana de transacciones trimestrales cae de **6 en t−2 a 5 en t−1 y 4 en t0**. Los productos pasan de una mediana de **6 en t−2 y t−1 a 5 en t0**. El uso de tarjeta se reduce de **20,64% a 14,78% y 9,94%**. El digital se mantiene alrededor de cuatro de cada diez clientes: **40,89%, 40,38% y 43,95%**.[^F5]

El saldo mediano se deteriora de **−$ 5.497,90 a −$ 7.009,22 y −$ 7.420,04**, la trayectoria terminal más desfavorable.[^F5]

### Abandonos observados

El perfil registra los mayores abandonos de **tenencia de tarjeta (14,52%)** y **canales digitales (14,52%)**, además de **11,59% en uso de tarjeta**. Los restantes son **7,01% en débitos automáticos, 3,06% en créditos, 2,80% en seguros, 1,91% en payroll, 1,27% en pago de servicios y 0,76% en inversiones**.[^F6]

La coexistencia de bajos niveles terminales y pérdidas recientes de canales refuerza la caracterización de desvinculación avanzada, sin demostrar qué causa la salida.

### Saldos y rentabilidad

El saldo terminal mediano es **−$ 7.420,04** y **88,54%** registra saldo negativo, los resultados más desfavorables. Sin embargo, la rentabilidad mensual mediana es positiva (**$ 1.902,38**), con intervalo intercuartílico de **$ 1.145,57 a $ 2.797,87**; **90,70%** presenta rentabilidad mensual positiva. La rentabilidad anual registrada mediana es **$ 12.759,96**.[^F8]

La combinación de deterioro de saldo y rentabilidad positiva es compatible con cargos, financiación u otros conceptos, pero no permite determinar su composición. No debe interpretarse como valor futuro alto ni como argumento suficiente para subsidiar una acción.

### Qué lo distingue y qué hipótesis permite plantear

Es el perfil de menor actividad y vinculación, con mayor deterioro terminal de saldo y pérdidas visibles de tarjeta y digital. La oportunidad de recuperación puede ser más limitada o costosa, pero eso no se estima en el notebook.

Como hipótesis, podría utilizarse un contacto diagnóstico de bajo costo y reservar incentivos para casos cuya recuperación sea plausible bajo criterios definidos. Deberían medirse tasa de contacto, recuperación de actividad, regularización y resultado neto. No se afirma que este enfoque evite la salida.

## Trayectorias comparadas durante los tres meses

Las siguientes cifras resumen exactamente los mismos clientes de cada perfil en t−2, t−1 y t0. En cada celda se informa una secuencia t−2 → t−1 → t0.[^F5]

| Perfil (denominador constante) | Transacciones trimestrales, mediana | Saldo, $ nominales, mediana | Productos, mediana | Uso de tarjeta, % | Uso digital, % |
|---|---:|---:|---:|---:|---:|
| Transaccionales multicanal (n=979) | 42 → 40 → 37 | 2.119,47 → 1.902,23 → 117,30 | 7 → 7 → 7 | 78,75 → 75,89 → 69,87 | 87,84 → 85,19 → 86,52 |
| Vinculados por nómina (n=135) | 85 → 78 → 72 | 19.791,00 → 23.033,22 → 7.178,78 | 7 → 7 → 7 | 60,00 → 62,22 → 56,30 | 92,59 → 92,59 → 93,33 |
| Vinculados por crédito (n=92) | 12,5 → 12 → 12 | −2.167,04 → −3.069,22 → −2.767,68 | 6 → 6 → 6 | 19,57 → 14,13 → 10,87 | 61,96 → 61,96 → 64,13 |
| Desvinculación avanzada (n=785) | 6 → 5 → 4 | −5.497,90 → −7.009,22 → −7.420,04 | 6 → 6 → 5 | 20,64 → 14,78 → 9,94 | 40,89 → 40,38 → 43,95 |

La trayectoria más claramente deteriorada en nivel terminal es la de Desvinculación avanzada. Nómina presenta la mayor caída absoluta de transacciones y una fuerte reducción final de saldo, pero termina con los mejores niveles. Crédito conserva casi estable la actividad, aunque desde una base baja y con saldo negativo. Multicanal se contrae de forma gradual y distribuida mientras todavía conserva uso significativo.

## Abandono total de productos y comportamientos

Cada porcentaje representa clientes que llegan a cero en t0 después de haber tenido un valor positivo en t−1 o t−2. El denominador es el total de cada perfil; los indicadores fueron auditados sin nulos ni valores no binarios.[^F6]

| Familia abandonada | Transaccionales multicanal (n=979) | Vinculados por nómina (n=135) | Vinculados por crédito (n=92) | Desvinculación avanzada (n=785) |
|---|---:|---:|---:|---:|
| Payroll | 3,17% | 33,33% | 1,09% | 1,91% |
| Tenencia de tarjeta | 6,84% | 5,19% | 5,43% | 14,52% |
| Uso de tarjeta | 10,32% | 8,15% | 8,70% | 11,59% |
| Débitos automáticos | 7,05% | 7,41% | 5,43% | 7,01% |
| Pago de servicios | 9,30% | 6,67% | 0,00% | 1,27% |
| Canales digitales | 6,84% | 2,96% | 6,52% | 14,52% |
| Créditos | 6,03% | 3,70% | 0,00% | 3,06% |
| Inversiones | 7,15% | 5,93% | 0,00% | 0,76% |
| Seguros | 3,98% | 1,48% | 5,43% | 2,80% |

La pérdida de una familia no permite distinguir entre cancelación voluntaria, vencimiento normal, cambio administrativo, decisión comercial u otra causa. En particular, la pérdida de crédito, inversión o seguro no demuestra insatisfacción.

## Saldos y valor económico descriptivo

El notebook comprueba que las variables económicas no forman una identidad contable universal. La igualdad `mcomisiones = mantenimiento + otras comisiones` se verifica en **642.692 de 983.061 registros comparables (65,38%)**. La igualdad entre rentabilidad y la suma de comisiones más márgenes se verifica en sólo **2.255 de 983.061 registros comparables (0,23%)**. Por ello, cada componente se informa por separado.[^F9]

| Indicador en t0, mediana salvo porcentaje | Transaccionales multicanal | Vinculados por nómina | Vinculados por crédito | Desvinculación avanzada |
|---|---:|---:|---:|---:|
| Rentabilidad mensual, $ nominales | 1.703,68 | 1.153,53 | 2.067,44 | 1.902,38 |
| P25–P75 de rentabilidad mensual, $ nominales | 397,50–2.578,16 | 477,88–3.074,62 | 414,30–3.001,23 | 1.145,57–2.797,87 |
| Rentabilidad mensual positiva | 85,50% | 86,67% | 88,04% | 90,70% |
| Rentabilidad anual registrada, $ nominales | 9.670,50 | 9.086,31 | 21.948,20 | 12.759,96 |
| Comisiones totales, $ nominales | 1.721,92 | 372,29 | 1.592,20 | 1.592,20 |
| Mantenimiento, $ nominales | 1.774,04 | 0,00 | 1.774,04 | 1.774,04 |
| Mantenimiento positivo | 68,64% | 17,78% | 73,91% | 75,80% |
| Otras comisiones, $ nominales | 1.721,92 | 372,29 | 1.592,20 | 1.592,20 |
| Margen de activos, $ nominales | −49,32 | −3,28 | 287,92 | 155,23 |
| Margen de pasivos, $ nominales | 153,18 | 359,65 | 10,36 | 1,06 |
| Saldo de cuentas, $ nominales | 117,30 | 7.178,78 | −2.767,68 | −7.420,04 |
| Saldo negativo | 43,51% | 22,22% | 83,70% | 88,54% |

Fuente de toda la tabla: salida `perfil_economico`; denominador de porcentajes: n de cada perfil. La auditoría general del notebook detecta nulos exclusivamente en variables asociadas con tarjetas, por lo que estos porcentajes económicos utilizan el n completo del perfil.[^F8][^F10]

No se calcula una “rentabilidad neta sin mantenimiento”, no se suman márgenes y comisiones, y no se interpreta la rentabilidad registrada como valor futuro, rentabilidad incremental o resultado ajustado por riesgo.

## Qué diferencia realmente a los cuatro perfiles

### Nómina frente a multicanal: mismo número de productos, distinta profundidad

Ambos perfiles tienen **7 productos medianos**, pero nómina posee **4 familias activas frente a 3**, **72 transacciones frente a 37**, **61,48% de transferencias frente a 32,18%** y **66,67% de payroll frente a 1,02%**. El saldo mediano también difiere: **$ 7.178,78 frente a $ 117,30**.[^F2] La separación no se explica por una simple cuenta de productos, sino por el anclaje de nómina y la intensidad de uso.

### Crédito frente a desvinculación avanzada: baja actividad con naturalezas distintas

Ambos grupos tienen actividad baja y saldo mayormente negativo. Sin embargo, crédito mantiene **100% de presencia crediticia, 2 familias activas medianas y 12 transacciones**, mientras que desvinculación avanzada presenta **0,51% de crédito, 1 familia y 4 transacciones**.[^F2] Crédito es una relación estrecha pero anclada; desvinculación avanzada es una relación residual y dispersa.

### Multicanal frente a desvinculación avanzada: conservación del ecosistema

Multicanal mantiene **69,87% de uso de tarjeta, 86,52% digital y 61,59% de débitos**, frente a **9,94%, 43,95% y 14,39%** en desvinculación avanzada. La diferencia de saldo es también sustantiva: **$ 117,30 frente a −$ 7.420,04**.[^F2] El perfil avanzado no sólo transacciona menos; conserva menos vías de relación.

### Rentabilidad frente a salud de la relación: señales cruzadas

Crédito y desvinculación avanzada presentan medianas de rentabilidad mensual superiores a nómina, aun cuando concentran saldos negativos. Nómina exhibe el mejor saldo y la mayor actividad, pero la menor rentabilidad mensual mediana.[^F8] Esto impide equiparar rentabilidad registrada con calidad, profundidad o potencial de retención.

## Ordenamiento razonado por dimensión

Los órdenes siguientes son descriptivos. No constituyen un puntaje individual ni un ranking causal.

1. **Actividad en t0:** Vinculados por nómina (**72**) > Transaccionales multicanal (**37**) > Vinculados por crédito (**12**) > Desvinculación avanzada (**4**), según la mediana de transacciones trimestrales.[^F2]

2. **Amplitud de vinculación:** Vinculados por nómina (**4 familias activas**) > Transaccionales multicanal (**3**) > Vinculados por crédito (**2**) > Desvinculación avanzada (**1**).[^F2]

3. **Posición de saldo, de mejor a peor:** Vinculados por nómina (**$ 7.178,78**) > Transaccionales multicanal (**$ 117,30**) > Vinculados por crédito (**−$ 2.767,68**) > Desvinculación avanzada (**−$ 7.420,04**). La proporción negativa —22,22%, 43,51%, 83,70% y 88,54%— confirma el mismo orden.[^F2]

4. **Deterioro terminal:** Desvinculación avanzada ocupa el extremo más deteriorado por nivel de actividad, productos, saldo y pérdida de tarjeta/digital. Crédito le sigue por su saldo mayormente negativo y baja actividad, aunque conserva estable el volumen transaccional. Multicanal muestra una contracción intermedia y distribuida. Nómina mantiene la mejor posición terminal, pero no debe considerarse “sin deterioro”: transacciones y saldo caen y un tercio abandona payroll.[^F5][^F6]

5. **Dependencia de un ancla específica:** Vinculados por crédito presenta la mayor concentración, porque el 100% mantiene crédito. Le sigue Vinculados por nómina, con 66,67% de payroll activo y 33,33% que lo abandona durante la ventana. Transaccionales multicanal depende menos de una sola familia porque su relación se reparte entre tarjeta, digital y débitos. Desvinculación avanzada no presenta un ancla dominante; su vínculo es residual.[^F2][^F6]

6. **Urgencia comercial de detección, si se prioriza una señal temprana y todavía accionable:** primero nómina, por la interrupción observable de payroll mientras subsiste alta actividad; segundo multicanal, por la contracción de un ecosistema aún activo; tercero crédito, donde la urgencia debe ser de revisión conjunta de riesgo y vinculación antes que de incentivo; cuarto desvinculación avanzada, para contacto diagnóstico de bajo costo y selección estricta de casos. Este orden es una hipótesis operativa, no una estimación de efectividad.

7. **Urgencia de revisión de riesgo, que responde a otra pregunta:** primero Vinculados por crédito, luego Desvinculación avanzada, por la elevada prevalencia de saldos negativos; después Transaccionales multicanal y finalmente Vinculados por nómina. Esta secuencia no debe confundirse con prioridad de retención.[^F2]

## Por qué los perfiles no forman una escala lineal

Los cuatro perfiles no pueden ordenarse de manera única desde “mejor” hasta “peor” porque las dimensiones se cruzan:

- Nómina lidera actividad, familias y saldo, pero tiene la menor rentabilidad mensual mediana.[^F2][^F8]
- Crédito tiene actividad baja y saldo negativo, pero conserva un ancla universal y la mayor rentabilidad mensual y anual registrada.[^F2][^F8]
- Multicanal y nómina comparten una mediana de 7 productos, aunque difieren ampliamente en payroll, transacciones, transferencias y saldo.[^F2]
- Crédito muestra una trayectoria transaccional casi estable, mientras multicanal y nómina se contraen desde niveles más altos.[^F5]
- Desvinculación avanzada es el extremo de deterioro terminal, pero todavía conserva uso digital en 43,95% y presenta rentabilidad mensual positiva en 90,70%.[^F2][^F8]

Por lo tanto, los perfiles representan **configuraciones**: amplitud de uso, producto ancla, nivel de saldo, trayectoria y composición económica. Convertirlos en una sola escala eliminaría precisamente las diferencias que la segmentación busca conservar.

## Hipótesis comerciales y medición requerida

| Perfil | Hipótesis a evaluar | Resultado que debería medirse | Riesgo de interpretación |
|---|---|---|---|
| Transaccionales multicanal | Contacto temprano ante caída conjunta de actividad, tarjeta y saldo; prueba de beneficios vinculados con productos todavía usados | Retención incremental, recuperación de actividad y costo por cliente retenido | Subsidiar clientes que habrían salido o permanecido de todas formas |
| Vinculados por nómina | Investigación de la interrupción de haberes y prueba condicionada a continuidad o recuperación de payroll | Recuperación de payroll, permanencia posterior y margen incremental | Confundir cambios laborales, estacionales o administrativos con intención de salida |
| Vinculados por crédito | Revisión conjunta de riesgo y vinculación antes de asistencia, refinanciación o beneficio | Regularización, riesgo incremental, permanencia y resultado económico neto | Aumentar exposición o confundir cargos registrados con valor sostenible |
| Desvinculación avanzada | Contacto diagnóstico de bajo costo; incentivos sólo en casos con recuperación plausible | Contactabilidad, recuperación de actividad, regularización y resultado neto | Financiar masivamente relaciones ya deterioradas |

Estas hipótesis requieren pilotos con grupo de control, asignación aleatoria y métricas definidas antes del contacto. La evidencia observacional del notebook no permite afirmar que una acción evitará la salida.

## Precauciones interpretativas

1. **Ausencia no equivale a cancelación contractual.** La baja estricta es una ausencia sostenida durante el seguimiento disponible. La pertenencia al paquete Premium también se asume por el contexto de la consigna, porque el dataset no contiene una variable que la compruebe directamente.[^F3]

2. **La población del informe es exactamente 1.991.** Las 1.986 bajas con historia parcial sólo reciben una proyección en otra sección del notebook; no reemplazan ni amplían las etiquetas observadas. Los 1.103 censurados y 23 intermitentes no son bajas confirmadas de la segmentación principal.[^F3][^F4]

3. **Los perfiles se mantienen tal como fueron aprobados.** Sus nombres, integrantes y tamaños no se alteran. La taxonomía es descriptiva y sus fronteras no deben interpretarse como categorías humanas naturales o permanentes.

4. **Asociación no implica causa.** Una caída de transacciones, payroll, tarjeta, saldo o productos puede preceder a la ausencia, pero no demuestra que la haya causado.

5. **Nulo no equivale a cero.** El notebook detecta nulos exclusivamente en variables de tarjetas y documenta casos con producto registrado pero consumo nulo, cero, positivo o negativo. También existen montos de descuentos nulos aun cuando la cantidad de descuentos es positiva. Esos nulos no deben reinterpretarse como ausencia de uso.[^F10]

6. **Los indicadores de presencia reproducen las reglas del notebook.** Expresan si una variable agregada es mayor que cero. Un porcentaje de 0,00% en una tabla de perfil no habilita a reemplazar con cero los nulos de los campos crudos.

7. **No se mezclan monedas.** No se agregan importes explícitamente identificados en pesos con importes identificados en dólares. Los valores aquí expuestos son los campos agregados que el notebook rotula como $ nominales; su moneda exacta no se redefine.

8. **Los componentes económicos no son aditivos.** La identidad de comisiones sólo se verifica en 65,38% de los registros comparables y la de rentabilidad en 0,23%. No se calcula rentabilidad neta restando mantenimiento ni sumando márgenes.[^F9]

9. **La rentabilidad no incorpora todo lo necesario para decidir.** No se estiman costo de riesgo, valor futuro, margen incremental, costo de contacto, costo de incentivo ni efecto causal.

10. **Las trayectorias son resúmenes grupales.** Una secuencia de medianas no representa necesariamente la trayectoria de un cliente típico ni permite restar medianas como si fueran cambios individuales.

11. **“Abandono” tiene una definición operacional limitada.** Significa pasar de positivo en t−2 o t−1 a cero en t0. No distingue cancelación voluntaria, vencimiento, decisión comercial, cambio laboral o falla de registro.[^F6]

12. **La condición VIP no permite generalizar.** Sólo 3 de los 1.991 clientes son VIP: 2 pertenecen a Transaccionales multicanal y 1 a Vinculados por nómina. No corresponde usar esos casos para renombrar perfiles, priorizar acciones o inferir asociaciones.[^F11]

13. **Los importes son nominales.** La ventana abarca marzo–agosto de 2021; no se realiza ajuste por inflación. La segmentación principal se apoya en bajas ancladas en mayo y junio: 1.129 y 862 clientes, respectivamente.[^F1][^F12]

14. **La unidad de antigüedad no está explicitada.** El notebook informa `cliente_antiguedad`, pero no documenta en una salida si se expresa en meses, años u otra unidad. Por esa razón no se convierte.

## Datos que no pueden verificarse directamente con el notebook

- La cancelación contractual efectiva del paquete Premium; sólo se observa ausencia sostenida.
- La pertenencia al paquete Premium mediante una variable específica; el notebook declara que se asume por contexto.
- La causa de cada ausencia o abandono de producto.
- La moneda exacta de los campos monetarios agregados rotulados como “$ nominales”, más allá de que los campos explícitos en pesos y dólares se mantienen separados.
- La unidad documental de `cliente_antiguedad`.
- La interpretación contable completa de rentabilidad, comisiones y márgenes, por falta de un diccionario contable que establezca identidades universales.
- El costo de riesgo, el valor futuro del cliente y el efecto incremental de cualquier intervención.
- La generalización poblacional de la condición VIP, porque sólo se observan 3 casos.[^F11]

## Registro de trazabilidad

[^F1]: `EDA_Miranda.ipynb`, §14 “Etiquetas finales y perfiles de negocio”, salida de la celda ejecutada 40: tabla de tamaños finales y hash de etiquetas.
[^F2]: `EDA_Miranda.ipynb`, §14, salida de la celda ejecutada 41: tabla `perfil_general` y composición de cohorte. Incluye n, porcentajes, medianas, presencia/uso, saldos y rentabilidad por perfil.
[^F3]: `EDA_Miranda.ipynb`, §6 “Definición de los eventos de salida y de las poblaciones”, salida de la celda ejecutada 13: 3.977 estrictas, 1.103 censuradas y 23 intermitentes; y §2, definición operacional de baja estricta.
[^F4]: `EDA_Miranda.ipynb`, §10 “Muestra de referencia”, salida de la celda ejecutada 28: 3.977 bajas estrictas, 1.991 con tres meses consecutivos, 1.986 con historia parcial, y 1.129/862 bajas ancladas en mayo/junio.
[^F5]: `EDA_Miranda.ipynb`, §15 “Trayectorias previas y abandono de familias”, salida de la celda ejecutada 43: `resumen_trayectorias`, con n constante, medianas y porcentajes en t−2, t−1 y t0.
[^F6]: `EDA_Miranda.ipynb`, §15.1, salida de la celda ejecutada 45: auditoría binaria y tabla completa de abandono por familia y perfil.
[^F7]: `EDA_Miranda.ipynb`, §15.1, texto metodológico previo a la celda ejecutada 45: definición de abandono y denominador.
[^F8]: `EDA_Miranda.ipynb`, §16 “Valor económico descriptivo”, salida de la celda ejecutada 47: tabla `perfil_economico`.
[^F9]: `EDA_Miranda.ipynb`, §16, salida de la celda ejecutada 46: auditoría de identidades entre comisiones, rentabilidad y márgenes sobre 983.061 registros comparables.
[^F10]: `EDA_Miranda.ipynb`, §7 “Auditoría de valores ausentes y validez numérica”, salidas de las celdas ejecutadas 16–21: patrones de nulidad, consumos nulos/cero/positivos/negativos, descuentos y validez numérica.
[^F11]: `EDA_Miranda.ipynb`, §20 “Distribución de clientes VIP”, salida de la celda ejecutada 58.
[^F12]: `EDA_Miranda.ipynb`, §4.3, salida de la celda ejecutada 7: seis fotografías mensuales entre 202103 y 202108.
