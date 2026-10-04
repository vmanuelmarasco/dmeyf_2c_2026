# Apuntes de estudio — SQL + DuckDB (Clase 01, `z101_target_sql.ipynb`)

> Extraído del notebook de la cátedra. La idea de este documento no es que lo leas una vez y listo — es que puedas explicar cada bloque **en 60 segundos, sin mirar la pantalla**, porque en diciembre te lo van a preguntar así.

---

## 1. ¿Qué es DuckDB y por qué lo usamos acá?

**Antes de leer la respuesta, pensalo vos:** ¿por qué la cátedra elige DuckDB para transformar un CSV en vez de, por ejemplo, hacerlo directo con pandas?

DuckDB es una base de datos **OLAP** (Online Analytical Processing) que corre **embebida** — no es un servidor al que te conectás por red como Postgres o MySQL, es una librería que vive adentro de tu proceso de Python (o R). La diferencia clave con pandas:

- Pandas carga todo en memoria como estructuras de Python/NumPy y vos programás en un lenguaje imperativo (fila por fila, o vectorizado si sos prolijo).
- DuckDB te deja escribir **SQL declarativo** ("quiero esto") y el motor decide **cómo** resolverlo de la forma más eficiente — optimiza el plan de ejecución, algo que pandas no hace.

Para este tipo de tarea (agrupar, hacer joins, funciones de ventana sobre millones de filas y cientos de columnas) SQL declarativo + un motor columnar como DuckDB suele ser más rápido y más legible que reescribir la misma lógica a mano en pandas.

En el notebook, la conexión se abre con:

```python
%load_ext sql
%sql duckdb:///
```

`duckdb:///` es una base **en memoria** (no se guarda en disco a menos que vos exportes algo, como al final con `COPY ... TO`). El `%load_ext sql` y `%%sql` son de la librería `jupysql`, que te deja escribir SQL directo en una celda de Jupyter en vez de mandarlo como un string de Python — es puro azúcar sintáctico para trabajar más cómodo en notebook, no es magia de DuckDB en sí.

**Cargar el CSV como tabla**, sin pasos intermedios de "importar":

```sql
create or replace table competencia_01_crudo as
select * from read_csv_auto("ruta/al/archivo.csv")
```

`read_csv_auto` infiere los tipos de cada columna solo (números, fechas, texto). Es cómodo, pero ojo: en datasets grandes con columnas ambiguas a veces infiere mal el tipo — algo para tener en el radar si más adelante te aparecen columnas con tipos raros.

---

## 2. La cláusula `WITH` (Common Table Expressions / CTEs)

**Pensalo vos primero:** si necesitás armar una consulta que depende del resultado de otra consulta, ¿qué alternativas tenés y qué problema tiene cada una?

Dos formas tradicionales y por qué el notebook las evita:

1. **Crear una tabla temporal** con el resultado intermedio → ensucia el esquema de la base con tablas que solo se usan una vez y ocupan espacio.
2. **`select` anidado dentro de otro `select`** (subquery) → funciona, pero es de lo peor visto por administradores de bases porque el motor a veces no lo puede optimizar bien y es difícil de leer cuando se anida varias veces.

La solución es `WITH`: le ponés nombre a uno o más resultados intermedios, y los usás en la consulta principal como si fueran tablas. Ejemplo del notebook (comparar la rentabilidad de un cliente contra su propio promedio histórico):

```sql
with promedios as (
    select numero_de_cliente, avg(mrentabilidad) as avg_mrentabilidad
    from competencia_01_crudo
    group by numero_de_cliente
)
select cp.numero_de_cliente, cp.foto_mes,
       if(cp.mrentabilidad > p.avg_mrentabilidad, 'mayor', 'menor') as comparacion
from competencia_01_crudo cp
join promedios p using (numero_de_cliente)
```

Beneficios: se lee de arriba hacia abajo como una receta ("primero calculo esto, después uso eso para calcular lo otro"), y el motor SQL entiende el plan completo y lo puede optimizar mejor que un subquery anidado.

**Dato de color:** las tablas temporales (`CREATE TEMP TABLE`) sí tienen sentido cuando ese resultado intermedio se va a reusar **varias veces** en el proceso. Si es de un solo uso, `WITH` es mejor porque el motor puede optimizar sabiendo todo el plan de antemano.

---

## 3. Funciones analíticas / funciones de ventana (`OVER`, `PARTITION BY`)

Esta es la parte que más te va a servir para el resto de la materia — se usa todo el tiempo para construir variables que dependen del histórico de cada cliente.

**Pensalo vos primero:** tenés una tabla con una fila por cliente por mes. Necesitás, para cada fila, el valor de una columna **del mes anterior o del mes siguiente**, sin perder la fila actual. ¿Cómo lo harías con SQL "básico" (group by, joins)? ¿Por qué es incómodo?

Con `group by` solo podés colapsar filas (una fila de salida por grupo), perdés el detalle fila por fila. Para lo que necesitamos —mantener cada fila individual pero calcular algo sobre una "ventana" de filas relacionadas— existen las funciones analíticas:

```sql
select
    numero_de_cliente, foto_mes, mrentabilidad,
    lead(mrentabilidad, 1) over (partition by numero_de_cliente order by foto_mes) as mrentabilidad_mas_1,
    lag(mrentabilidad, 1)  over (partition by numero_de_cliente order by foto_mes) as mrentabilidad_menos_1
from competencia_01_crudo
where numero_de_cliente = 12159854
order by foto_mes
```

Desglosado:

- `partition by numero_de_cliente` — separá los datos en "ventanas" independientes, una por cliente. Sin esto, `lead`/`lag` mezclarían datos entre clientes distintos.
- `order by foto_mes` (dentro del `over`) — dentro de cada ventana, ordená por mes. `lead`/`lag` necesitan saber qué es "antes" y "después".
- `lead(columna, n)` — trae el valor de esa columna, `n` filas **hacia adelante** dentro de la partición.
- `lag(columna, n)` — trae el valor `n` filas **hacia atrás**.

Es la herramienta central para construir el target: en vez de mirar "el futuro" con un join complicado, `lead` te deja mirar N meses adelante en la misma fila.

---

## 4. El problema de los "casos raros" — por qué no alcanza con `lead`/`lag` solo

El notebook muestra el caso del cliente `60112254`: en la foto `202104` **no aparece** — no es que tenga NULL en sus columnas, directamente **no hay fila** para ese cliente ese mes (se fue del banco y volvió, o hay un hueco de datos).

**Pensalo vos:** si usás `lead`/`lag` directo sobre `competencia_01_crudo` tal cual está, y a un cliente le falta la fila de un mes intermedio, ¿qué pasa con el cálculo de "cuántos meses faltan hasta que se da de baja"? ¿`lead` se da cuenta de que faltó un mes, o simplemente salta a la próxima fila que exista?

Ese es el problema: `lead`/`lag` operan sobre las filas que **existen**, no sobre el calendario. Si a un cliente le falta el mes 202104, `lead(mrentabilidad, 1)` en la fila de 202103 te va a traer el valor de 202105 (la próxima fila que *sí* existe), no un NULL indicando "faltó un mes". Eso arruinaría cualquier lógica de "está o no está el mes que viene".

**La solución del notebook: generar el calendario completo primero.**

```sql
with periodos as (
  select distinct foto_mes from competencia_01_crudo
), clientes as (
  select distinct numero_de_cliente from competencia_01_crudo
)
select numero_de_cliente, foto_mes from clientes cross join periodos
```

`cross join` arma **todas las combinaciones posibles** entre dos conjuntos (todo cliente x todo período), sin importar si esa combinación existe en los datos reales. Es la única forma de tener, para cada cliente, una fila por **cada mes que existió en el dataset**, exista o no el dato real.

Después, con un `left join` contra la tabla real usando `if(c.numero_de_cliente is null, 0, 1)`, marcás con `1`/`0` si el cliente **estuvo presente de verdad** ese mes. Ahí sí, `lead`/`lag` sobre esa bandera (`mes_0`, `mes_1`, `mes_2` en el notebook) te dice con certeza si el cliente va a estar presente 1 o 2 meses adelante, sin que un hueco te arruine la cuenta.

---

## 5. `PIVOT` y `COPY ... TO`

- `PIVOT tabla ON columna USING agregacion GROUP BY otra_columna` — convierte valores de una columna en columnas nuevas. En el notebook se usa para ver, por cada `foto_mes`, cuántos casos hay de cada clase (`CONTINUA`/`BAJA+1`/`BAJA+2`) sin tener que escribir un `group by` + `case when` a mano.
- `COPY tabla TO 'archivo.csv' (FORMAT CSV, HEADER)` — exporta el resultado de una tabla/consulta a disco. Es la forma de persistir el trabajo hecho en memoria.

---

## 6. Lo que el notebook te deja como tarea (y que NO te resuelvo acá)

La celda final arma el esqueleto:

```sql
, if(c.numero_de_cliente is null, 0, 1) as mes_0
, lead(mes_0, 1) over (partition by t.numero_de_cliente order by foto_mes) as mes_1
, lead(mes_0, 2) over (partition by t.numero_de_cliente order by foto_mes) as mes_2
, null as clase_ternaria -- Reemplazar null por la lógica que genera el target
```

Tenés tres banderas por fila: `mes_0` (¿existe este mes? — siempre 1, porque el filtro final es `where mes_0 = 1`), `mes_1` (¿existe el mes siguiente?), `mes_2` (¿existe el mes después de ese?).

Armar la fórmula de `clase_ternaria` a partir de `mes_1` y `mes_2` es **la tarea de la Primera Asignación**, y es una decisión de modelado — no infraestructura. Por eso no te la resuelvo yo: es exactamente lo que te van a poder preguntar en el examen del 10 de diciembre, y tiene que ser algo que puedas defender como propio.

Para guiarte sin dártela armada, las preguntas que te tenés que hacer (mirá el slide de "Ayuda de Michelina" de la Clase 01 si te trabás):

1. Si un cliente está este mes (`mes_0=1`) pero **no** el que viene (`mes_1=?`), ¿qué tan rápido se fue? ¿Con qué clase se corresponde eso?
2. Si sigue estando el mes que viene pero no el que sigue después, ¿qué clase es?
3. Si sigue estando en los dos meses siguientes, ¿qué clase le corresponde?
4. `mes_2` depende de `mes_1` — es decir, `lead(mes_0, 2)` calcula sobre la bandera de presencia, no sobre si el cliente "sigue existiendo en general". ¿Qué pasa con esa columna cuando el cliente directamente no tiene datos futuros porque el dataset se termina (los últimos meses del dataset)? ¿Vas a tener NULLs ahí, y qué deberías hacer con esos casos?

Con `if()` (que ya usaste en el ejemplo de `WITH`) o `case when` podés armar la lógica una vez que tengas clara la respuesta a esas cuatro preguntas.

---

## 7. Autoevaluación — la pregunta de examen

Cerrá este bloque de estudio explicándote a vos mismo, en voz alta, sin mirar el notebook ni este documento:

- ¿Por qué se usa `WITH` en vez de subqueries anidados o tablas temporales de un solo uso?
- ¿Qué hace `partition by` dentro de un `over()`, y qué pasaría si te olvidás de ponerlo?
- ¿Por qué hace falta el `cross join` de clientes x períodos antes de poder calcular el target con `lead`? ¿Qué se rompería si lo saltearas y usaras `lead` directo sobre la tabla cruda?
- ¿Qué significan `mes_0`, `mes_1` y `mes_2`, y por qué no alcanza con mirar solo `mes_1` para decidir entre `BAJA+1` y `BAJA+2`?

Si en alguna te trabás, es señal de que hay que volver a leer esa sección — no de memorizar la respuesta.
