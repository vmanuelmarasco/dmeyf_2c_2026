[Página 1]
Universidad  de  Buenos Aires
Maestría en  Explotación de Datos
y Descubrimiento del Conocimiento
Data Mining en Economía y Finanzas
Libro de la Asignatura
Edición 2026 
Did you ask a good question today?
última actualización  2026-08-07  07:13
1
[Página 2]
Tabla de Contenidos
1La Asignatura..................................................................................................................................................6
1.1Sobre este documento..........................................................................................................................6
1.2Links fundamentales año 2026.............................................................................................................7
1.3Cronograma de la Asignatura año 2026................................................................................................8
1.4Dedicación Estimada.............................................................................................................................9
1.5UBA Modalidad y Criterios de Aprobación y Evaluación.....................................................................10
1.5.1Modalidad y Criterios de Aprobación y Evaluación Competencias..............................................11
1.5.2Experimentos Colaborativos........................................................................................................12
1.5.3Evaluación participación en foro/chat Zulip.................................................................................13
1.5.4Evaluación video presentaciones.................................................................................................14
1.5.5Evaluación  Individual..................................................................................................................16
1.6Misión de la Asignatura......................................................................................................................17
1.6.1Objetivos de la Asignatura...........................................................................................................17
1.7 Unidades Temáticas...........................................................................................................................18
1.7.1Tecnologías a enseñar y reforzar..................................................................................................19
1.8Bibliografía General de la Asignatura..................................................................................................20
2Metodología de Enseñanza...........................................................................................................................22
2.1Generalidades.....................................................................................................................................22
2.2Manifiesto Pedagógico.......................................................................................................................22
2.3Pensamiento Creativo – Tinkering......................................................................................................23
2.4Dinámica Constructivista en el Aula....................................................................................................24
2.5Authentic Learning.............................................................................................................................24
2.6Flipped Classroom Forte.....................................................................................................................24
2.7Conectivismo......................................................................................................................................25
2.8Active Learning...................................................................................................................................25
2.9Evaluación Continua...........................................................................................................................26
3Abróchense los cinturones...........................................................................................................................28
3.1Alta en Plataformas............................................................................................................................28
3.1.1Zulip  herramienta de  chat y foros discusión..............................................................................28
3.1.2Comisión Jueves : Google Sheet Colaborativa de la Asignatura...................................................29
3.1.3Plataforma Hypothes.is................................................................................................................29
3.1.4Plataforma ChatGPT/Claude........................................................................................................29
3.1.5Plataforma GitHub.......................................................................................................................30
3.1.6Check In  a la Asignatura   en Zulip..............................................................................................31
4Herramientas, Conceptos, Operación y Buenas pácticas..............................................................................34
4.1UBA Preguntas de código, Google Cloud, Git, etc  en clases presenciales..........................................34
4.2UBA Preguntas de código, Google Cloud, Git, etc  en Sesiones de Tutorias........................................34
4.3Zulip....................................................................................................................................................34
4.3.1Buenas prácticas para el uso de Zulip..........................................................................................36
4.3.2Participaciones extraordinariamente significativas en Zulip........................................................37
4.3.3Participaciones significativas en Zulip..........................................................................................37
4.3.4Mensajes Privados a los profesores Permitidos...........................................................................37
5Tutorías.........................................................................................................................................................39
5.1Objetivo..............................................................................................................................................39
5.2Tuplas de tutoría.................................................................................................................................39
5.3Agenda de sesiones de tutorías..........................................................................................................39
5.4Trabajo previo.....................................................................................................................................39
5.5Sesión de tutoría.................................................................................................................................40
6Git y GitHub..................................................................................................................................................41
2
[Página 3]
6.1.1conceptos y operación.................................................................................................................41
6.2Librería  data.table  del lenguaje R.....................................................................................................42
7Niveles de complejidad.................................................................................................................................44
7.1Etapa Preprocesamiento de Datos......................................................................................................44
7.1.1Armado del Target.......................................................................................................................44
7.1.2Dimensionalidad del dataset.......................................................................................................44
7.1.3Tratamiento Valores faltantes......................................................................................................44
7.1.4Calidad de Datos..........................................................................................................................44
7.1.5Data Drifting................................................................................................................................44
7.1.6Feature Engineering intra mes.....................................................................................................44
7.1.7Feature Engineering histórico......................................................................................................44
7.1.8Aceleración Procesamiento.........................................................................................................44
7.2Etapa Modelado.................................................................................................................................45
7.2.1Clase Target.................................................................................................................................45
7.2.2Estrategia Entrenamiento............................................................................................................45
7.2.3Algoritmo.....................................................................................................................................45
7.2.4Optimizacion de Hiperparámetros...............................................................................................45
7.3Etapa Puesta en Produccion...............................................................................................................45
8Experimentos Colaborativos.........................................................................................................................45
8.1Documento Colaborativo  Google Slides.............................................................................................47
8.2Problemas disponibles y alumnos avanzados.....................................................................................48
8.3Configuración de los grupos...............................................................................................................48
8.4Forma de pre registración...................................................................................................................48
8.5Negociación de Problemas : vacancia.................................................................................................48
8.6Negociación de Problemas : sobreoferta............................................................................................49
8.7Asignación Grupo A vs Grupo B..........................................................................................................49
8.8Antagonismo de los grupos................................................................................................................49
8.9Perídos de análisis de los grupos........................................................................................................50
8.10Aprobación de Diseños Experimentales...........................................................................................51
8.11Ceteris Paribus..................................................................................................................................52
8.12Capítulo Conclusiones......................................................................................................................52
8.13Video Presentaciones Ejecutivas.......................................................................................................52
9¿Cómo seguir?..............................................................................................................................................53
9.1Mantenerse actualizado.....................................................................................................................53
9.2Datasets Públicos................................................................................................................................53
10Empiricismo en la Ciencia de Datos............................................................................................................54
10.1.1Bibliografía Introductoria...........................................................................................................54
11Las dos culturas en el modelado predictivo................................................................................................55
12Estimando la Ganancia de un modelo predictivo........................................................................................56
12.1Bibliografía Introductoria..................................................................................................................56
12.2Sesgo Varianza en la estadística clásica.............................................................................................56
12.2.1Bibliografía Inicial......................................................................................................................56
12.3Sesgo Varianza en el moderno Machine Learning............................................................................57
12.3.1Bibliografía Inicial......................................................................................................................57
13Análisis exploratorio de datos.....................................................................................................................58
13.1Consistencia longitudinal de los atributos........................................................................................58
13.2Drifting de atributos.........................................................................................................................58
13.3Tratamiento de variables "rotas"......................................................................................................58
13.4Tratamiento de nulos........................................................................................................................58
13.5Tratamiento de outliers....................................................................................................................58
14Algoritmo: Árbol de Decisión......................................................................................................................59
14.1Bibliografía Inicial.............................................................................................................................59
3
[Página 4]
14.2El hiperparámetro cp en la librería rpart..........................................................................................59
15Comparación de Modelos Predictivos........................................................................................................60
15.1Bibliografía Inicial.............................................................................................................................60
16Optimización de Hiperparámetros..............................................................................................................61
16.1Bibliografía Introductoria..................................................................................................................61
16.2Grid Search.......................................................................................................................................61
16.3Bayesian Optimization......................................................................................................................61
17MLOPS........................................................................................................................................................62
17.1Bibliografía Inicial.............................................................................................................................62
18Curva ROC...................................................................................................................................................63
18.1Bibliografía........................................................................................................................................63
18.2Curva ROC de un split en un árbol de decisión.................................................................................63
18.3Curva ROC de un modelo predictivo.................................................................................................63
18.4Relación de la Curva ROC con la función de ganancia.......................................................................63
19Controlando el False Discovery Rate...........................................................................................................64
19.1Bibliografía Introductoria..................................................................................................................64
20Feature Engineering....................................................................................................................................65
20.1Bibliografía Introductoria..................................................................................................................65
20.2Cambiando la clase...........................................................................................................................65
20.3Variables Manuales...........................................................................................................................65
20.4Transformaciones tradicionales........................................................................................................65
20.5Variables históricas...........................................................................................................................66
20.6Lags y delta lags................................................................................................................................66
20.7Tendencias........................................................................................................................................66
20.8Medias móviles.................................................................................................................................66
20.9Ajuste por Inflación...........................................................................................................................66
20.10Variables hojas de un Random Forest.............................................................................................66
20.11Reducción de la dimensionalidad para acelerar.............................................................................66
21Ensembles de Modelos...............................................................................................................................67
21.1Bibliografía Introductoria..................................................................................................................67
22Algoritmo: Random Forest..........................................................................................................................68
22.1Bibliografía Introductoria..................................................................................................................68
23Algoritmo: Gradient Boosting of Decision Trees.........................................................................................69
23.1Bibliografía Introductoria..................................................................................................................69
23.2XGBoost............................................................................................................................................69
23.2.1Bibliografía XGBoost..................................................................................................................69
23.3Optimización hiperparámetros XGBoost...........................................................................................69
23.4Bibliografía LightGBM.......................................................................................................................69
23.5Optimización hiperparámetros LightGBM........................................................................................69
24Estrategia de Entrenamiento......................................................................................................................70
24.1Determinación de los mejores períodos para entrenar....................................................................70
24.2Acelerando el entrenamiento : evitando el cross validation.............................................................70
24.3Acelerando el entrenamiento : Undersampling de la clase mayoritaria...........................................70
25Importancia de Variables............................................................................................................................71
25.1Bibliografía Introductoria..................................................................................................................71
25.2Valores Shapley.................................................................................................................................71
26Ensembles de Modelos Segunda Parte.......................................................................................................72
26.1Semilleríos........................................................................................................................................72
26.2Hibridación de Semilleríos................................................................................................................72
27Acelerando el procesamiento.....................................................................................................................73
27.1Bibliografía Inicial.............................................................................................................................73
27.2El problema de la calibración en Gradient Boosting.........................................................................73
4
[Página 5]
27.3Otra forma de cortar.........................................................................................................................73
28Clustering jerárquico basado en la distancia de Random Forest.................................................................74
29Storytelling.................................................................................................................................................75
29.1Historia.............................................................................................................................................75
29.2Bibliografía Introductoria Storytelling...............................................................................................77
29.3Herramientas para grabar videos......................................................................................................77
29.4Grabación sencilla de videos con Microsoft Teams...........................................................................78
29.5Torneo de Videos..............................................................................................................................78
30Artículos de negocios..................................................................................................................................79
31El Recuperatorio.........................................................................................................................................79
32Hoja de Ruta clases jueves en construcción................................................................................................80
32.1Introducción y construcción Primera Competencia..........................................................................80
32.2Construcción  Ultima Competencia..................................................................................................82
5
[Página 6]
1
La Asignatura
Esta asignatura en su versión 2026 será una intensa y personalizada experiencia emocional, social y 
cognitiva, con una dinámica totalmente distinta a las que está acostumbrado; le demandará un gran 
involucramiento. Los instructores intentarán sorprenderlo en cada tutoría y clase.
Aprenderá a resolver un problema real, con datos reales (millones de registros y cientos de columnas), 
procesará en la nube Google Cloud y utilizará los algoritmos estado del arte como ser la Bayesian 
Optimization y LightGBM. Será motivado a pensar out of the box en Feature Engineering, organizando el 
trabajo bajo la metodología de MLOps. 
La materia posee competencias de ciencias de datos con el objetivo de construir el mejor modelo predictivo
que resuelva el problema de retención de clientes financieros.
La metodología es la de flipped classroom,  deberá venir con los experimentos corridos a las clases , 
mantendrá contacto asincrónico con los profesores y compañeros con la herramienta de foro/chat Zulip y 
periódicamente deberá asistir a tutorías virtuales para recibir retroalimentación sobre el avance de su 
modelado. En las clases presenciales se desarrollarán dinámicas de active learning en contraposición a la 
tradicional dinámica de clases magistrales. 
El centro de esta asignatura es usted formando con sus pares una comunidad de aprendizaje 
colaborativo, recibiendo retroalimentación personalizada en tutorías.
1.1 Sobre este documento
No se sienta abrumado por la extensión y detalle de este documento, en la primera clase se le explicará 
coloquialmente que se espera de usted.
No imprima en papel este documento;  está solo completo tan solo hasta el capítulo 3,  qué es lo que 
necesita para iniciar la materia. Se le irá agregando contenido los próximos días a los primeros capítulos, y el
resto a medida que avance la materia.
6
[Página 7]
1.2 Links fundamentales año 2026
UBA  DMEyF 2026
Links Fundamentales
Zulip
https://uba26.zulip.rebelare.com/
repositorio GitHub oficial
https://github.com/dmecoyfin/dmeyf2026
Diccionario de Datos
https://storage.googleapis.com/open-courses/dmeyf2026-
9c6f/DMEyF_2026.odt
Dataset Primera Competencia 
https://storage.googleapis.com/open-courses/dmeyf2026-
9c6f/competencia_01_crudo.csv
Google Sheet Colaborativa 
Comisión Jueves
https://docs.google.com/spreadsheets/d/1Q03J0AfgDxS2wo79ZUlon8
Q32_jfAhyx-f1e0r6sXws/edit?usp=sharing
Google Slides de Experimentos 
Colaborativos
Aún no definido
Dataset Ultima Competencia
Aún no definido
7
[Página 8]
1.3 Cronograma de la Asignatura año 2026
Eventos Importantes
ambas comisiones
DMEyF 2026
Fecha
Hora
Canal
Actividad
Mie 05-ago
20:00
email
Envio por email de este documento a alumnos
Lun 10-ago
19:00 a 22:00
Presencial
Clase 01  comision Lunes
Jue 13-ago
19:00 a 22:00
Presencial
Clase 01  comision Jueves
Mar 18-ago
19:00 a 22:00
Presencial
Comisión lunes, clase movida por feriado
Lun 15-sep
random
Zulip
Disponibilizacion Instructivo instalacion ambiente 
Google Cloud
dom 27-.sep
'23:59
Zulip
Entrega video Miranda Wintour
sem 22-sep
semana
Presencial
Lanzamiento Experimentos Colaborativos
sem 28-sep
semana
Virtual
Lanzamiento Tutorias, primer encuentro
Dom 04-oct
11:59:00 PM
Zulip
Cierre Primera Competencia
sem 05-oct
semana
Virtual
Tutorias   Disaster Autopsy
Mar 13-oct
19:00 a 22:00
Presencial
Comisión lunes, clase movida por feriado
Jue 05-nov
19:00 a 22:00
Presencial
Clase Plenaria  (AMBAS comisiones)
Mar 21-nov
19:00 a 22:00
Presencial
Comisión lunes, clase movida por feriado
Mie 09-dic
23:59
Zulip
Cierre  Última Competencia
Jue 10-dic
19:00 a 21:59
Presencial
Examen Individual
Dom 13-dic
23:59
Zullip
Fecha límite entrega del video a Michelina Porto
Lun 14-dic
11:00
Zulip
Lanzamiento Torneo de Videos
Sab 19-dic
23:30
Zulip
Fin Torneo de Videos
Dom 20-dic
23:30
Zulip
Publicación de Notas en la cartelera de Zulip
Lun 21-dic
23:30
email
Emails con recuperatorios
8
[Página 9]
1.4 Dedicación Estimada
La materia en el año 2026  consta de 54 horas de encuentros  presenciales. 
Además de las horas presenciales hacen falta, en promedio, otras 54 horas entre asistencia a las tutorias 
virtuales y trabajo  asincrónico.  Resumiendo, usted deberá dedicar en promedio 54 + 54 = 108 horas a esta 
materia
La asistencia a las tutorías virtuales es obligatoria.
Se estima que el alumno en promedio deberá dedicarle otras 54 horas fuera de las clases presenciales a :
•
diseñar, ejecutar, interpretar los resultados de experimentos que intenten mejorar los modelos 
predictivos.  En función de esa interpretación, idear nuevos experimentos superadores
•
avanzar etapas del proyecto para mostrar en la sesión de tutoria
•
asistir a la sesiones tutorías virtuales, de 55 minutos cada una.
•
instalar, aprender a manejar herramientas como Git, GitHub, Google Cloud
•
ver videos/lecturas previas para estar preparado para la siguiente clase sincrónica, comentado en 
Zulip lo leído.
•
responder quizzes asincrónicos en Zulip
•
Zulip
◦
leer los mensajes de los profesores y compañeros
◦
preguntar a los profesores
◦
responder a sus compañeros
•
programar scripts, realizar modificaciones a los scripts oficiales de la materia
•
operar el entorno Google Cloud, administrando efectivamente las restricciones
•
participar de  "Experimentos Colaborativos”
Todas las horas anteriores no contemplan temas que los alumnos quieren profundizar por su cuenta.
Contemplar pasivamente la pantalla a la espera que una virtual machine de Google Cloud termine de 
procesar no cuenta para el cómputo de horas humanas dedicadas.
Históricamente hay un puñado de alumnos outliers que llegan con un ímpetu extraordinario a la maestría,  
la asignatura les resulta de interés para su futuro laboral, disponen del tiempo, que dedican la descomunal 
cantidad de 150 horas humanas asincrónicas a la asignatura. 
9
[Página 10]
1.5 UBA Modalidad y Criterios de Aprobación y Evaluación
La nota mínima de aprobación de la materia es  5.50 , la nota final oficial se redondea a un número entero,  
la máxima nota final posible es 10.
 Nota Ordinaria
Contribución
Nota
Actividades Obligatorias
10%
Primera Competencia. Tarea grupal de 1 o 2 personas
40%
Última Competencia. Tarea grupal 1 o 2 personas
20%
Experimentos Colaborativos. Tarea Grupal
8%
Participación significativa en conversaciones el foro de la materia con 
herramienta Zulip .Tarea Individual
9%
Video Presentación de 5 minutos con storytelling a la Directora Comercial 
explicando perfiles de clientes que se dan de baja y proponiendo acciones 
de retención. Tarea Individual.
10%
Video Presentación de 5 minutos sobre lo más relevante de la confección 
del modelo de la Tercera Competencia, con storytelling dirigido la Gerenta 
de Business Intelligence. Tarea Individual
1%
Tutorias virtuales, asistir con grado de avance en el proyecto y participar del
intercambio de ideas. Asistencia mandatoria.
1%
Evaluación individual comprobación de autoría de la Segunda Competencia
1%
Torneo de Videos,  ser jurado en "duelos" de videos para calificar videos de 
pares. Tarea Individual
100%
Total
Todas las actividades anteriores son obligatorias. La no participación/entrega en cualquiera de ellas implica 
el desaprobar la materia.
La detección de no autoría de la segunda competencia implica desaprobar la materia.
10
[Página 11]
1.5.1 Modalidad y Criterios de Aprobación y Evaluación Competencias
En las competencias se puede participar formando grupo de una sola persona, o grupos de dos personas; 
no están permitidos los grupos de tres o más personas.  Puede formar grupos distintos en cada 
competencia, trabajar en algunas solo y en otras en grupo, etc
Cada competencia tendrá limitaciones sobre los algoritmos y datos que se pueden utilizar, usted deberá 
observar un cumplimiento estricto de dichas normas.
Es parte de la filosofía de la materia la reproducibilidad de los experimentos.  Su repositorio personal 
GitHub de la materia deberá permanecer completamente público y abierto, accesible para los profesores y 
sus compañeros, durante todo el transcurso de la asignatura hasta cinco meses después de la entrega de las
notas finales (servirá como material de referencia para quienes deban realizar un recuperatorio). 
Usted deberá subir a su repositorio personal de la materia en GitHub los scripts y experimentos que 
partiendo del dataset original permitan a los profesores replicar exactamente sus entregas finales en las 
competencias , esto incluye a las semillas aleatorias.
Su solución definitiva será analizada por los profesores para:
•
verificar que su entrega final fue generada por su script/notebook
•
verificar que los scripts/notebooks cumplen con las condiciones de la competencia
•
garantizar que está haciendo aportes significativos a la soluciones de la cátedra y a los 
"Experimentos Colaborativos"
•
garantizar que la solución de su equipo es original y sustancialmente distinta a la de los otras 
personas de la cohorte.
Es una práctica común la existencia de meta grupos, desde el mismo inicio prevea que para lograr entregas 
significativamente distintas deberá en algún momento cortar esa simbiosis.
Una vez cumplido exitosamente el requisito de originalidad, se pasa a evaluar la ganancia en los datos del 
mes del futuro.
Para cada competencia existirá una ganancia llamada  “Línea de Muerte”  que usted deberá superar para 
obtener  un 5.  Las ganancias por encima de la Línea de Muerte se normalizan y mapean al intervalo [5,10] 
de notas.
11
[Página 12]
1.5.2 Experimentos Colaborativos
Ver el capítulo homónimo para entender en que consisten los experimentos colaborativos.
Rúbrica de Experimentos Colaborativos
Porcentaje Concepto a Evaluar
10%
Hipótesis Experimental,  será brindada por los profesores
Los alumnos buscarán bibliografía que soporta, o no, la hipótesis experimental.
30%
Diseño experimental y materiales.
El diseño del experimento muestra una acabada comprensión del problema a resolver, los 
experimentos están precisamente especificados y detallados, los scripts son adecuados.
20%
Procedimientos y recopilación de resultados
La ejecución de los experimentos es correcta, los parámetros iniciales y resultados 
capturados son los adecuados.
20%
Análisis de los resultados.
Se interpretan correctamente los resultados obtenidos
20%
Resultados, conclusiones y discusión con pares
La conclusión determina claramente si la hipótesis experimental se cumple o no.
Se presentan los resultados en forma clara y convincente, se debaten los mismos con el resto
de los alumnos.
Se relacionan los resultados con la bibliografía original.
Se proponen nuevos experimentos en caso de cuestionamientos externos.
Cada alumno participa en forma individual, los profesores le asignan a cada alumno problemas/tareas 
específicas.
12
[Página 13]
1.5.3 Evaluación participación en foro/chat Zulip
Zulip es el sistema nervioso de la asignatura. 
Pasan más cosas en Zulip que en las clases sincrónicas.
Ver el capítulo Zulip dentro de Herramientas, Conceptos y Operación el detalle de lo que se espera de usted 
en Zulip
Rúbrica de participación en foro Zulip
Porcentaje Concepto a Evaluar
30%
Contenido. Propone ideas profundas, signfiicativas, claras, fáciles de implementar y con 
potencial para el proyecto de la asignatura. Compartiendo resultados logra atraer la atención
y colaboración de sus compañeros a sus posts.
Idealmente se transforma en un líder conceptual del grupo.
30%
Contribución al dinamismo de la comunidad. Plantea preguntas interesantes y relevantes, 
intenta motivar las discusiones grupales sobre tópicos relevantes a la asignatura, debate 
positivamente, participa de conversaciones iniciadas por otros.
20%
Colaboración  en dudas operativas, colabora rápida y acertadamente con compañeros que 
requieren algún tipo de asistencia operativa en alguna de las herramientas del curso, 
lenguaje de programación, etc
Idealmente se transforma en un referente en un tema específico del grupo.
20%
Frecuencia. Todas las semanas participa con al menos tres posteos relevantes.
Cada alumno participa con su usuario en forma individual en Zulip
13
[Página 14]
1.5.4 Evaluación video presentaciones
Debe leer la historia que genera la necesidad de las video presentaciones en el capítulo Storytelling.
Las video presentaciones deben cumplir mandatoriamente con las siguientes condiciones, en caso contrario
se le asignará una nota de cero y no participará del Torneo de Videos:
•
La duración esperada es de alrededor de 5:00 minutos;  jamás podrá exceder los 5:30 minutos.
•
su cara, mostrando claramente sus gestos, debe aparecer durante al menos el 95% del tiempo,  y 
debe ocupar al menos el 10% de la superficie del cuadro. Usted debe hablar mirando a la cámara.
•
En el video a Michelina Porto:
•
No puede exceder de 30 segundos la presentación del problema.
•
No debe explicar el significado de los términos de Ciencia de Datos, Michelina Porto ya los 
conoce.
•
Debe quedar clara la justificación de cada decisión tomada, si obedece a resultados de 
experimentos previos o a algún tipo de dogma revelado.
•
Es fundamental que narre los experimentos que le dieron resultados sub óptimos, esa es la 
forma en que el equipo aprende.
Rúbrica Videos Presentación
Porcentaje Concepto a Evaluar
10%
Entretenimiento.  La video presentación es apasionante y no hay sección que aburra, son 5 
minutos de pura adrenalina.
10%
Audiencia.  La video presentación tiene totalmente en cuenta la audiencia a la cual está 
dirigida y saca provecho de las características únicas de esa audiencia.
30%
Historia .  La presentación narra una historia,   hay una clara introducción con un "gancho" 
que invita a ver el video, un desarrollo adecuado con una continuidad argumental lógica y un 
desenlace concreto. La narrativa está organizada en torno a las etapas de la pirámide de 
Freytag o estructura de similar complejidad donde la emoción y la sorpresa juegan un papel 
fundamental.
25%
Consistencia  del contenido.  Lo presentado refleja fielmente el conocimiento descubierto en 
los experimentos efectivamente realizados y las conclusiones están sustentadas en datos que 
aparecen presentados adecuadamente.
25%
Originalidad del contenido.  Las ideas presentadas son originales, ingeniosas, basadas en una 
profunda comprensión del problema y su solución.
Cada alumno debe hacer  dos video presentaciones, es una tarea individual. Por más que se haya formado 
grupo de dos personas para la/s competencia/s de la asignatura, los videos deben ser distintos así como el 
material que los soporta.
14
[Página 15]
La entrega de los videos se hace por Zulip, enviando un mensaje al Stream z-entregafinal  que se 
disponibilizará a su debido tiempo. El mensaje deberá contener un link de acceso público que no haga falta 
ni usuario ni password ni estar registrado en ninguna plataforma para quienes lo vean. 
Los formatos preferidos para los videos son: YouTube, Prezi, Loom  y  Twitch.  No se acepta la entrega de 
links a Google Drive o Dropbox.  El video debe ser accesible para todos por lo menos hasta cinco meses 
después de haber sido entregadas las notas, para que tengan acceso a todos ellos los alumnos que deben 
realizar el recuperatorio de la materia.
Es parte fundamental de la tarea que los alumnos investiguen la forma de hacer una video presentación 
efectiva e intercambien ideas en Zulip.  Ese descubrimiento será asistido por los profesores.
La calificación de los videos se realizará mediante un Torneo de Videos en donde votarán alumnos y 
también profesores, normalizando los puntajes obtenidos al intervalo [ 0, 10 ]. Se tomarán los recaudos para
garantizar el más alto grado de imparcialidad en las votaciones, como por ejemplo, no se podrá ser juez de 
un duelo de videos donde participan personas con las que se ha formado grupo.
Habrá una nota para el video a Michelina Porto y otra nota al video a Miranda Wintour.
15
[Página 16]
1.5.5 Evaluación  Individual
Esta instancia de evaluación individual es obligatoria para todos los alumnos, presencial, el día estipulado en
el cronograma, y se presenta en dos formatos mutuamente excluyentes:
•
examen oral individual
•
examen escrito individual
en ambos casos  los alumnos deberán responder preguntas personalizadas sobre cómo hicieron la 
predicción de la Ultima Competencia.  No se realizan preguntas teóricas. El objetivo es garantizar que el 
alumno realmente sea autor del trabajo de  la Ultima Competencia y conoce a la perfección cada parte de 
su código y ha sido él quien criteriosamente ha tomado cada una de las decisiones.
Aún en el caso que la Ultima Competencia fue realizada en un grupo de dos personas, ambas personas 
deberán conocer a la perfección todos los aspectos de la solución presentada y haber participado 
activamente en la toma de todas las decisiones de construcción del modelo. En el caso que ambos 
integrantes rindan examen oral, será individual.
En el Examen Escrito no se podrá disponer de ningún artefacto electrónico: teléfonos móbiles, laptops, 
smartwatches, anteojos inteligentes, etc  los que se deberán dejar sobre el escritorio del aula de examen.
El Examen Escrito deberá ser escrito de puño y letra en hojas de papel.
En el Examen Oral, el alumno deberá tener su laptop con acceso offline al código con el que ha generado el 
submit de la Ultima Competencia, ya que se le podrá pedir que explique detalles del mismo. No podrá 
acceder a  teléfonos móbiles, smartwatches, anteojos inteligentes, ni tampoco tendrá conexión a internet 
desde su laptop.
En ninguno de los casos podrá acceder a inteligencia artificial generativa del tipo LLM ( ChatGPT, Claude, 
etc)
El dia del examen, minutos antes del comienzo, cada alumno será notificado sobre la modalidad oral o 
escrita de su examen.
Más allá de la evaluación individual, la segunda competencia debe ser realizada con notebooks/scripts que 
impliquen un aporte significativo a los notebooks brindados por la cátedra. No es aceptable que la solución 
final sea la corrida de un notebook con el simple cambio de la semilla primigenia.
16
[Página 17]
1.6 Misión de la Asignatura
Intentar que l@s estudiantes se enamoren de la ciencia de datos.
Lograr que los alumnos sean capaces de resolver un problema de dimensiones reales del mercado 
argentino con una excelencia que sorprenda a sus pares.
Despertar la llama de la pasión sostenible por explorar con espíritu crítico nuevos saberes.
1.6.1 Objetivos de la Asignatura
1.
Resolver un problema de dimensiones reales del mercado local utilizando las herramientas
tecnológicas para manejar grandes volúmenes de datos y ser capaz de generar una predicción
competitiva.
2.
Aplicar en forma práctica a un problema concreto con datos reales los conocimientos vistos en
las materias teóricas de los cuatrimestres anteriores, enfrentando eficazmente todas las etapas
del data mining.
3.
Desarrollar una fuerte cultura de la experimentación siendo capaz de cuestionar todo lo que le han 
enseñado.
4.
Desarrollar la creatividad para el Feature Engineering
5.
Conocer y utilizar efectivamente las técnicas “estado del arte” en cuanto a 
1.
Algoritmos y librerías de última generación de modelado predictivo sobre datos estructurados
2.
Optimización de hiperparámetros
3.
Interpretación de modelos predictivos
4.
Metodología Machine Learning Operations
5.
Procesamiento de grandes volúmenes de datos en la nube
17
[Página 18]
1.7 Unidades Temáticas
1.
Nociones elementales de la actividad bancaria, ciclo de vida de un cliente, valor vida de un cliente, 
retención de clientes, campañas de marketing directo.
2.
Metodologías CRISP, Six sigma y SEMMA.
3.
Comparación de modelos predictivos
1.
El problema de las múltiples comparaciones
2.
Estimación Montecarlo
3.
Validación Cruzada
4.
Curva ROC, concepto de área bajo la curva AUC
5.
Breve reseña de Árboles de Decisión y de Árboles de Estimación de Probabilidad.
6.
Tratamiento de datasets desbalanceados
7.
Optimización de hiperparámetros
1.
Red de Búsqueda (Grid Search)
2.
Optimización Bayesiana
8.
Métodos de Ensamblado de Modelos Predictivos
1.
Voting
2.
Bagging
3.
Boosting
4.
Stacking
9.
Algoritmos de Ensemble
1.
Random Forest
2.
Gradient Boosting of Decision Trees
1.
XGBoost (año 2016)
2.
LightGBM (año 2017)
3.
CatBoost (año 2018)
4.
NGBoost (año 2019)
10. Ingeniería de Atributos ( Feature Engineering )
1.
 Tratamiento de nulos
2.
 Selección de variables
3.
 Variables derivadas, el problema de los cortes paralelos de los árboles de decisión.
4.
 Incorporación de variables históricas
11. Metodología de Walk Forward Validation
12. Interpretación de predicciones por el método “Explicaciones Locales Aditivas de valores Shapley”
18
[Página 19]
13. Clustering Jerárquico basado en la distancia derivada de Random Forest
14. Nociones básicas de MLOPs
1.
 Administración de datos
2.
 Plataforma de Experimentación
3.
 Puesta en producción
4.
 Monitoreo
15. Nociones básicas de Storytelling
1.
armado de una narración, Pirámide de Freytag, Camino del Héroe de Joseph Campbell
2.
 buenas prácticas para el armado de gráficos
3.
 buenas prácticas para el armado de video presentaciones
1.7.1 Tecnologías a enseñar y reforzar
•
Lenguajes Python,  R,  SQL
•
Vibe Coding
•
Plataforma Jupyter Lab
•
Control de versiones de código Git y GitHub
•
Plataforma open source de MLOps : MLFlow
•
Procesamiento en Google Cloud, creación de máquinas virtuales, imágenes del sistema operativo, 
instancias spot
•
Rudimentos de la terminal de Linux
•
Docker
•
Control de Versiones de Datos Git LFS
19
[Página 20]
1.8 Bibliografía General de la Asignatura
La siguiente es la bibliografía general; en cada clase se brindará bibliografía específica sobre los temas 
vistos.
Libro de cabecera    https://hastie.su.domains/ElemStatLearn/
Hastie, T, Tibshirani, R. Friedman, J. The Elements of Statistical Learning, Second Edition, Springer Series
in Statistics, Springer, 2017
Sitios con muy buenos artículos
https://machinelearningmastery.com/  muy precisos los artículos de Jason Brownlee
https://www.kdnuggets.com/
Sitios populares fáciles de leer que hay que tomar con pinzas sus artículos ya que por lo general carecen de 
profundidad, y lo que es muy grave hay muchos con enormes errores conceptuales y prácticos:
https://towardsdatascience.com/
https://medium.com/tag/data-science
Papers y otros Libros
Chen, Tianqui, XGBoost: A Scalable Tree Boosting System, Proceedings of the 22Nd ACM SIGKDD 
International Conference on Knowledge Discovery and Data Mining, pages 785–794. ACM, 2016. 
Duan, Tony, NGBoost: Natural Gradient Boosting for Probabilistic Prediction, Proceedings of 
the37thInternational Conference on Machine Learning,  2020
Fawcett, Tom, ROC Graphs: Notes and Practical Considerations for Data Mining Researchers. Technical 
Report HPL-2003-4, HP Labs, 2003.
Flach, Peter Putting Things in Order, On the fundamental role of ranking in classification and probability 
estimation. , 18th European Conference on Machine Learning, 2007
Gareth, James, An Introduction to Statistical Learning with Applications in R, Springer Texts in Statistics, 
Springer, 2017
Guo, P, Kim, Jm Rubin, R, How video production affects student engagement: An empirical study of MOOC 
videos, Proceedings of the first ACM conference on Learning @ scale conference, 2014
Hastie, T, Tibshirani, R. Friedman, J. The Elements of Statistical Learning, Second Edition, Springer Series in 
Statistics, Springer, 2017
He, K.; Zhang, X.; Ren, S.; Sun, J. Deep residual learning for image recognition. In Proceedings of the 
Conference on Computer Vision and Pattern Recognition, Las Vegas, NV, USA, 30 June 2016.
Ke, Guolin, LightGBM: A Highly Efficient Gradient Boosting Decision Tree, NIPS'17: Advances in Neural 
Information Processing Systems Conference, 3148-3156 , 2017
20
[Página 21]
Krizhevsky, A.; Sutskever, I.; Hinton, G.E. Imagenet classification with deep convolutional neural networks.
Commun. ACM 2017, 60, 84–90.
Knaflic, C. (2015). Storytelling with data : a data visualization guide for business professionals . Hoboken, 
New Jersey: Wiley. 
Kuhn, Max, Feature Engineering and Selection: A Practical Approach for Predictive Models 
Chapman&Hall/CRC Data Science Series, 2019
Imani, M.; Arabnia, H.R. Hyperparameter optimization and combined data sampling techniques in machine 
learning for customer churn prediction: A comparative analysis. Technologies 2023, 11, 167.
Lundberg, Scott, A Unified Approach to Interpreting Model Predictions, 31st Conference on Neural 
Information Processing Systems, 2017
Pang-Ning Tan, Introduction to Data Mining second edition, What's New in Data mining Series, Pearson, 
2018
Pearl, Judea, Causality, Models, Reasoning and Inference, Cambridge University Press, 2009
Piotr Skalski, David Sutton, Stuart Burrell, Iker Perez, and Jason Wong. 2023. Towards a foundation 
purchasing model: Pretrained generative autoregression on transaction sequences. In Proceedings of the 
Fourth ACM International Conference on AI in Finance. 141–149. 
Provost, Foster, Robust Classification for Imprecise Environments. Proceedings of the Fifteenth National 
Conference on Artificial Intelligence (pp. 706-713). Menlo Park, CA: AAAI Press, 1998.
Prokhorenkova, Liudmila,  CatBoost: unbiased boosting with categorical features, NIPS'18: Proceedings of 
the 32nd International Conference on Neural Information Processing Systems, 2018
Raj, E, Engineering MLOps: Rapidly build, test, and manage production-ready machine learning life cycles at 
scale, Packt Publishing, 2021
Reshef, David, Detecting Novel Associations in Large Datasets, Science 2011, Vol. 334 no. 6062 pp. 1518-
1524, 2011
Rokach, L, Ensemble Learning: Pattern Classification Using Ensemble Methods, Second Edition, Series in 
Machine Perception and Artificial Intelligence, WSPC, 2019
Salzberg, Steven, On Comparing Classifiers: Pitfalls to Avoid and a Recommended Approach. Data Mining 
and Knowledge Discovery Journal, Kluwer Academic Publishers, 1, 317-327. 1997.
Zhao, X.; Liu, Y.; Zhao, Q. Improved LightGBM for extremely imbalanced data and application to credit card 
fraud detection. IEEE Access 2024, 12, 159316–159335. 
Z.-H. Zhou, Ensemble methods: foundations and algorithms, Second Edition, Chapman and Hall/CRC, 2025. 
Para la búsqueda de papers se recomienda primero buscarlos en Google Scholar  
https://scholar.google.com/  y si no está disponible bajarlo del inframundo https://sci-hub.ru/
Página para bajar libros  https://annas-archive.pk/
Herramienta para visualizar conexiones entre papers  https://www.connectedpapers.com/
21
[Página 22]
2
Metodología de Enseñanza
2.1 Generalidades
Basados en el  authentic learning se plantea un problema de negocios de la vida real tal cual se podría 
encontrar en una empresa en Argentina, con una cantidad de datos real, y una forma de evaluación igual a 
la que el proyecto sería evaluado en una empresa :  ganancia en pesos argentinos de la campaña de 
retención de clientes y exposición de los resultados con storytelling.
La asignatura se dicta con el espíritu de la teoría del conectivismo, donde los alumnos deben crear 
colaborativamente contenido, reflexionar sobre el contenido de sus pares y tutores, discutirlo y de esa 
forma generar nuevo contenido.  
La dinámica de las clases presenciales (opcionales) es la del flipped classroom (aula invertida)  donde los 
alumnos ven los materiales más simples y hacen experimentos asincrónicamente antes de la clase 
sincrónica,  y una vez en la clase sincrónica se utiliza una dinámica de active learning  basado en pares, más 
algunas exposiciones de los profesores para los temas más complejos. Dado que los alumnos pueden 
consultar por Zulip en forma asincrónica antes de la clase, técnicamente el  nombre correcto de la 
metodología es  active learning forte.
Las tutorias son encuentros virtuales  (mandatorios) de un tutor y tres estudiatnes a los que el/la estudiante
llegan a recibir retroalimentación personalizada de sus avances del proyecto de predicción.
2.2 Manifiesto Pedagógico
Si una sola frase debiera resumir la pedagogía de esta asignatura  sería la expresión de Plutarco :
Enseñar, más que llenar un recipiente es encender un fuego.
Similarmente, si la pedagogía de esta asignatura sigue el siguiente video de Seymour Papert (46 segundos) 
https://www.youtube.com/watch?v=9qj1E-uhFYc
Las filosofías pedagógicas se pueden dividir a grandes rasgos en estas teorias antagónicas:
•
el conductismo- instructivismo de Edward Thorndike y B. F. Skinner que es la predominate en el 
mundo y Argentina 
•
el constructivismo social de John Dewey y Seymour Papert que será utilizada en esta asignatura.
El estilo en esta asignatura es el del aprendizaje basado en pares, en donde a pesar de existir un sendero 
diseñado por los profesores se alienta continuamente a los alumnos a recorrer su propio camino, observar, 
valorar y aprender de sus pares, diseñar creativamente sus propios experimentos reflexionando 
profundamente sobre los resultados que se van obteniendo, cuestionando al establishment. Como 
referencia se puede tener la filosofía y comunidad que existe alrededor del lenguaje de programación 
Scratch y los conceptos de experiencias educativas de Mitchel Resnick.
La asignatura es una experiencia de gran intensidad intelectual y emocional que busca constantemente 
alentar a que cada alumno supere lo que se espera de él; se intenta lograr un efecto de resonancia grupal 
que amplifique el aprendizaje.
El estilo de dictado está basado en la frase "la emoción es el timón de la razón" donde cada tema es 
22
[Página 23]
presentado de forma que genere gran emocionalidad en los alumnos, por momentos usted sentirá 
profundo amor y fascinación por las ideas que se estarán viendo,  en otros literalmente sentirá odio.
Finalmente, aunque no menos importante, la asignatura busca la reflexión profunda y el cuestionamiento 
crítico de lo enseñado en otras asignaturas e incluso en ella misma. Citando a Richard Feynman
 "The problem is not people being uneducated. The problem is that people are educated just 
enough  to believe what they have been taught, and not educated enough to question anything 
from what they have been  taught".   
Dos videos y un artículo que reflejan aspectos de la forma de enseñanza
https://www.youtube.com/watch?v=uRxD-pe3PN0
Mitchel Resnick, Lifelong Kindergarten Group, 
Media Lab, MIT  
https://web.media.mit.edu/~mres/papers/constructionism-2014.pdf  Mitchel Resnick, Construtionism
https://www.youtube.com/watch?v=yx5VHpaW8sQ   George Siemens, Connectivism
2.3 Pensamiento Creativo – Tinkering
Para ilustrar lo que se espera del alumno en esta asignatura, van unos párrafos del libro Resnick, Mitchel , 
Lifelong Kindergarten: Cultivating Creativity through Projects, Passion, Peers, and Play, The MIT Press, 2018
One of the students, named Nicky, started by building a car out of LEGO bricks. After racing the car down a 
ramp several times, Nicky added a motor to the car and connected it to the computer. When he 
programmed the motor to turn on, the car moved forward a bit—but then the motor fell off the body of the
car and began vibrating across the table on its own. 
Rather than trying to repair the car, Nicky became intrigued with the vibration of the motor. He played and 
experimented with the vibrating motor, and began to wonder whether he might be able to use the 
vibrations to power a vehicle. Nicky mounted the motor on a platform atop four “legs” (LEGO axles).
After some experimentation. he realized that he needed some way to amplify the motor vibrations.
To do that, he drew upon some personal experiences. Nicky enjoyed riding a skateboard, and he 
remembered that swinging his arms gave him an extra push on the skateboard. He figured that a swinging 
arm might accentuate the vibrations of the motor as well, so he connected two LEGO axles with a hinged 
joint to create an arm and attached it to the motor. As the motor turned, the arm whipped around—and 
amplified the motor vibrations, just as Nicky had hoped. In fact, the system vibrated so strongly that it 
frequently tipped over.
A classmate suggested that Nicky create a more stable base by placing a LEGO tire horizontally at the 
bottom of each leg. Nicky made the revision, and his “vibrating walker” worked perfectly. Nicky was even 
able to steer the walker. When he programmed the motor to turn in one direction, the walker vibrated 
forward and to the right. When he programmed the motor to turn in the other direction, the walker 
vibrated forward and to the left.
I was impressed with Nicky’s vibrating walker—but even more impressed by the strategies he used in 
creating it. As Nicky worked on his project, he was constantly tinkering. Throughout the process, he was 
playfully experimenting, trying out new ideas, reassessing his goals, making refinements, and imagining new
possibilities.
23
[Página 24]
Like all good tinkerers, Nicky was:
•Taking advantage of the unexpected. When the motor fell off of his car, Nicky didn’t see it as a sign of 
failure; he saw it as an opportunity for new explorations.
•Drawing on personal experience. When Nicky needed to amplify the vibrations of the motor, he relied 
on his experiences as a skateboarder and knowledge of his own body.
•Using familiar materials in unfamiliar ways. Most people don’t imagine LEGO axles as arms or legs, nor 
do they imagine LEGO wheels as feet—but Nicky was able to look at objects in the world around him and 
see them in new ways.
Usted no jugará con bloques de un  LEGO , sino que configurará y reconectará scripts y bloques de código 
brindados por la cátedra, y si tiene la fortuna de ser atrapado por el entusiamo y motivación de Nicky, 
desarmará scripts y creará algo totalmente iinnovador.
2.4 Dinámica Constructivista en el Aula
En caso de preguntarse cómo concretamente difieren el conductismo del constructivismo, aquí va un 
ejemplo de dinámica sobre como medir para niños de 6 años
•
https://storage.googleapis.com/open-courses/videos-d0a6/whale_measuring.mp4  (4 minutos)
Schifter, Deborah, A constructivist perspective on teaching and learning mathematics,  Phi Delta 
Kappan; Bloomington Vol. 77, Iss. 7, (Mar 1996): 492-499
Sin embargo esta dinámica en el aula puede ser muy dificl para algunos alumnos adulto, ya que suele estar 
acostumbrado a la "Educacion Colonial"  https://storage.googleapis.com/open-courses/videos-
d0a6/colonial_education.mp4  (1 minuto)   y esperar “recetas” del profesor.
2.5 Authentic Learning
El enfoque pedagógico de esta asignatura está fuertemente relacionado con el Authentic Learning e 
indirectamente al Constructivismo Social en cuanto a:
•
El aprendizaje comienza con el planteo de un problema del mundo real, de dimensiones reales, que 
la mayoría de alumnos muy probablemente deban enfrentar su actividad profesional.
•
Algunos aspectos del problema están intencionalmente definidos de forma difusa para que los 
alumnos reflexionen profundamente sobre las mejores soluciones posibles.
•
La evaluación de la asignatura refleja la evaluación del mundo real profesional, basada en la 
rentabilidad, comunicación de resultados y colaboración con pares.
•
La teoría está al servicio de la práctica, y aparece justo a tiempo; jamás a la inversa.
•
El profesor colabora con Instructional Scaffolding ayudando a escalar los distintos niveles de 
complejidad conceptual.
2.6 Flipped Classroom Forte
La idea es que el alumno reciba la ayuda directa del profesor en los momentos de aprendizaje que más lo 
24
[Página 25]
necesita que generalmente es cuando el alumno se enfrenta a resolver un problema en soledad (no en el 
aula); para generar esa disponibilidad de ese tiempo lo fácil, la transmisión de información, se pasa para 
antes de la clase en modalidad asincrónica, cuando los profesores no están disponibles en tiempo real. 
Además se evalúa lo asincrónico con quizzes y con las submits a las competencias, permitiendo a los 
profesores responder por el foro/chat Zulip preguntas simples, y dejar lo que no ha sido entendido para 
reforzarlo en la siguiente clase presencial. Los quizzes muestran que se está entendiendo y que no, por más 
que el alumno crea que está entendiendo todo lo que lee.
Una presentación sobre Flipped Learing Forte   clara,  concisa y carente del usual laberinto literario de las 
ciencias sociales, es  https://www.slideshare.net/alfredo.prietomartin/intercambio-hispano-chileno-de-
experiencias-de-flipped-learning-en-enseanza-blended-y-online  junto con este artículo  
https://revistaventanaabierta.es/flipped-classroom-sus-limitaciones-y-aparicion-de-una-nueva-variante-
flipped-learning-forte/
2.7 Conectivismo
Las ideas da teoría del aprendizaje del conectivismo tienen gran semejanza con el deep learning, 
https://www.youtube.com/watch?v=yx5VHpaW8sQ (3 min)
•
El aprendizaje ocurre a traves de la red neuranal del cerebro, redes de humanos y redes de 
entidades no humanas.
•
La red de conocimento se desarrolla a partir de las interacciones con otras entidades de la red y el 
mundo en general. El objetivo de la enseñanza es estimular dichas interacciones.
•
La red crece:
◦
nodos
▪
incorporando nuevos nodos
▪
eliminado nodos ya no útiles
◦
conexiones entre nodos
▪
creando conexiones entre nodos que antes no estaban conectados
▪
reforzando la conexción entre dos nodos, al ser transitado frecuentemente
▪
disminuyendo el peso de la conexión, al ser poco transitado ese link
•
El conocimiento reside en los pesos de las conexiones
•
Aprender es el proceso de conectar nodos específicos o fuentes de información
•
La capacidad de aprender cosas nuevas es más importante de lo que se ya se conoce.
•
Tomar decisiones es un proceso de aprendizaje, de transitar reiteradamente y en formas novedosas 
la red
2.8 Active Learning
Active Learning  significa que se llevará al mínimo lo que en la jerga se denomina "clase magistral" que es 
básicamente un profesor hablando y los alumnos escuchando; los alumnos se deberán involucrar 
activamente durante la clase sincrónica con actividades diseñadas y guiadas por el profesor, en donde  
25
[Página 26]
deban reflexionar profundamente sobre el tema que se está viendo en clase. 
Active Learning es la metodología en la que toda la comunidad educativa está de acuerdo que brinda 
resultados superadores.  Puede verse este artículo  "Active learning increases student performance in 
science, engineering, and mathematics" https://www.pnas.org/doi/full/10.1073/pnas.1319030111 
Es tan distinto a la mecánica tradicional que muchos alumnos se sentirán incómodos al comienzo porque 
jamás han sido expuestos a la misma. No se confunda, active learning no tiene que ver con ser extrovertido,
sino con ser reflexivo !!
Videos cortos que explican los fundamentos del active learning:
https://www.youtube.com/watch?v=zoa2pKYp_fk
Janet Rankin, MIT
https://www.youtube.com/watch?v=Z9orbxoRofI
Eric Mazur, Harvard University 
Un artículo sobre aulas de active learning  https://er.educause.edu/articles/2017/12/creating-active-
learning-classrooms-is-not-enough-lessons-from-two-case-studies
Esta es una lista de ejemplo no vinculante del tipo de actividades que se llevarán a cabo  (ir a Large Group ) 
https://www.queensu.ca/teachingandlearning/modules/active/12_exmples_of_active_learning_activities.h
tml  siendo la actividad más tradicional el Think Pair Share  que es la que habla Eric Mazur
Al siguiente extenso video  es una clase en el MIT donde en un posgrado de educación la profesora  
demuestra a los alumnos diversas dinámicas de active learning https://www.youtube.com/watch?
v=hGBNi4P9OfA , no hace falta que lo vea,  ya lo experimentará.
2.9 Evaluación Continua
Habrá notado que la asignatura, en lugar de un gran examen o proyecto final, cuenta con varias pequeñas 
instancias minuciosamente detalladas de evaluación, distribuídas a lo largo de todo su desarrollo. Usted 
conoce exactamente cómo será evaluado, y las etapas de evaluación son pensadas para ayudarle en la 
subida de los andamios. 
De cada competencia aprenderá qué es lo que funciona y que no y deducirá como elegir un modelo que 
consistentemente funcione bien en datos nuevos del futuro. Cada competencia pone en práctica un 
conjunto específico de algoritmos y técnicas, de los resultados de cada competencia usted tendrá claro qué 
conocimientos deberá reforzar para el siguiente desafío y los profesores lo ayudarán en el proceso.
El participar del torneo de videos dirigidos a la  directora comercial Miranda Wintour, le permitirá observar 
la forma en que exponen sus compañeros, las herramientas que utilizan, y el feedback que le darán los 
profesores le servirá para generar un video muy superador para la gerenta de business intelligence 
Michelina Porto.
El participar en la tarea de "Experimentos Colaborativos", que dura varias semanas, lo retroalimentará 
continuamente sobre las mejores decisiones para la construcción de un excelente modelo predictivo. Usted 
recibirá feedback de sus compañeros y profesores sobre los problemas que deba resolver y experimentos 
que le corresponda realizar.
Durante el transcurso de la asignatura se le dará retroalimentación sobre lo relevante que está siendo su 
participación en el foro/chat Zulip, y se lo alentará continuamente a contribuir a la comunidad.
26
[Página 27]
Todo lo anterior, debido a que facilita que el alumno lleve al día la asignatura, permite intervenir a tiempo 
en las debilidades, evitará que pierda el hilo de la enseñanza, y aumentará para muchas personas la 
probabilidad de aprobar. 
Siempre habrá alumnos que no necesitan que les enseñen, sino simplemente que les indiquen qué vale la 
pena aprender primero.
27
[Página 28]
3
Abróchense los cinturones
En la materia trabajaremos con una gran variedad de herramientas,  plataformas y conjuntos de datos, las 
que deben ser instaladas,  configuradas, interconectadas y finalmente usted deberá aprender a utilizarlas. 
No se sienta desbordado por el shock inicial, aprenderá a utilizarlas. En palabras de un ex alumno "todas las 
semanas tuve en simultáneo la frustración de no entender alguna implementación y la satisfacción de 
dominar lo que una semana atrás parecía imposible."
Intente avanzar por su cuenta con los siguientes pasos de instalación, aunque no entienda en detalle lo que 
está haciendo, ya se le explicará el uso de las herramientas en clase.
En caso de tener dificultades con algún punto, o consúltelo en Zulip o espere a la primera clase en donde se 
verán conceptualmente estos pasos.
Palabras de un ex alumno : "Unos pocos empezamos la maestría con PC de Empresa con restricciones, 
bloqueos y rigideces. La fuimos "piloteando" pero cada vez se fue haciendo más cuesta arriba... Creo que en
esta materia finalmente todos terminamos con PC libre. En mi caso fue un salto cuántico de comodidad."
3.1 Alta en Plataformas
3.1.1 Zulip  herramienta de  chat y foros discusión
En la materia no se utilizará el email, toda la comunicación y discusión se llevará a cabo en Zulip. Los 
profesores serán por lejos quienes más participen en las conversaciones de Zulip.
Para darse de alta en el Zulip de la materia usted debe ingresar al link de Zulip de la sección 1.1 Links 
Fundamentales  y presionar el link de Sign Up o Regístrese (abajo a la derecha del recuadro)
Usted puede acceder a Zulip de tres formas:
•
En su PC desde algun browser
•
En su PC utilizando la app, instálela desde https://zulip.com/apps/
•
Desde su smartphone, busque la app en su App Store ( alerta : no es buena la app)
En el capítulo Herramientas, Conceptos y Operación  se le enseñará cómo utilizar efectivamente Zulip a lo 
largo de la asignatura, ya que parte de la nota final depende de ello.
Una vez dentro de Zulip, vaya a la rueda dentada ubicada arriba a la derecha,  Manage streams, elija el 
stream cartelera, y a la derecha marque Email notifications
28
[Página 29]
3.1.2 Comisión Jueves : Google Sheet Colaborativa de la Asignatura
Para proceder con este paso es necesario que usted deberá contar con una cuenta de gmail  la que además 
le será indispensable para trabajar con la plataforma Google Cloud.
Siga el link de Google Sheet Colaborativa  de la sección 1.2 Links Fundamentales de este documento.
Le aparecerá una pantalla de este estilo
presione el boton de  Solicitar acceso ,  a las pocas horas recibirá un email en su gmail informándole que ya 
tiene acceso a la planilla colaborativa, en donde deberá ir cargando resultados de las corridas que realice a 
lo largo de la asignatura y podrá comparar sus resultados con los de sus compañeros.
3.1.3 Plataforma Hypothes.is
Hypothes.is es una herramienta de anotación colaborativa de páginas web y documentos .pdf,  que también
utilizaremos para anotar scripts y Jupyter Notebooks . Le permitirá ver anotaciones que el profesor ha 
realizado en papers a utilizar en la asignatura, y usted mismo podrá agregar anotaciones.
La siguiente es una razonable introducción a Hypothes.is  https://www.youtube.com/watch?v=87h0nYi-i9o 
presta atención a cómo se hacen las anotaciones y a cómo se participa en una discusión sobre una 
anotación.
Ir a la página  https://web.hypothes.is/start/   y seguir los pasos para crear un usuario, se recomienda 
enfáticamente crear un usuario con el formato que sea   <su nombre>  punto   <su apellido>  . 
Adicionalmente instale la extensión para el navegador Google Chrome (la que podrá utilizar también en 
Brave), que le será indispensable para la anotación de  documentos .pdf
No haremos públicas las anotaciones en la asignatura, sino que las acotaremos al grupo privado dmeyf,  
para sumarse a estos grupos estando logueado a  Hypothes.is en su browser y luego  seguir este link 
https://hypothes.is/groups/oZ5iroYw/dmeyf
3.1.4 Plataforma ChatGPT/Claude
Utilizaremos el novedoso ChatGPT para que explique que hacen los scripts provistos por la cátedra y si 
tenemos suerte escriba scripts simples por nosotros.
Ingrese  a  https://chat.openai.com/auth/login y cree un usuario gratuito.  Dada la alta demanda de 
ChatGPT es posible que deba probar en distintos horarios hasta lograrlo.
29
[Página 30]
3.1.5 Plataforma GitHub
Para poder compartir nuestros scripts y experimentos habrá un repositorio oficial de la asignatura y uno 
suyo personas, ambos residirán en la nube. Para ello usted dará de alta una cuenta en los servicios de la 
plataforma GitHub
Registrarse en  https://github.com/signup?ref_cta=Sign+up
Una ayuda simple es https://www.classicpress.net/github-desktop-a-really-really-simple-tutorial/
Una vez ya registrado y dentro de su sesión de GitHub, es momento de crear su Personal Access Token que 
será algo así como su usuario y clave para acceder desde su PC local y desde Google Cloud.  
Dentro de su sesión de GitHub vaya a la página  https://github.com/settings/tokens  ,  elija la opcion que 
está a la izquieda de la pantalla  Tokens (classic)  y presione el botón  que está a la derecha  de la 
pantalla  Generate new token (classic)
•
donde dice Note escriba  "mi primer token",
•
en  Expiration marque  Custom  y elija como fecha el 31-dic-2026  (realmente es muy recomendable
que ponga una fecha de expiración)
•
 luego en  Select Scopes marque  TODOS los recuadros que le aparecen
•
al final de la página presione el botón   Generate Token 
Le aparecerá una pantalla como esta:
allí copie y guarde en un lugar seguro el token, que en el ejemplo de la imagen es el string   
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
usted volverá a utilizar ese token cuando instale Google Cloud, guárdelo en un lugar seguro por favor
30
[Página 31]
Una vez hecho el login a su GitHub usted no tendrá ningún repositorio creado inicialmente, ir al repositorio 
GitHub de la materia  cuyo link está en el capítulo  Links Fundamentales  y hacer un fork del repositorio  
presionando el botón que está arriba a la derecha de la página de GitHub . Por favor NO cambie el nombre 
del repositorio.
Le aparecerá un formulario para crear el fork
De esta forma usted pasará a tener en su cuenta de GitHub una copia del repositorio oficial  de la 
asignatura. Ambos están en la nube, pero usted no puede trabajar directamente con esos archivos en la 
nube sino que debe generar una copia en su computadora de su repositorio GitHub local.
3.1.6 Check In  a la Asignatura   en Zulip
Para los profesores es muy importante conocer quién es usted de forma de personalizar los mensajes en 
Zulip, recomendar lecturas personalizadas acordes a su formación, experiencia e intereses.
Adicionalmente, el CheckIn sirve para conocer a sus compañeros, en particular con el alto grado de 
virtualidad que ha tenido el primer cuatrimestre.
Enviar un gran mensaje Zulip al channel  #z-CheckIn   topic  CheckIn con los siguientes puntos  (recuerde
que es público)
1.
Identificación
1.
Nombre y Apellido
2.
Linkedin
3.
Usuario GitHub.com
4.
Edad
2.
Estudios previos
1.
Carrera de grado, Universidad, año de graduación
31
[Página 32]
2.
Posgrados realizados o en curso (no incluya a este posgrado...)
3.
Educación previa en ciencia de datos ( Coursera, edX, Udemy, Digital House, etc ) detalle
3.
La vida más allá de  Cero + Infinito
1.
¿Cuál fue el artista que más escuchó en su Spotify Wrapped 2025? Si no
utiliza Spotify, mencione el artista o grupo que más escuchó durante el
último año. 
2.
¿Qué comida o ingrediente le resulta irresistible? 
3.
¿Qué actividad, hobby, habilidad o rasgo importante de su vida sería
imposible descubrir leyendo su LinkedIn? 
4.
Tuvo alguna cábala durante el mundial de fútbol 2026? Cuál? si tuvo más
de una elija la que mejores resultados le dio. 
5.
¿Hubo algún tema, actividad, objeto o fenómeno con el que se haya
obsesionado particularmente durante el último año? ¿Cuál? 
4.
Cursada de esta maestría
1.
Del cuatrimestre anterior, relate en gran detalle alguna actividad de una materia que logró 
despespertar en usted pasión.  ¿Cómo interactuó con esa actividad?
2.
¿Qué materias de la maestría está cursando en este cuatrimestre en paralelo a DMEyF?
5.
Experiencia laboral
1.
¿Posee experiencia docente? explíquela
2.
¿A que se dedica en su trabajo actual? ¿Trabaja en ciencia de datos o en algo relacionado? En 
caso afirmativo detalle.
3.
¿En su trabajo actual tiene profesionales ( o equivalente)  que le reportan directamente a 
usted?
4.
¿Cuántas horas ha dedicado a programar en algún lenguaje en lo que va del año 2026, sin tener 
en cuenta este posgrado? Relate lenguajes y tareas.
5.
¿Detalle el uso que da a LLMs ( Claude , ChatGPT, Gemini,  etc) en su actividad profesional y 
académica? ¿Qué uso le da para programar?¿Paga mensualmente alguna de las herramientas?
6.
¿Cómo prefiere aprender usted, como el hijo o como la hija de Feynman?  Vea el siguiente video de 
Richard Feynman, ganador del Premio Nobel por sus aportes en Física Cuántica,  considerado el 
mejor profesor del siglo XX    ttps://www.youtube.com/watch?v=BY6VntTmtIo
7.
¿Tiene experiencia en la metodología y realización de experimentos, ya sea de ciencias naturales, 
físicas, o simulaciones informáticas?
8.
Primero lea en detalle https://subscription.packtpub.com/book/business-and-
other/9781787287037/1/ch01lvl1sec13/12-developer-learning-curve-why-learning-how-to-code-
takes-so-long
y luego relate su experiencia con lenguajes/entornos de programación/algoritmos
1.
lenguaje R 
2.
Tidyverse y dplyr en R
3.
data.table  en R
32
[Página 33]
4.
Git / GitHub
5.
Cloud Computing (Google Cloud, Azure, AWS, ...)
6.
lenguaje Python
7.
lenguaje Julia
8.
SQL
9.
Sistema operativo Linux
10. Árboles de Decisión
11. Gradient Boosting (XGBoost/LightGBM)
12. Utiliza algún entorno en particular de R ( Rstudio, R Base, etc )
13. Utiliza algún entorno en particular de Python ( VS Code, PyCharm )
9.
Elija CINCO números primos,  100003 < p <  999983 ,  intentando que sus números no se solapen 
con los que elijan sus compañeros de curso. Serán utilizados como sus semillas de generadores 
números pseudoaleatorios a lo largo de la asignatura. 
10. Aqui le haremos meditar profundamente. 
Enviar un gran mensaje Zulip al channel  #z-CheckIn   topic  Pregunta Trascendental usando
el  formato < Contexto, Pregunta> 
a partir de lo cursado en la primera mitad del año realizar la pregunta más trascendental que se le 
ocurra sobre los temas de ciencia de datos vistos, que sea respondible en esta maestría. Debe ser 
una pregunta conceptual que REALMENTE no sepa la respuesta y que considere que no le sería 
posible responder con facilidad (ej una consulta a Google o ChatGPT)
33
[Página 34]
4
Herramientas, Conceptos, Operación y Buenas pácticas
4.1 UBA Preguntas de código, Google Cloud, Git, etc  en clases presenciales
En las clases presenciales se verán temas conceptuales de Ciencia de Datos. En algunos casos se mostrará 
código para explicar o señalar algún concepto importante, en otras se hará una demostración de cómo 
correr una máquina virtual de Google Cloud. Otras veces si se detecta algún problema de código recurrente 
en los alumnos se explicará la solución conceptual en la clase presencial.   Incluso en casos extremos de 
alumnos con una pobre computer literacy en donde el diálogo vía Zulip ya no sea suficiente (el alumno no 
percibe lo que aparece en su pantalla) se realizará una videoconferencia instructor-alumno vía Jitsi Meet, 
desde el mismo Zulip.
Sin embargo no se utilizará tiempo de la clase presencial para responder a preguntas puntuales sobre 
código, problemas puntuales de Google Cloud, Git & GitHub, ya sea a todo el curso o a algún alumno que se
acerque con su laptop, ni minutos antes de la clase, ni después.  Todas las preguntas técnicas deben 
canalizarse públicamente a través de Zulip, deben quedan registradas en para ambas comisiones, de forma 
que todos los alumnos se puedan enriquecer. En caso de ser necesario se coordinará una videoconferencia 
de soporte técnico.
Por supuesto, las clases presenciales están completamente abiertas a todas las preguntas de Ciencias de 
Datos y los profesores harán todo lo posible, hasta el límite de su conocimiento, por responder en algún 
momento de la clase presencial, o luego de una pertinente investigación.
4.2 UBA Preguntas de código, Google Cloud, Git, etc  en Sesiones de Tutorias
En las sesiones de tutoria al igual que en las clases presenciales no se utilizará el tiempo para responder 
preguntas preguntas puntuales sobre código, problemas puntuales de Google Cloud, Git & GitHub. El 
soporte técnico siempre se brindará por Zulip y de ser necesario videollamadas.
4.3 Zulip
Zulip es el sistema nervioso de la asignatura.
Utilizaremos Zulip para
•
Solicitar y brindar soporte técnico.
•
Mantener discusiones en foros, asincrónicos fuera de las clases.
•
Intercambiar mensajes, links, artículos,  archivos entre alumnos entre sí,  alumnos y profesores
•
quizzes en la clase sincrónica y también en la tarea para el hogar asincrónica
•
Cartelera de información de la asignatura
•
resultados de Experimentos Colaborativos : 23 problemas
•
Entregas de la materia
Zulip es una aplicación, open source, de chat y foros de discusión con videoconferencia integrada, que  
34
[Página 35]
permite organizar las conversaciones en streams y topics posibilitando mantener intercambio de ideas en 
paralelo, lo que es una gran ventaja si no se está online todo el tiempo. La existencia de hilos de 
conversación (streams)y tópicos  permite una eficiente conversación no lineal y asincrónica. 
Zulip es open source y está corriendo en un servidor propio, Slack no es open source.   Zulip no está tan 
pulido y es menos intuitivo que Slack.  Sin embargo Slack tiene la contra que revisar canales con mucha 
actividad se vuelve muy tedioso ya que no se dividen en tópicos.
Zulip permite enviar estos tipos de mensajes :
•
privados  a otro usuario o grupo de usuarios o  a un stream privado
•
públicos,  a streams públicos, que actúan como salas de chat/foros de discusión
Todas las preguntas a profesores, sobre cualquier tema de la asignatura, deben ser siempre públicas. Es 
mandatoria esta transparencia.  Escribir mensajes privados es caer en el oscurantismo de los emails, es 
dejar de compartir con el resto y evita que se pueda leer lo que otros están pensando.
Lo más probable es que usted acceda a Zulip desde su computadora y además tenga instalada la app en su 
smartphone (Android y iOS están soportados)
Zulip como aplicación permite esto https://zulip.com/features/  , una relativa "falencia" de Zulip es que no 
permite enviar mensajes de voz en forma nativa.
Un video que muestra como utilizar Zulip es https://www.youtube.com/watch?v=xWa56KdgYZM
Stream
Utilidad
z-CheckIn
CheckIn , será utilizado solamente para enviar el check in a la materia, la 
información es pública, y servirá de consulta constante a los profesores para 
personalizar la comunicación con cada alumno.
cartelera
alumnos no pueden enviar mensajes, profesores anuncian
•
recordatorios fechas importantes
•
commits del repositorio oficial
•
actualizaciones Libro de la Asignatura
•
bibliografía nueva
•
comunicaciones generales
Clase n
Cada clase tendrá su correspondiente stream, y estos dos tópicos estarán siempre 
presentes:
•
antes  :  de que tratará la clase y que debo traer hecho/leido
•
despues:  resumen de lo que fue visto en clase, de las conclusiones
SoporteTecnico
Soporte técnico  sobre R, Rstudio, scripts oficiales, librerías de R generales, no 
incluidos en streams específicos
data.table
Temas específicos y soporte técnico sobre  data.table
35
[Página 36]
Google Cloud
Temas específicos y soporte técnico de Google Cloud
Git & GitHub
Temas específicos y soporte técnico de Git y GitHub
Exp Colaborativos
Intercambio Experimentos Colaborativos
general
Diálogos generales.
memes
memes
off-topíc
estados de ánimo generales, todo lo que no valdría la pena ir a buscar a los threads 
oficiales
zero2hero
temas sobre  From Zero to Hero
z-entregafinal
será utilizado para entregar la video presentación al Gerente de Ciencia de Datos
Los streams en fondo amarillo serán habilitados a su debido momento.
Si usted desea que un profesor preste atención a su mensaje, simplemente incluya  @profesores en el 
texto de su mensaje.
Alumnos de promociones anteriores han manifestado la importancia que tiene elegir correctamente el 
thread y tópico de cada mensaje : poder encontrarlos en el futuro, tener lo de un mismo tema todo en un 
solo lugar.
4.3.1 Buenas prácticas para el uso de Zulip
Zulip es una herramienta asincrónica, sus compañeros y profesores pueden escribir en cualquier momento 
del dia o la noche, incluso durante el fin de semana.
En caso que usted no pueda acceder en forma diaria a Zulip, al ingresar se sentirá abrumado por la enorme 
cantidad de mensajes que han enviado sus compañeros y se planteará : “¿Qué es lo importante y que es lo 
que puedo obviar?
El stream cartelera es fundamental, ya que es ahi donde los profesores pasan unos pocos anuncios 
vitales.
Todo mensaje privado que reciba de compañeros, también será vital que los responda cuanto antes.  Toda 
respuesta a un previo mensaje suyo también será importante que la vea cuanto antes.  Si configuró bien 
Zulip, en estos dos casos usted recibirá un email para avisarle que tiene algo importante.
Los streams  clase n  y sus tópicos  antes y despues  tambien son vitales y debe leer lo antes posible 
todo mensaje que aparezca,  ya que lo orientarán.
Es una maravillosa práctica que usted se cree un stream personal en Zulip con su nombre completo, que 
solamente usted podra ver y escribir en él, totalmente invisible a todos, incluso invisible a los profesores. En
ese stream personal usted podrá escribir sus ideas, y también guardarse links (punteros)  a otros mensajes 
que encontró interesantes.
Para todo el resto, le recomendamos esta estrategia que le ha resultado muy útil a alumnos de años 
previos:
1.
Empezar a leer superficialmente todos los mensajes no leídos
2.
Si el mensaje vale la pena, marcarlo con la estrella  “Star this message Ctrl + s” 
36
[Página 37]
3.
Luego, cuando disponga un momento de tranquilidad, ir a la carpeta de  
 Starred Messages 
★
 
y leer cada mensaje en detalle, quizas hasta preguntando algo a su autor. Una vez leido a conciencia
el mensaje, si es muy importante guardar un link al mismo en su stream privado (el que tiene su 
nombre)  y  lo desmarca.
4.
Con este método,  
 Starred Messages
★
 contiene los mensajes que temporalmente están 
pendientes de revision, y su stream privado contiene punteros a los mensajes que le interesaron.
4.3.2 Participaciones extraordinariamente significativas en Zulip
Muchos alumnos experimentan ansiedad sobre como participar en Zulip dado que no poseen 
conocimientos previos y suponen no tienen nada interesante para aportar.
Esta es la historia de Isidor I. Rabi  premio Nobel de Física en 1944 por la invención de lo que luego 
permitiría la Resonancia Magnética.
Interviewer: Why did you become a scientist, rather than a doctor or lawyer or 
businessman, like the other immigrant kids in your neighborhood?
Rabi: My mother made me a scientist without ever intending it.
Every other Jewish mother in Brooklyn would ask her child after school: 'So? 
Did you learn anything today?' But not my mother. She always asked me a 
different question.
 'Izzy,' she would say, 'did you ask a good question today?' 
That difference, asking good questions, made me become a scientist!" 
Son participaciones extraordinariamente significativas :
•
Realizar una pregunta conceptual (no de soporte técnico).
•
Responder una pregunta técnica a un compañero.
•
Recomendar bibliografía junto con un comentario por haber sido leida peviamente.
•
Compartir resultados de análisis exploratorio de datos.
•
Compartir un problema que se tuvo y la forma en que se solucionó.
•
Compartir tips sobre como operar eficientemente alguna aplicación.
4.3.3 Participaciones significativas en Zulip
Son participaciones significativas en Zulip:
•
Una pregunta técnica que aún no fue realizada  por un compañero
La cátedra prefiera que usted realice preguntas ténicas por Zulip en lugar del grupo privado de whatsapp 
para así poder entender los temas que no están quedando claros y reforzarlos ya sea por Zulip mismo o en 
la clase.
Es preferible curar a ese paciente aún con vida, antes que una vez finalizada la materia realizar una autopsia
para entender que salió mal y salvar vidas futuras.
37
[Página 38]
4.3.4 Mensajes Privados a los profesores Permitidos
Todas las preguntas a profesores, sobre cualquier tema de la asignatura, deben ser siempre públicas.
La heterogeneidad de alumnos en las maestrías de ciencias de datos genera que por un lado nos 
encontramos con alumnos que ya saben programar, se encuentran trabajando en ciencia de datos y realizan
preguntas avanzadas en clase,  mientras que otros recién están aprendiendo lo básico. 
Por lo general, varios alumnos se sienten tentados a preguntar en forma privada al profesor, manifiestan 
tener miedo que su pregunta sea elemental, o incluso mal formulada, que pueda ser mal vista por el resto. 
Se sienten incómodos con dicha exposición.  Le doy una maravillosa noticia, el 65% de los alumnos está en 
su misma situación, con lo que al hacer en forma pública la pregunta que considera elemental, dará una 
alentadora señal al resto y se producirá la propiedad emergente que muchos más se animen a preguntar, y 
tal cual lo expresa  Eric Mazur  será muy posible que obtenga una clara respuesta de un compañero que 
acaba de aprender el tema hace muy poco.
Estos son ejemplos de mensajes privados dirigidos a los profesores, que por no hablar de la asignatura, y ser
sobre temas privados, si están permitidos.
"Me surgió una posibilidad laboral en Madrid y la acepté, por lo que me voy a estar yendo en las próximas 
semanas y estoy haciendo gestiones a destajo. Como te podrás imaginar voy a tener que dejar el posgrado, 
aunque pienso empezar otro allá. Y en este punto, si sabés y tenés ganas de pasarme info, me vendría muy 
bien recomendaciones de buenos programas de maestría en ciencia de datos."
"Más adelante... si es posible y no es molestia, quisiera consultarte si podrías orientarme en encontrar una 
aplicación (PROYECTO DE TESIS) en la Industria Petroquímica / Generación de Energía....que es donde me 
desempeño actualmente.
Por supuesto que encontrar la aplicación es mi tarea, pero quizás algún exalumno hizo o está haciendo algo 
dentro del campo...o hay algún paper que se me esté escapando... . Cualquier dato es bienvenido. "
38
[Página 39]
5
Tutorías
Asistir a las clases presenciales es optativo;  asistir a las sesiones de tutoría mandatorio.
5.1 Objetivo
La asignatura intenta replicar un proyecto profesional real mediante la modalidad de Authentic Learning, en
donde debe construirse iterativamente un excelente modelo predictivo y explicar resultados mediante 
vídeos a dos  actores de la empresa.
A fin de garantizar el éxito del proyecto cada alumn@  participará periódicamente  (especialmente previo y 
posterior a entregas) de sesiones de tutorías, en donde junto a otros dos alum@s y un@ tutor@  recibirá 
retroalimentación constructiva personalizada sobre el avance de su proyecto incentivando mediante el 
análisis crítico la mejora.
5.2 Tuplas de tutoría
La cátedra en las primeras semanas de cursada definirá tuplas de tutoría  dadas por  <tutor, 
estudiante1, estudiante2, estudiante3>  donde los estudiantes tendrán un nivel equivalente y la 
tupla presenta altas posibilidades de sinergias tutor-estudiante y estudiante-estudiante incluyendo la 
compatibilidad de agendas. 
Se intentará mantener las tuplas intactas durante toda la asignatura.  Las tuplas  pueden llegar a estar 
formadas por alumnos pertenecientes a distintas comisiones si la cátedra así lo considera.
Los estudiantes no participan en el armado de las tuplas de tutoría.
Estas tuplas de tutoría no necesariamente guardan relación alguna con los grupos de estudio de l@s 
estudiantes ni tampoco con los grupos  que forman para las competencias de la asignatura, los cuales se 
suelen forman por afinidad y los estudiantes poseen completa libertad en su armado. La riqueza radica en 
que los grupos de estudio integren la diversidad de perspectivas adquiridas en las distintas tutorías. 
Los tuplas de tutoría más avanzadas exploran temas muy distintos a las tuplas que están lidiando con 
conceptos  básicos, generando soluciones y enfoques diferenciados.
En caso que alumnos abandonen la materia, se reorganizarán algunas tuplas.
5.3 Agenda de sesiones de tutorías
Cada tutor negociará con los estudiantes  la fecha y hora de la siguiente tutoría especificando claramente 
por escrito en el correspondiente canal dde Zulip los temas que deben ser trabajados previamente por los 
estudiantes para ser debatidos durante la sesión de tutoría.
Las sesiones de tutorías serán (salvo excepciones) en un horario distinto al de las clases sincrónicas. Sin 
embargo la cátedra deja  abierta la posibilidad a que cierto dia puntual coincidente con una clase sincrónica 
sea reemplazada por una maratón de tutorías.
5.4 Trabajo previo
Los estudiantes siempre deben asistir a la sesión de tutoría habiendo realizado el trabajo previo acordado 
con su tutor, generalmente experimentos para mejorar el modelo predictivo.
39
[Página 40]
5.5 Sesión de tutoría
La sesión de tutoría consiste de una videollamada de 55 minutos de duración en la que se produce un 
diálogo entre los cuatro integrantes de la tupla de tutoría fomentando el pensamiento crítico, abordando 
los problemas desde múltiples perspectivas analizando soluciones, identificando posibles errores y 
refinando el razonamiento mediante preguntas desafiantes y retroalimentación constructiva.
El tutor suele hablar el 30% del tiempo generalmente formulando preguntas y los tres estudiantes en forma 
equitativa el 70% restante.
El tutor abre la sesión preguntando a los estudiantes sobre avances y problemas encontrados en la etapa 
del proyecto de modelado en generar y también sobre otros micro proyectos  agendados para esa sesión de
tutoría. 
Los estudiantes explican lo realizado, el racional detrás de las decisiones tomadas a lo que el tutor indaga  
en profundidad, solicita detalles, cuestiona todas las decisiones, detecta razonamientos incorrectos y los 
marca constructivamente a los estudiantes, solicita al estudiante que lleva a la palabra (y a los otros dos)  
soluciones alternativas y su análisis crítico. El tutor indaga constantemente el  “por qué”.
En casos puntuales el tutor clarifica temas conceptuales de las clases sincrónicas sin embargo el tutor jamas
utiliza el tiempo de la tutoría para dictar una “mini clase”  acaparando el 50% del tiempo.
El tutor no hace “spoonfeeding” en la sesión de tutoría, jamás brinda instantáneamente la receta que 
resuelve el problema o mejora el modelo. Ante una decisión a tomar por l@s estudiantes el tutor debe 
enseñar a abordar criticamente el problema y proveer a l@s estudiantes las herramientas para que ellos 
resuelvan.
El estudiante jamás participa de la tutoría sin haber preparado la tarea asignada, ya que la tutoría se basa 
en darle retroalimentación constructiva sobre lo hecho y producir nuevos avances.
El estudiante no espera una respuesta directa a su pregunta, sabe que está ahí para razonar él mismo la 
respuesta. El estudiante dialoga, cuestiona lo que el tutor y los otros estudiantes dicen,  llega con ciertas 
ideas a la sesión y las transforma en superadoras durante la sesión.
La filosofía de las tutorías es  "Not what to think,  but how to think." 
40
[Página 41]
6
Git y GitHub
6.1.1 conceptos y operación
o desde la línea de comando de la PC local
git checkout main
git pull
git fetch upstream
git checkout main
git merge  -Xtheirs  upstream/main   -m "Z manda"
git checkout main
git add .
git commit -m "otro commit"
git push
41
GitHub  Repositorio 
de la  Materia
(nube)
GitHub Repositorio
personal
(nube)
3 add
4 commit
5 push
Fetch 
Upstream
&
merge
PC local
Repositorio 
(su computadora)
PC local
Repositorio 
(su computadora)
modificaciiones
[Página 42]
6.2 Librería  data.table  del lenguaje R
En el lenguaje R tradicional se utiliza el objeto  dataframe que posee severas deficiencias en cuanto a 
performance y rebuscada complicada.  En esta asignatura utilizamos la librería data.table  que es 
ampliamente superadora y que permite manejar grandes volúmenes de datos, tiene una sintaxis más 
simple PERO muy distinta a la de dataframes, por lo que es necesario aprenderla.
Leer los siguientes artículos
•
https://towardsdatascience.com/data-table-rs-best-data-object-c95b7d5f0104
•
https://towardsdatascience.com/blazing-fast-data-wrangling-with-r-data-table-
de5045cc4b4d
•
introduction to data table  https://cran.r-
project.org/web/packages/data.table/vignettes/datatable-intro.html
•
data.table  cheat sheet  https://www.beoptimized.be/pdf/R_Data_Transformation.pdf
42
[Página 43]
43
[Página 44]
7
Niveles de complejidad
A lo largo de los años se han observado escalones cognitivos que para cada uno de los temas l@s 
estudiantes deben escalar en función de sus conocimientos previos, sesgos cognitivos y dedicación a la 
asignatura.
7.1 Etapa Preprocesamiento de Datos
7.1.1 Armado del Target
Nivel Uno,  utiliza Positivos = {“BAJA+2”}    Negativos = {“BAJA+1”, “CONTINUA”}   y para cortar utiliza el 
umbral de probabilidad  PROB(“BAJA+2”) >= 0.025   ( 1/40)
7.1.2 Dimensionalidad del dataset
7.1.3 Tratamiento Valores faltantes
7.1.4 Calidad de Datos
7.1.5 Data Drifting
7.1.6 Feature Engineering intra mes
7.1.7 Feature Engineering histórico
7.1.8 Aceleración Procesamiento
44
[Página 45]
7.2 Etapa Modelado
7.2.1 Clase Target
7.2.2 Estrategia Entrenamiento
7.2.3 Algoritmo
7.2.4 Optimizacion de Hiperparámetros
7.3 Etapa Puesta en Produccion
8
Experimentos Colaborativos
El gran objetivo es encontrar cuál es la mejor configuración de las etapas del workflow que genere  la mejor 
predicción.
Para ello se parte de una buena solución, la configuración workflow-inicial  y basado en ella mediante la 
experimentación se prueban alternativas a la configuración inicial, o incluso, para el caso de alumnos 
avanzados, se agrega nueva funcionalidad a los scripts del workflow.
Aclaración importante : el objetivo de estos experimentos es el capitalismo salvaje, lográr ganar más dinero 
que el workflow-inicial.  Decididamente NO nos son experimentos académicos por el amor al conocimiento 
abstracto.
Cada problema es abordado por dos grupos experimentadores antagónicos, el Grupo A y el Grupo B, que 
utilizando diseños experimentales distintos intentan responder la Hipótesis Experimental. Se puede leer lo 
siguiente para entender esta usual práctica de la reproducción de un experimento en el mundo científico  
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5778115/
Se ha observado que la etapa inicial de elección de un problema despierta en ocasiones ciertas batallas y 
malos entendidos entre los alumnos, para lo cual se han diseñado un conjunto de reglas simples orientadas 
a suavizar las fricciones.
Es el objetivo de la cátedra que los experimentos realizados sean auténticamente útiles para encontrar el 
mejor modelo predictivo. Lograr lo anterior demanda experimentar en el entorno de una buena 
configuración general de parámetros la que es “computacionalmente pesada”.
Por la naturaleza de la ciencia de datos se cumple generalmente  que utilizando algoritmos de Gradient 
Boosting of Decision Trees, entrenar en mayor cantidad de datos ( mas historia y  más columnas) produce 
mejores modelos predictivos.
45
[Página 46]
Si se entrena en datasets pequeños se llega a parámetros óptimos de las etapas del workflow que luego no 
se corresponden con los que se encuentran si el entrenamiento se hace en un dataset grande de varios 
meses.
Las corridas con grandes volúmenes de datos son naturalmente más lentas, más propensas a fallas de una 
máquina virtual, requieren de más control, y por lo general intentan ser evitadas por los alumnos.
Es tarea de la cátedra enseñarle a buscar y encontrar el procedimiento que produce los mejores resultados 
y que usted sea capaz de ponderar el costo beneficio de la experimentación. 
No intente buscar atajos suboptimos en la experimentación !
46
[Página 47]
8.1 Documento Colaborativo  Google Slides
En el documento colaborativo de Google Slides se encuentran las siguientes secciones:
•
Indice
•
Plantilla de Problema, en donde hay un slide para cada uno de los capítulos
1.
Portada
2.
Hipótesis Experimental
3.
Bibliografía
4.
Sesgos Cognitivos
5.
Diseño Experimental
6.
Limitaciones
7.
Resultados
8.
Discusión de los resultados
9.
Conclusiones
10. Recomendación Concreta
11. Futuros problemas y experimentos
12. Anexo
•
Ejemplo completo de un Problema resuelto por un grupo A y la contraparte grupo B
•
14 Potenciales Problemas a resolver, con su portada, hipótesis experimental y en algunos casos algo
de bibliografía.
Los capítulos de la Plantilla del Problema y su estricto orden deben respetarse siempre; por supuesto 
algunos de los capítulos podrán demandar más de un slide. 
Será en este mismo documento colaborativo donde los alumnos escriban sus presentaciones.
El Grupo A tendrá su presentación completa con todos los capítulos y a continuación vendrá la presentación
del Grupo B la que también incluirá todos los capítulos, incluirá su propia Portada e Hipótesis Experimental 
que deberán ser idénticas a las del Grupo A
Lo más razonable es que usted elija uno de los Potenciales Problemas que ya aparecen en el documento, los
que superan la cantidad de grupos posibles de esta cohorte. Si usted quisiera resolver un problema que no 
está planteado en el documento, deberá proponérselo a los profesores y éstos darle su autorización en 
función de los temas que fueron elegidos por el resto de los alumnos, que los problemas más interesantes 
tengan grupos asignados que los resuelvan, que usted tenga el know how suficiente en este momento para 
enfrentar el problema, etc
47
[Página 48]
8.2 Problemas disponibles y alumnos avanzados
En el documento colaborativo de Google Slides la cátedra ha propuesto una serie de problemas cuyos 
experimentos son muy relevantes para la generación de un modelo que auemnte la ganancia y factibles de 
realizar en el tiempo disponible.
Si usted es una alumno avanzado con conocimientos previos de Ciencia de Datos o de Estadística Clásica, se 
lo autoriza a que :
•
Proponga a los profesores un nuevo Problema, junto con su diseño experimental
•
Convenza a uno o dos alumnos más, no necesariamente estos avanzados, para poder conformar los 
grupos antagónicos A y B
8.3 Configuración de los grupos
A continuación las únicas dos configuraciones posibles en cuanto a la cantidad de integrantes de los grupos 
para un problema:
Configuraciones Válidas
# Integrantes por equipos
Configuración
Grupo A
Grupo B
Solitaria
1
1
Asimétrica
2
1
Por favor notar que siempre deben existir los dos grupos antagónicos para un Problema, los que realizan 
diseños experimentales distintos.
No pueden trabajar cuatro personas en un mismo problema, es un desperdicio de materia gris.
8.4 Forma de pre registración
Simplemente anote su nombre en la portada del problema que es de su interés. 
Podrá anotar su nombre en un único problema. Siempre podrá cambiarlo por otro problema en la medida 
que se cumpla el proceso que sigue a continuación.
8.5 Negociación de Problemas : vacancia
Se cumple lo siguiente:
48
[Página 49]
•
Cada problema requiere como mínimo dos personas, una que integre el Equipo A y otra que integre
el Equipo B. 
•
Hay una oferta de Potenciales Problemas mayor a la cantidad de equipos posibles
Las condiciones anteriores producen por lo general la situación de personas que anotan su nombre en uno 
de los  Potenciales Problemas y ese problema no le interesa a nadie más, pasan los días y nadie más anota 
su nombre en esa portada.
Esta persona deberá :
•
convencer a otras que se pasen a su problema, al menos una persona más
•
o abandonar ese problema y unirse  a algún otro problema/grupo que esté en etapa de formación.
8.6 Negociación de Problemas : sobreoferta
Esta es la situación que produce mas fricciones y sensación de injusticia entre los alumnos.
Se sumple lo siguiente:
•
A pesar de haber sobreabundancia de Potenciales Problemas, hay algunos muy populares en el 
imaginario de los alumnos que son pretendidos por más de tres alumnos.
•
Hay algunos alumnos que ingresan al Google Slides varios dias después del lanzamiento de la 
registración y desean unirse a un problema con sobreoferta de alumnos.
Negocie sin intentar sacar ventajas solo por haber llegado antes.
8.7 Asignación Grupo A vs Grupo B
El grupo A posee cierta ventaja ya que es quien primero establece el diseño experimental, y una vez que 
este fue aprobado por los profesores deberá el Grupo B proponer su diseño experimental, 
mandatoriamente distinto, el que también deberá ser aprobado por los profesores, deberán garantizar que 
los diseños experimentales sean lo suficientemente disjuntos.
Dicha mínima ventaja a favor del Grupo A tiene el potencial de generar algun tipo de conflicto. Se sugiere 
en este caso tirar una moneda al aire.
8.8 Antagonismo de los grupos
Para un problema dado, el Grupo A y el B son antagónicos. Ellos intentan llegar a resultados distintos.
Solo se comunican al principio, cuando el Grupo A establece su diseño experimental y el B está obligado a 
hacer uno completamente distinto. A partir de allí, nunca más interactuan, evitan por todos los medios de 
contaminar sus mentes con potenciales ideas equivocadas del otro grupo.
49
[Página 50]
8.9 Perídos de análisis de los grupos
Los grupos A y B analizarán distintos períodos con el objetivo de determinar si lo que mejor funciona lo 
hace lamentablemente para un solo período o si por el contrario es una propiedad que generaliza.
 
50
[Página 51]
8.10 Aprobación de Diseños Experimentales
El procedimiento a seguir es el siguiente
1.
Al Grupo A deberá cuanto antes escribir su diseño experimental en el Google Slides colaborativo, y 
solicitar rápidamente su revisión a los profesores, por medio de Zulip. En esta etapa podrá haber 
una brevísima conference con el profesor.
2.
Luego de un ida y vuelta, a la mayor brevedad el profesor, por Zulip, aprueba el diseño experimental
del Grupo A. En ese momento el Grupo A ya puede comenzar la experimentación.
3.
El Grupo B, viendo ese diseño experimental aprobado, construye un diseño experimental 
alternativo, y solicita la aprobación del profesor.
4.
Luego de un ida y vuelta, el profesor, por Zulip, aprueba el diseño experimental del Grupo B, y éste 
puede comenzar la experimentacion.
8.11 Ceteris Paribus
Cada Potencial Problema hace referencia unicamente a una parte de un script.
Los grupos que intenten resolver una Hipótesis Experimental deberán hacer modificaciones a los 
parámetros del correspondiente script, o en caso de tener el conocimiento suficiente al código del script, 
pero deberán mantener sin modificación el resto de los scripts que no son objeto de su experimentación, 
excepto : 
•
las semillas generales de todos los scripts que siempre deben ser cambiadas a las propias.
•
Las pruebas que se realicen en otros períodos “future”, donde se elije un mes para el que se conoce
la clase y se pueden dibujar las curvas
Concretamente, en los scripts de workflow-inicial se entrena en 18 meses de historia ya que esto produce 
buenas ganancias. Excepto que su Problema sea el de probar la cantidad óptima de meses donde entrenar, 
usted deberá correr el worflow respetando esos 18 meses y no podrá tomar ningun atajo de utilizar menos 
meses para que sus procesos corran m+as rápido.
Una vez que haga las corridas esperadas con esos 18 meses, podrá agregar a los experimentos, como algo 
adicional, alguna corrida en menos meses.
8.12 Capítulo Conclusiones
El capítulo de las conclusiones, en alguna parte se debe dar una receta bien concreta :  ¿Cómo se deben 
setear los parámetros de esa etapa del Workflow? Incluso para el caso que los experimentos no sean 
concluyentes se debe dar una clara receta.
Por supuesto, quienes lean el experimento simplemente lo tomarán como un punto de partida a sus propios
“últimos experimentos”.
8.13 Video Presentaciones Ejecutivas
Los links a las video presentaciones de los distintos problemas deberán disponiblizarse antes de la fecha 
51
[Página 52]
indicada en el Cronograma de la Asignatura, y se harán directamente a partir del documento colaborativo 
de Google Slides.
52
[Página 53]
9
¿Cómo seguir?
9.1 Mantenerse actualizado
¿En que sitios es posible mantenerse actualizado luego de finalizada la asignatura ?
•
https://machinelearningmastery.com/
•
https://www.kdnuggets.com/
•
https://www.analyticsvidhya.com/
•
https://neptune.ai/blog
•
https://www.fast.ai/#category=technical
•
https://openai.com/blog
•
https://www.deepmind.com/blog
•
https://blog.ml.cmu.edu/
•
https://news.mit.edu/topic/machine-learning
9.2 Datasets Públicos
¿Dónde encontrar datasets públicos para una Tesis de Maestría o practicar ?
•
https://cloud.google.com/datasets 
•
https://github.com/awesomedata/awesome-public-datasets 
•
https://www.kaggle.com/datasets 
•
https://www.datos.gob.ar/dataset 
•
https://data.buenosaires.gob.ar/dataset/ 
•
https://data.gov/ 
•
https://careerfoundry.com/en/blog/data-analytics/where-to-find-free-datasets/ 
•
https://medium.com/analytics-vidhya/top-100-open-source-datasets-for-data-science-
cd5a8d67cc3d 
•
https://www.v7labs.com/open-datasets 
•
https://azure.microsoft.com/en-us/products/open-datasets/ 
•
https://registry.opendata.aws/ 
53
[Página 54]
10 Empiricismo en la Ciencia de Datos
En el Machine Learning muchos algoritmos son encontrados por prueba y error, al inicio se los prueba en 
decenas de datasets, se divulgan, los usa la comunidad, ganan competencias Kaggle, pasan a ser usados por
practicantes en innumerables situaciones, pero carecen de demostraciones matemáticas sobre su robustez- 
Recién luego de varios años, o más de una década se demuestra la razón por la que funcionan, los efectos 
que se observan y las razones de su poder (por ejemplo que converjan o que el overfitting esté controlado).
Esta situación colisiona violentamente con quienes provienen de la estadística clásica.
10.1.1 Bibliografía Introductoria
•
https://www.kdnuggets.com/2015/07/deep-learning-triumph-empiricism-over-theoretical-
mathematical-guarantees.html
•
https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.2016.0153
•
https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124
•
https://ris.utwente.nl/ws/files/177935074/Boon_2020_BioTechnoPractic_How_scientists_are_brou
ght_back_into_science_preprint_May_9_2019.pdf
•
https://hdsr.mitpress.mit.edu/pub/l39rpgyc/release/1
•
https://www.youtube.com/watch?v=oONHlua2gBY
•
https://royalsocietypublishing.org/doi/10.1098/rsta.2018.0145
Arquetípico ejemplo de empiricismo, acontecido en la medicina
•
https://www.redalyc.org/journal/920/92044744013/html/  leer la sección  Anexo Semmelweis 
y la fiebre puerperal  (está después de la bibliografía)  
54
[Página 55]
11 Las dos culturas en el modelado predictivo
•
El paper original de Leo Breiman  Statistical Modeling, The Two Cultures  
http://www2.math.uu.se/~thulin/mm/breiman.pdf
  https://towardsdatascience.com/thoughts-on-the-two-cultures-of-statistical-modeling-
72d75a9e06c2
55
[Página 56]
12 Estimando la Ganancia de un modelo predictivo
12.1 Bibliografía Introductoria
Bibliografía
•
https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-
algorithms/
•
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.183.534&rep=rep1&type=pdf
12.2 Sesgo Varianza en la estadística clásica
 
12.2.1 Bibliografía Inicial
•
https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-
machine-learning/ 
•
http://scott.fortmann-roe.com/docs/BiasVariance.html  (15 minutos)
56
[Página 57]
12.3 Sesgo Varianza en el moderno Machine Learning
 
12.3.1 Bibliografía Inicial
•
https://medium.com/mlearning-ai/double-descent-8f92dfdc442f
•
https://arxiv.org/pdf/1812.11118.pdf
57
[Página 58]
13 Análisis exploratorio de datos
13.1 Consistencia longitudinal de los atributos
13.2 Drifting de atributos
En el período (mes) donde debo aplicar el modelo para hacer la predicción final, ¿existen variables con una 
distribución distinta a los períodos anteriores?
•
https://towardsdatascience.com/data-drift-part-1-types-of-data-drift-16b3eb175006
•
https://towardsdatascience.com/data-drift-part-2-how-to-detect-data-drift-1f8bfa1a893e
•
https://towardsdatascience.com/machine-learning-in-production-why-you-should-care-about-data-
and-concept-drift-d96d0bc907fb
•
https://medium.com/data-from-the-trenches/a-primer-on-data-drift-18789ef252a6
•
https://arxiv.org/pdf/2004.05785.pdf
•
https://wires.onlinelibrary.wiley.com/doi/epdf/10.1002/widm.1381
13.3 Tratamiento de variables "rotas"
¿Qué funciona mejor con una variable "rota" para un mes? ¿ Imputar su valor interpolando, o asignarle 
nulo? sí, ha leído bien, estamos proponiendo asignar NA a una variable !
13.4 Tratamiento de nulos
¿Debo tratar a los nulos de alguna forma en especial?
¿Es razonable eliminar un registro donde algún campo importante sea nulo?
¿Qué funciona mejor, imputar nulos con alguna metodología o permitir que XGBoost/LightGBM trabajen 
con ellos?
13.5 Tratamiento de outliers
¿Debo tratar a los outliers de alguna forma en especial?
¿Qué funciona mejor, tratar los outliers con alguna metodología o permitir que XGBoost/LightGBM trabajen
con ellos?
¿Qué pasa con los outliers en el bining que realiza LightGBM?
58
[Página 59]
14 Algoritmo: Árbol de Decisión
14.1 Bibliografía Inicial
Capítulo 9.2 de  https://hastie.su.domains/Papers/ESLII.pdf
Optimización de hiperparámetros  https://bradleyboehmke.github.io/HOML/DT.html
Explicacion sobre rpart  https://www.learnbymarketing.com/tutorials/rpart-decision-trees-in-r/
14.2 El hiperparámetro cp en la librería rpart
En esta materia "un experimento no se le niega a nadie" , con lo cual independientemente de lo que digan 
papers, libros, otras materias, profesores con gran cantidad de tiras en su uniforme, towardsdatascience e 
incluso el infame Medium, haremos el experimento, y que una optimización de hiperparámetros diga si el 
hiperparámetro cp óptimo es positivo o negativo.
Luego, ante la evidencia experimental, si un cp negativo es el óptimo, buscaremos una explicación, aquellos 
que tengan ganas y tiempo harán nuevos experimentos.
Seamos fieles a "The problem is that people are educated just enough to believe what they have been 
taught, and not educated enough to question anything from what they have been taught" y maravillarnos 
ante nuestra propia ignorancia y tendencia a ponernos cómodos techos.
"I am tormented with an everlasting itch for things remote. I love to sail forbidden seas. " Moby Dick
59
[Página 60]
15 Comparación de Modelos Predictivos
15.1 Bibliografía Inicial
Cross Validation https://machinelearningmastery.com/k-fold-cross-validation/
Receta  https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-
algorithms/
Paper https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.183.534&rep=rep1&type=pdf
60
[Página 61]
16 Optimización de Hiperparámetros
16.1 Bibliografía Introductoria
16.2 Grid Search
16.3 Bayesian Optimization
•
https://towardsdatascience.com/bayesian-optimization-concept-explained-in-layman-
terms-1d2bcdeaf12f
•
https://towardsdatascience.com/the-beauty-of-bayesian-optimization-explained-in-simple-
terms-81f3ee13b10f
•
Taking the Human out of the Loop
 
    (para quienes quieran leer en profundidad en algún 
momento tranquilo de la vida)
61
[Página 62]
17 MLOPS
17.1 Bibliografía Inicial
•
https://towardsdatascience.com/a-gentle-introduction-to-mlops-7d64a3e890ff
62
[Página 63]
18 Curva ROC
18.1 Bibliografía
•
Ling, C. X., Huang, J., & Zhang, H. (2003, August). AUC: a statistically consistent and more discriminating measure 
than accuracy. In Ijcai (Vol. 3, pp. 519-524). https://www.ijcai.org/Proceedings/03/Papers/077.pdf
18.2 Curva ROC de un split en un árbol de decisión
18.3 Curva ROC de un modelo predictivo
18.4 Relación de la Curva ROC con la función de ganancia
63
[Página 64]
19 Controlando el False Discovery Rate
19.1 Bibliografía Introductoria
•
https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124
•
https://www.youtube.com/watch?v=6ZxIzVjV1DE (subtitulos en ingles activables)
64
[Página 65]
20 Feature Engineering
20.1 Bibliografía Introductoria
•
https://www.youtube.com/watch?v=X4pWmkxEikM
•
https://www.youtube.com/watch?v=06-AZXmwHjo   video Andrew Ng
20.2 Cambiando la clase
La clase a predecir es  POS = { BAJA+2 }  , NEG = { BAJA+1, CONTINUA }
La forma natural de predecir es con POS = { BAJA+2 }  ,  sin embargo  ¿Es esa la mejor forma, o me 
convendría entrenar con otra clase?
20.3 Variables Manuales
Artículos donde resolvieron un problema similar
•
https://www.researchgate.net/profile/Xingsen-
Li/publication/224759968_The_Analysis_on_the_Customers_Churn_of_Charge_Email_Based_on_D
ata_Mining_Take_One_Internet_Company_for_Example/links/5891e87a92851cda256a0358/The-
Analysis-on-the-Customers-Churn-of-Charge-Email-Based-on-Data-Mining-Take-One-Internet-
Company-for-Example.pdf
Tesis de maestría donde se solucionó un problema parecido.
•
https://core.ac.uk/download/pdf/83461632.pdf
•
http://research.sabanciuniv.edu/39116/1/AneelaTanveer_10236886.pdf
•
https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1218&context=scschcomdis
20.4 Transformaciones tradicionales
¿Qué efectos tienen en un árbol de decisión y algoritmos derivados, transformaciones del tipo: ?
•
transformación lineal    x' = ax + b
•
x' = log( x )
•
normalización
65
[Página 66]
20.5 Variables históricas
20.6 Lags y delta lags
20.7 Tendencias
20.8 Medias móviles
20.9 Ajuste por Inflación
20.10 Variables hojas de un Random Forest
20.11 Reducción de la dimensionalidad para acelerar
66
[Página 67]
21 Ensembles de Modelos
21.1 Bibliografía Introductoria
•
BBC - The Code - The Wisdom of the Crowd https://www.youtube.com/watch?v=iOucwX7Z1HU 
(5 minutos)
•
https://www.all-about-psychology.com/the-wisdom-of-crowds.html (10 minutos)
•
https://machinelearningmastery.com/what-is-ensemble-learning/  (10 minutos)
67
[Página 68]
22 Algoritmo: Random Forest
22.1 Bibliografía Introductoria
•
https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
•
68
[Página 69]
23 Algoritmo: Gradient Boosting of Decision Trees
23.1 Bibliografía Introductoria
23.2 XGBoost
23.2.1 Bibliografía XGBoost
Inicial
•
https://towardsdatascience.com/xgboost-regression-explain-it-to-me-like-im-10-
2cf324b0bbdb
•
https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-
she-may-rein-edd9f99be63d
•
https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
Avanzada
•
paper original de XGBoost  https://dl.acm.org/doi/pdf/10.1145/2939672.2939785
•
23.3 Optimización hiperparámetros XGBoost
•
https://www.hackerearth.com/practice/machine-learning/machine-learning-
algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
•
Narrowing the Search: Which Hyperparameters Really Matter?
23.4 Bibliografía LightGBM
Avanzada
•
Paper original  de LightGBM  https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-
Paper.pdf 
•
23.5 Optimización hiperparámetros LightGBM
•
https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
•
http://devdoc.net/bigdata/LightGBM-doc-2.2.2/Parameters-Tuning.html
•
https://neptune.ai/blog/lightgbm-parameters-guide
69
[Página 70]
24 Estrategia de Entrenamiento
24.1 Determinación de los mejores períodos para entrenar
24.2 Acelerando el entrenamiento : evitando el cross validation
24.3 Acelerando el entrenamiento : Undersampling de la clase mayoritaria
70
[Página 71]
25 Importancia de Variables
25.1 Bibliografía Introductoria
25.2 Valores Shapley
•
ttps://www.youtube.com/watch?v=ngOBhhINWb8
•
https://www.youtube.com/watch?v=B-c8tIgchu0
71
[Página 72]
26 Ensembles de Modelos Segunda Parte
Bibliografía de Model Stacking
•
https://www.youtube.com/watch?v=DCrcoh7cMHU
•
https://www.youtube.com/watch?v=8T2emza6g80
•
https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html
•
https://medium.com/ml-research-lab/stacking-ensemble-meta-algorithms-for-improve-
predictions-f4b4cf3b9237
•
https://citeseerx.ist.psu.edu/viewdoc/download?
doi=10.1.1.56.1533&rep=rep1&type=pdf el paper original del año 1992
26.1 Semilleríos
26.2 Hibridación de Semilleríos
72
[Página 73]
27 Acelerando el procesamiento
27.1 Bibliografía Inicial
•
https://medium.com/hmif-itb/overfitting-the-leaderboard-da25172ac62e
•
https://www.kaggle.com/general/54610
•
http://gregpark.io/blog/Kaggle-Psychopathy-Postmortem/
27.2
El problema de la calibración en Gradient Boosting
27.3
Otra forma de cortar
73
[Página 74]
28 Clustering jerárquico basado en la distancia de Random Forest
Un modelo de Random Forest es un ensemble de árboles de decisión. Ese ensemble puede servirnos para 
definir una distancia entre registros del dataset. Esta distancia que decididamente no es la euclídea, es 
insensible a los outliers,  a los nulos y a las diferencias de escala de las variables; no hace falta normalizar los
atributos del dataset.
Utilizando esa distancia podemos realizar una partición por el método de clusters jerárquicos.
74
[Página 75]
29 Storytelling
29.1 Historia
Al finalizar su primer año del posgrado en ciencia de datos usted ingresa como data scientist en la compañía
al sector de Business Intelligence.
El sector está liderado por la gerenta  Michelina Porto y está organizado de la siguiente forma:
•
Advanced Analytics, son 4 data scientists ( una economista, un actuario, una informática los tres con
una maestría en ciencia de datos más una  PhD en física) los que poseen entre 3 y 12 años de 
experiencia en ciencia de datos. Usted acaba de ingresar como la quinta persona, la más junior de 
ese equipo aún sin finalizar la maestría. Todo este sector reporta directamente a Michelina Porto, ya
que advanced analytics es donde ella más interactúa.
•
Data Engineering  7 personas, todos con sólida formación en sistemas (hay especialistas en Big Data,
en cloud computing, en arquitectura de data lakes, en infraestructura y en MLOPs que se encargan 
de poner en producción los modelos desarrollados por los data scientists)
•
Data Verticals  10 personas provenientes de las más diversas carreras, parte de ellos son data 
analysts
•
CRM & Campaign Managing    3 personas
•
Reporting,  6 personas que mantienen los reportes y además hacen análisis simples ad-hoc .
•
Data Governance  1 vigilante
El sector ha existido en la organización por más de 20 años, evolucionando con diversas responsabilidades, 
nombres y estructuras. Decididamente usted no ha llegado como la salvación del sector que viene a 
reinventar la forma de trabajo y a reescribir todo de cero. El sector ya tiene una metodología de trabajo, 
decenas de repositorios de datos, cientos de scripts, ecosistema en el cual usted debe aprender a 
manejarse y agregar valor.
Todo el sector considera muy importante la reutilización de código. Todas las mejoras, previamente 
documentadas son bienvenidas.  Reescribir de cero un script que ya está en el repositorio y es eficiente es 
visto como muy antieconómico en el sector.
La reproducibilidad de los experimentos es considerada de capital importancia en el sector. Es sagrado que 
quede la historia de los experimentos sub óptimos y se enseña a todos los integrantes a documentar y a 
aprender de lo que no funciona tan bien.
Luego de cuatro meses aprendiendo del ecosistema, le encargaron el proyecto del modelo predictivo de 
retención proactiva de clientes, lo ha terminado y ahora debe comunicar dos aspectos muy distintos de lo 
analizado a dos interlocutores.
Se deberán confeccionar dos videos, uno dirigido la gerenta de business intelligence y otro a la directora 
comercial. La duración de los videos debe estar en el entorno de los 5 minutos.
Michelina Porto, 38 años,  es la gerenta de business intelligence de la compañía, es su jefa directa y fue 
quien hizo su contratación hace 4 meses.
Michelina posee un título de grado de Actuaria, desde su graduación se dedicó a análisis econométricos en  
75
[Página 76]
riesgo crediticio,  en el año 2011 cursó una maestría en ciencia de datos e ingresó a trabajar a la compañía  
en el año 2017,  ya en 2020 la ascendieron a gerenta en medio de una reorganización general de toda el 
área comercial.  Michelina es una persona muy metódica y organizada, pausada en su hablar, elige sus 
palabras con gran precisión, reflexiva, considera muchas opciones antes de tomar una decisión, ante una 
situación difícil a resolver escribe un cuadro de excel con las alternativas a las que  les estima una 
probabilidad. Michelina no transparenta fácilmente sus emociones. En las reuniones va escribiendo la 
minuta en tiempo real y la disponibiliza a todos ni bien termina.  Es una persona muy focalizada y ninguna 
idea foránea la distrae del problema que debe resolver. Prefiere aprender en forma estructurada y abordar 
los temas desde lo abstracto.
La frase de cabecera de Michelina es : “no me cuentes que encontraste un patrón en los datos si no me 
traés como lo podemos utilizar para aumentar la rentabilidad”.
Michelina posee un elevado sentido de la ética y la justicia.
Al momento de comenzar el proyecto usted ya le hizo una primera presentación a Michelina, en donde le 
planteó el problema a resolver, la campaña de retención proactiva, la clase { BAJA+1, BAJA+2, CONTINUA } , 
que a los BAJA+1 se llega tarde y no se los puede retener, la fórmula de ganancia, la cantidad de datos 
disponible,  el momento del mes en que los datos están disponibles, esto NO lo podrá volver a repetir en la 
segunda presentación.
Ha pasado el tiempo, usted ha concluido el modelo de attrition y Michelina espera de usted un segundo 
video con una breve presentación de alrededor de 5 minutos en donde con un storytelling le cuente la 
forma en que resolvió el problema, qué funcionó en forma sub óptima, que funciono mejor, los hallazgos 
más importantes y cómo eso se va a transformar en rentabilidad.  Esta no es una tesis de maestría, usted no
debe explicar el algoritmo árbol de decisión ni gradient boosting, usted debe ir al grano con Michelina, pero
sorprenderla, y debe narrar lo que no funcionó.
Michelina reporta a la directora comercial y es la "gerenta de menos peso". Sus pares son las siguientes 
gerencias :  sucursales (1000 comerciales), productos pasivos, productos activos, canales digitales,  
marketing, telemarketing y atención al cliente.
En las reuniones semanales que Michelina mantiene con su jefa contándole el avance de los múltiples 
proyectos de sector, ella le pide información adicional sobre el proyecto de retención proactiva de clientes y
Michelina le solicita a usted hacer esa presentación en un video para que empiece a figurar en el radar de la
compañía.
Miranda Wintour, argentina, 48 años, dos hijas gemelas pre adolescentes,  es la directora comercial de la 
compañía desde hace dos años y medio, y en su meteórica carrera se pronostica que llegará a la gerencia 
general en dos años más.
Ambos padres de Miranda son nacidos fuera de Argentina dedicados en su momento a la actividad consular.
Miranda de joven emigró, concluyó  sus estudios secundarios  en el UWC Atlantic College en Gales, se 
graduó con honores en Ciencias Políticas en la Sorbonne Université de París y cursó una maestría en 
economía en la London School of Economics and Political Science.
Miranda practica yachting desde su infancia, actividad fomentada por su padre quien le inculcó el trabajo en
equipo y la competitividad.  En su juventud participó de varias competencias internacionales, siendo un 
punto de inflexión en su vida la accidentada carrera de 1998  54th Sydney to Hobart  Yacht Race.  En su 
oficina posee un cuadro de muy importantes dimensiones con una fotografía de esa carrera en donde se 
aprecia a una joven Miranda formando parte de un numeroso equipo sobre una embarcación, al pie del 
cuadro reza una enorme leyenda “Las regatas se ganan en tierra”.
Miranda se unió desde muy joven a la compañía en Europa, estuvo a cargo del área Fintech europea y la 
convencieron de hacerse cargo en Argentina de una transformación radical.  Su primera tarea fue 
76
[Página 77]
desarticular el anquilosado club de amigos que había formado su antecesor.
Aunque va con una sonrisa y su tono de voz es muy bajo y sereno, todos tienen una especie de temor hacia 
ella. Se dice cuando luego de una exposición Miranda le dice al disertante "buen trabajo" sonriendo, antes 
de los tres meses esa persona ya no está más en la compañía.
Usted jamás ha participado en una reunión con ella y esta será su gran oportunidad de darse a conocer, 
Miranda busca ideas revolucionarias que le permitan aumentar la posición de liderazgo que ya ostenta la 
compañía.
Un total de 2500 personas dependen indirectamente de Miranda, estando el grueso en la red de sucursales.
telemarketing y atención al cliente.
Usted debe realizar un video presentación a Miranda que no puede exceder 5 minutos en donde le explique
los perfiles de los clientes que se dan de baja de paquete premium y proponga acciones para revertirlo. 
Miranda no sabe (ni le interesa saber) sobre Ciencia de Datos pero está absolutamente convencida de las 
posibilidades que brinda la tecnología para tratar uno a uno a los clientes.
Debe quedarle muy claro que el video a Miranda no es contarle lo mismo que Michelina pero en palabras 
de negocios, Miranda le está pidiendo los perfiles de los clientes que dejan de tener Paquete Premium e 
ideas de como evitarlo, ideas basadas en evidencia que debe salir de los datos. 
Debe quedarle claro que la tarea de un modelo predictivo es, situados dos meses antes,  diferenciar a los 
que van a vivir de los que van a morir; Miranda le está pidiendo algo distinto : "en que se diferencian entre 
sí los que ya han muerto"
29.2 Bibliografía Introductoria Storytelling
•
https://www.youtube.com/watch?v=3KQaWHesWwc
•
https://jtappler.medium.com/effective-data-storytelling-and-visualization-81ad24229745  (5 minutos)
•
https://www.youtube.com/watch?v=9UNlHFfy4hs  (4 minutos)
•
https://www.themuse.com/advice/public-speaking-tips  (10 minutos)
•
https://www.researchgate.net/publication/262393281_How_video_production_affects_student_engagement
_An_empirical_study_of_MOOC_videos  (15 minutos)
https://www.youtube.com/watch?v=ba-CB6wVuvQ  (15 minutos)
29.3 Herramientas para grabar videos
•
Herramientas online,  all included
◦
https://prezi.com/
◦
https://www.loom.com/
•
Herramientas locales
◦
grabación sencilla ( ver siguiente capítulo )
◦
https://obsproject.com/
◦
77
[Página 78]
29.4 Grabación sencilla de videos con Microsoft Teams
Esta forma de grabar videos en Windows fue propuesta por un alumno y se hizo popular entre sus 
compañeros
1.
descargar microsoft teams 1. https://www.microsoft.com/es-ww/microsoft-teams/download-app 
2.
abrir teams, e ir a  “Meet Now” para crear una sala vacía con tu cámara.
3.
abrir la meet estará usted solo y nadie más
4.
dar click a “Share” y elegir tu tipo de pantalla: Standout, side-by-side o Reporter, y elegir Window 
para que muestre una pantalla y al lado su cámara
5.
elegir tamaño de cámara (su imagen) y posición left-right de la cámara, recordar que debe ser al 
menos el 10% de la superficie  ( se puede elegir otro tipo de composición, que no recorte el fondo, 
dando un resultado más formal)
6.
Presionar las teclas Windows + G, para abrir consola de grabación de Windows 
7.
Elegir bien la fuente (su pantalla compartida del Microsoft Teams) y darle a REC (el botón redondito 
que debería ser rojo pero ahí es blanco)
8.
No olvidarse de activar el audio para que grabe con audio 
9.
también se puede grabar directamente usando el shortcut ALT+WIN+R y arranca a grabar 
directamente 
10. Arrancar el speech pitch del video con la ppt 
11. A finalizar, detener el video 
12. Volver a abrir la consola de grabación con windows + G para encontrar el video guardado: poner 
open file location: 
13. ingresar a youtube y subir el video 
14. detalle: revisar que cuando se inicia la grabación con Windows G, el recording sea SOBRE el 
meeting, así  graba el video que está sucediendo dentro de la ventana del meeting. 
29.5 Torneo de Videos
Los videos se calificarán mediante la metodología  Swiss-system tournament con el método de apareo 
Dutch system ( ver https://en.wikipedia.org/wiki/Swiss-system_tournament )  este método implica la 
existencia de rondas y que en cada ronda hay duelos de videos donde se intenta que compitan videos de 
similar calidad.  Solo se pasa a la ronda siguiente una vez que todos los duelos hayan sido calificados.
Habrá un torneo para el video a Michelina Porto y otro torneo para el video a Miranda Wintour.  Habrá una 
ronda diaria de cada uno de los dos torneos.
Los jueces de cada duelo serán los mismos alumnos y también los profesores. Se espera que cada alumno 
sea juez en al menos 6 duelos de cada torneo. Esto significa que cada alumno deberá ver  
2*log2( cantidad_alumnos )  videos dirigidos a Michelina Porto y la misma cantidad de Miranda 
Wintour.  Nadie será juez de sus propios videos ni de compañeros.  La votación en los duelos de videos es 
secreta.
La nota que se asigna por ser juez en el torneo de videos es simbólica ya que apenas representa el 2% de la 
nota final de la asignatura y está regida por el cumplimiendo en su rol como jurado. Se pretende que TODOS
los alumnos participen en todas las rondas como jueces.
La frase del informe   “En cuanto a limpieza de datos si bien hicimos tratamientos de nulos con rpart, no lo 
78
[Página 79]
convalidamos con la ganancia por lo que lo desestimamos. En una primera instancia nos dio resultados 
positivos pero repitiendo las pruebas esto no sucedió.“   no me queda clara ¿Podría explicar en gran 
detalle ?
30 Artículos de negocios
◦https://storage.googleapis.com/dmeyf/annotation/general/Target.html
◦https://storage.googleapis.com/dmeyf/annotation/general/why_business_fail_at_mac
hine_learning.html
◦https://storage.googleapis.com/dmeyf/annotation/general/why_so_many_data_scienti
sts_are_leaving_their_jobs.html
◦https://storage.googleapis.com/dmeyf/annotation/general/MostMisunderstoodHero.ht
ml
31 El Recuperatorio
El recuperatorio está destinado a personas que habiendo asistido al 75% de las clases, o no cumplieron con 
la entrega de todas las actividades obligatorias o 
79
[Página 80]
32 Hoja de Ruta clases jueves en construcción
Hoja de ruta de la Comisión 2, clases dictadas los jueves, por Gustavo Denicolay 
Las clases de los lunes y jueves no están sincronizadas, ni siquiera se dictan los mismos temas, y por 
supuesto cada profesor tiene su formato y estilo propios.
Esta hoja de ruta es tentativa, se irá actualizando en función del conocimiento previo que tengan los 
alumnos lo que se indagará en la primera clase, el avance real que se logre en cada clase, la resistencia que 
se encuentre en los alumnos a abandonar conocimientos previos erróneos.
32.1 Introducción y construcción Primera Competencia
Clase 01,  jueves 13-agosto, presencial
•
presentación de la materia, la metodología de enseñanza, herramientas a utilizar, lenguajes y 
ambientes.  
•
Indagación sobre lo aprendido en materias del primer cuatrimestre, especialmente por la ausencia 
de Aprendizaje Automático este 2026
•
Presentación del problema de negocios a resolver. Presentación de la Primera Competencia.
•
Primera actividad de active learning sobre la fórmula de ganancia 
Clase 02,  jueves 20-agosto, presencial
•
hiperparámetros del  árbol librería  rpart
•
training/testing/ datos del futuro
•
efecto de la semilla en training/testing,  actividad de active learning
•
Montecarlo Estimation, Cross Validation, Test de Wilcoxon
•
Tarea para el hogar : Grid Search, experimentos encargados a los alumnos : corrida de 12+ horas de 
duración
Clase 03,  jueves  27-agosto, presencial
•
análisis de los resultados de las corrida de Grid Search, actividad de active learning, el infame 
hiperparámetro  cp
•
problemas de overfitting
•
posibles optimizaciones al Grid Search,  actividad de active learning
•
Bayesian Optimization
•
Tarea para el Hogar:  corridas de Bayesian Optimization de árboles de decisión
80
[Página 81]
Clase 04,  jueves  03-septiembre, presencial
•
Análisis corridas Bayesian Optimization de árboles de decisión
•
Métodos de Ensemble
◦
Arboles Azarosos
◦
Random Forest
◦
Gradient Boosting of Decision Trees
•
Tarea para el Hogar: Bayesian Optimization de LightGBM
Clase 05,  jueves 10-septiembre, presencial
◦
Análisis corrida Bayesian Optimization de LightGBM
◦
Profundizacion en hiperparámetros de LightGBM
Clase 06,  jueves  17-septiembre, presencial
•
Storytelling (Silvana Contreras)
•
Distancia Random Forest, Clustering Jerárquico
Clase 07,  jueves  24-septiembre, presencial
•
Experimento:  Clase Ternaria vs Clase Binaria.  Curva de Ganancia
•
Experimento: Feature Engineering Histórico, lags y delta-lags
domingo 27 septiembre, entrega video Miranda Wintour
Clase 08,  jueves  01-octubre, presencial
•
Experimento: Feature Engineering : creando variables para ser utilizadas por LightGBM a partir de 
Random Forest
•
Data Drifting
domingo 04 octubre, cierre Primera Competencia
lunes 05 ocutbre, lanzamiento Última Competencia
Clase 09,  jueves  08-octubre, presencial
•
checkpoint de la materia, escuchar a los alumnos proponer cambios y mejoras
•
exposición ganadores de la Primera Competencia
81
[Página 82]
•
análisis de los resultados de la Primera Competencia
32.2 Construcción  Ultima Competencia
Clase 10,  jueves  15-octubre, presencial
•
Exposición de experimentos colaborativos,  dinámica active learning
•
Feature Engineering de variables históricas
◦
Medias Móviles
◦
Tendencias
◦
¿Ajuste por inflación?
◦
Rankeo de Variables
◦
Efecto de las variables históricas en Random Forest
Clase 11,  jueves  22-octubre, presencial
•
Exposición de experimentos colaborativos,  dinámica active learning
•
¿En qué meses conviene entrenar?
•
Undersampling de la clase mayoritaria para acelerar el procesamiento. ¿El undersampling mejora el 
poder predictivo del modelo?
•
técnica de Walk Forward Validation
•
experimentos encargados a alumnos : profundización en la Training Strategy
Clase 11,  jueves  29-octubre, presencial
•
Exposición de experimentos colaborativos,  dinámica active learning
•
¿En qué meses conviene entrenar?
•
Undersampling de la clase mayoritaria para acelerar el procesamiento. ¿El undersampling mejora el 
poder predictivo del modelo?
•
técnica de Walk Forward Validation
•
experimentos encargados a alumnos : profundización en la Training Strategy
Clase 13,  jueves  05-noviembre, presencial, clase plenaria, ambas comisiones
•
Tema Sorpresa
Clase 14,  jueves   12-noviembre, presencial
•
Ensemble stacking
82
[Página 83]
•
Valores Shapley
•
Exposición de experimentos colaborativos,  dinámica active learning
Clase 15,  jueves  18 noviembre, presencial
•
Exposición de experimentos colaborativos,  dinámica active learning
Clase 16,  jueves  26-noviembre, presencial
•
Exposición de experimentos colaborativos,  dinámica active learning
Clase 17,  jueves  03-diciembre, presencial
•
Exposición de experimentos colaborativos,  dinámica active learning
Clase 18, jueves  10-diciembre, presencial
•
examen individual escrito de puño y letra escrito / examen oral individual presencial
domingo 13-diciembre  entrega de video Michelina Porto
lunes   14-diciembre Inicio Torneo de Videos
sábado 19-diciembre Fin Torneo de Videos
domingo 20-diciembre   publicacion notas en cartelera Zulip
83