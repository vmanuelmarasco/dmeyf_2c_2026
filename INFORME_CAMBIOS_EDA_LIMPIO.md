# Informe de cambios — `EDA_limpio.ipynb`

## Alcance

Se reconstruyó el análisis en `EDA_limpio.ipynb` sin sobrescribir `EDA.ipynb` ni modificar el dataset crudo. El notebook nuevo contiene 61 celdas: 31 Markdown y 30 de código.

El archivo original conserva el SHA-256 `359b9f130118a486f0e05808073165c02e417d9e66b3642491995b5e011b5c6f` y la fecha de modificación 2026-09-25 21:26:42.

## Cambios metodológicos principales

1. Se recuperó la consigna oficial y se separaron descripción, segmentación, predicción e inferencia causal.
2. Se reconstruyeron las poblaciones desde el panel completo:
   - 3.977 bajas estrictas;
   - 1.103 censurados a derecha;
   - 23 intermitentes.
3. Se construyó el feature engineering sobre el panel antes de seleccionar `t0`; todas las ventanas se particionan por cliente y utilizan solamente presente y pasado.
4. Se exigieron tres meses calendario consecutivos. Resultaron 1.991 bajas con historia completa y 1.986 con historia parcial.
5. Se reemplazó la representación dependiente del último split por proximidad real de Random Forest: proporción de árboles en los que dos clientes caen en la misma hoja.
6. Se fijó un orden determinístico de filas, `float32`, cinco semillas declaradas, una única matriz promedio, un único linkage de enlace promedio y una única asignación final.
7. Se balancearon bajas y continuadores dentro del mes de anclaje, con una fila por cliente.
8. Se reevaluó `k` mediante separación, tamaños, estabilidad entre semillas, estabilidad al retirar un bosque y utilidad comercial.
9. Se mantuvieron separadas las etiquetas observadas, proyectadas y compatibles mediante `origen_cluster`.
10. Se añadió una sensibilidad estructural que vuelve a clusterizar al incorporar censurados completos.
11. Se descartó la descomposición contable no demostrada. Rentabilidad, comisiones y márgenes se presentan como indicadores descriptivos no aditivos.
12. Las ramas exploratorias quedaron en el anexo y las conclusiones fueron reescritas desde los resultados reproducibles.

## Resultados reconstruidos

La solución principal utiliza `k=4`:

| Perfil | Clientes | Porcentaje | Evidencia central en `t0` |
|---|---:|---:|---|
| Vinculados multicanal | 450 | 22,6% | 44 transacciones; 100% usa tarjeta; 91,6% usa canales digitales |
| Digitales selectivos | 541 | 27,2% | 38 transacciones; 44,7% usa tarjeta; 83,5% usa canales digitales |
| Vinculados por nómina | 104 | 5,2% | 92,3% con payroll; 57 transacciones; saldo mediano $5.988 |
| Desvinculación avanzada | 896 | 45,0% | 5 transacciones; una familia activa mediana; saldo mediano $-6.512 |

Los tamaños originales informados —aproximadamente 101, 808, 377 y 705— no se conservaron porque provenían de una cadena no reproducible. Las etiquetas nuevas no son una mera renumeración de aquellas.

`k=4` presenta silhouette 0,066, ARI promedio 0,61 entre semillas y 0,68 al retirar un bosque. `k=3` presenta 0,059, 0,54 y 0,53, respectivamente. No se afirma que `k=4` sea matemáticamente óptimo.

## Generalización y sensibilidad

- La fidelidad fuera de mes del clasificador de extensión es 0,89 de balanced accuracy y 0,89 de F1 macro.
- Las 1.986 bajas con historia parcial se proyectan sin reemplazar las 1.991 etiquetas observadas.
- La distribución proyectada es cercana a la observada: 25,4% multicanal, 26,8% digitales selectivos, 4,1% nómina y 43,7% desvinculación avanzada.
- Los 1.103 censurados se mantienen como tales. Su distribución compatible es similar, pero al incorporarlos estructuralmente el ARI de las bajas estrictas es 0,49.
- La sensibilidad a dos muestras alternativas de continuadores produce ARI 0,45–0,49.
- Entre los 23 intermitentes, 13 son compatibles con el perfil multicanal; este dato se presenta como descripción de casos, no como generalización.

La segmentación es, por lo tanto, útil para organizar decisiones sobre la población principal, pero provisional y dependiente del grupo de comparación. Debe validarse con más meses.

## Auditoría económica

La identidad `mrentabilidad = mcomisiones + mactivos_margen + mpasivos_margen` se cumple solamente en 0,22% de las filas. La identidad de comisiones se cumple en 65,37%. No se calcula “rentabilidad sin mantenimiento” ni se interpretan los campos como una suma contable válida.

Las recomendaciones se formulan como hipótesis de experimentación. No se atribuye causalidad y se exige medir costo, riesgo, retención incremental y punto de equilibrio.

## Validación técnica

- Dos ejecuciones desde kernel nuevo reprodujeron los mismos objetos principales.
- Conteos de ejecución finales: 1 a 30, sin saltos.
- Celdas con error: 0.
- Hash de identificadores ordenados: `2048c0f97c03e1ff`.
- Hash de proximidad final: `9681de68cab1b9de`.
- Hash de etiquetas finales: `a1a8b756ce522202`.
- Filas de la proximidad y de la asignación: 1.991.
- Atributos del bosque: 147, con una única definición.
- Los ocho gráficos se regeneran en `eda_limpio_outputs/`.

## Archivos entregados

- `EDA_limpio.ipynb`: notebook final ejecutado y con outputs.
- `eda_limpio_outputs/`: ocho gráficos finales.
- `build_eda_limpio.py`: generador trazable del notebook.
- `execute_eda_limpio.py`: ejecutor secuencial desde kernel nuevo.
- `INFORME_CAMBIOS_EDA_LIMPIO.md`: este informe.

## Riesgos que permanecen

- Sólo se observan seis meses.
- La baja se infiere por ausencia del panel y no por una cancelación contractual.
- La pertenencia al paquete Premium proviene del contexto de la consigna, no de una variable explícita.
- Inflación, aguinaldo, estacionalidad y efecto cohorte pueden afectar importes nominales.
- La sensibilidad a continuadores y censurados limita la transportabilidad de los perfiles.
- No existe respaldo contable suficiente para valorar rentabilidad incremental.

