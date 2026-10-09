"""Etapa 00 - Configuracion unica del pipeline.

POR QUE EXISTE ESTE ARCHIVO
---------------------------
Hasta ahora las semillas, la formula de ganancia y los meses estaban repetidos
en competencia.ipynb, competenciav2.ipynb, competencia_modelado_v2.ipynb,
preprocesamiento.py y modelo_referencia.py. Cuando un valor vive en cinco
lugares, tarde o temprano dos de esos lugares dicen cosas distintas: de hecho
competenciav2.ipynb quedo con folds de gap 1 mientras modelo_referencia.py
usaba gap 2. Un unico archivo de configuracion hace que esa contradiccion sea
imposible por construccion.

POR QUE ESTA PARAMETRIZADO POR VENTANA DE MESES
-----------------------------------------------
La Ultima Competencia usa un dataset distinto, con mas meses. Si los meses
estan hardcodeados, el pipeline no se reusa. Si estan aca, cambiar de
competencia es cambiar tres constantes.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------

# parent.parent porque este archivo vive en pipeline/ y queremos la raiz del
# repo, que es donde estan los datos y las salidas historicas.
PROYECTO = Path(__file__).resolve().parent.parent
PIPELINE = Path(__file__).resolve().parent

# La base DuckDB se persiste en un archivo en vez de vivir en memoria.
# Razon: cada etapa es un proceso separado (asi se puede relanzar una sola sin
# recorrer todo), y procesos separados no comparten memoria. Ademas, si una
# etapa larga se cae, lo ya calculado queda en disco.
BASE_DUCKDB = PIPELINE / "competencia.duckdb"

SALIDAS = PIPELINE / "salidas"
SALIDAS.mkdir(exist_ok=True)

# Ruta del CSV crudo. Se resuelve contra una lista de candidatos porque el
# dataset vive en Google Drive montado, que puede no estar disponible.
RUTAS_FUENTE_CANDIDATAS = (
    PROYECTO / "data" / "competencia_01_crudo.csv",
    PROYECTO / "competencia_01_crudo.csv",
    Path(
        "/Users/manuelmarasco/Library/CloudStorage/"
        "GoogleDrive-hola@manuelmarasco.com/Mi unidad/DMEyF/2026/"
        "notebooks/data/competencia_01_crudo.csv"
    ),
)

# --------------------------------------------------------------------------
# El problema de negocio
# --------------------------------------------------------------------------

# Estimular a un BAJA+2 paga; estimular a cualquier otro cuesta. La asimetria
# es de 39 a 1, y es la razon por la que no se optimiza accuracy ni AUC.
GANANCIA_BAJA2 = 1_072_500
COSTO_ESTIMULO = 27_500

# Punto de indiferencia: la probabilidad a la que estimular deja de ser
# rentable. Sale de igualar el valor esperado a cero:
#   p * GANANCIA - (1-p) * COSTO = 0  =>  p = COSTO / (GANANCIA + COSTO)
# Da exactamente 0,025 = 1/40, que es el Nivel Uno de la catedra.
UMBRAL_TEORICO = COSTO_ESTIMULO / (GANANCIA_BAJA2 + COSTO_ESTIMULO)

# --------------------------------------------------------------------------
# La ventana temporal
# --------------------------------------------------------------------------

# Las 6 fotos del dataset de la Primera Competencia.
MESES_DATASET = (202103, 202104, 202105, 202106, 202107, 202108)

# Mes a predecir. No tiene etiqueta y nunca puede entrar a entrenamiento.
MES_SCORING = 202108

# Meses con clase_ternaria completa. Un mes M necesita observar M+1 y M+2 para
# saber si fue BAJA+2, asi que el ultimo mes etiquetable es MES_SCORING - 2.
# 202107 queda afuera: se puede ver si desaparecio en 202108 (eso lo haria
# BAJA+1), pero para distinguir BAJA+2 de CONTINUA haria falta 202109.
MESES_TARGET_MADURO = (202103, 202104, 202105, 202106)

# Cantidad de meses que el target necesita para madurar. Es la constante que
# gobierna el gap de validacion de toda la materia.
MESES_MADURACION = 2

# --------------------------------------------------------------------------
# Semillas
# --------------------------------------------------------------------------

# Las cinco primas declaradas en el CheckIn. Son las UNICAS que pueden
# aparecer en la prediccion que se entrega: la catedra reproduce la entrega
# desde el crudo y tiene que dar bit a bit lo mismo.
SEMILLAS_ENTREGA = (100129, 100183, 100151, 500041, 900007)

# Semillas adicionales para MEDIR. No entran en la entrega.
# Por que separarlas: el Experimento 01 mostro que el desvio de ganancia entre
# semillas en junio es de 13,18 M. Con 5 semillas el error estandar de la
# media es 5,9 M, asi que no se puede distinguir un bloque que aporta 5 M de
# uno que aporta 0. Ensamblar satura en 5 semillas; medir necesita 20.
SEMILLAS_MEDICION = SEMILLAS_ENTREGA + (
    103687, 150151, 212777, 265543, 318211,
    371299, 424243, 477551, 530177, 583651,
    636851, 690037, 743251, 796553, 849839,
)

# --------------------------------------------------------------------------
# Folds de validacion
# --------------------------------------------------------------------------

def indice_mes(foto_mes: int) -> int:
    """Convierte 202104 en un entero continuo de meses.

    Por que: 202112 y 202201 son meses contiguos, pero restados como enteros
    dan 89. Pasando a (anio * 12 + mes) la resta vuelve a ser la distancia
    real. Sin esto, cualquier chequeo de gap falla al cruzar un fin de anio.
    """
    return (foto_mes // 100) * 12 + (foto_mes % 100)


# Los dos unicos folds que esta ventana admite con el gap correcto.
# El gap replica la distancia real del problema: el ultimo mes etiquetado es
# 202106 y hay que predecir 202108, o sea 2 meses. Un fold con gap 1 entrena
# bajo una relacion temporal distinta a la que va a enfrentar.
FOLDS = (
    {"nombre": "mayo", "train": (202103,), "validacion": 202105},
    {"nombre": "junio", "train": (202103, 202104), "validacion": 202106},
)


def verificar_gap(fold: dict) -> None:
    """Falla si el fold permite solapamiento de ventanas de resultado.

    La etiqueta de un mes M codifica la presencia del cliente en M+1 y M+2.
    Si el ultimo mes de train esta a 1 mes de la validacion, las dos etiquetas
    comparten un mes de observacion: la de train revela parte de la de
    validacion. Exigiendo distancia >= MESES_MADURACION las ventanas quedan
    disjuntas, igual que entre 202106 (ultimo train real) y 202108 (scoring).
    """
    distancia = indice_mes(fold["validacion"]) - indice_mes(max(fold["train"]))
    if distancia < MESES_MADURACION:
        raise ValueError(
            f"Fold '{fold['nombre']}' invalido: train hasta "
            f"{max(fold['train'])}, validacion {fold['validacion']}, "
            f"distancia {distancia} < {MESES_MADURACION} requerida."
        )


def resolver_ruta_fuente() -> Path:
    """Devuelve la primera ruta candidata que exista, o falla con la lista."""
    for ruta in RUTAS_FUENTE_CANDIDATAS:
        if ruta.is_file():
            return ruta
    raise FileNotFoundError(
        "No se encontro el CSV crudo. Candidatas probadas:\n  "
        + "\n  ".join(str(r) for r in RUTAS_FUENTE_CANDIDATAS)
    )


# --------------------------------------------------------------------------
# Valores esperados: los gates del pipeline
# --------------------------------------------------------------------------

# Conteo de filas por mes del crudo. Si el dataset cambia, el pipeline tiene
# que gritar en la etapa 01 y no producir silenciosamente otro resultado.
FILAS_POR_MES_ESPERADAS = {
    202103: 162_900, 202104: 163_284, 202105: 163_768,
    202106: 164_114, 202107: 164_348, 202108: 164_647,
}
FILAS_TOTALES_ESPERADAS = 983_061

# Casos de BAJA+2 por mes maduro. Vienen del EDA ya validado y sirven como
# test de regresion: si la construccion del target cambia, esto se rompe.
BAJA2_POR_MES_ESPERADOS = {
    202103: 960, 202104: 1_139, 202105: 870, 202106: 1_098,
}
