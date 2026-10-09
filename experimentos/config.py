"""Configuracion compartida del Experimento 01.

Separa tres cosas que antes estaban mezcladas:

1. SEMILLAS_ENTREGA: las cinco declaradas en el CheckIn. Son las unicas que
   pueden aparecer en la prediccion que se entrega.
2. SEMILLAS_MEDICION: instrumento de medicion. Mas semillas = menos ruido en
   la comparacion entre brazos. No entran en la entrega.
3. FOLDS: las dos unicas particiones con gap de 2 meses que admite la ventana.
"""

from pathlib import Path

PROYECTO = Path(__file__).resolve().parent.parent
SALIDAS = Path(__file__).resolve().parent / "salidas"
PREDICCIONES = Path(__file__).resolve().parent / "predicciones"
CACHE_BASE = Path(__file__).resolve().parent / "base_modelado.pkl"

GANANCIA_BAJA2 = 1_072_500
COSTO_ESTIMULO = 27_500
UMBRAL_TEORICO = COSTO_ESTIMULO / (GANANCIA_BAJA2 + COSTO_ESTIMULO)  # 0.025

SEMILLAS_ENTREGA = (100129, 100183, 100151, 500041, 900007)

# Primos propios en [100003, 999983], generados una sola vez y congelados aca
# para que la corrida sea reproducible. Las cinco de entrega van primero.
SEMILLAS_MEDICION = SEMILLAS_ENTREGA + (
    103687, 150151, 212777, 265543, 318211,
    371299, 424243, 477551, 530177, 583651,
    636851, 690037, 743251, 796553, 849839,
)

# Gap de 2 meses entre el ultimo mes de train y el mes de validacion, igual que
# el gap real entre el ultimo mes etiquetado (202106) y el mes a predecir
# (202108). Es la unica configuracion que mide lo que vamos a hacer de verdad.
FOLDS = (
    {"nombre": "mayo", "train": (202103,), "validacion": 202105},
    {"nombre": "junio", "train": (202103, 202104), "validacion": 202106},
)

# Trial 25 congelado. Ceteris paribus: en el Experimento 01 NO se toca.
PARAMETROS_TRIAL25 = {
    "objective": "binary",
    "metric": "None",
    "boosting_type": "gbdt",
    "boost_from_average": True,
    "feature_pre_filter": False,
    "first_metric_only": True,
    "verbosity": -1,
    "bagging_freq": 1,
    "max_depth": 12,
    "num_leaves": 57,
    "learning_rate": 0.014449396240655157,
    "min_data_in_leaf": 1614,
    "feature_fraction": 0.2613620757159252,
    "bagging_fraction": 0.5276823217951291,
    "lambda_l1": 6.957248085997317e-05,
    "lambda_l2": 4.968141077185522e-06,
    "min_gain_to_split": 1.5074952598260436,
    "max_bin": 31,
}

MAX_ITERACIONES = 3000
EARLY_STOPPING = 100


def indice_mes(foto_mes: int) -> int:
    return (foto_mes // 100) * 12 + (foto_mes % 100)


def verificar_gap(fold) -> None:
    """El target BAJA+2 necesita dos meses para madurar: si el ultimo mes de
    train esta a menos de 2 del mes de validacion, las ventanas de resultado se
    solapan y la validacion deja de medir lo que va a pasar en agosto."""
    distancia = indice_mes(fold["validacion"]) - indice_mes(max(fold["train"]))
    if distancia < 2:
        raise ValueError(
            f"Gap temporal insuficiente en fold {fold['nombre']}: "
            f"train hasta {max(fold['train'])}, validacion {fold['validacion']}, "
            f"distancia {distancia} (se requiere >= 2)"
        )
