"""Entrenamiento y evaluacion reproducibles del candidato LightGBM trial 25."""

from __future__ import annotations

import gc
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score


GANANCIA_ACIERTO = 1_072_500
COSTO_ESTIMULO = 27_500
BREAK_EVEN = COSTO_ESTIMULO / (GANANCIA_ACIERTO + COSTO_ESTIMULO)

MESES_ENTRENAMIENTO_REFERENCIA: Tuple[int, int] = (202103, 202104)
MES_VALIDACION_REFERENCIA = 202106

PARAMETROS_TRIAL_25: Mapping[str, Union[int, float]] = {
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
ITERACIONES_POR_SEMILLA: Mapping[int, int] = {
    100129: 129,
    100183: 130,
    100151: 150,
    500041: 164,
    900007: 143,
}
SEMILLAS: Tuple[int, ...] = tuple(ITERACIONES_POR_SEMILLA)

REFERENCIA_ENSEMBLE: Mapping[str, Mapping[str, float]] = {
    "PROMEDIO_5_SEMILLAS": {
        "ganancia": 408_732_500.0,
        "auc": 0.901335,
        "clientes_estimulados": 8_617,
    },
    "MEDIANA_5_SEMILLAS": {
        "ganancia": 415_332_500.0,
        "auc": 0.901260,
        "clientes_estimulados": 8_537,
    },
}


@dataclass
class ParticionTemporal:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_valid: pd.DataFrame
    y_valid: np.ndarray
    clases_valid: np.ndarray
    clientes_valid: np.ndarray
    ganancia_fila_valid: np.ndarray
    meses_train: Tuple[int, ...]
    mes_validacion: int


@dataclass
class ResultadoReferencia:
    metricas_semillas: pd.DataFrame
    metricas_ensemble: pd.DataFrame
    correlacion_semillas: pd.DataFrame
    predicciones_validacion: pd.DataFrame
    meses_train: Tuple[int, ...]
    mes_validacion: int
    features: Tuple[str, ...]
    configuracion: Dict[str, object]


def ganancia_por_fila(clase_ternaria: Sequence[object]) -> np.ndarray:
    """Valor economico exacto; nunca debe construirse desde el target binario."""

    clases = np.asarray(clase_ternaria, dtype=object)
    clases_validas = {"CONTINUA", "BAJA+1", "BAJA+2"}
    observadas = set(pd.Series(clases, dtype="object").dropna().unique())
    if pd.isna(clases).any() or not observadas.issubset(clases_validas):
        inesperadas = sorted(str(valor) for valor in observadas - clases_validas)
        raise ValueError(
            "La ganancia requiere clases maduras no nulas. "
            f"Clases inesperadas: {inesperadas}; nulos={int(pd.isna(clases).sum())}."
        )
    return np.where(
        clases == "BAJA+2",
        GANANCIA_ACIERTO,
        -COSTO_ESTIMULO,
    )


def _validar_alineacion(
    probabilidades: Sequence[float],
    ganancia_fila: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    probabilidades_np = np.asarray(probabilidades)
    ganancia_np = np.asarray(ganancia_fila)
    if len(probabilidades_np) != len(ganancia_np):
        raise ValueError(
            "Vectores desalineados: "
            f"probabilidades={len(probabilidades_np)}, "
            f"ganancias={len(ganancia_np)}"
        )
    return probabilidades_np, ganancia_np


def ganancia(
    probabilidades: Sequence[float],
    ganancia_fila: Sequence[float],
    threshold: float = BREAK_EVEN,
) -> float:
    """Ganancia con el umbral economico fijo 0.025 del proyecto."""

    probabilidades_np, ganancia_np = _validar_alineacion(
        probabilidades, ganancia_fila
    )
    return float(ganancia_np[probabilidades_np >= threshold].sum())


def cargar_y_validar_configuracion(
    ruta: Union[str, Path],
) -> Dict[str, object]:
    """Carga el JSON y falla si difiere del trial 25 congelado."""

    with Path(ruta).open("r", encoding="utf-8") as archivo:
        configuracion = json.load(archivo)

    errores = []
    if configuracion.get("trial") != 25:
        errores.append(f"trial={configuracion.get('trial')!r}; se esperaba 25")
    if configuracion.get("features") != 238:
        errores.append(
            f"features={configuracion.get('features')!r}; se esperaban 238"
        )
    if configuracion.get("meses_entrenamiento_final") != [
        202103,
        202104,
        202105,
        202106,
    ]:
        errores.append(
            "meses_entrenamiento_final no coincide con [202103, 202104, "
            "202105, 202106]"
        )
    if configuracion.get("mes_prediccion") != 202108:
        errores.append(
            f"mes_prediccion={configuracion.get('mes_prediccion')!r}; "
            "se esperaba 202108"
        )

    parametros = configuracion.get("parametros", {})
    if set(parametros) != set(PARAMETROS_TRIAL_25):
        errores.append(
            "las claves de parametros no coinciden: "
            f"observadas={sorted(parametros)}, "
            f"esperadas={sorted(PARAMETROS_TRIAL_25)}"
        )
    else:
        for nombre, esperado in PARAMETROS_TRIAL_25.items():
            observado = parametros[nombre]
            if isinstance(esperado, float):
                coincide = np.isclose(
                    float(observado), esperado, rtol=1e-15, atol=0.0
                )
            else:
                coincide = observado == esperado
            if not coincide:
                errores.append(
                    f"parametro {nombre}={observado!r}; se esperaba {esperado!r}"
                )

    semillas = tuple(configuracion.get("semillas", []))
    if semillas != SEMILLAS:
        errores.append(f"semillas={semillas}; se esperaban {SEMILLAS}")

    iteraciones = {
        int(semilla): int(valor)
        for semilla, valor in configuracion.get("iteraciones_por_semilla", {}).items()
    }
    if iteraciones != dict(ITERACIONES_POR_SEMILLA):
        errores.append(
            f"iteraciones={iteraciones}; se esperaban {dict(ITERACIONES_POR_SEMILLA)}"
        )

    if errores:
        raise ValueError("Configuracion final incompatible:\n - " + "\n - ".join(errores))
    return configuracion


def dividir_temporalmente(
    base: pd.DataFrame,
    features: Sequence[str],
    meses_train: Sequence[int] = MESES_ENTRENAMIENTO_REFERENCIA,
    mes_validacion: int = MES_VALIDACION_REFERENCIA,
) -> ParticionTemporal:
    """Centraliza la division temporal; numero_de_cliente nunca es predictor."""

    meses_train = tuple(int(mes) for mes in meses_train)
    mes_validacion = int(mes_validacion)
    if len(features) != 238 or len(set(features)) != 238:
        raise ValueError(
            "La division temporal requiere exactamente 238 features unicas."
        )
    prohibidas = {
        "numero_de_cliente",
        "foto_mes",
        "target_baja2",
        "clase_ternaria",
    }
    encontradas = sorted(
        feature
        for feature in features
        if feature in prohibidas or feature.startswith("_")
    )
    if encontradas:
        raise ValueError(
            "Hay columnas de control prohibidas entre los predictores: "
            + ", ".join(encontradas)
        )
    if mes_validacion in meses_train:
        raise ValueError("El mes de validacion no puede formar parte de train")

    faltantes = [feature for feature in features if feature not in base]
    if faltantes:
        raise ValueError("Faltan features para dividir: " + ", ".join(faltantes))

    mascara_train = base["foto_mes"].isin(meses_train)
    mascara_valid = base["foto_mes"].eq(mes_validacion)
    if not mascara_train.any():
        raise ValueError("La particion de entrenamiento esta vacia")
    if not mascara_valid.any():
        raise ValueError("La particion de validacion esta vacia")
    if base.loc[mascara_train, "target_baja2"].isna().any():
        raise ValueError("El entrenamiento contiene targets nulos")
    if base.loc[mascara_valid, "target_baja2"].isna().any():
        raise ValueError("La validacion contiene targets nulos")

    X_train = base.loc[mascara_train, list(features)]
    X_valid = base.loc[mascara_valid, list(features)]
    no_numericas = sorted(
        set(X_train.select_dtypes(exclude="number").columns)
        | set(X_valid.select_dtypes(exclude="number").columns)
    )
    if no_numericas:
        raise TypeError("LightGBM recibio columnas no numericas: " + ", ".join(no_numericas))

    y_train = (
        base.loc[mascara_train, "target_baja2"].astype("int8").to_numpy()
    )
    y_valid = (
        base.loc[mascara_valid, "target_baja2"].astype("int8").to_numpy()
    )
    clases_valid = base.loc[mascara_valid, "clase_ternaria"].astype(object).to_numpy()
    clientes_valid = base.loc[mascara_valid, "numero_de_cliente"].to_numpy()
    valores_target = set(np.unique(np.concatenate([y_train, y_valid])).tolist())
    if not valores_target.issubset({0, 1}):
        raise ValueError(f"El target no es binario: {sorted(valores_target)}")
    baja2_desde_clase = clases_valid == "BAJA+2"
    if not np.array_equal(y_valid.astype(bool), baja2_desde_clase):
        raise ValueError(
            "target_baja2 no coincide con clase_ternaria en validacion"
        )

    if meses_train == MESES_ENTRENAMIENTO_REFERENCIA and len(X_train) != 326_184:
        raise ValueError(
            f"La particion de referencia tiene {len(X_train):,} filas de train; "
            "se esperaban 326,184"
        )
    if mes_validacion == MES_VALIDACION_REFERENCIA and len(X_valid) != 164_114:
        raise ValueError(
            f"La particion de referencia tiene {len(X_valid):,} filas de validacion; "
            "se esperaban 164,114"
        )

    return ParticionTemporal(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        clases_valid=clases_valid,
        clientes_valid=clientes_valid,
        ganancia_fila_valid=ganancia_por_fila(clases_valid),
        meses_train=meses_train,
        mes_validacion=mes_validacion,
    )


def _parametros_semilla(
    parametros_trial: Mapping[str, Union[int, float]],
    semilla: int,
) -> Dict[str, object]:
    return {
        "objective": "binary",
        "metric": "None",
        "boosting_type": "gbdt",
        "boost_from_average": True,
        "feature_pre_filter": False,
        "first_metric_only": True,
        "verbosity": -1,
        "bagging_freq": 1,
        **dict(parametros_trial),
        "seed": semilla,
        "feature_fraction_seed": semilla,
        "bagging_seed": semilla,
        "data_random_seed": semilla,
    }


def entrenar_y_evaluar(
    particion: ParticionTemporal,
    configuracion: Mapping[str, object],
    directorio_modelos: Optional[Union[str, Path]] = None,
) -> ResultadoReferencia:
    """Entrena las cinco semillas y evalua individuales, promedio y mediana."""

    parametros = configuracion["parametros"]
    semillas = tuple(int(valor) for valor in configuracion["semillas"])
    iteraciones = {
        int(clave): int(valor)
        for clave, valor in configuracion["iteraciones_por_semilla"].items()
    }

    ruta_modelos = Path(directorio_modelos) if directorio_modelos else None
    if ruta_modelos is not None:
        ruta_modelos.mkdir(parents=True, exist_ok=True)

    predicciones: Dict[int, np.ndarray] = {}
    filas_metricas = []
    for semilla in semillas:
        dataset_train = lgb.Dataset(
            particion.X_train,
            label=particion.y_train,
        )
        modelo = lgb.train(
            params=_parametros_semilla(parametros, semilla),
            train_set=dataset_train,
            num_boost_round=iteraciones[semilla],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        probabilidades = modelo.predict(particion.X_valid)
        predicciones[semilla] = probabilidades
        filas_metricas.append(
            {
                "semilla": semilla,
                "iteraciones": iteraciones[semilla],
                "ganancia": ganancia(
                    probabilidades, particion.ganancia_fila_valid
                ),
                "auc": float(roc_auc_score(particion.y_valid, probabilidades)),
                "clientes_estimulados": int(
                    (probabilidades >= BREAK_EVEN).sum()
                ),
            }
        )
        if ruta_modelos is not None:
            modelo.save_model(str(ruta_modelos / f"trial25_semilla_{semilla}.txt"))
        del modelo
        del dataset_train
        gc.collect()

    matriz = np.column_stack([predicciones[semilla] for semilla in semillas])
    promedio = matriz.mean(axis=1)
    mediana = np.median(matriz, axis=1)

    filas_ensemble = []
    for metodo, probabilidades in (
        ("PROMEDIO_5_SEMILLAS", promedio),
        ("MEDIANA_5_SEMILLAS", mediana),
    ):
        referencia = REFERENCIA_ENSEMBLE[metodo]
        ganancia_observada = ganancia(
            probabilidades, particion.ganancia_fila_valid
        )
        auc_observado = float(roc_auc_score(particion.y_valid, probabilidades))
        estimulados = int((probabilidades >= BREAK_EVEN).sum())
        filas_ensemble.append(
            {
                "metodo": metodo,
                "ganancia": ganancia_observada,
                "auc": auc_observado,
                "clientes_estimulados": estimulados,
                "ganancia_referencia": referencia["ganancia"],
                "delta_ganancia": ganancia_observada - referencia["ganancia"],
                "auc_referencia": referencia["auc"],
                "delta_auc": auc_observado - referencia["auc"],
                "estimulados_referencia": int(referencia["clientes_estimulados"]),
                "delta_estimulados": estimulados
                - int(referencia["clientes_estimulados"]),
            }
        )

    columnas_prob = [f"probabilidad_{semilla}" for semilla in semillas]
    predicciones_validacion = pd.DataFrame(
        {
            "numero_de_cliente": particion.clientes_valid,
            "foto_mes": particion.mes_validacion,
            "clase_ternaria": particion.clases_valid,
            "target_baja2": particion.y_valid,
            "probabilidad_promedio": promedio,
            "probabilidad_mediana": mediana,
            "desvio_entre_semillas": matriz.std(axis=1),
        }
    )
    for posicion, columna in enumerate(columnas_prob):
        predicciones_validacion[columna] = matriz[:, posicion]

    correlacion = pd.DataFrame(matriz, columns=columnas_prob).corr()
    correlacion.index.name = "probabilidad"
    return ResultadoReferencia(
        metricas_semillas=pd.DataFrame(filas_metricas),
        metricas_ensemble=pd.DataFrame(filas_ensemble),
        correlacion_semillas=correlacion,
        predicciones_validacion=predicciones_validacion,
        meses_train=particion.meses_train,
        mes_validacion=particion.mes_validacion,
        features=tuple(particion.X_train.columns),
        configuracion=json.loads(json.dumps(configuracion)),
    )


def comparar_con_validacion_historica(
    predicciones_nuevas: pd.DataFrame,
    ruta_historica: Union[str, Path],
) -> pd.DataFrame:
    """Compara probabilidades nuevas y antiguas cliente a cliente."""

    historica = pd.read_csv(ruta_historica)
    if historica["numero_de_cliente"].duplicated().any():
        raise ValueError("La validacion historica tiene clientes duplicados")
    if predicciones_nuevas["numero_de_cliente"].duplicated().any():
        raise ValueError("La validacion nueva tiene clientes duplicados")

    columnas = [
        columna
        for columna in predicciones_nuevas.columns
        if columna.startswith("probabilidad_")
        and columna in historica.columns
    ]
    union = predicciones_nuevas[["numero_de_cliente", *columnas]].merge(
        historica[["numero_de_cliente", *columnas]],
        on="numero_de_cliente",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_nueva", "_historica"),
    )
    if not union["_merge"].eq("both").all():
        raise ValueError(
            "La poblacion nueva no coincide con la validacion historica: "
            + str(union["_merge"].value_counts().to_dict())
        )

    filas = []
    for columna in columnas:
        nueva = union[f"{columna}_nueva"].to_numpy()
        historica_col = union[f"{columna}_historica"].to_numpy()
        diferencia = nueva - historica_col
        filas.append(
            {
                "prediccion": columna,
                "correlacion": float(np.corrcoef(nueva, historica_col)[0, 1]),
                "error_absoluto_medio": float(np.abs(diferencia).mean()),
                "error_absoluto_maximo": float(np.abs(diferencia).max()),
                "decisiones_distintas": int(
                    ((nueva >= BREAK_EVEN) != (historica_col >= BREAK_EVEN)).sum()
                ),
            }
        )
    return pd.DataFrame(filas)


def guardar_resultados(
    resultado: ResultadoReferencia,
    directorio: Union[str, Path],
    reporte_correcciones: Optional[pd.DataFrame] = None,
    auditoria_preprocesamiento: Optional[pd.DataFrame] = None,
    auditoria_features: Optional[Mapping[str, object]] = None,
    nulos_por_familia: Optional[pd.DataFrame] = None,
    comparacion_historica: Optional[pd.DataFrame] = None,
) -> Path:
    """Guarda resultados v2 sin sobrescribir los artefactos historicos."""

    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    sufijo = str(resultado.mes_validacion)
    resultado.metricas_semillas.to_csv(
        directorio / f"metricas_semillas_{sufijo}.csv", index=False
    )
    resultado.metricas_ensemble.to_csv(
        directorio / f"metricas_ensemble_{sufijo}.csv", index=False
    )
    resultado.correlacion_semillas.to_csv(
        directorio / f"correlacion_semillas_{sufijo}.csv"
    )
    resultado.predicciones_validacion.to_csv(
        directorio / f"predicciones_validacion_{sufijo}.csv", index=False
    )
    if reporte_correcciones is not None:
        reporte_correcciones.to_csv(
            directorio / "correcciones_calidad.csv", index=False
        )
    if auditoria_preprocesamiento is not None:
        auditoria_preprocesamiento.to_csv(
            directorio / "auditoria_preprocesamiento.csv", index=False
        )
    if nulos_por_familia is not None:
        nulos_por_familia.to_csv(
            directorio / "nulos_por_familia.csv", index=False
        )
    if comparacion_historica is not None:
        comparacion_historica.to_csv(
            directorio / "comparacion_validacion_historica.csv", index=False
        )

    manifiesto = {
        "particion": {
            "meses_entrenamiento": list(resultado.meses_train),
            "mes_validacion": resultado.mes_validacion,
        },
        "umbral_ganancia": BREAK_EVEN,
        "ganancia_acierto": GANANCIA_ACIERTO,
        "costo_estimulo": COSTO_ESTIMULO,
        "versiones": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "features": list(resultado.features),
        "configuracion_modelo": resultado.configuracion,
        "auditoria_features": dict(auditoria_features or {}),
    }
    with (directorio / "manifiesto_ejecucion.json").open(
        "w", encoding="utf-8"
    ) as archivo:
        json.dump(manifiesto, archivo, ensure_ascii=False, indent=2)
    return directorio
