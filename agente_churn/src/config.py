"""
src/config.py — Carga de config/experimento.yaml y utilidades compartidas.

Todo lo que se repite entre scripts vive acá: la ganancia, las semillas primas,
las columnas que nunca son variables y las rutas del experimento.
"""

from pathlib import Path

import numpy as np
import yaml

GANANCIA_ACIERTO = 1_072_500
COSTO_ESTIMULO = 27_500

# Columnas que nunca entran al modelo
NO_FEATURES = ["numero_de_cliente", "foto_mes", "clase_ternaria", "clase01", "clase_binaria"]


def cargar_config(path: str = "config/experimento.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    exp_dir = Path(cfg["datos"]["carpeta_exp"]) / f"exp{cfg['experimento']}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg["exp_dir"] = exp_dir
    cfg["fe_path"] = exp_dir / "fe.parquet"
    return cfg


def meses_necesarios(cfg: dict) -> list[int]:
    m = cfg["meses"]
    return sorted(set(m["train"]) | {m["test"]} | set(m["train_final"]) | {m["future"]})


def semillas_primas(semilla_primigenia: int, n: int) -> list[int]:
    """n números primos de 6 dígitos elegidos al azar a partir de la semilla primigenia."""
    limite = 1_000_000
    criba = np.ones(limite, dtype=bool)
    criba[:2] = False
    for i in range(2, int(limite**0.5) + 1):
        if criba[i]:
            criba[i * i :: i] = False
    primos = np.nonzero(criba)[0]
    primos = primos[primos >= 100_000]
    rng = np.random.default_rng(semilla_primigenia)
    return [int(p) for p in rng.choice(primos, size=n, replace=False)]


def ganancia_por_cliente(clase_ternaria) -> np.ndarray:
    """+1.072.500 si es BAJA+2, -27.500 en cualquier otro caso."""
    return np.where(np.asarray(clase_ternaria) == "BAJA+2", GANANCIA_ACIERTO, -COSTO_ESTIMULO)
