#!/usr/bin/env python3
"""
tests/make_synthetic_data.py — Dataset CRUDO sintético con la estructura de la competencia.

Sirve SOLO para verificar que el pipeline corre de punta a punta (test de humo).
No sirve para evaluar modelos: las relaciones entre variables son inventadas.

- Meses 202101 a 202108, sin clase_ternaria (la calcula el paso 01, como en z101).
- Cada cliente tiene un riesgo propio; cuando se va, deja de aparecer en el dataset.
- En los 2 meses previos a irse, su actividad cae (señal aprendible).
- Las variables monetarias (m*) crecen con una inflación mensual.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MESES = [202101, 202102, 202103, 202104, 202105, 202106, 202107, 202108]
MONETARIAS = [("mcaja_ahorro", 50_000), ("mcuenta_corriente", 30_000), ("mpasivos_margen", 2_000),
              ("mactivos_margen", 1_500), ("mrentabilidad_annual", 20_000), ("mtarjeta_visa_consumo", 15_000),
              ("mtarjeta_master_consumo", 8_000), ("Visa_msaldototal", 25_000), ("Master_msaldototal", 10_000),
              ("Visa_mconsumototal", 15_000), ("Master_mconsumototal", 8_000)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_path", default="tests/tmp/competencia_crudo.csv")
    p.add_argument("--n_clientes", type=int, default=8000)
    p.add_argument("--inflacion_mensual", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=100003)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    n = a.n_clientes
    riesgo = rng.beta(1, 30, n)                       # probabilidad mensual de irse
    ultimo = np.full(n, len(MESES) - 1)               # índice del último mes presente
    for i in range(n):
        for t in range(len(MESES) - 1):
            if rng.random() < riesgo[i]:
                ultimo[i] = t
                break
    antig0 = rng.integers(1, 300, n)

    filas = []
    for t, mes in enumerate(MESES):
        vivos = np.flatnonzero(ultimo >= t)
        faltan = ultimo[vivos] - t                    # meses que le quedan
        se_va = (ultimo[vivos] < len(MESES) - 1) & (faltan <= 2)
        act = np.where(se_va, 0.35, 1.0) * (1 - 3 * riesgo[vivos]).clip(0.2)
        infl = (1 + a.inflacion_mensual) ** t
        m = len(vivos)
        df = pd.DataFrame({
            "numero_de_cliente": vivos + 1,
            "foto_mes": mes,
            "cliente_antiguedad": antig0[vivos] + t,
            "active_quarter": (rng.random(m) < act).astype(int),
            "ctrx_quarter": rng.poisson(40 * act + 1),
            "cpayroll_trx": rng.poisson(1.2 * act),
            "Visa_status": np.where(rng.random(m) < np.where(se_va, 0.3, 0.03), 9.0, 0.0),
            "Master_status": np.where(rng.random(m) < 0.3, np.nan, 0.0),
        })
        for col, escala in MONETARIAS:
            df[col] = rng.gamma(1.5, escala, m) * act * infl
        df.loc[rng.random(m) < 0.05, "mcuenta_corriente"] = np.nan
        filas.append(df)
    data = pd.concat(filas, ignore_index=True).sample(frac=1, random_state=1)  # desordenado, como el real

    out = Path(a.output_path); out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out, index=False)
    print(f"✅ Crudo sintético: {len(data):,} filas, {n:,} clientes → {out}")


if __name__ == "__main__":
    main()
