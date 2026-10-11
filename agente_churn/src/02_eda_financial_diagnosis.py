#!/usr/bin/env python3
"""
src/02_eda_financial_diagnosis.py — Agente 2: EDA y Diagnóstico Financiero Bancario
Alineado con las notebooks y metodología de DMEyF 2026 (z201, z401, z701).

## Propósito
Realizar el Análisis Exploratorio de Datos (EDA) y Diagnóstico Financiero del dataset bancario.
Analizar patrones de abandono (churn), evaluar métricas de discriminación inicial (Lift, distribuciones por clase), auditar proporciones de ceros/nulos y extraer insights de negocio alineados con la cátedra de DMEyF 2026.

## Directivas
1. Cargar el dataset con clase (`data/dataset_con_clase.parquet`).
2. Analizar la actividad transaccional del cliente (`ctrx_quarter`, `active_quarter`).
3. Diagnosticar el estado y uso de tarjetas (`Visa_status`, `Master_status`, consumos y límites) calculando el Lift sobre `BAJA+2`.
4. Evaluar saldos y márgenes financieros (`mcaja_ahorro`, `mcuenta_corriente`, `mpasivos_margen`) mediante percentiles y detección de vaciamiento de cuentas.
5. Auditar la estabilidad de ceros y nulos a lo largo de las fotos mensuales (`foto_mes`).
6. Rankear todas las variables por su AUC univariado sobre BAJA+2, mes a mes, y marcar las inestables.
7. Lo que pide z201: resumen de TODAS las variables (estadísticas, valores únicos, faltantes),
   tablas cruzadas de Visa_status/Master_status, densidades y boxplots por clase de las variables
   del ranking, y un árbol simple con sus variables más importantes.
8. Exportar un reporte en consola, un resumen JSON (`data/eda_financial_summary.json`), tablas CSV,
   gráficos en `data/eda_graficos/` y un borrador de hallazgos (`data/eda_informe.md`) para
   Zulip #Code>EDA y el video. Las conclusiones las escribe el alumno.
Los análisis contra la clase usan solo los meses con clase completa.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

# Carpeta donde se guardan las tablas CSV (la misma del resumen JSON; se fija en main)
OUT_DIR = Path("data")

# Parámetros del negocio bancario según la cátedra
GANANCIA_ACIERTO = 1_072_500  # Pesos por retenido exitoso (BAJA+2)
COSTO_ESTIMULO = 27_500       # Pesos por costo de envío de campaña


def parse_args():
    parser = argparse.ArgumentParser(
        description="Agente 2: EDA y Diagnóstico Financiero de Churn Bancario."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/dataset_con_clase.parquet",
        help="Ruta al dataset ingestado en formato parquet",
    )
    parser.add_argument(
        "--output_summary",
        type=str,
        default="data/eda_financial_summary.json",
        help="Ruta donde se guardará el resumen JSON con insights del EDA",
    )
    return parser.parse_args()


def load_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"❌ No se encontró el archivo ingestado: {input_path}")
    print(f"📥 Cargando dataset para EDA desde: {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"✅ Carga lista: {len(df):,} filas y {len(df.columns)} columnas.")
    return df


def analyze_transactional_activity(df: pd.DataFrame) -> dict:
    """Diagnóstico sobre ctrx_quarter (transacciones trimestrales)."""
    print("\n" + "=" * 65)
    print("📊 1. DIAGNÓSTICO TRANSACCIONAL (ctrx_quarter)")
    print("=" * 65)

    summary = {}
    if "ctrx_quarter" in df.columns:
        stats = df.groupby("clase_ternaria")["ctrx_quarter"].agg(
            ["count", "mean", "median", "std", lambda x: (x == 0).sum()]
        )
        stats.columns = ["count", "mean", "median", "std", "zeros_count"]
        stats["pct_zeros"] = (stats["zeros_count"] / stats["count"]) * 100

        print(stats.to_string())

        mean_baja2 = stats.loc["BAJA+2", "mean"] if "BAJA+2" in stats.index else 0
        mean_continua = stats.loc["CONTINUA", "mean"] if "CONTINUA" in stats.index else 0

        print(f"\n💡 Insight: Los clientes CONTINUA promedian {mean_continua:.1f} transacciones,")
        print(f"   mientras que los clientes en BAJA+2 caen a un promedio de {mean_baja2:.1f}.")

        summary["ctrx_quarter"] = {
            "mean_baja2": float(mean_baja2),
            "mean_continua": float(mean_continua),
            "drop_ratio": float(mean_baja2 / max(mean_continua, 1e-5)),
        }
    else:
        print("⚠️ Columna 'ctrx_quarter' no presente en el dataset.")

    return summary


def analyze_card_status_and_lift(df: pd.DataFrame) -> dict:
    """Diagnóstico de tarjetas (Visa_status) y cálculo de Lift sobre BAJA+2."""
    print("\n" + "=" * 65)
    print("💳 2. DIAGNÓSTICO DE TARJETAS Y CÁLCULO DE LIFT (Visa_status)")
    print("=" * 65)

    results = {}
    if "Visa_status" in df.columns and "clase_ternaria" in df.columns:
        total_clientes = len(df)
        baja2_total = (df["clase_ternaria"] == "BAJA+2").sum()
        p_baja2_global = baja2_total / total_clientes

        # Regla de negocio simple: Visa_status > 0
        df_visa_gt0 = df[df["Visa_status"] > 0]
        cant_visa_gt0 = len(df_visa_gt0)
        tp = (df_visa_gt0["clase_ternaria"] == "BAJA+2").sum()
        fp = cant_visa_gt0 - tp
        fn = baja2_total - tp

        p_baja2_visa = tp / max(cant_visa_gt0, 1)
        lift = p_baja2_visa / max(p_baja2_global, 1e-9)
        recall = tp / max(baja2_total, 1)
        ganancia_regla = (tp * GANANCIA_ACIERTO) - (fp * COSTO_ESTIMULO)

        print(f"Total clientes analizados:          {total_clientes:,}")
        print(f"Tasa global de BAJA+2:               {p_baja2_global:.6f} ({p_baja2_global*100:.2f}%)")
        print(f"Clientes con Visa_status > 0:       {cant_visa_gt0:,}")
        print(f"Aciertos (TP - BAJA+2 en regla):    {tp:,}")
        print(f"P(BAJA+2 | Visa_status > 0):        {p_baja2_visa:.6f}")
        print(f"🚀 LIFT de la regla Visa_status > 0: {lift:.2f}x")
        print(f"Recall capturado por la regla:     {recall*100:.2f}%")
        print(f"💰 Ganancia Neta teórica regla:     ${ganancia_regla:,.0f}")

        results["visa_status_lift"] = {
            "lift": float(lift),
            "recall": float(recall),
            "tp": int(tp),
            "fp": int(fp),
            "ganancia": float(ganancia_regla),
        }
    else:
        print("⚠️ Columna 'Visa_status' o 'clase_ternaria' no encontrada.")

    return results


def analyze_financial_balances(df: pd.DataFrame) -> dict:
    """Análisis de saldos y márgenes pasivos (mpasivos_margen, mcaja_ahorro)."""
    print("\n" + "=" * 65)
    print("🏦 3. DIAGNÓSTICO DE SALDOS Y MÁRGENES (mpasivos_margen, mcaja_ahorro)")
    print("=" * 65)

    summary = {}
    cols_saldos = [c for c in ["mpasivos_margen", "mcaja_ahorro", "mcuenta_corriente"] if c in df.columns]

    for col in cols_saldos:
        print(f"\n--- Atributo: {col} ---")
        p5 = df[col].quantile(0.05)
        p95 = df[col].quantile(0.95)

        # Filtro de percentiles 5 y 95 para ignorar outliers extremos en el resumen
        df_filtered = df[(df[col] >= p5) & (df[col] <= p95)]
        grouped = df_filtered.groupby("clase_ternaria")[col].agg(["median", "mean", "std"])
        print(grouped.to_string())

        summary[col] = {
            "p5": float(p5),
            "p95": float(p95),
            "medians": grouped["median"].to_dict(),
        }

    return summary


def audit_data_quality_across_months(df: pd.DataFrame) -> dict:
    """Auditoría preliminar de ceros y nulos por foto_mes para prevenir Data Drift."""
    print("\n" + "=" * 65)
    print("🛡️ 4. AUDITORÍA PRELIMINAR DE CALIDAD Y DRIFT (Por foto_mes)")
    print("=" * 65)

    quality_report = {}
    if "foto_mes" in df.columns:
        meses = sorted(df["foto_mes"].unique())
        print(f"Fotos mensuales detectadas: {meses}")

        # Evaluar porcentaje de nulos por mes
        null_counts = df.groupby("foto_mes").apply(lambda g: g.isnull().mean() * 100)
        max_null_diffs = (null_counts.max() - null_counts.min()).sort_values(ascending=False)

        top_drift_nulls = max_null_diffs.head(5)
        print("\nVariables con mayor variación mensual en % de NULOS:")
        for var, diff in top_drift_nulls.items():
            print(f"  • {var:<30}: Δ Max-Min = {diff:.2f}%")

        # Evaluar porcentaje de ceros por mes en numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        zero_counts = df.groupby("foto_mes")[numeric_cols].apply(lambda g: (g == 0).mean() * 100)
        max_zero_diffs = (zero_counts.max() - zero_counts.min()).sort_values(ascending=False)

        top_drift_zeros = max_zero_diffs.head(5)
        print("\nVariables con mayor variación mensual en % de CEROS:")
        for var, diff in top_drift_zeros.items():
            print(f"  • {var:<30}: Δ Max-Min = {diff:.2f}%")

        null_counts.T.to_csv(OUT_DIR / "eda_nulos_por_mes.csv")
        zero_counts.T.to_csv(OUT_DIR / "eda_ceros_por_mes.csv")
        print(f"\n   Tablas completas por variable y mes: {OUT_DIR}/eda_nulos_por_mes.csv y eda_ceros_por_mes.csv")

        quality_report["top_null_drift_cols"] = top_drift_nulls.to_dict()
        quality_report["top_zero_drift_cols"] = top_drift_zeros.to_dict()

    return quality_report


def rank_univariate_variables(df: pd.DataFrame, output_csv: Path, top: int = 25) -> dict:
    """Poder discriminante de cada variable por separado sobre BAJA+2 (AUC univariado),
    calculado mes a mes para ver si es estable. Un AUC de 0,5 no separa nada;
    lejos de 0,5 (hacia 1 o hacia 0) separa bien. Los nulos se tratan como el valor más bajo."""
    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 65)
    print("🔎 5. RANKING UNIVARIADO DE VARIABLES (AUC sobre BAJA+2, por mes)")
    print("=" * 65)
    excluir = {"numero_de_cliente", "foto_mes", "clase_binaria", "clase_binaria2", "clase01"}
    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluir]
    meses = sorted(df["foto_mes"].unique())
    filas = []
    for col in cols:
        aucs = {}
        for mes in meses:
            g = df[df["foto_mes"] == mes]
            y = (g["clase_ternaria"] == "BAJA+2").values
            x = g[col].astype(float).values
            if y.sum() == 0 or np.nanstd(x) == 0 or np.isnan(x).all():
                continue
            x = np.where(np.isnan(x), np.nanmin(x) - 1, x)
            aucs[mes] = roc_auc_score(y, x)
        if not aucs:
            continue
        vals = np.array(list(aucs.values()))
        filas.append({"variable": col, "auc_medio": vals.mean(), "separacion": abs(vals.mean() - 0.5),
                      "auc_min": vals.min(), "auc_max": vals.max(), "auc_desvio": vals.std(),
                      **{f"auc_{m}": a for m, a in aucs.items()}})
    tabla = pd.DataFrame(filas).sort_values("separacion", ascending=False)
    tabla.to_csv(output_csv, index=False)
    print(tabla.head(top)[["variable", "auc_medio", "auc_min", "auc_max", "auc_desvio"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n   AUC < 0,5 significa que valores BAJOS se asocian a la baja. Tabla completa: {output_csv}")
    inestables = tabla[tabla["auc_desvio"] > 0.03]
    if len(inestables):
        print(f"   ⚠️  {len(inestables)} variables con AUC inestable entre meses (desvío > 0,03): "
              f"{', '.join(inestables['variable'].head(10))}")
    return {"top_variables": tabla.head(top)["variable"].tolist(),
            "inestables": inestables["variable"].tolist()}


def summarize_all_variables(df: pd.DataFrame) -> dict:
    """z201 'Sumarizando': describe + valores únicos + faltantes de TODAS las variables."""
    print("\n" + "=" * 65)
    print("📋 6. RESUMEN DE TODAS LAS VARIABLES (z201)")
    print("=" * 65)
    num = df.select_dtypes(include=[np.number])
    tabla = num.describe(percentiles=[0.05, 0.5, 0.95]).T
    tabla["unicos"] = df[num.columns].nunique()
    tabla["faltantes"] = df[num.columns].isnull().sum()
    tabla["pct_faltantes"] = tabla["faltantes"] / len(df) * 100
    tabla["pct_ceros"] = (num == 0).mean() * 100
    salida = OUT_DIR / "eda_resumen_variables.csv"
    tabla.to_csv(salida)
    # Pocas categorías en una variable numérica = probable categórica codificada como número
    categoricas = tabla[(tabla["unicos"] <= 10) & (tabla.index != "foto_mes")].index.tolist()
    muy_faltantes = tabla[tabla["pct_faltantes"] > 50].index.tolist()
    print(f"   {len(tabla)} variables numéricas. Tabla completa: {salida}")
    print(f"   Probables categóricas (≤ 10 valores distintos): {', '.join(categoricas)}")
    print(f"   Más de 50% faltantes: {', '.join(muy_faltantes) or 'ninguna'}")
    return {"probables_categoricas": categoricas, "mas_50pct_faltantes": muy_faltantes}


def crosstab_status(df: pd.DataFrame) -> dict:
    """z201: tabla cruzada de Visa_status / Master_status contra la clase, con tasa de BAJA+2."""
    print("\n" + "=" * 65)
    print("🧾 7. TABLAS CRUZADAS DE ESTADO DE TARJETAS vs CLASE (z201)")
    print("=" * 65)
    resultado = {}
    for col in ["Visa_status", "Master_status"]:
        if col not in df.columns:
            continue
        ct = pd.crosstab(df[col].fillna(-1), df["clase_ternaria"])
        ct.index = ["sin tarjeta (NA)" if i == -1 else i for i in ct.index]
        ct["total"] = ct.sum(axis=1)
        ct["pct_baja2"] = ct.get("BAJA+2", 0) / ct["total"] * 100
        print(f"\n--- {col} ---")
        print(ct.to_string(float_format=lambda v: f"{v:.2f}"))
        ct.to_csv(OUT_DIR / f"eda_crosstab_{col}.csv")
        resultado[col] = {str(k): float(v) for k, v in ct["pct_baja2"].items()}
    return resultado


def plot_top_variables(df: pd.DataFrame, variables: list[str], n: int = 8) -> list[str]:
    """z201: densidad por clase y boxplot recortado al 5-95% para las variables del ranking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    carpeta = OUT_DIR / "eda_graficos"
    carpeta.mkdir(parents=True, exist_ok=True)
    clases = ["CONTINUA", "BAJA+1", "BAJA+2"]
    colores = {"CONTINUA": "#2a78d6", "BAJA+1": "#eda100", "BAJA+2": "#d43d51"}
    archivos = []
    for var in variables[:n]:
        x = df[var].astype(float)
        lo, hi = x.quantile(0.05), x.quantile(0.95)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
        recorte = x[x.between(lo, hi)].dropna()
        bins = np.histogram_bin_edges(recorte, bins=min(40, max(recorte.nunique(), 2)))  # mismos bins para las 3 clases
        for c in clases:
            v = x[(df["clase_ternaria"] == c) & x.between(lo, hi)].dropna()
            if len(v):
                ax1.hist(v, bins=bins, density=True, histtype="step", linewidth=1.8, label=c, color=colores[c])
        ax1.set_title(f"{var}: densidad por clase (recorte 5–95%)")
        ax1.legend()
        datos = [x[(df["clase_ternaria"] == c) & x.between(lo, hi)].dropna() for c in clases]
        ax2.boxplot(datos, showfliers=False)
        ax2.set_xticks(range(1, len(clases) + 1), clases)
        ax2.set_title(f"{var}: boxplot por clase (recorte 5–95%)")
        fig.tight_layout()
        ruta = carpeta / f"{var}.png"
        fig.savefig(ruta, dpi=110)
        plt.close(fig)
        archivos.append(str(ruta))
    print(f"\n🖼️  8. Gráficos de densidad y boxplot (z201) de {len(archivos)} variables en: {carpeta}")
    return archivos


def tree_feature_importance(df: pd.DataFrame, mes: int, top: int = 15) -> dict:
    """z201: un árbol de decisión simple sobre un mes y sus variables más importantes."""
    from sklearn.tree import DecisionTreeClassifier

    print("\n" + "=" * 65)
    print(f"🌳 9. ÁRBOL DE DECISIÓN SIMPLE SOBRE {mes} Y SUS VARIABLES IMPORTANTES (z201)")
    print("=" * 65)
    d = df[df["foto_mes"] == mes]
    excluir = {"numero_de_cliente", "foto_mes", "clase_binaria", "clase_binaria2", "clase01"}
    cols = [c for c in d.select_dtypes(include=[np.number]).columns if c not in excluir]
    X = d[cols].astype(float).fillna(-1e12)  # los nulos van como un valor extremo, separable
    y = d["clase_ternaria"]
    arbol = DecisionTreeClassifier(max_depth=6, min_samples_leaf=200, class_weight="balanced", random_state=0)
    arbol.fit(X, y)
    imp = pd.Series(arbol.feature_importances_, index=cols).sort_values(ascending=False)
    imp = imp[imp > 0]
    imp.to_csv(OUT_DIR / "eda_importancia_arbol.csv", header=["importancia"])
    print(imp.head(top).to_string(float_format=lambda v: f"{v:.4f}"))
    print("   (árbol de profundidad 6 con clases balanceadas: es para explorar, no para predecir)")
    return {"mes": int(mes), "top": imp.head(top).round(4).to_dict()}


def write_report(s: dict, completos: list, path: Path):
    """Borrador de una página con los hallazgos: base para Zulip #Code>EDA y el video para Miranda."""
    t = s.get("transactional", {}).get("ctrx_quarter", {})
    v = s.get("cards_lift", {}).get("visa_status_lift", {})
    r = s.get("ranking_univariado", {})
    res = s.get("resumen_variables", {})
    arbol = s.get("arbol", {})
    calidad = s.get("data_quality", {})
    lineas = [
        "# EDA — borrador de hallazgos",
        "",
        "> Borrador generado automáticamente a partir de los números del EDA. Revisarlo, completarlo con el",
        "> diccionario de datos y reescribirlo con tus palabras antes de compartirlo.",
        "",
        f"**Meses con clase completa analizados:** {', '.join(str(m) for m in completos)}.",
        "",
        "## Actividad transaccional",
        f"- Los CONTINUA promedian {t.get('mean_continua', 0):.0f} transacciones en el trimestre (`ctrx_quarter`); "
        f"los BAJA+2, {t.get('mean_baja2', 0):.0f} ({t.get('drop_ratio', 0)*100:.0f}% de los CONTINUA).",
        "",
        "## Tarjetas",
        f"- Regla `Visa_status > 0`: lift {v.get('lift', 0):.1f}x sobre la tasa base de BAJA+2, "
        f"pero captura solo el {v.get('recall', 0)*100:.1f}% de los BAJA+2. Ganancia de la regla: ${v.get('ganancia', 0):,.0f}.",
        "- Ver la tabla cruzada de `Visa_status` y `Master_status` para la tasa de BAJA+2 por estado.",
        "",
        "## Variables que más separan a los BAJA+2 (AUC univariado)",
        *[f"- `{x}`" for x in r.get("top_variables", [])[:10]],
        f"- Inestables entre meses: {', '.join(r.get('inestables', [])[:10]) or 'ninguna'}.",
        "",
        "## Variables más importantes de un árbol simple",
        *[f"- `{k}` ({val:.3f})" for k, val in list(arbol.get("top", {}).items())[:8]],
        "",
        "## Calidad de datos",
        f"- Probables categóricas codificadas como número: {', '.join(res.get('probables_categoricas', [])[:15])}.",
        f"- Con más de 50% de faltantes: {', '.join(res.get('mas_50pct_faltantes', [])) or 'ninguna'}.",
        f"- Mayor variación mensual de ceros: {', '.join(list(calidad.get('top_zero_drift_cols', {}))[:5])}.",
        "",
        "## Preguntas abiertas (para completar vos)",
        "- ¿Qué significa cada una de las variables top según el diccionario de datos?",
        "- ¿Qué historia cuentan juntas sobre cómo se va un cliente?",
        "- ¿Qué variables nuevas (lags, tendencias, ratios) podrían anticipar mejor la baja?",
    ]
    path.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"\n📝 Borrador de hallazgos: {path}")


def save_summary(data_dict: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)
    print(f"\n💾 Resumen del Diagnóstico Financiero guardado en: {output_path}")


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_summary_path = Path(args.output_summary)

    global OUT_DIR
    OUT_DIR = output_summary_path.parent
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)

    eda_summary = {}
    # Los análisis contra la clase usan solo meses con clase COMPLETA (los que tienen CONTINUA).
    # Los últimos dos meses no tienen clase completa y distorsionarían tasas, lift y recall.
    completos = sorted(df.loc[df["clase_ternaria"] == "CONTINUA", "foto_mes"].unique())
    df_lab = df[df["foto_mes"].isin(completos)]
    print(f"🏷️  Meses con clase completa (para análisis contra la clase): {completos}")

    eda_summary["transactional"] = analyze_transactional_activity(df_lab)
    eda_summary["cards_lift"] = analyze_card_status_and_lift(df_lab)
    eda_summary["balances"] = analyze_financial_balances(df_lab)
    eda_summary["data_quality"] = audit_data_quality_across_months(df)
    eda_summary["ranking_univariado"] = rank_univariate_variables(
        df_lab, OUT_DIR / "eda_ranking_variables.csv")
    eda_summary["resumen_variables"] = summarize_all_variables(df_lab)
    eda_summary["crosstab_tarjetas"] = crosstab_status(df_lab)
    eda_summary["graficos"] = plot_top_variables(df_lab, eda_summary["ranking_univariado"]["top_variables"])
    # z201 trabaja sobre 202104; si no está, el último mes con clase completa
    mes_arbol = 202104 if 202104 in completos else completos[-1]
    eda_summary["arbol"] = tree_feature_importance(df_lab, mes_arbol)
    write_report(eda_summary, completos, OUT_DIR / "eda_informe.md")

    save_summary(eda_summary, output_summary_path)
    print("\n✅ Agente 2: EDA y Diagnóstico Financiero finalizado con éxito.")


if __name__ == "__main__":
    main()