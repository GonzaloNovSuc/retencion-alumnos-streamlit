"""Pestaña 3 · DESCRIBIR: ¿cómo se comportan, qué segmentos hay y qué KPIs los describen?"""

import pandas as pd
import streamlit as st


def mostrar(df: pd.DataFrame):
    st.header("3. Estadística descriptiva")
    st.markdown(
        "**Pregunta:** ¿Cómo se comportan los alumnos, qué segmentos existen "
        "y qué KPIs los describen mejor?"
    )

    # Variable binaria 0/1 para poder promediarla como "tasa de abandono"
    df = df.assign(abandono_num=(df["abandono"] == "Sí").astype(int))

    # --- KPIs globales -----------------------------------------------------
    st.subheader("KPIs generales")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tasa de abandono", f"{df['abandono_num'].mean():.1%}")
    k2.metric("Asistencia promedio", f"{df['asistencia_pct'].mean():.1f} %")
    k3.metric("Entregas a tiempo", f"{df['entregas_a_tiempo_pct'].mean():.1f} %")
    k4.metric("Días activos/mes", f"{df['dias_activo_plataforma_mes'].mean():.1f}")

    # --- Tabla resumen por segmento ---------------------------------------
    st.subheader("Perfil de cada segmento")
    resumen = df.groupby("segmento").agg(
        alumnos=("id_alumno", "count"),
        tasa_abandono=("abandono_num", "mean"),
        asistencia=("asistencia_pct", "mean"),
        entregas=("entregas_a_tiempo_pct", "mean"),
        dias_activo=("dias_activo_plataforma_mes", "mean"),
        horas_estudio=("horas_estudio_sem", "mean"),
    ).sort_values("tasa_abandono", ascending=False)
    resumen["participacion"] = resumen["alumnos"] / resumen["alumnos"].sum()

    st.dataframe(
        resumen.style.format({
            "tasa_abandono": "{:.1%}", "participacion": "{:.1%}", "asistencia": "{:.1f}",
            "entregas": "{:.1f}", "dias_activo": "{:.1f}", "horas_estudio": "{:.1f}",
        }),
        width="stretch",
    )

    # --- Gráfica 1: tasa de abandono por segmento -------------------------
    st.subheader("Tasa de abandono por segmento")
    st.bar_chart(
        (resumen["tasa_abandono"] * 100).rename("Tasa de abandono (%)"),
        x_label="Segmento", y_label="Abandono (%)",
    )
    peor = resumen["tasa_abandono"].idxmax()
    st.caption(
        f"El segmento **{peor}** tiene la tasa más alta "
        f"({resumen.loc[peor, 'tasa_abandono']:.1%}), muy por encima del promedio "
        f"({df['abandono_num'].mean():.1%})."
    )

    # --- Gráfica 2: KPI elegido por el usuario ----------------------------
    st.subheader("Compara un KPI entre segmentos")
    opciones = {
        "Asistencia (%)": "asistencia", "Entregas a tiempo (%)": "entregas",
        "Días activos al mes": "dias_activo", "Horas de estudio por semana": "horas_estudio",
    }
    eleccion = st.selectbox("KPI:", list(opciones), key="kpi_desc")
    st.bar_chart(resumen[opciones[eleccion]].rename(eleccion),
                 x_label="Segmento", y_label=eleccion)

    # --- Gráfica 3: abandono según una variable categórica ----------------
    st.subheader("Abandono según una característica")
    cat = st.selectbox("Variable:", ["trabaja", "beca", "modalidad", "programa", "canal_ingreso"],
                       key="cat_desc")
    tasa_cat = df.groupby(cat)["abandono_num"].mean() * 100
    st.bar_chart(tasa_cat.rename("Abandono (%)"), x_label=cat, y_label="Abandono (%)")

    st.success(
        "**Interpretación:** los segmentos *Desconectado* y *Sobrecargado* concentran "
        "el abandono; se distinguen por baja asistencia, pocas entregas y pocos días "
        "activos. Estos KPIs de hábito (no los datos demográficos) son los que mejor "
        "separan a los grupos."
    )
