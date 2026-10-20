"""Pestaña 2 · LIMPIAR: ¿qué tan confiables son los datos?"""

import pandas as pd
import streamlit as st


def mostrar(df_crudo: pd.DataFrame, df_limpio: pd.DataFrame, bitacora: pd.DataFrame):
    st.header("2. Limpiar los datos")
    st.markdown("**Pregunta:** ¿Qué tan confiables son los datos y qué hacemos con cada problema?")

    # --- Diagnóstico ANTES -------------------------------------------------
    st.subheader("Diagnóstico inicial")
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros duplicados", int(df_crudo.duplicated().sum()))
    c2.metric("Celdas con valor faltante", int(df_crudo.isnull().sum().sum()))
    c3.metric("Asistencia > 100 %", int((df_crudo["asistencia_pct"] > 100).sum()))

    nulos = df_crudo.isnull().sum()
    nulos = nulos[nulos > 0].rename("Nulos").reset_index()
    nulos.columns = ["Variable", "Nulos"]
    nulos["% de la columna"] = (nulos["Nulos"] / len(df_crudo) * 100).round(1)
    st.write("**Nulos por variable:**")
    st.dataframe(nulos, width="stretch")

    # --- Decisiones --------------------------------------------------------
    st.subheader("Decisión tomada con cada problema")
    bitacora = bitacora.rename(columns={
        "problema": "Problema", "cantidad": "Casos", "decision": "Decisión"})
    st.dataframe(bitacora, width="stretch")

    # --- Comparación de estrategias para nulos (antes vs. después) --------
    st.subheader("¿Por qué imputar y no eliminar filas con nulos?")
    sin_dup = df_crudo.drop_duplicates()
    eliminando = sin_dup.dropna()
    comp = pd.DataFrame({
        "Estrategia": ["Eliminar filas con nulos", "Imputar (mediana / 'Sin dato')"],
        "Filas que quedan": [len(eliminando), len(df_limpio)],
        "Alumnos perdidos": [len(sin_dup) - len(eliminando), 0],
    })
    st.dataframe(comp, width="stretch")
    st.info(
        "Eliminar filas haría perder alumnos y podría sesgar el análisis (quizá "
        "quienes tienen datos incompletos son justo los más desconectados). "
        "Por eso se imputa con la **mediana**, que no se distorsiona con extremos."
    )

    # --- Resultado DESPUÉS -------------------------------------------------
    st.subheader("Resultado después de limpiar")
    d1, d2, d3 = st.columns(3)
    d1.metric("Filas", len(df_limpio), delta=len(df_limpio) - len(df_crudo))
    d2.metric("Nulos restantes", int(df_limpio.isnull().sum().sum()))
    d3.metric("Duplicados restantes", int(df_limpio.duplicated().sum()))
