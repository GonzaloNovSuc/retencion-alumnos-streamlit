"""Pestaña 1 · ENTENDER: ¿quiénes son los alumnos y qué información tenemos?"""

import pandas as pd
import streamlit as st

# Variables que más importan para la pregunta guía y por qué (se muestra en la app)
VARIABLES_CLAVE = {
    "abandono": "Variable objetivo (Sí/No): lo que queremos predecir.",
    "asistencia_pct": "Participación en clase; suele caer antes de abandonar.",
    "entregas_a_tiempo_pct": "Señal temprana de desconexión académica.",
    "dias_activo_plataforma_mes": "Uso real de la plataforma, difícil de 'maquillar'.",
    "trabaja": "Posible causa externa: falta de tiempo.",
    "segmento": "Perfil de comportamiento para decidir acciones por grupo.",
}


def mostrar(df: pd.DataFrame):
    st.header("1. Entender los datos")
    st.markdown("**Pregunta:** ¿Quiénes son nuestros alumnos y qué información tenemos de ellos?")

    # --- Métricas generales ------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Número de alumnos (filas)", len(df))
    c2.metric("Número de variables", len(df.columns))
    c3.metric("Tasa de abandono observada", f"{(df['abandono'] == 'Sí').mean():.1%}")

    # --- Vista previa ------------------------------------------------------
    st.subheader("Vista previa del dataset")
    st.dataframe(df.head(10), width="stretch")

    # --- Diagnóstico: tipo, únicos y faltantes de cada variable -----------
    st.subheader("Variables y tipos de datos")
    diagnostico = pd.DataFrame({
        "Variable": df.columns,
        "Tipo": df.dtypes.astype(str).values,
        "Valores únicos": df.nunique().values,
        "Valores faltantes": df.isnull().sum().values,
    })
    st.dataframe(diagnostico, width="stretch")

    # --- Variables clave vinculadas con la pregunta guía ------------------
    st.subheader("Variables clave para la decisión")
    st.table(pd.DataFrame(
        {"Variable": list(VARIABLES_CLAVE), "Por qué importa": list(VARIABLES_CLAVE.values())}
    ))

    # --- Dudas sobre calidad (el dato más valioso para el análisis crítico) -
    st.warning(
        "**Dudas detectadas:** hay valores faltantes en `promedio_parcial`, "
        "`horas_estudio_sem` y `canal_ingreso`; "
        f"{int(df.duplicated().sum())} filas duplicadas; y "
        f"{int((df['asistencia_pct'] > 100).sum())} registros con asistencia mayor a 100 %. "
        "Además, el dataset es **sintético**: refleja patrones que yo mismo definí."
    )

    # --- Explorador interactivo de una variable ---------------------------
    st.subheader("Explorar una variable")
    variable = st.selectbox("Selecciona una variable:", df.columns, key="variable_entender")
    if pd.api.types.is_numeric_dtype(df[variable]):
        st.write("**Tipo:** numérica")
        st.dataframe(df[variable].describe(), width="stretch")
    else:
        st.write("**Tipo:** categórica")
        frec = df[variable].value_counts(dropna=False).reset_index()
        frec.columns = ["Valor", "Frecuencia"]
        st.dataframe(frec, width="stretch")
