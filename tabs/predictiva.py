"""Pestaña 4 · PREDECIR: ¿qué caracteriza a quien abandona y podemos predecirlo?"""

import pandas as pd
import streamlit as st

from utils import entrenar_modelo


def mostrar(df: pd.DataFrame):
    st.header("4. Estadística predictiva")
    st.markdown(
        "**Pregunta:** ¿Qué características tienen quienes abandonaron "
        "y podemos predecir quién abandonará?"
    )

    res = entrenar_modelo(df)
    m = res["metricas"]

    # --- Calidad del modelo ------------------------------------------------
    st.subheader("¿Qué tan bueno es el modelo?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Exactitud (prueba)", f"{m['accuracy']:.1%}")
    c2.metric("AUC (prueba)", f"{m['auc']:.2f}")
    c3.metric("Referencia: 'nadie abandona'", f"{m['base']:.1%}")
    st.caption(
        f"Entrenado con {m['n_train']} alumnos y evaluado con {m['n_test']} que no vio. "
        "La exactitud debe compararse con la referencia: si todos fueran 'No abandona' "
        "se acertaría esa proporción sin modelo. El AUC (0.5 = azar, 1 = perfecto) "
        "mide qué tan bien ordena a los alumnos de menor a mayor riesgo."
    )

    # --- Variables que más pesan ------------------------------------------
    st.subheader("Variables que más pesan")
    coef = res["coeficientes"]
    top = coef.head(10).set_index("variable")["coeficiente"]
    st.bar_chart(top.rename("Coeficiente"), x_label="Variable", y_label="Coeficiente")
    st.caption(
        "Coeficiente **positivo** = más riesgo de abandono; **negativo** = protege. "
        "Las numéricas están estandarizadas, así que son comparables entre sí."
    )
    tabla = coef[["variable", "coeficiente"]].head(10).rename(
        columns={"variable": "Variable", "coeficiente": "Coeficiente"})
    st.dataframe(tabla.style.format({"Coeficiente": "{:+.2f}"}), width="stretch")

    # --- Alcance del modelo (honestidad sobre sus límites) ----------------
    st.warning(
        "**Alcance:** el modelo encuentra *asociaciones*, no causas. Además los datos "
        "son sintéticos, así que el rendimiento refleja las reglas con que se "
        "generaron y probablemente sería menor con datos reales.\n\n"
        "**Ojo al interpretar:** variables como `programa` o `modalidad` aparecen con peso "
        "aunque en la simulación no influyen en el abandono; con solo ~500 alumnos el "
        "modelo puede tomar el azar por señal. Los pesos de asistencia, entregas y trabajo "
        "sí coinciden con las reglas del negocio."
    )
