"""Pestaña 5 · PRESCRIBIR: ¿qué acción tomar con cada caso?"""

import pandas as pd
import streamlit as st

from utils import datos_con_riesgo

# Regla de negocio: acción recomendada por segmento (se usa también en la pestaña 6)
ACCIONES_SEGMENTO = {
    "Desconectado": "Contacto personal inmediato (llamada de un tutor) y plan de reincorporación.",
    "Sobrecargado": "Flexibilizar fechas de entrega y ofrecer clases grabadas / tutorías en horario nocturno.",
    "Intermitente": "Recordatorios automáticos semanales y metas pequeñas con seguimiento quincenal.",
    "Comprometido": "Mantener y reconocer; invitarlos como mentores de otros alumnos.",
}

ACCIONES_NIVEL = {
    "Alto": "Intervención prioritaria esta semana (tutor + llamada).",
    "Medio": "Seguimiento quincenal y recordatorios automáticos.",
    "Bajo": "Monitoreo pasivo; sin intervención adicional.",
}


def mostrar(df: pd.DataFrame):
    st.header("5. Estadística prescriptiva")
    st.markdown("**Pregunta:** ¿Qué acción deberíamos tomar con cada alumno o grupo?")

    d = datos_con_riesgo(df)
    # Pasos para ordenar los niveles de forma lógica (Alto, Medio, Bajo)
    orden = ["Alto", "Medio", "Bajo"]

    # --- Acción por nivel de probabilidad ---------------------------------
    st.subheader("Acción por nivel de riesgo")
    por_nivel = d.groupby("nivel_riesgo").agg(
        alumnos=("id_alumno", "count"),
        prob_media=("prob_abandono", "mean"),
        abandono_real=("abandono", lambda s: (s == "Sí").mean()),
    ).reindex(orden)
    por_nivel["accion_recomendada"] = por_nivel.index.map(ACCIONES_NIVEL)
    st.dataframe(
        por_nivel.style.format({"prob_media": "{:.1%}", "abandono_real": "{:.1%}"}),
        width="stretch",
    )
    st.caption(
        "Umbrales: Alto ≥ 50 %, Medio 25–50 %, Bajo < 25 % de probabilidad de abandono. "
        "Si 'abandono real' crece con el nivel, el modelo ordena bien el riesgo."
    )

    # --- Acción por segmento ----------------------------------------------
    st.subheader("Acción por segmento")
    por_seg = d.groupby("segmento").agg(
        alumnos=("id_alumno", "count"),
        prob_media=("prob_abandono", "mean"),
        en_riesgo_alto=("nivel_riesgo", lambda s: (s == "Alto").sum()),
    ).sort_values("prob_media", ascending=False)
    por_seg["accion_recomendada"] = por_seg.index.map(ACCIONES_SEGMENTO)
    st.dataframe(por_seg.style.format({"prob_media": "{:.1%}"}), width="stretch")

    # --- Lista priorizada de alumnos --------------------------------------
    st.subheader("Alumnos a contactar primero")
    n = st.slider("¿Cuántos mostrar?", 5, 50, 15, key="n_prioridad")
    cols = ["id_alumno", "segmento", "prob_abandono", "asistencia_pct",
            "entregas_a_tiempo_pct", "trabaja"]
    prioridad = d.sort_values("prob_abandono", ascending=False).head(n)[cols]
    st.dataframe(prioridad.style.format({"prob_abandono": "{:.1%}"}), width="stretch")
