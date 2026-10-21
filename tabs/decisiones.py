"""Pestaña 6 · DECIDIR: ¿qué mensaje mostramos a cada tipo de alumno?"""

import pandas as pd
import streamlit as st

from tabs.prescriptiva import ACCIONES_SEGMENTO
from utils import datos_con_riesgo, entrenar_modelo

# Mensaje dirigido al alumno según su segmento (tono distinto para cada perfil)
MENSAJES = {
    "Desconectado": "Te extrañamos en clase. Tu lugar sigue aquí: ¿platicamos 10 minutos "
                    "para encontrar cómo retomarlo juntos?",
    "Sobrecargado": "Sabemos que trabajar y estudiar a la vez es mucho. Te ofrecemos "
                    "clases grabadas y más flexibilidad en tus entregas para que no tengas que elegir.",
    "Intermitente": "Vas avanzando: un paso pequeño esta semana te mantiene en ritmo. "
                    "Te enviaremos una meta corta cada lunes.",
    "Comprometido": "Tu constancia se nota. ¿Te gustaría compartir tu método como mentor "
                    "de otros compañeros?",
}


def _error_por_subgrupo(d: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula, para cada subgrupo, qué tan bien acierta el modelo.
    `d` debe traer prob_abandono (fuera de muestra) y la columna real `abandono`.
    """
    d = d.copy()
    d["real"] = d["abandono"] == "Sí"             # lo que ocurrió
    d["predicho"] = d["prob_abandono"] >= 0.50    # lo que el modelo marcó como riesgo
    d["acierto"] = d["real"] == d["predicho"]
    # Rangos de edad para poder agrupar la variable numérica
    d["rango_edad"] = pd.cut(d["edad"], [0, 27, 33, 120],
                             labels=["27 o menos", "28 a 33", "34 o más"])

    filas = []
    for variable in ["trabaja", "beca", "rango_edad"]:
        for valor, g in d.groupby(variable, observed=True):
            abandonaron = g[g["real"]]              # solo quienes sí abandonaron
            filas.append({
                "variable": variable,
                "grupo": str(valor),
                "alumnos": len(g),
                "abandono_real": g["real"].mean(),
                "exactitud": g["acierto"].mean(),
                # de los que abandonaron, ¿cuántos había marcado el modelo?
                "abandonos_detectados": abandonaron["predicho"].mean() if len(abandonaron) else float("nan"),
            })
    return pd.DataFrame(filas)


def mostrar(df: pd.DataFrame):
    st.header("6. Toma de decisiones")
    st.markdown("**Pregunta:** ¿Qué mensaje deberíamos mostrarle a cada tipo de alumno?")

    d = datos_con_riesgo(df)
    segmento = st.selectbox("Selecciona un segmento:", list(MENSAJES), key="seg_decision")
    sub = d[d["segmento"] == segmento]

    # Datos propios de la app que sustentan la decisión
    prob = sub["prob_abandono"].mean()
    abandono_real = (sub["abandono"] == "Sí").mean()
    prom_general = (d["abandono"] == "Sí").mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Alumnos en el segmento", len(sub))
    c2.metric("Abandono observado", f"{abandono_real:.1%}",
              delta=f"{(abandono_real - prom_general) * 100:+.1f} pts vs. promedio",
              delta_color="inverse")
    c3.metric("Probabilidad media (modelo)", f"{prob:.1%}")

    st.subheader("Acción recomendada")
    st.info(ACCIONES_SEGMENTO[segmento])

    st.subheader("Mensaje para el alumno")
    st.success(f"“{MENSAJES[segmento]}”")

    # Justificación construida con los números reales del segmento
    st.subheader("Justificación")
    ref = sub[["asistencia_pct", "entregas_a_tiempo_pct", "dias_activo_plataforma_mes"]].mean()
    top_vars = entrenar_modelo(df)["coeficientes"].head(3)["variable"].tolist()
    st.write(
        f"- El segmento **{segmento}** abandona en **{abandono_real:.1%}** de los casos "
        f"(promedio general: {prom_general:.1%}).\n"
        f"- Su asistencia media es {ref['asistencia_pct']:.0f} %, entrega a tiempo "
        f"{ref['entregas_a_tiempo_pct']:.0f} % y se conecta {ref['dias_activo_plataforma_mes']:.0f} "
        "días al mes.\n"
        f"- Las variables que más pesan en el modelo son: {', '.join(top_vars)}; "
        "la acción busca mover justo esos hábitos."
    )

    # ------------------------------------------------------------------
    # Equidad: ¿el modelo falla más con algún grupo? (idea surgida de la
    # revisión de una compañera: "evaluar el error por subgrupo")
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("¿El modelo falla más con algún grupo?")
    st.caption(
        "Se compara la predicción (riesgo ≥ 50 % = 'abandonará') con lo que realmente "
        "ocurrió, separando por características que podrían generar sesgo."
    )
    tabla_equidad = _error_por_subgrupo(d)
    st.dataframe(
        tabla_equidad.style.format({
            "abandono_real": "{:.1%}", "exactitud": "{:.1%}", "abandonos_detectados": "{:.1%}",
        }),
        width="stretch", hide_index=True,
    )
    st.info(
        "**Cómo leerla:** 'exactitud' es el % de alumnos bien clasificados y "
        "'abandonos_detectados' es, de quienes sí abandonaron, cuántos el modelo había "
        "marcado como riesgo. Una diferencia grande entre grupos indicaría que el modelo "
        "protege a unos y deja sin ver a otros. Con muestras pequeñas, la diferencia "
        "puede ser azar."
    )

    # ------------------------------------------------------------------
    # Reflexión de transferencia (Momento 7): límites con datos reales
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Límites y qué cambiaría con datos reales")
    st.markdown(
        """
**1. Calidad de datos.** Aquí los datos son sintéticos y los errores se pusieron a propósito.
Con datos reales habría que auditar el origen de cada campo (¿quién captura la asistencia?),
definir qué significa "abandono" (baja formal vs. dejar de asistir) y validar la imputación
con el área escolar en lugar de usar solo la mediana.

**2. Sesgos.** Variables como `trabaja`, `beca` o `edad` pueden hacer que el modelo etiquete
como "riesgo" a ciertos grupos y reciban más (o menos) atención por razones ajenas a su
esfuerzo. Mediría el error del modelo por grupo y usaría las predicciones solo para
**ofrecer apoyo**, nunca para sancionar o negar oportunidades.

**3. Privacidad.** Un modelo con datos personales de alumnos exige consentimiento informado,
mínimo de datos necesarios, ids seudonimizados, acceso restringido a tutores autorizados
y una política de retención; el repositorio público nunca debe contener datos reales.

**4. Alcance del modelo.** Mide asociaciones, no causas, y se entrenó con una sola cohorte.
Antes de usarlo habría que probar las intervenciones con un piloto (grupo con acción vs. grupo
sin acción) para saber si el mensaje realmente reduce el abandono.
        """
    )
