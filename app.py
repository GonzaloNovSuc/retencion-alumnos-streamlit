"""
app.py · Punto de entrada de la aplicación.

Ejecutar con el entorno activo:
    streamlit run app.py

Estructura: este archivo solo configura la página, carga/limpia los datos UNA vez
y reparte el trabajo a un módulo por pestaña (carpeta tabs/).
"""

import streamlit as st

from tabs import decisiones, descriptiva, entender, limpiar, predictiva, prescriptiva
from utils import cargar_datos, limpiar_datos

# --- Configuración general de la página (debe ser la primera llamada a st) ---
st.set_page_config(
    page_title="Retención de alumnos",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Retención de alumnos: ¿quién está en riesgo de abandonar?")
st.markdown(
    """
**Contexto:** programa de posgrado y cursos cortos.
**Pregunta guía:** ¿Qué alumnos tienen mayor probabilidad de abandonar y qué apoyo
deberíamos ofrecer a cada grupo?

> ℹ️ **Datos sintéticos.** El autor no dispone de una base real extensa, por lo que el
> dataset fue generado con `generar_datos.py` (520 alumnos más duplicados, nulos y errores
> intencionales). No contiene datos personales.
"""
)

# --- Datos: crudos (para entender/limpiar) y limpios (para el resto) ---
df_crudo = cargar_datos()
df_limpio, bitacora = limpiar_datos(df_crudo)

# --- Seis pestañas, una por fase del framework ---
t1, t2, t3, t4, t5, t6 = st.tabs([
    "1. Entender", "2. Limpiar", "3. Descriptiva",
    "4. Predictiva", "5. Prescriptiva", "6. Decisiones",
])

with t1:
    entender.mostrar(df_crudo)            # Se analiza el dato tal como llegó
with t2:
    limpiar.mostrar(df_crudo, df_limpio, bitacora)
with t3:
    descriptiva.mostrar(df_limpio)        # Desde aquí se usa el dato ya limpio
with t4:
    predictiva.mostrar(df_limpio)
with t5:
    prescriptiva.mostrar(df_limpio)
with t6:
    decisiones.mostrar(df_limpio)
