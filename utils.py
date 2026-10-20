"""
utils.py
========
Funciones compartidas por todas las pestañas: cargar datos, limpiarlos y
entrenar el modelo. Se centralizan aquí para que cada pestaña use EXACTAMENTE
los mismos datos limpios y las mismas probabilidades (una sola fuente de verdad).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ruta construida a partir de la ubicación de este archivo, así la app funciona
# sin importar desde qué carpeta se ejecute `streamlit run`.
RUTA_CSV = Path(__file__).parent / "data" / "dataset_alumnos_riesgo_abandono.csv"

# Variables que usará el modelo (se excluyen id_alumno, que no tiene significado,
# y segmento, que ya es un resumen de las demás variables de hábitos).
COLS_NUMERICAS = [
    "edad", "horas_estudio_sem", "asistencia_pct", "entregas_a_tiempo_pct",
    "dias_activo_plataforma_mes", "promedio_parcial",
]
COLS_CATEGORICAS = ["programa", "modalidad", "canal_ingreso", "trabaja", "beca"]


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    """Lee el CSV crudo. `cache_data` evita releerlo en cada interacción."""
    return pd.read_csv(RUTA_CSV)


@st.cache_data
def limpiar_datos(df: pd.DataFrame):
    """
    Aplica las decisiones de limpieza y devuelve (df_limpio, bitacora).

    La bitácora es una lista de dicts con lo que se hizo, para mostrarlo en la
    pestaña 2 y justificar cada decisión.
    """
    limpio = df.copy()
    bitacora = []

    # 1) DUPLICADOS: filas idénticas = el mismo alumno capturado dos veces.
    #    Se eliminan para no contarlo doble en métricas ni en el modelo.
    n_dup = int(limpio.duplicated().sum())
    limpio = limpio.drop_duplicates().reset_index(drop=True)
    bitacora.append(dict(problema="Registros duplicados", cantidad=n_dup,
                         decision="Eliminar (conservar la primera aparición)"))

    # 2) VALORES IMPOSIBLES: asistencia mayor a 100 % no existe.
    #    Se topan en 100 (probable error de captura de "asistencia completa").
    n_imp = int((limpio["asistencia_pct"] > 100).sum())
    limpio["asistencia_pct"] = limpio["asistencia_pct"].clip(upper=100)
    bitacora.append(dict(problema="Asistencia > 100 %", cantidad=n_imp,
                         decision="Topar en 100 %"))

    # 3) NULOS NUMÉRICOS: imputar con la MEDIANA (robusta a valores extremos)
    #    en lugar de borrar filas, porque se perdería información de alumnos.
    for col in ["promedio_parcial", "horas_estudio_sem"]:
        n_nulos = int(limpio[col].isna().sum())
        limpio[col] = limpio[col].fillna(limpio[col].median())
        bitacora.append(dict(problema=f"Nulos en {col}", cantidad=n_nulos,
                             decision="Imputar con la mediana"))

    # 4) NULOS CATEGÓRICOS: no se inventa un canal; se marca como "Sin dato".
    n_canal = int(limpio["canal_ingreso"].isna().sum())
    limpio["canal_ingreso"] = limpio["canal_ingreso"].fillna("Sin dato")
    bitacora.append(dict(problema="Nulos en canal_ingreso", cantidad=n_canal,
                         decision="Etiquetar como 'Sin dato'"))

    return limpio, pd.DataFrame(bitacora)


def _armar_pipeline() -> Pipeline:
    """
    Pipeline = preprocesamiento + modelo en un solo objeto.
    - Numéricas: se estandarizan (media 0, desv. 1) para que los coeficientes
      sean comparables entre sí.
    - Categóricas: one-hot (una columna 0/1 por categoría).
    - Modelo: regresión logística (sencilla e interpretable).
    """
    pre = ColumnTransformer([
        ("num", StandardScaler(), COLS_NUMERICAS),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), COLS_CATEGORICAS),
    ])
    return Pipeline([("pre", pre), ("modelo", LogisticRegression(max_iter=1000))])


@st.cache_data
def entrenar_modelo(df_limpio: pd.DataFrame):
    """
    Entrena la regresión logística y devuelve un dict con:
      - metricas: accuracy y AUC en un conjunto de prueba (20 %) que el modelo no vio
      - coeficientes: peso de cada variable (positivo = aumenta riesgo de abandono)
      - probabilidades: probabilidad de abandono de CADA alumno, calculada con
        validación cruzada (cada alumno es predicho por un modelo que no lo vio)
    """
    X = df_limpio[COLS_NUMERICAS + COLS_CATEGORICAS]
    y = (df_limpio["abandono"] == "Sí").astype(int)  # 1 = abandonó

    # Evaluación honesta: separar prueba (20 %), manteniendo la proporción de abandono
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    modelo = _armar_pipeline().fit(X_tr, y_tr)
    prob_te = modelo.predict_proba(X_te)[:, 1]
    metricas = dict(
        accuracy=accuracy_score(y_te, prob_te >= 0.5),
        auc=roc_auc_score(y_te, prob_te),
        n_train=len(X_tr), n_test=len(X_te),
        base=float(1 - y_te.mean()),  # accuracy de "predecir que nadie abandona"
    )

    # Probabilidades fuera de muestra para todos los alumnos (5 particiones)
    probabilidades = cross_val_predict(
        _armar_pipeline(), X, y, cv=5, method="predict_proba"
    )[:, 1]

    # Modelo final con todos los datos solo para leer los coeficientes
    final = _armar_pipeline().fit(X, y)
    nombres = final.named_steps["pre"].get_feature_names_out()
    coef = pd.DataFrame({
        "variable": [n.split("__", 1)[1] for n in nombres],
        "coeficiente": final.named_steps["modelo"].coef_[0],
    })
    coef["peso_absoluto"] = coef["coeficiente"].abs()
    coef = coef.sort_values("peso_absoluto", ascending=False).reset_index(drop=True)

    return dict(metricas=metricas, coeficientes=coef, probabilidades=probabilidades)


def nivel_riesgo(prob: float) -> str:
    """Convierte una probabilidad en un nivel de riesgo accionable."""
    if prob >= 0.50:
        return "Alto"
    if prob >= 0.25:
        return "Medio"
    return "Bajo"


def datos_con_riesgo(df_limpio: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el df limpio con columnas `prob_abandono` y `nivel_riesgo`."""
    resultado = entrenar_modelo(df_limpio)
    out = df_limpio.copy()
    out["prob_abandono"] = resultado["probabilidades"]
    out["nivel_riesgo"] = out["prob_abandono"].apply(nivel_riesgo)
    return out
