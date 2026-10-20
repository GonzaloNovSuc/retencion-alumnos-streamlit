"""
generar_datos.py
================
Genera un dataset SINTÉTICO de alumnos para el proyecto de retención.

¿Por qué sintético?
    El autor no cuenta con una base real extensa (su trabajo no genera datasets
    de cientos de filas), así que se simula una población con la misma
    estructura que tendría un registro académico real. NINGÚN dato proviene de
    personas reales ni contiene nombres, correos o teléfonos.

Uso (con el entorno activo):
    python generar_datos.py

Es reproducible: la semilla fija (SEMILLA) produce siempre el mismo CSV.
"""

import numpy as np
import pandas as pd

SEMILLA = 42            # Semilla del generador aleatorio -> resultados repetibles
N_ALUMNOS = 520         # Filas "base" (luego se añaden duplicados a propósito)
RUTA_SALIDA = "data/dataset_alumnos_riesgo_abandono.csv"

rng = np.random.default_rng(SEMILLA)

# ---------------------------------------------------------------------------
# 1. SEGMENTOS LATENTES
# Cada alumno pertenece a un perfil de comportamiento. De ese perfil dependen
# sus hábitos de estudio y, por tanto, su probabilidad de abandonar.
# Parámetros por segmento:
#   peso: proporción en la población
#   horas / asist / entregas / dias: medias de cada hábito
#   p_abandono: probabilidad base de abandonar
# ---------------------------------------------------------------------------
SEGMENTOS = {
    "Comprometido":      dict(peso=0.35, horas=12, asist=93, entregas=92, dias=22, p_abandono=0.05),
    "Intermitente":      dict(peso=0.30, horas=7,  asist=78, entregas=70, dias=13, p_abandono=0.25),
    "Sobrecargado":      dict(peso=0.20, horas=4,  asist=70, entregas=60, dias=10, p_abandono=0.45),
    "Desconectado":      dict(peso=0.15, horas=2,  asist=48, entregas=35, dias=4,  p_abandono=0.75),
}

nombres = list(SEGMENTOS.keys())
pesos = [SEGMENTOS[s]["peso"] for s in nombres]
segmento = rng.choice(nombres, size=N_ALUMNOS, p=pesos)


def por_segmento(clave):
    """Devuelve un arreglo con el parámetro `clave` del segmento de cada alumno."""
    return np.array([SEGMENTOS[s][clave] for s in segmento])


# ---------------------------------------------------------------------------
# 2. VARIABLES NUMÉRICAS (hábitos de estudio)
# Se muestrean alrededor de la media del segmento y se acotan a rangos válidos.
# ---------------------------------------------------------------------------
horas_estudio = np.clip(rng.normal(por_segmento("horas"), 2.5), 0, 30).round(1)
asistencia = np.clip(rng.normal(por_segmento("asist"), 8), 10, 100).round(1)
entregas = np.clip(rng.normal(por_segmento("entregas"), 10), 0, 100).round(1)
dias_activo = np.clip(rng.normal(por_segmento("dias"), 3), 0, 30).round().astype(int)

# Promedio parcial (0-10) correlacionado con entregas y asistencia
promedio = np.clip(
    4 + entregas * 0.03 + asistencia * 0.025 + rng.normal(0, 0.6, N_ALUMNOS), 0, 10
).round(1)

edad = np.clip(rng.normal(29, 6, N_ALUMNOS), 21, 55).round().astype(int)

# ---------------------------------------------------------------------------
# 3. VARIABLES CATEGÓRICAS
# ---------------------------------------------------------------------------
programa = rng.choice(["Maestría", "Diplomado", "Curso corto"], N_ALUMNOS, p=[0.5, 0.3, 0.2])
modalidad = rng.choice(["En línea", "Presencial", "Híbrida"], N_ALUMNOS, p=[0.5, 0.2, 0.3])
canal = rng.choice(["Recomendación", "Redes sociales", "Sitio web", "Empresa"], N_ALUMNOS,
                   p=[0.35, 0.25, 0.25, 0.15])
# Los "Sobrecargado" casi siempre trabajan tiempo completo
trabaja = np.where(
    (segmento == "Sobrecargado") | (rng.random(N_ALUMNOS) < 0.5), "Sí", "No"
)
beca = rng.choice(["Sí", "No"], N_ALUMNOS, p=[0.25, 0.75])

# ---------------------------------------------------------------------------
# 4. VARIABLE OBJETIVO: abandono (Sí/No)
# Probabilidad = base del segmento, ajustada por beca (protege) y trabajo (riesgo).
# ---------------------------------------------------------------------------
p = por_segmento("p_abandono").astype(float)
p = p - 0.07 * (beca == "Sí") + 0.04 * (trabaja == "Sí")
p = np.clip(p, 0.01, 0.95)
abandono = np.where(rng.random(N_ALUMNOS) < p, "Sí", "No")

# ---------------------------------------------------------------------------
# 5. ENSAMBLAR DATAFRAME con ID anónimo (código, no nombre)
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "id_alumno": [f"AL{i:04d}" for i in range(1, N_ALUMNOS + 1)],
    "edad": edad,
    "programa": programa,
    "modalidad": modalidad,
    "canal_ingreso": canal,
    "trabaja": trabaja,
    "beca": beca,
    "horas_estudio_sem": horas_estudio,
    "asistencia_pct": asistencia,
    "entregas_a_tiempo_pct": entregas,
    "dias_activo_plataforma_mes": dias_activo,
    "promedio_parcial": promedio,
    "segmento": segmento,
    "abandono": abandono,
})

# ---------------------------------------------------------------------------
# 6. "SUCIEDAD" INTENCIONAL (para que la pestaña Limpiar tenga trabajo real)
#    - nulos en 3 columnas
#    - registros duplicados
#    - valores imposibles (asistencia > 100) -> dudas sobre calidad
# ---------------------------------------------------------------------------
def poner_nulos(col, fraccion):
    """Convierte una fracción aleatoria de la columna en valores faltantes."""
    idx = rng.choice(df.index, size=int(len(df) * fraccion), replace=False)
    df.loc[idx, col] = np.nan

poner_nulos("promedio_parcial", 0.06)
poner_nulos("horas_estudio_sem", 0.04)
poner_nulos("canal_ingreso", 0.03)

# 3 valores de asistencia imposibles (captura errónea)
idx_mal = rng.choice(df.index, size=3, replace=False)
df.loc[idx_mal, "asistencia_pct"] = [105.0, 112.5, 130.0]

# 10 filas duplicadas exactas
df = pd.concat([df, df.sample(10, random_state=SEMILLA)], ignore_index=True)
df = df.sample(frac=1, random_state=SEMILLA).reset_index(drop=True)  # mezclar orden

df.to_csv(RUTA_SALIDA, index=False, encoding="utf-8")
print(f"Dataset generado: {RUTA_SALIDA}  ->  {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"Tasa de abandono: {(df['abandono'] == 'Sí').mean():.1%}")
