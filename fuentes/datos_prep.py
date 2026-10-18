# TAREA 5
# Archivo: datos_prep.py
# Proyecto: Predicción de PM2.5 en la Ciudad de México

# Funciones para validar y preparar el dataset RAMA.
# El archivo original no se modifica.

import numpy as np
import pandas as pd

SEMILLA = 42

VARIABLES_PREDICTORAS = [
    "CO", "NO", "NO2", "NOX", "O3", "PM10", "PM25", "SO2",
    "PM25_rezago_1d",
    "PM25_rezago_2d",
    "PM25_rezago_3d",
    "PM25_rezago_7d",
    "PM25_media_7d",
    "PM25_media_30d",
    "mes_seno",
    "mes_coseno",
    "dia_semana_prediccion"
]

VARIABLE_OBJETIVO = "PM25_siguiente_dia"

# Función que comprueba la estructura y continuidad del dataset RAMA.
# Devuelve una copia ordenada cronológicamente.
def validar_datos(df):
    datos = df.copy()

    columnas_requeridas = [
        "fecha", "CO", "NO", "NO2", "NOX",
        "O3", "PM10", "PM25", "SO2"
    ]

    faltantes = [
        col for col in columnas_requeridas
        if col not in datos.columns
    ]

    if faltantes:
        raise ValueError(
            f"Columnas no encontradas: {faltantes}"
        )

    datos["fecha"] = pd.to_datetime(
        datos["fecha"],
        format="%Y-%m-%d",
        errors="raise"
    )

    datos = datos.sort_values(
        "fecha"
    ).reset_index(drop=True)

    if datos["fecha"].duplicated().any():
        raise ValueError("Existen fechas duplicadas.")

    if not datos["fecha"].diff().dropna().eq(
        pd.Timedelta(days=1)
    ).all():
        raise ValueError(
            "La serie presenta días sin registro."
        )

    columnas_numericas = columnas_requeridas[1:]

    if datos[columnas_numericas].isna().any().any():
        raise ValueError(
            "Existen valores faltantes en los contaminantes."
        )

    if (datos[columnas_numericas] < 0).any().any():
        raise ValueError(
            "Existen concentraciones negativas."
        )

    return datos

# Función que genera variables históricas, temporales y el objetivo
# de predicción de PM2.5 del día siguiente.
def crear_variables(df):
    datos = df.copy()

    # Variable objetivo: concentración del día siguiente.
    datos[VARIABLE_OBJETIVO] = datos["PM25"].shift(-1)

    # Concentraciones históricas.
    for dias in [1, 2, 3, 7]:
        datos[f"PM25_rezago_{dias}d"] = (
            datos["PM25"].shift(dias)
        )

    # Promedios móviles hasta el día actual.
    datos["PM25_media_7d"] = (
        datos["PM25"]
        .rolling(window=7, min_periods=7)
        .mean()
    )

    datos["PM25_media_30d"] = (
        datos["PM25"]
        .rolling(window=30, min_periods=30)
        .mean()
    )

    # Variables temporales del día que se desea predecir.
    fecha_prediccion = (
        datos["fecha"] + pd.Timedelta(days=1)
    )

    datos["mes_prediccion"] = fecha_prediccion.dt.month

    datos["dia_semana_prediccion"] = (
        fecha_prediccion.dt.dayofweek
    )

    # Representación cíclica del mes.
    datos["mes_seno"] = np.sin(
        2 * np.pi * datos["mes_prediccion"] / 12
    )

    datos["mes_coseno"] = np.cos(
        2 * np.pi * datos["mes_prediccion"] / 12
    )

    return datos

# Función que ejecuta la validación y preparación completa.
# Devuelve el dataset preparado, la lista de variables
# predictoras y el nombre de la variable objetivo.
def preparar_dataset(df):
    datos = validar_datos(df)
    datos = crear_variables(datos)

    # Se excluyen únicamente los registros sin historial
    # suficiente o sin valor objetivo del día siguiente.
    columnas_modelo = (
        VARIABLES_PREDICTORAS + [VARIABLE_OBJETIVO]
    )

    datos = (
        datos.dropna(subset=columnas_modelo)
        .reset_index(drop=True)
    )

    return datos, VARIABLES_PREDICTORAS, VARIABLE_OBJETIVO

# Función que divide cronológicamente el conjunto preparado.
# No realiza una separación aleatoria.
def dividir_datos(datos, proporcion=0.80):  
    if not 0 < proporcion < 1:
        raise ValueError(
            "La proporción debe estar entre 0 y 1."
        )

    punto_corte = int(len(datos) * proporcion)

    entrenamiento = datos.iloc[:punto_corte].copy()
    prueba = datos.iloc[punto_corte:].copy()

    X_train = entrenamiento[VARIABLES_PREDICTORAS].copy()
    y_train = entrenamiento[VARIABLE_OBJETIVO].copy()

    X_test = prueba[VARIABLES_PREDICTORAS].copy()
    y_test = prueba[VARIABLE_OBJETIVO].copy()

    return X_train, X_test, y_train, y_test
