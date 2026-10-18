# TAREA 5. 
# Archivo: fuentes/train.py
# Proyecto: Predicción de PM2.5 en la Ciudad de México

# Entrenamiento, ajuste, evaluación y registro en MLflow.
# Ejecución desde la raíz del repositorio:
# python fuentes/train.py

import hashlib
import json
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datos_prep import preparar_dataset, dividir_datos

# Configuración general
SEMILLA = 42

RAIZ = Path(__file__).resolve().parent.parent

RUTA_DATOS = (
    RAIZ / "datos" /
    "ebc079e5-bd11-4830-b595-14292f753575.csv"
)

RUTA_RESULTADOS = RAIZ / "resultados"

RUTA_MLFLOW = RAIZ / "mlflow_train"

NOMBRE_EXPERIMENTO = "RAMA_PM25_TrainScript"

METRICAS = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}

CONFIGURACIONES = {
    "Ridge Regression": {
        "estimador": Pipeline([
            ("escalador", StandardScaler()),
            ("modelo", Ridge())
        ]),
        "parametros": {
            "modelo__alpha": [0.1, 1.0, 10.0, 100.0]
        }
    },
    "Random Forest Regressor": {
        "estimador": Pipeline([
            ("modelo", RandomForestRegressor(
                random_state=SEMILLA,
                n_jobs=-1
            ))
        ]),
        "parametros": {
            "modelo__n_estimators": [100, 200],
            "modelo__max_depth": [5, 10, None],
            "modelo__min_samples_leaf": [1, 3, 5]
        }
    }
}

# Función que calcula las métricas de regresión 
def calcular_metricas(reales, estimados):
    return {
        "MAE": float(mean_absolute_error(reales, estimados)),
        "RMSE": float(
            np.sqrt(mean_squared_error(reales, estimados))
        ),
        "R2": float(r2_score(reales, estimados))
    }

# Función que obtiene la huella digital de un archivo.
def calcular_sha256(ruta):
    hash_archivo = hashlib.sha256()

    with open(ruta, "rb") as archivo:
        for bloque in iter(
            lambda: archivo.read(1024 * 1024), b""
        ):
            hash_archivo.update(bloque)

    return hash_archivo.hexdigest()

# Carga y preparación de datos
def cargar_datos():

    print("\n" + "=" * 65)
    print("1. CARGA Y PREPARACIÓN DE DATOS")
    print("=" * 65)

    if not RUTA_DATOS.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {RUTA_DATOS}"
        )

    originales = pd.read_csv(RUTA_DATOS)

    preparados, variables, objetivo = preparar_dataset(
        originales
    )

    X_train, X_test, y_train, y_test = dividir_datos(
        preparados,
        proporcion=0.80
    )

    assert len(preparados) == 3043
    assert len(variables) == 17
    assert len(X_train) == 2434
    assert len(X_test) == 609

    print("Registros originales:", len(originales))
    print("Registros preparados:", len(preparados))
    print("Variables predictoras:", len(variables))
    print("Entrenamiento:", len(X_train))
    print("Prueba:", len(X_test))

    return preparados, X_train, X_test, y_train, y_test

# Configuración de MLFLOW
def configurar_mlflow():

    print("\n" + "=" * 65)
    print("2. CONFIGURACIÓN DE MLFLOW")
    print("=" * 65)

    RUTA_RESULTADOS.mkdir(
        parents=True, exist_ok=True
    )

    RUTA_MLFLOW.mkdir(
        parents=True, exist_ok=True
    )

    ruta_db = RUTA_MLFLOW / "mlflow.db"
    ruta_artefactos = RUTA_MLFLOW / "artefactos"

    ruta_artefactos.mkdir(
        parents=True, exist_ok=True
    )

    mlflow.set_tracking_uri(
        f"sqlite:///{ruta_db.resolve()}"
    )

    experimento = mlflow.get_experiment_by_name(
        NOMBRE_EXPERIMENTO
    )

    if experimento is None:
        mlflow.create_experiment(
            NOMBRE_EXPERIMENTO,
            artifact_location=ruta_artefactos.resolve().as_uri()
        )

    mlflow.set_experiment(NOMBRE_EXPERIMENTO)

    print("Experimento:", NOMBRE_EXPERIMENTO)
    print("MLflow:", mlflow.__version__)
    print("Base de datos:", ruta_db)

# Entrenamiento y ajuste de modelos
def entrenar_modelos(X_train, X_test, y_train, y_test):

    print("\n" + "=" * 65)
    print("3. ENTRENAMIENTO Y GRID SEARCH")
    print("=" * 65)

    validacion = TimeSeriesSplit(
        n_splits=5,
        gap=1
    )

    resultados = []

    for nombre, configuracion in CONFIGURACIONES.items():

        print("\n" + "-" * 65)
        print("Modelo:", nombre)
        print("-" * 65)

        busqueda = GridSearchCV(
            estimator=configuracion["estimador"],
            param_grid=configuracion["parametros"],
            scoring=METRICAS,
            refit="MAE",
            cv=validacion,
            n_jobs=1,
            error_score="raise",
            return_train_score=True
        )

        inicio = time.perf_counter()

        busqueda.fit(X_train, y_train)

        tiempo = time.perf_counter() - inicio

        indice = busqueda.best_index_

        metricas_cv = {
            "CV_MAE": float(
                -busqueda.cv_results_[
                    "mean_test_MAE"
                ][indice]
            ),
            "CV_RMSE": float(
                -busqueda.cv_results_[
                    "mean_test_RMSE"
                ][indice]
            ),
            "CV_R2": float(
                busqueda.cv_results_[
                    "mean_test_R2"
                ][indice]
            )
        }

        modelo = busqueda.best_estimator_

        predicciones = modelo.predict(X_test)

        metricas_prueba = calcular_metricas(
            y_test,
            predicciones
        )

        nombre_corto = (
            "ridge_pm25"
            if nombre == "Ridge Regression"
            else "random_forest_pm25"
        )

        with mlflow.start_run(
            run_name=f"{nombre_corto}_GridSearch"
        ) as ejecucion:

            mlflow.set_tag("modelo", nombre)
            mlflow.set_tag("fase", "Entrenamiento reproducible")
            mlflow.set_tag("dataset", "RAMA_CDMX")

            mlflow.log_param("semilla", SEMILLA)
            mlflow.log_param("n_splits", 5)
            mlflow.log_param("gap", 1)
            mlflow.log_param("registros_train", len(X_train))
            mlflow.log_param("registros_test", len(X_test))

            for parametro, valor in busqueda.best_params_.items():
                mlflow.log_param(parametro, str(valor))

            mlflow.log_metrics(metricas_cv)

            mlflow.log_metric(
                "Tiempo_GridSearch_segundos",
                tiempo
            )

            mlflow.log_metric(
                "Test_MAE",
                metricas_prueba["MAE"]
            )

            mlflow.log_metric(
                "Test_RMSE",
                metricas_prueba["RMSE"]
            )

            mlflow.log_metric(
                "Test_R2",
                metricas_prueba["R2"]
            )

            firma = infer_signature(
                X_test,
                predicciones
            )

            # cloudpickle evita el error de serialización
            # que encontramos con Random Forest.

            modelo_mlflow = mlflow.sklearn.log_model(
                sk_model=modelo,
                name=nombre_corto,
                signature=firma,
                input_example=X_test.head(3),
                serialization_format="cloudpickle"
            )

            # Guardar tabla completa de Grid Search.
            tabla_cv = pd.DataFrame(
                busqueda.cv_results_
            )

            for columna in tabla_cv.columns:
                if columna.startswith("param_"):
                    tabla_cv[columna] = (
                        tabla_cv[columna].astype(str)
                    )

            ruta_cv = (
                RUTA_RESULTADOS /
                f"gridsearch_{nombre_corto}.csv"
            )

            tabla_cv.to_csv(
                ruta_cv,
                index=False
            )

            mlflow.log_artifact(
                str(ruta_cv),
                artifact_path="gridsearch"
            )

            # Guardar predicciones.
            ruta_predicciones = (
                RUTA_RESULTADOS /
                f"predicciones_{nombre_corto}.csv"
            )

            pd.DataFrame({
                "PM25_real": y_test.to_numpy(),
                "PM25_predicho": predicciones
            }).to_csv(
                ruta_predicciones,
                index=False
            )

            mlflow.log_artifact(
                str(ruta_predicciones),
                artifact_path="evaluacion"
            )

            # Verificar recuperación del modelo.
            modelo_recuperado = mlflow.sklearn.load_model(
                modelo_mlflow.model_uri
            )

            np.testing.assert_allclose(
                predicciones,
                modelo_recuperado.predict(X_test),
                rtol=1e-12,
                atol=1e-12
            )

            resultados.append({
                "Modelo": nombre,
                **metricas_cv,
                "Test_MAE": metricas_prueba["MAE"],
                "Test_RMSE": metricas_prueba["RMSE"],
                "Test_R2": metricas_prueba["R2"],
                "Tiempo_segundos": tiempo,
                "Mejores_parametros": str(
                    busqueda.best_params_
                ),
                "Run_ID": ejecucion.info.run_id,
                "Modelo_URI": modelo_mlflow.model_uri
            })

        print("Mejores parámetros:", busqueda.best_params_)
        print("MAE validación:", round(metricas_cv["CV_MAE"], 4))
        print("MAE prueba:", round(metricas_prueba["MAE"], 4))
        print("Modelo MLflow: VERIFICADO")

    return pd.DataFrame(resultados)

# Ejecución general
def main():

    np.random.seed(SEMILLA)

    print("=" * 65)
    print("TAREA 5. ENTRENAMIENTO REPRODUCIBLE RAMA")
    print("=" * 65)

    preparados, X_train, X_test, y_train, y_test = (
        cargar_datos()
    )

    configurar_mlflow()

    resultados = entrenar_modelos(
        X_train,
        X_test,
        y_train,
        y_test
    )

    resultados = resultados.sort_values(
        "Test_MAE"
    ).reset_index(drop=True)

    ruta_resumen = (
        RUTA_RESULTADOS / "resultados_modelos.csv"
    )

    resultados.to_csv(
        ruta_resumen,
        index=False
    )

    print("\n" + "=" * 65)
    print("4. RESULTADOS FINALES")
    print("=" * 65)

    print(
        resultados[
            [
                "Modelo",
                "CV_MAE",
                "Test_MAE",
                "Test_RMSE",
                "Test_R2"
            ]
        ].round(4).to_string(index=False)
    )

    print("\nSHA-256 dataset original:")
    print(calcular_sha256(RUTA_DATOS))

    print("\nArchivo de resultados:", ruta_resumen)
    print("Entrenamiento reproducible finalizado.")


if __name__ == "__main__":
    main()
