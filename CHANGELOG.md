# Historial de cambios del proyecto (CHANGELOG)

## Actividad 5. Gestión de Proyectos de Inteligencia Artificial

**Proyecto:** Predicción de la concentración de partículas PM2.5 en la Ciudad de México mediante aprendizaje automático.

Este archivo documenta los principales cambios realizados durante el desarrollo del proyecto, desde la creación del repositorio hasta la validación independiente de los modelos entrenados.

Su propósito es mantener un registro de las modificaciones, las decisiones técnicas y los resultados obtenidos, facilitando la trazabilidad y reproducción de los experimentos.

Las versiones 0.1.0 a 0.8.0 representan etapas documentales del proyecto. No corresponden necesariamente a etiquetas de versión publicadas en Git. Las fechas exactas de los cambios pueden consultarse en el historial de commits de GitHub.

---

## [0.1.0] - Creación y configuración inicial del proyecto

### Repositorio y entorno de trabajo

- Creación del repositorio `Actividad-5_ABG` en GitHub.
- Organización inicial mediante las carpetas `datos/` y `fuentes/`.
- Incorporación de los archivos de documentación `README.md` y `CHANGELOG.md`.
- Creación del notebook `fuentes/entrena.ipynb` para desarrollar los experimentos en Google Colab.
- Configuración del entorno de Python y las bibliotecas necesarias para el análisis de datos y el aprendizaje automático.
- Establecimiento de una semilla aleatoria fija de 42 para favorecer la reproducibilidad.

### Incorporación del dataset original

- Selección de un conjunto de datos públicos asociado con la Red Automática de Monitoreo Atmosférico (RAMA) de la Ciudad de México.
- Incorporación del archivo CSV original a la carpeta `datos/`.
- Conservación del archivo original sin modificar sus mediciones.
- Identificación de 3,073 registros diarios y nueve columnas.
- Verificación del periodo comprendido entre el 1 de enero de 2015 y el 31 de mayo de 2023.

---

## [0.2.0] - Diagnóstico de calidad y análisis exploratorio

### Evaluación inicial de los datos

- Revisión de la estructura y los tipos de datos.
- Corrección de la interpretación de fechas mediante el formato `YYYY-MM-DD`.
- Confirmación de que no existen fechas duplicadas ni interrupciones en la secuencia diaria.
- Identificación de cero valores faltantes y cero registros duplicados.
- Verificación de que no existen concentraciones negativas ni valores iguales a cero.
- Cálculo de estadísticas descriptivas para los ocho contaminantes.
- Identificación de posibles valores atípicos mediante el método del rango intercuartílico (IQR).

### Decisiones de preparación

Se decidió conservar inicialmente los valores atípicos porque, en un conjunto de datos de contaminación atmosférica, las concentraciones elevadas pueden representar episodios reales y no necesariamente errores de medición.

Por este motivo, no se realizaron eliminaciones automáticas basadas únicamente en criterios estadísticos.

### Análisis temporal de PM2.5

- Visualización de las concentraciones diarias de PM2.5.
- Cálculo de un promedio móvil de 30 días.
- Comparación de promedios anuales.
- Análisis del comportamiento mensual.
- Identificación de los diez días con mayores concentraciones.
- Elaboración de diagramas de caja para comparar la distribución anual.

El análisis permitió identificar variaciones temporales y posibles patrones estacionales, que posteriormente se consideraron para la construcción de variables predictoras.

---

## [0.3.0] - Preparación de datos e ingeniería de características

### Construcción del conjunto para modelado

- Ordenamiento cronológico de los registros.
- Creación de la variable objetivo `PM25_siguiente_dia`.
- Incorporación de rezagos históricos de PM2.5 de 1, 2, 3 y 7 días.
- Cálculo de promedios móviles de 7 y 30 días.
- Generación de variables relacionadas con el calendario.
- Incorporación de representaciones cíclicas del mes mediante funciones seno y coseno.
- Selección de 17 variables predictoras.

### Tratamiento de registros incompletos

Se excluyeron 30 registros como consecuencia de la construcción de variables históricas y de la ausencia de la concentración objetivo correspondiente al último día.

Después de las transformaciones, el conjunto preparado quedó integrado por 3,043 registros y 20 columnas.

### División de entrenamiento y prueba

Se realizó una separación cronológica de los datos:

| Conjunto | Registros | Proporción |
|---|---:|---:|
| Entrenamiento | 2,434 | 80% |
| Prueba | 609 | 20% |

Esta decisión permitió reservar las observaciones más recientes para evaluar los modelos después de seleccionar sus hiperparámetros.

---

## [0.4.0] - Reproducibilidad y versionamiento de datos

### Script de preparación

- Creación del archivo `fuentes/datos_prep.py`.
- Implementación de funciones para validar registros, generar variables y dividir cronológicamente los datos.
- Importación y ejecución del script desde Google Colab.
- Comparación de los resultados del script con los obtenidos previamente en el notebook.
- Confirmación de coincidencia exacta entre ambos procedimientos.
- Verificación de que el dataset original permaneció sin modificaciones.

### Generación del dataset preparado

- Exportación del conjunto preparado al archivo `datos/datos_limp.csv`.
- Conservación de 3,043 registros y 20 columnas.
- Verificación de fechas, estructura y equivalencia numérica.
- Cálculo de la huella digital SHA-256.
- Incorporación del archivo preparado al repositorio GitHub.

**SHA-256 de `datos_limp.csv`:**

`93c76aa30597174fce7bbd49092aafdd586bd56786415091bb3e79177dab1d8b`

Esta huella permite identificar el contenido exacto del archivo y comprobar si posteriormente ha sido modificado.

---

## [0.5.0] - Configuración de MLflow y entrenamiento inicial

### Configuración del seguimiento de experimentos

- Configuración de MLflow 3.17.1 en Google Colab.
- Creación del experimento `RAMA_PM25_Actividad5`.
- Configuración de almacenamiento local mediante SQLite.
- Registro de una ejecución inicial para comprobar el funcionamiento de MLflow.
- Verificación del registro de parámetros, métricas y artefactos.
- Generación de un primer respaldo del seguimiento.

### Entrenamiento inicial de modelos

Se seleccionaron dos algoritmos de regresión:

- Ridge Regression.
- Random Forest Regressor.

Ambos se evaluaron utilizando cinco particiones de validación cruzada temporal mediante `TimeSeriesSplit`, con una separación de una observación entre entrenamiento y validación.

Los resultados iniciales fueron:

| Modelo | MAE promedio | RMSE promedio | R² promedio |
|---|---:|---:|---:|
| Ridge Regression | 4.3075 | 5.8610 | 0.5253 |
| Random Forest Regressor | 4.4741 | 5.9944 | 0.4965 |

Ridge Regression obtuvo inicialmente un menor error promedio y un tiempo de evaluación considerablemente inferior al de Random Forest.

Estos resultados se utilizaron como referencia para evaluar posteriormente las mejoras obtenidas mediante Grid Search.

---

## [0.6.0] - Ajuste de hiperparámetros y evaluación final

### Grid Search con validación cruzada temporal

- Implementación de `GridSearchCV` para ambos modelos.
- Utilización de MAE como métrica principal de selección.
- Registro complementario de RMSE y R².
- Evaluación de cuatro configuraciones de Ridge Regression.
- Evaluación de 18 configuraciones de Random Forest Regressor.
- Realización de 110 ajustes de validación cruzada, además de los reajustes finales de las búsquedas.
- Registro de los resultados de ajuste en MLflow.

### Hiperparámetros seleccionados

| Modelo | Configuración seleccionada |
|---|---|
| Ridge Regression | `alpha=100.0` |
| Random Forest Regressor | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=5` |

### Resultados después del ajuste

| Modelo | MAE CV | RMSE CV | R² CV |
|---|---:|---:|---:|
| Ridge Regression | 4.2367 | 5.7668 | 0.5405 |
| Random Forest Regressor | 4.3197 | 5.8419 | 0.5264 |

El ajuste redujo el MAE promedio de Ridge aproximadamente un 1.64% y el de Random Forest un 3.45% respecto a sus configuraciones iniciales.

### Evaluación sobre el conjunto de prueba

Los modelos ajustados se evaluaron sobre los 609 registros reservados.

También se incorporó un modelo de referencia denominado persistencia, que utiliza la concentración del día actual como estimación para el día siguiente.

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge Regression | 3.6493 | 4.9894 | 0.4861 |
| Random Forest Regressor | 3.6743 | 5.1004 | 0.4629 |
| Persistencia | 4.1401 | 5.9459 | 0.2701 |

Ridge Regression obtuvo el menor error promedio de prueba y se seleccionó como modelo candidato principal del experimento.

Sin embargo, la diferencia de MAE respecto a Random Forest fue pequeña y no se estableció su significancia estadística.

### Análisis de concentraciones elevadas

Se utilizó el percentil 90 del conjunto de entrenamiento para identificar concentraciones elevadas de PM2.5.

El umbral calculado fue de 32.568 en las unidades del archivo y permitió identificar 20 observaciones elevadas dentro del conjunto de prueba.

| Modelo | MAE en concentraciones elevadas |
|---|---:|
| Persistencia | 9.9416 |
| Random Forest Regressor | 10.7245 |
| Ridge Regression | 10.7461 |

Los resultados mostraron que ambos modelos de aprendizaje automático presentaron mayores errores durante las concentraciones elevadas y que, en ese subconjunto, la persistencia obtuvo el menor MAE.

Este hallazgo se identificó como una limitación relevante para cualquier posible aplicación operativa.

---

## [0.7.0] - Registro de modelos y recuperación de incidencia en MLflow

### Registro de resultados experimentales

- Registro de los mejores hiperparámetros de Ridge Regression y Random Forest.
- Incorporación de métricas de validación cruzada y prueba.
- Almacenamiento de archivos CSV con predicciones.
- Generación de resúmenes JSON de los modelos.
- Registro de modelos entrenados mediante MLflow.
- Verificación de recuperación y equivalencia de predicciones.

### Incidencia técnica de Random Forest

Durante el primer intento de almacenamiento de Random Forest se presentó un error de serialización relacionado con la validación de tipos utilizados por MLflow.

La incidencia produjo una ejecución y un registro de modelo con estado `FAILED`.

Se comprobó que las métricas y los hiperparámetros originales permanecían disponibles, por lo que no fue necesario repetir el entrenamiento ni Grid Search.

Para resolver la incidencia:

- Se identificó el registro incompleto del modelo.
- Se conservó la ejecución fallida como parte del historial.
- Se utilizó `cloudpickle` para serializar el modelo.
- Se recuperó el registro mediante una nueva ejecución vinculada con la original.
- Se verificó que el modelo almacenado podía cargarse y reproducir las predicciones.

### Ejecuciones definitivas

| Modelo | Run ID definitivo |
|---|---|
| Ridge Regression | `73339b012c7640e6bb99744dc3018c2d` |
| Random Forest Regressor | `3c164f2beea644eba2f4068cb3b7beb6` |

Ambas ejecuciones definitivas terminaron con estado `FINISHED`.

Los modelos almacenados se recuperaron correctamente y generaron predicciones equivalentes a las originales.

### Respaldo de experimentos

- Verificación de las métricas y los artefactos de evaluación.
- Comprobación de la huella SHA-256 del dataset preparado.
- Generación de una copia consistente de la base de datos SQLite.
- Creación del archivo `mlflow_rama_final.zip`.
- Verificación estructural del ZIP y descarga desde Google Colab.

La restauración integral del respaldo en un entorno nuevo permanece pendiente de comprobación.

---

## [0.8.0] - Entrenamiento independiente y documentación del entorno

### Script de entrenamiento

- Creación del archivo `fuentes/train.py`.
- Implementación de la carga y preparación de datos mediante `datos_prep.py`.
- Incorporación de Ridge Regression y Random Forest Regressor.
- Reproducción de las configuraciones de Grid Search.
- Aplicación de validación cruzada temporal.
- Cálculo de MAE, RMSE y R².
- Registro independiente de parámetros, métricas, modelos y artefactos en MLflow.

El script utiliza el experimento:

`RAMA_PM25_TrainScript`

Este experimento se mantiene separado del utilizado originalmente en el notebook.

### Validación independiente

Se ejecutó `train.py` desde una copia independiente del repositorio dentro de Google Colab.

La ejecución terminó con código de salida 0.

Posteriormente se compararon las métricas del script con las obtenidas en el notebook original.

Las diferencias fueron inferiores a 0.0001 en las métricas contrastadas, considerando el redondeo de los valores originales.

También se verificó que MLflow registrara dos ejecuciones con estado `FINISHED` y que ambos modelos pudieran recuperarse.

Esta validación confirmó la reproducibilidad dentro del entorno utilizado, aunque no constituye una prueba completa de portabilidad en otro sistema.

### Dependencias y documentación

- Creación de `requirements.txt`.
- Documentación de las versiones principales de las bibliotecas.
- Identificación de `cloudpickle==3.1.2`.
- Actualización del archivo `README.md`.
- Incorporación de resultados finales, instrucciones de reproducción y limitaciones del proyecto.
- Consolidación del notebook `fuentes/entrena.ipynb` con 14 celdas definitivas.
- Eliminación de celdas temporales utilizadas para las comprobaciones auxiliares.

---

## Actividades pendientes de cierre

Las siguientes actividades corresponden a la preparación final de los entregables:

- Comprobar la instalación de las dependencias en un entorno limpio.
- Verificar la restauración integral del respaldo de MLflow.
- Obtener y organizar las capturas de pantalla de los experimentos.
- Confirmar documentalmente las unidades y la metodología de agregación del dataset original.
- Elaborar el reporte técnico académico con metodología, resultados, interpretación, limitaciones, deuda técnica y recomendaciones MLOps.
- Integrar las referencias de información consultada en formato APA.
- Preparar el archivo ZIP final y comprobar que contiene todos los entregables solicitados.

---

## Consideraciones de reproducibilidad y trazabilidad

Los archivos originales y preparados se conservan por separado para facilitar la identificación de las transformaciones realizadas.

La preparación puede reproducirse mediante `datos_prep.py`, mientras que el entrenamiento y registro de modelos pueden ejecutarse mediante `train.py`.

Los experimentos utilizan una semilla fija de 42, una división cronológica y validación cruzada temporal.

Las versiones principales de las dependencias se documentan en `requirements.txt`.

El registro de parámetros, métricas y modelos se realizó mediante MLflow, conservando los identificadores de las ejecuciones y los artefactos asociados.

La incidencia de serialización de Random Forest se mantuvo documentada para preservar el historial técnico del proyecto.

La validación independiente confirmó la reproducción de los resultados en el entorno utilizado, mientras que la portabilidad completa y la restauración del respaldo continúan como verificaciones pendientes.
