
# Actividad 5. Gestión de Proyectos de Inteligencia Artificial

## Predicción de partículas PM2.5 en la Ciudad de México mediante aprendizaje automático

### 1. Descripción del proyecto

En este proyecto se desarrolla una solución de aprendizaje automático para estimar la concentración de partículas PM2.5 del día siguiente en la Ciudad de México, utilizando registros históricos de contaminación atmosférica.

El propósito es analizar si la información de contaminantes y las mediciones de días anteriores permiten anticipar el comportamiento de las partículas PM2.5.

Para ello, se realiza la preparación y el versionamiento de los datos, así como el entrenamiento y la comparación de dos modelos de regresión: Ridge Regression y Random Forest Regressor.

Los experimentos se registrarán mediante MLflow, con el objetivo de facilitar la comparación de resultados y mantener la trazabilidad de los modelos.

### 2. Objetivo

Desarrollar y comparar dos modelos de aprendizaje automático capaces de estimar la concentración de PM2.5 del día siguiente, utilizando datos públicos de calidad del aire y aplicando procedimientos reproducibles de preparación, entrenamiento, evaluación y registro de experimentos.

### 3. Fuente y características del dataset

**Fuente:** Red Automática de Monitoreo Atmosférico (RAMA), Secretaría del Medio Ambiente de la Ciudad de México.

**Portal oficial:**
https://www.datos.cdmx.gob.mx/dataset/red-automatica-de-monitoreo-atmosferico

El archivo utilizado contiene registros diarios correspondientes al periodo del 1 de enero de 2015 al 31 de mayo de 2023.

| Característica | Dataset original | Dataset preparado |
|---|---|---|
| Registros | 3,073 | 3,043 |
| Columnas | 9 | 20 |
| Frecuencia | Diaria | Diaria |
| Valores faltantes | 0 | 0 |
| Fechas duplicadas | 0 | 0 |
| Periodo | 2015-01-01 a 2023-05-31 | 2015-01-30 a 2023-05-30 |

El dataset original incluye la fecha y ocho variables de contaminantes: CO, NO, NO2, NOX, O3, PM10, PM25 y SO2.

Las unidades de medición y la metodología de agregación del archivo deberán confirmarse con la documentación oficial antes de interpretar las concentraciones en términos ambientales.

### 4. Estructura del repositorio

```text
Actividad-5_ABG/
|
|-- datos/
|   |-- ebc079e5-bd11-4830-b595-14292f753575.csv
|   |-- datos_limp.csv
|
|-- fuentes/
|   |-- entrena.ipynb
|   |-- datos_prep.py
|   |-- train.py                  [pendiente]
|
|-- README.md
|-- CHANGELOG.md
```

**Descripción de los archivos:**

- `datos/ebc079e5-bd11-4830-b595-14292f753575.csv`: registros originales utilizados en el proyecto.
- `datos/datos_limp.csv`: conjunto preparado para el entrenamiento.
- `fuentes/entrena.ipynb`: notebook de Google Colab con el desarrollo experimental.
- `fuentes/datos_prep.py`: funciones de validación y preparación de datos.
- `fuentes/train.py`: script de entrenamiento y registro de experimentos, pendiente de implementación.
- `README.md`: descripción y guía de reproducción del proyecto.
- `CHANGELOG.md`: historial de cambios y decisiones realizadas.

### 5. Diagnóstico y preparación de datos

Se realizó un análisis inicial para verificar la estructura, la continuidad temporal y la calidad de los registros.

Los resultados mostraron que no existen valores faltantes, registros duplicados, fechas repetidas ni concentraciones negativas.

También se identificaron observaciones estadísticamente atípicas mediante el método del rango intercuartílico (IQR).

Estas observaciones se conservaron inicialmente porque podrían representar episodios reales de contaminación y no necesariamente errores de medición.

Posteriormente se realizó un análisis temporal de PM2.5, incluyendo su evolución diaria, promedios anuales, comportamiento mensual y distribución de valores.

### 6. Ingeniería de características

Para preparar el conjunto de entrenamiento se realizaron las siguientes transformaciones:

- Creación de la variable objetivo `PM25_siguiente_dia`.
- Incorporación de concentraciones históricas de PM2.5 con rezagos de 1, 2, 3 y 7 días.
- Cálculo de promedios móviles de 7 y 30 días.
- Generación de variables temporales relacionadas con el mes y el día de la semana.
- Representación cíclica del mes mediante funciones seno y coseno.

El conjunto final contiene 17 variables predictoras.

Se excluyeron 30 registros debido a la falta de historial suficiente para calcular determinadas variables y a la ausencia del valor objetivo del último día.

El dataset preparado contiene 3,043 registros y 20 columnas.

### 7. División de entrenamiento y prueba

Los datos se dividieron cronológicamente para evitar mezclar observaciones pasadas y futuras.

| Conjunto | Registros | Proporción |
|---|---:|---:|
| Entrenamiento | 2,434 | 80% |
| Prueba | 609 | 20% |

**Periodo de entrenamiento:** 30 de enero de 2015 al 28 de septiembre de 2021.

**Periodo de prueba:** 29 de septiembre de 2021 al 30 de mayo de 2023.

Las fechas corresponden al día de las variables predictoras. La concentración objetivo corresponde al día siguiente.

La estandarización de variables se realizará durante el entrenamiento, utilizando exclusivamente los datos correspondientes a cada partición de entrenamiento.

### 8. Herramientas utilizadas

El proyecto se desarrolla principalmente en Google Colab, utilizando Python y las siguientes bibliotecas:

| Herramienta | Versión registrada |
|---|---|
| Python | 3.13.16 |
| Pandas | 2.2.3 |
| NumPy | 2.1.3 |
| Matplotlib | 3.10.0 |
| Seaborn | 0.13.2 |
| Scikit-learn | 1.6.1 |
| MLflow | 3.17.1 |

Se utiliza una semilla aleatoria de 42 para favorecer la reproducibilidad de los experimentos.

Las versiones corresponden al entorno utilizado durante la preparación inicial del proyecto.

### 9. Reproducción de la preparación de datos

Para reproducir la preparación:

1. Descargar o clonar el repositorio.
2. Disponer de un entorno Python con las dependencias necesarias.
3. Cargar el CSV original desde la carpeta `datos/`.
4. Importar las funciones de `fuentes/datos_prep.py`.
5. Ejecutar `preparar_dataset()` sobre los datos originales.
6. Verificar que se obtengan 3,043 registros y 17 variables predictoras.
7. Ejecutar `dividir_datos()` para obtener los conjuntos de entrenamiento y prueba.

Ejemplo de ejecución desde la raíz del repositorio:

```python
import pandas as pd
from fuentes.datos_prep import preparar_dataset, dividir_datos

archivo = "datos/ebc079e5-bd11-4830-b595-14292f753575.csv"

df_original = pd.read_csv(archivo)

df_preparado, variables, objetivo = preparar_dataset(
    df_original
)

X_train, X_test, y_train, y_test = dividir_datos(
    df_preparado
)

print("Registros preparados:", len(df_preparado))
print("Variables predictoras:", len(variables))
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))
```

La preparación fue verificada comparando los resultados del script con los obtenidos previamente en Google Colab.

### 10. Integridad y versionamiento

El archivo original se conserva sin modificaciones.

El dataset preparado fue exportado y verificado mediante una comparación de su contenido con el resultado generado por `datos_prep.py`.

**SHA-256 de datos_limp.csv:**

`93c76aa30597174fce7bbd49092aafdd586bd56786415091bb3e79177dab1d8b`

El historial de modificaciones se documenta en `CHANGELOG.md` y en los commits del repositorio.

### 11. Entrenamiento y evaluación de modelos

**Estado: pendiente de implementación.**

Se contempla entrenar y comparar:

- Ridge Regression.
- Random Forest Regressor.

Se utilizará Grid Search con validación cruzada temporal para ajustar los hiperparámetros.

Las métricas previstas son:

- MAE: error absoluto medio.
- RMSE: raíz del error cuadrático medio.
- R²: coeficiente de determinación.

Los parámetros, métricas y modelos se registrarán mediante MLflow.

### 12. Limitaciones y consideraciones

- El dataset contiene registros históricos hasta mayo de 2023.
- Los resultados no deben interpretarse como predicciones operativas en tiempo real.
- Las unidades de medición y el procedimiento de agregación requieren confirmación documental.
- Las observaciones atípicas se conservaron inicialmente y deberán considerarse durante la evaluación.
- El modelo supondrá que los promedios de contaminantes del día actual están disponibles antes de predecir el día siguiente.
- La capacidad de generalización se evaluará sobre observaciones cronológicamente posteriores al entrenamiento.

### 13. Estado del proyecto

**Completado:**

- Creación y organización del repositorio.
- Incorporación de datos originales.
- Diagnóstico de calidad y análisis exploratorio.
- Ingeniería de características.
- Preparación y división cronológica.
- Validación del script `datos_prep.py`.
- Generación y versionamiento del dataset preparado.

**Pendiente:**

- Implementación de `train.py`.
- Entrenamiento y ajuste de modelos.
- Configuración y registro de experimentos en MLflow.
- Análisis comparativo de resultados.
- Evidencias de ejecución y reporte técnico final.

### 14. Referencias

Secretaría del Medio Ambiente de la Ciudad de México. (s. f.). *Red Automática de Monitoreo Atmosférico*. Portal de Datos Abiertos de la Ciudad de México. https://www.datos.cdmx.gob.mx/dataset/red-automatica-de-monitoreo-atmosferico

Scikit-learn developers. (s. f.). *Scikit-learn: Machine learning in Python*. https://scikit-learn.org/stable/

MLflow. (s. f.). *MLflow documentation*. https://mlflow.org/docs/latest/
