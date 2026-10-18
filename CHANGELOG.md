
# Historial de cambios del proyecto (CHANGELOG)

## Actividad 5. Gestión de Proyectos de Inteligencia Artificial

**Proyecto:** Predicción de la concentración de partículas PM2.5 en la Ciudad de México mediante aprendizaje automático.

Este archivo documenta los principales cambios realizados durante el desarrollo del proyecto, con el propósito de mantener un registro de las modificaciones, facilitar su seguimiento y contribuir a la reproducibilidad de los resultados.

---

## [0.1.0] - Preparación inicial del proyecto

### Repositorio y entorno de trabajo

- Creación del repositorio público `Actividad-5_ABG` en GitHub.
- Inicialización del historial de versiones mediante commits en la rama `main`.
- Creación de las carpetas `datos/` y `fuentes/`.
- Incorporación del archivo `README.md`.
- Creación del notebook `fuentes/entrena.ipynb` para trabajar en Google Colab.
- Configuración inicial de Python y de las bibliotecas Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn y MLflow.
- Establecimiento de una semilla aleatoria fija de 42.

### Incorporación del dataset original

- Descarga e incorporación de los registros públicos de la Red Automática de Monitoreo Atmosférico (RAMA) de la Ciudad de México.
- Conservación del archivo CSV original sin modificar sus mediciones.
- Identificación de 3,073 registros diarios y nueve columnas.
- Verificación del periodo comprendido entre el 1 de enero de 2015 y el 31 de mayo de 2023.

---

## [0.2.0] - Diagnóstico y análisis exploratorio

### Evaluación de calidad de los datos

- Verificación del formato de las fechas.
- Confirmación de que no existen fechas duplicadas ni interrupciones en la secuencia diaria.
- Identificación de cero valores faltantes y cero registros duplicados.
- Comprobación de que no existen concentraciones negativas ni valores iguales a cero.
- Cálculo de estadísticas descriptivas para los ocho contaminantes.
- Identificación de posibles valores atípicos mediante el método del rango intercuartílico (IQR).
- Conservación inicial de los valores atípicos para evitar eliminar posibles episodios reales de contaminación.

### Análisis temporal de PM2.5

- Visualización de las concentraciones diarias y del promedio móvil de 30 días.
- Cálculo y comparación de promedios anuales.
- Análisis del comportamiento mensual de las concentraciones.
- Identificación de los diez días con mayores concentraciones registradas.
- Elaboración de diagramas de caja para comparar la distribución anual.
- Identificación de posibles patrones estacionales y variaciones temporales.

---

## [0.3.0] - Preparación del dataset para modelado

### Ingeniería de características

- Creación de la variable objetivo `PM25_siguiente_dia`.
- Incorporación de rezagos históricos de PM2.5 de 1, 2, 3 y 7 días.
- Cálculo de promedios móviles de 7 y 30 días.
- Generación de variables temporales relacionadas con el mes y el día de la semana.
- Incorporación de representaciones cíclicas del mes mediante funciones seno y coseno.
- Selección de 17 variables predictoras.

### Preparación y división de los datos

- Exclusión de 30 registros por falta de historial suficiente o ausencia del valor objetivo del día siguiente.
- Generación de un conjunto preparado de 3,043 registros.
- División cronológica del conjunto de datos:
  - Entrenamiento: 2,434 registros (80%).
  - Prueba: 609 registros (20%).
- Conservación de los valores atípicos identificados durante el diagnóstico.
- Preparación de los datos sin aplicar todavía estandarización ni entrenamiento.

---

## [0.4.0] - Reproducibilidad y versionamiento

### Script de preparación

- Creación del archivo `fuentes/datos_prep.py`.
- Implementación de funciones reutilizables para validar los registros, crear variables y dividir cronológicamente los datos.
- Importación y ejecución del script desde Google Colab.
- Verificación de coincidencia exacta entre los resultados del script y los obtenidos previamente en el notebook.
- Confirmación de que el dataset original permanece sin modificaciones.

### Dataset preparado

- Generación del archivo `datos/datos_limp.csv`.
- Exportación de 3,043 registros y 20 columnas.
- Verificación de las fechas, columnas y equivalencia numérica del archivo exportado.
- Cálculo de la huella SHA-256 para identificar el contenido exacto del dataset preparado.
- Incorporación del archivo preparado al repositorio GitHub.

**SHA-256 de datos_limp.csv:**

`93c76aa30597174fce7bbd49092aafdd586bd56786415091bb3e79177dab1d8b`

---

## Actividades pendientes

- Documentar las dependencias necesarias para reproducir el entorno.
- Completar el archivo README.md.
- Configurar y validar el registro de experimentos en MLflow.
- Implementar el script `fuentes/train.py`.
- Entrenar y comparar los modelos Ridge Regression y Random Forest Regressor.
- Realizar el ajuste de hiperparámetros mediante Grid Search con validación cruzada temporal.
- Registrar parámetros, métricas y modelos en MLflow.
- Analizar comparativamente los resultados.
- Documentar las limitaciones, la deuda técnica y las recomendaciones MLOps.
- Preparar las evidencias y el reporte técnico final.

---

## Consideraciones de reproducibilidad

Los archivos originales y preparados se conservan por separado para facilitar la trazabilidad de las transformaciones.

La preparación del dataset puede reproducirse mediante el script `datos_prep.py`. La coincidencia de resultados fue verificada desde Google Colab.

Los identificadores de versión 0.1.0 a 0.4.0 representan etapas documentales del proyecto. No corresponden a etiquetas de Git publicadas.

Las fechas exactas de los cambios pueden consultarse en el historial de commits del repositorio.
