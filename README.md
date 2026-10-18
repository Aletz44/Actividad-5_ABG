# Actividad 5. Gestión de Proyectos de Inteligencia Artificial

## Predicción de partículas PM2.5 en la Ciudad de México mediante aprendizaje automático

### 1. Descripción del proyecto

En este proyecto desarrollé un ejercicio de aprendizaje automático para estimar la concentración de partículas PM2.5 del día siguiente en la Ciudad de México, utilizando registros históricos de contaminación atmosférica.

El problema seleccionado es relevante porque las partículas PM2.5 forman parte de los contaminantes que se monitorean para evaluar la calidad del aire. Poder anticipar sus concentraciones puede aportar información útil para comprender su comportamiento y apoyar futuros sistemas de monitoreo ambiental.

Para desarrollar el proyecto utilicé un conjunto de datos públicos asociado con la Red Automática de Monitoreo Atmosférico (RAMA) de la Ciudad de México.

El trabajo comprendió la revisión de la calidad de los datos, su preparación, la creación de variables históricas, el entrenamiento de modelos de regresión, el ajuste de hiperparámetros y la evaluación de los resultados.

Los algoritmos seleccionados fueron Ridge Regression y Random Forest Regressor. Ambos se compararon mediante métricas estandarizadas y validación cruzada temporal.

También utilicé MLflow para registrar los experimentos, conservar sus parámetros y métricas, almacenar los modelos entrenados y comprobar que podían recuperarse posteriormente.

Finalmente, desarrollé un script independiente de entrenamiento y verifiqué que reproducía los resultados obtenidos originalmente en Google Colab.

### 2. Objetivo

Desarrollar, entrenar y comparar dos modelos de aprendizaje automático para estimar la concentración de partículas PM2.5 del día siguiente en la Ciudad de México, utilizando información histórica de calidad del aire.

Como parte del objetivo, se aplicaron procedimientos de preparación y versionamiento de datos, validación cruzada temporal, ajuste de hiperparámetros y registro de experimentos mediante MLflow.

El propósito no fue únicamente obtener predicciones, sino también documentar el procedimiento y comprobar que los resultados podían reproducirse.

### 3. Fuente y características del dataset

**Fuente de referencia:** Red Automática de Monitoreo Atmosférico (RAMA), Secretaría del Medio Ambiente de la Ciudad de México.

**Portal de datos abiertos:**

https://www.datos.cdmx.gob.mx/dataset/red-automatica-de-monitoreo-atmosferico

El archivo CSV utilizado en el proyecto contiene registros diarios comprendidos entre el 1 de enero de 2015 y el 31 de mayo de 2023.

El conjunto original contiene 3,073 registros y nueve columnas: una fecha y ocho variables relacionadas con contaminantes atmosféricos.

| Característica | Dataset original | Dataset preparado |
|---|---:|---:|
| Registros | 3,073 | 3,043 |
| Columnas | 9 | 20 |
| Frecuencia | Diaria | Diaria |
| Valores faltantes | 0 | 0 |
| Fechas duplicadas | 0 | 0 |
| Fecha inicial | 2015-01-01 | 2015-01-30 |
| Fecha final | 2023-05-31 | 2023-05-30 |

Las variables originales son:

- `fecha`: fecha correspondiente al registro.
- `CO`: monóxido de carbono.
- `NO`: óxido nítrico.
- `NO2`: dióxido de nitrógeno.
- `NOX`: óxidos de nitrógeno.
- `O3`: ozono.
- `PM10`: partículas suspendidas PM10.
- `PM25`: partículas finas PM2.5.
- `SO2`: dióxido de azufre.

La variable principal del proyecto es PM2.5, mientras que los demás contaminantes se utilizan como información predictora.

Las unidades de medición, la procedencia exacta de los agregados diarios y la metodología de agregación del archivo utilizado requieren confirmación con la documentación de la fuente antes de interpretar los resultados como concentraciones ambientales oficiales.

El dataset preparado conserva las observaciones originales necesarias y añade variables para realizar la predicción del día siguiente.

### 4. Estructura del repositorio

El proyecto se encuentra organizado de la siguiente manera:

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
|   |-- train.py
|
|-- README.md
|-- CHANGELOG.md
|-- requirements.txt
```

Las carpetas `datos/` y `fuentes/` también pueden contener archivos `.gitkeep`, utilizados durante la creación inicial de la estructura.

**Descripción de los archivos:**

- `datos/ebc079e5-bd11-4830-b595-14292f753575.csv`: conjunto original de datos utilizado para el experimento.
- `datos/datos_limp.csv`: conjunto preparado para el entrenamiento y la evaluación.
- `fuentes/entrena.ipynb`: notebook de Google Colab con las 14 celdas definitivas del desarrollo experimental.
- `fuentes/datos_prep.py`: funciones reutilizables para validar, transformar y dividir los datos.
- `fuentes/train.py`: script independiente para reproducir el entrenamiento, Grid Search, evaluación y registro en MLflow.
- `README.md`: documentación general y guía de reproducción.
- `CHANGELOG.md`: historial de cambios y decisiones del proyecto.
- `requirements.txt`: dependencias principales y versiones del entorno utilizado.

El repositorio se administra mediante GitHub, utilizando commits para conservar el historial de modificaciones.

### 5. Diagnóstico y preparación de datos

Antes de entrenar los modelos realicé un diagnóstico para conocer la estructura y calidad del conjunto original.

Las verificaciones incluyeron:

- Cantidad de registros y variables.
- Tipos de datos.
- Interpretación de fechas.
- Continuidad de la secuencia diaria.
- Identificación de valores faltantes.
- Detección de registros duplicados.
- Revisión de valores negativos e iguales a cero.
- Cálculo de estadísticas descriptivas.
- Identificación de posibles valores atípicos.

Los resultados mostraron que el conjunto original no contenía valores faltantes, registros duplicados, fechas repetidas ni concentraciones negativas.

También se verificó que la secuencia diaria no presentaba interrupciones durante el periodo analizado.

Para identificar posibles valores atípicos utilicé el método del rango intercuartílico (IQR).

Este procedimiento permitió identificar observaciones alejadas del comportamiento habitual de algunas variables. Sin embargo, decidí conservarlas porque podrían corresponder a episodios reales de contaminación y no necesariamente a errores de medición.

Posteriormente realicé un análisis exploratorio de PM2.5 mediante:

- Evolución diaria de concentraciones.
- Promedio móvil de 30 días.
- Comparación de promedios anuales.
- Comportamiento mensual.
- Identificación de días con concentraciones elevadas.
- Diagramas de caja para comparar distribuciones anuales.

El análisis permitió observar variaciones temporales y posibles patrones estacionales que justificaron la incorporación de variables históricas y temporales durante la preparación del conjunto de entrenamiento.

### 6. Ingeniería de características

Para transformar el problema en una tarea de regresión supervisada, creé la variable objetivo:

`PM25_siguiente_dia`

Esta variable representa la concentración de PM2.5 correspondiente al día posterior al de las variables predictoras.

Las transformaciones principales fueron:

- Creación de la variable objetivo del día siguiente.
- Incorporación de rezagos históricos de PM2.5 de 1, 2, 3 y 7 días.
- Cálculo de promedios móviles de 7 y 30 días.
- Incorporación de variables temporales relacionadas con el calendario.
- Representación cíclica del mes mediante funciones seno y coseno.
- Selección de 17 variables predictoras.

La creación de variables históricas permite que los modelos utilicen información de días anteriores para estimar el comportamiento del día siguiente.

Se excluyeron 30 registros debido a la falta de historial suficiente para determinadas transformaciones y a la ausencia del valor objetivo correspondiente al último día.

Después de la preparación, el conjunto final quedó integrado por:

- 3,043 registros.
- 20 columnas.
- 17 variables predictoras.
- Una variable objetivo denominada `PM25_siguiente_dia`.

El procedimiento fue implementado en el notebook y posteriormente reproducido mediante `datos_prep.py`.

Se verificó que ambos procedimientos generaran el mismo conjunto preparado.

### 7. División de entrenamiento y prueba

Debido a que los registros corresponden a una secuencia temporal, decidí realizar una división cronológica en lugar de distribuir aleatoriamente las observaciones.

Esta decisión permite evaluar los modelos sobre un periodo posterior al utilizado para entrenarlos.

| Conjunto | Registros | Proporción |
|---|---:|---:|
| Entrenamiento | 2,434 | 80% |
| Prueba | 609 | 20% |
| Total | 3,043 | 100% |

**Periodo de entrenamiento:** 30 de enero de 2015 al 28 de septiembre de 2021.

**Periodo de prueba:** 29 de septiembre de 2021 al 30 de mayo de 2023.

Las fechas corresponden al día de las variables predictoras. La concentración que se busca estimar pertenece al día siguiente.

Durante el entrenamiento se utilizó validación cruzada temporal mediante `TimeSeriesSplit`, con cinco particiones y una separación de una observación entre entrenamiento y validación.

En Ridge Regression se utilizó `StandardScaler` dentro de un `Pipeline`, de manera que la estandarización se ajustara únicamente con los registros de entrenamiento correspondientes a cada partición.

Random Forest Regressor se entrenó sin estandarización, ya que su funcionamiento se basa en árboles de decisión.

El conjunto de prueba permaneció reservado durante la selección de hiperparámetros y se utilizó posteriormente para la evaluación final.

### 8. Herramientas y dependencias

El proyecto fue desarrollado principalmente en Google Colab, utilizando Python y bibliotecas especializadas en análisis de datos y aprendizaje automático.

Las versiones registradas durante la ejecución fueron:

| Herramienta | Versión |
|---|---|
| Python | 3.13.16 |
| Pandas | 2.2.3 |
| NumPy | 2.1.3 |
| Matplotlib | 3.10.0 |
| Seaborn | 0.13.2 |
| Scikit-learn | 1.6.1 |
| MLflow | 3.17.1 |
| Cloudpickle | 3.1.2 |

Se utilizó una semilla aleatoria fija de 42 para favorecer la reproducibilidad.

Las dependencias principales se encuentran documentadas en `requirements.txt`.

Para instalarlas desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
```

Se recomienda utilizar un entorno virtual para evitar conflictos con bibliotecas de otros proyectos.

La instalación completa en un entorno limpio, así como la compatibilidad de todas las dependencias transitivas, todavía requiere una verificación adicional.

### 9. Reproducción del proyecto

El proyecto puede consultarse mediante el notebook de Google Colab o ejecutarse utilizando los scripts Python del repositorio.

#### 9.1. Descargar el repositorio

Desde una terminal con Git instalado:

```bash
git clone https://github.com/Aletz44/Actividad-5_ABG.git
cd Actividad-5_ABG
```

#### 9.2. Instalar las dependencias

```bash
python -m pip install -r requirements.txt
```

#### 9.3. Reproducir la preparación de datos

El archivo `fuentes/datos_prep.py` contiene las funciones utilizadas para validar y preparar el dataset.

El notebook `fuentes/entrena.ipynb` documenta la ejecución de estas funciones y las comprobaciones realizadas.

El procedimiento genera 3,043 registros preparados y selecciona 17 variables predictoras.

#### 9.4. Ejecutar el entrenamiento independiente

Desde la raíz del repositorio:

```bash
python fuentes/train.py
```

El script realiza automáticamente:

1. Carga del archivo original.
2. Preparación de variables.
3. División cronológica de entrenamiento y prueba.
4. Configuración de Ridge Regression y Random Forest Regressor.
5. Ajuste de hiperparámetros mediante Grid Search.
6. Validación cruzada temporal con cinco particiones.
7. Evaluación final con MAE, RMSE y R².
8. Registro de experimentos y modelos en MLflow.
9. Generación de archivos CSV con resultados y predicciones.
10. Recuperación de modelos para comprobar su funcionamiento.

El script utiliza un experimento independiente denominado:

`RAMA_PM25_TrainScript`

Su información se almacena localmente en la carpeta `mlflow_train/`, creada durante la ejecución.

Los archivos CSV de resultados se generan en `resultados/`.

Estas carpetas son productos de la ejecución y no forman parte de la estructura inicial obligatoria del repositorio.

#### 9.5. Validación independiente realizada

Para comprobar la reproducibilidad, ejecuté `train.py` desde una copia independiente del repositorio en Google Colab.

El proceso terminó con código de salida 0 y generó los resultados esperados para ambos modelos.

Posteriormente comparé las métricas obtenidas con las del notebook original.

Las diferencias fueron inferiores a 0.0001 en las métricas contrastadas, dentro de la tolerancia utilizada para considerar el redondeo de los resultados.

También verifiqué que el experimento independiente de MLflow contuviera dos ejecuciones finalizadas, con sus métricas registradas y modelos recuperables.

Esta prueba confirmó la reproducibilidad dentro del entorno utilizado. No representa todavía una validación completa de portabilidad en un sistema operativo o entorno Python diferente.

#### 9.6. Restauración y validación de dependencias

Como comprobación adicional de reproducibilidad, realicé una prueba de restauración de los experimentos de MLflow en una nueva sesión de Google Colab, utilizando el respaldo `mlflow_rama_final.zip`.

El procedimiento se documentó en el notebook complementario `fuentes/Validacion_Restauracion_MLflow_ABG.ipynb`.

La prueba incluyó la extracción del respaldo, la verificación de integridad de SQLite, la actualización de las rutas locales de almacenamiento y la recuperación de los dos modelos definitivos.

Ridge Regression se recuperó mediante su archivo `model.skops`, mientras que Random Forest Regressor se recuperó mediante `model.pkl`, utilizando los formatos de serialización `skops` y `cloudpickle`, respectivamente.

Ambos modelos conservaron su estado `READY` y pudieron generar predicciones en la nueva sesión.

Posteriormente se recuperaron los archivos de evaluación de los dos modelos. Se verificaron las métricas registradas y se recalcularon MAE, RMSE y R² a partir de los 609 registros de prueba de cada modelo.

Los resultados coincidieron con los valores almacenados originalmente en MLflow, sin necesidad de repetir el entrenamiento.

Adicionalmente, se realizó una instalación de las dependencias de `requirements.txt` en un entorno virtual aislado dentro de Google Colab, utilizando Python 3.13.16.

La instalación finalizó correctamente, `pip check` no identificó dependencias incompatibles y las siete bibliotecas principales pudieron importarse con las versiones documentadas.

Para ejecutar Matplotlib fuera de la interfaz interactiva de Colab se utilizó el backend `Agg`.

Estas comprobaciones fortalecen la reproducibilidad del proyecto, aunque no equivalen a una validación en todos los sistemas operativos o configuraciones posibles.

### 10. Integridad y versionamiento

El dataset original se conserva sin modificar sus mediciones.

El conjunto preparado se encuentra en:

`datos/datos_limp.csv`

Durante la exportación se verificaron:

- Cantidad de registros.
- Cantidad y orden de columnas.
- Fechas.
- Equivalencia de valores numéricos.
- Integridad del archivo mediante SHA-256.

**SHA-256 de datos_limp.csv:**

`93c76aa30597174fce7bbd49092aafdd586bd56786415091bb3e79177dab1d8b`

La huella permite comprobar si el contenido exacto del archivo ha cambiado.

Los cambios del proyecto se documentan mediante commits de GitHub y el archivo `CHANGELOG.md`.

Para facilitar la trazabilidad, los experimentos también incluyen identificadores de ejecución, parámetros, métricas y artefactos registrados en MLflow.

### 11. Entrenamiento, ajuste y evaluación de modelos

Se entrenaron y compararon dos algoritmos de regresión:

- Ridge Regression.
- Random Forest Regressor.

Ridge Regression utiliza una relación lineal regularizada entre las variables predictoras y la concentración estimada.

Random Forest Regressor combina múltiples árboles de decisión y permite representar relaciones no lineales.

#### 11.1. Entrenamiento inicial

Primero evalué ambos modelos con configuraciones iniciales, utilizando cinco particiones de validación cruzada temporal.

Los resultados fueron:

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge Regression | 4.3075 | 5.8610 | 0.5253 |
| Random Forest Regressor | 4.4741 | 5.9944 | 0.4965 |

Ridge Regression obtuvo inicialmente un menor error promedio y un tiempo de evaluación considerablemente inferior.

Estos resultados sirvieron como referencia para medir las mejoras posteriores.

#### 11.2. Ajuste de hiperparámetros

Utilicé `GridSearchCV` con cinco particiones de validación temporal y MAE como criterio principal de selección.

Para Ridge Regression se evaluaron cuatro valores de regularización:

`alpha = [0.1, 1.0, 10.0, 100.0]`

Para Random Forest se evaluaron combinaciones de:

- `n_estimators`: 100 y 200 árboles.
- `max_depth`: 5, 10 y sin límite explícito.
- `min_samples_leaf`: 1, 3 y 5 observaciones.

En total se evaluaron 22 configuraciones mediante 110 ajustes de validación cruzada, además de los reajustes finales de cada búsqueda.

Los mejores hiperparámetros fueron:

| Modelo | Hiperparámetros seleccionados |
|---|---|
| Ridge Regression | `alpha=100.0` |
| Random Forest | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=5` |

#### 11.3. Resultados después de Grid Search

| Modelo | MAE CV | RMSE CV | R² CV |
|---|---:|---:|---:|
| Ridge Regression | 4.2367 | 5.7668 | 0.5405 |
| Random Forest Regressor | 4.3197 | 5.8419 | 0.5264 |

El ajuste redujo el MAE promedio de Ridge aproximadamente un 1.64% y el de Random Forest un 3.45% respecto a sus configuraciones iniciales.

Aunque Random Forest presentó una mejora porcentual mayor, Ridge mantuvo el menor error promedio de validación.

#### 11.4. Evaluación final sobre el conjunto de prueba

Después de seleccionar los hiperparámetros, evalué los modelos sobre los 609 registros reservados.

También incorporé un modelo de referencia denominado persistencia, que estima la concentración del día siguiente utilizando la concentración observada el día actual.

Los resultados fueron:

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge Regression | 3.6493 | 4.9894 | 0.4861 |
| Random Forest Regressor | 3.6743 | 5.1004 | 0.4629 |
| Persistencia | 4.1401 | 5.9459 | 0.2701 |

Ridge Regression obtuvo el menor MAE y RMSE, así como el mayor R².

Respecto al modelo de persistencia, Ridge redujo el MAE aproximadamente un 11.86%, mientras que Random Forest consiguió una reducción del 11.25%.

Estos resultados muestran que ambos algoritmos mejoraron la predicción promedio frente a la estrategia de referencia durante el periodo evaluado.

La diferencia entre Ridge y Random Forest fue pequeña, por lo que no se considera demostrada una superioridad estadística de Ridge.

#### 11.5. Evaluación de concentraciones elevadas

También analicé el desempeño durante observaciones con concentraciones elevadas de PM2.5.

Para ello utilicé el percentil 90 del conjunto de entrenamiento, equivalente a un umbral de 32.568 en las unidades del archivo.

En el conjunto de prueba se identificaron 20 observaciones que alcanzaron o superaron ese umbral.

Los resultados fueron:

| Modelo | MAE en concentraciones elevadas |
|---|---:|
| Persistencia | 9.9416 |
| Random Forest Regressor | 10.7245 |
| Ridge Regression | 10.7461 |

En estas observaciones, la persistencia obtuvo un menor error que ambos modelos de aprendizaje automático.

Este resultado es importante porque demuestra que un modelo con buen desempeño promedio puede presentar limitaciones durante episodios de contaminación elevada.

Debido al número reducido de observaciones, los resultados deben interpretarse con cautela.

#### 11.6. Selección del modelo candidato

Con base en las métricas obtenidas, seleccioné Ridge Regression como modelo candidato principal del experimento.

La decisión se fundamenta en:

- Menor MAE promedio de validación cruzada.
- Menor MAE y RMSE en el conjunto de prueba.
- Mayor R² entre los dos modelos entrenados.
- Menor tiempo de ajuste de hiperparámetros.
- Mayor sencillez de implementación e interpretación.

Sin embargo, esta selección corresponde al alcance académico del proyecto y no implica que el modelo esté preparado para utilizarse en un sistema operativo de alertas ambientales.

#### 11.7. Registro de experimentos en MLflow

Utilicé MLflow para registrar los hiperparámetros, las métricas de validación y prueba, los archivos de evaluación y los modelos entrenados.

El experimento principal del notebook se denominó:

`RAMA_PM25_Actividad5`

Las ejecuciones definitivas fueron:

| Modelo | Run ID definitivo |
|---|---|
| Ridge Regression | `73339b012c7640e6bb99744dc3018c2d` |
| Random Forest Regressor | `3c164f2beea644eba2f4068cb3b7beb6` |

Durante el registro inicial de Random Forest se presentó una incidencia de serialización que dejó una ejecución y un modelo con estado `FAILED`.

La ejecución original se conservó y el modelo se recuperó mediante una nueva ejecución vinculada con la anterior, utilizando `cloudpickle`.

Posteriormente se verificó que ambos modelos podían recuperarse desde MLflow y generar las mismas predicciones originales.

También se generó un respaldo del seguimiento denominado `mlflow_rama_final.zip`, cuya integridad estructural fue comprobada.

La restauración integral de ese respaldo en un entorno nuevo continúa pendiente de verificación.

### 12. Limitaciones y consideraciones

Aunque el proyecto permitió obtener resultados reproducibles, identifiqué algunas limitaciones importantes.

**Periodo de los datos.** El conjunto utilizado contiene registros hasta mayo de 2023. Por tanto, los resultados corresponden al comportamiento histórico incluido en el experimento y no deben interpretarse como predicciones actuales.

**Unidades y agregación.** Antes de interpretar las predicciones en términos de calidad del aire, es necesario confirmar las unidades de medición y el procedimiento mediante el cual se obtuvieron los valores diarios del archivo original.

**Disponibilidad de información.** El experimento supone que las variables del día actual están disponibles al momento de estimar la concentración del día siguiente. Una implementación operativa tendría que verificar los horarios reales de publicación y disponibilidad de cada contaminante.

**Concentraciones elevadas.** Ambos modelos presentaron mayores errores durante las observaciones de PM2.5 elevadas. Por ello, no sería adecuado utilizar sus resultados directamente para emitir alertas ambientales sin validaciones adicionales.

**Generalización.** La evaluación se realizó sobre un periodo histórico reservado. Aunque los modelos mejoraron frente a persistencia, esto no garantiza que mantengan el mismo desempeño ante cambios futuros en las condiciones ambientales.

**Alcance de Grid Search.** La búsqueda se limitó a un conjunto definido de hiperparámetros. En Ridge, el mejor valor de `alpha` coincidió con el límite superior evaluado, por lo que podrían existir configuraciones adicionales de interés.

**Reproducibilidad del entorno.** Se verificó la instalación de las dependencias principales en un entorno virtual aislado y la ejecución independiente del script de entrenamiento dentro de Google Colab. La compatibilidad con otros sistemas operativos y configuraciones de Python no fue evaluada.

**Persistencia de experimentos.** Se comprobó la restauración funcional de los experimentos y modelos de MLflow en una nueva sesión de Google Colab. Para recuperar los artefactos fue necesario actualizar las rutas absolutas de almacenamiento en la copia restaurada. Por tanto, el respaldo requiere este ajuste cuando se utiliza en una ubicación diferente.

Como recomendaciones para una posible continuación del proyecto, considero conveniente ampliar las pruebas durante episodios de contaminación elevada, incorporar información meteorológica, verificar la disponibilidad operativa de los datos y establecer mecanismos de monitoreo del desempeño de los modelos.

También sería necesario evaluar periódicamente la degradación del modelo, mantener el versionamiento de datos y dependencias y definir criterios para su actualización.

### 13. Estado del proyecto

El proyecto ha completado las etapas de preparación y versionamiento de datos, desarrollo experimental, entrenamiento, ajuste de hiperparámetros, evaluación de modelos y registro de experimentos mediante MLflow.

También se han realizado satisfactoriamente las comprobaciones de reproducibilidad, incluyendo la ejecución independiente del script de entrenamiento, la restauración funcional de los modelos desde un respaldo y la validación de las dependencias en un entorno virtual aislado.

Actualmente, el proyecto se encuentra en la fase de cierre académico y preparación de los entregables finales.

**Actividades completadas:**

**Preparación y análisis de datos**

- Creación y organización del repositorio `Actividad-5_ABG` en GitHub.
- Incorporación y conservación del dataset original RAMA.
- Diagnóstico de calidad y análisis exploratorio de los registros.
- Verificación de valores faltantes, duplicados, fechas y posibles valores atípicos.
- Análisis temporal de las concentraciones de PM2.5.
- Ingeniería de características y creación de la variable objetivo `PM25_siguiente_dia`.
- Preparación del dataset para modelado, con 3,043 registros y 17 variables predictoras.
- División cronológica de entrenamiento y prueba.
- Implementación y validación del script `datos_prep.py`.
- Generación, verificación y versionamiento de `datos_limp.csv`.

**Entrenamiento y evaluación de modelos**

- Desarrollo del notebook principal `entrena.ipynb`.
- Entrenamiento inicial de Ridge Regression y Random Forest Regressor.
- Configuración de validación cruzada temporal mediante `TimeSeriesSplit`.
- Ajuste de hiperparámetros mediante `GridSearchCV`.
- Evaluación de los modelos utilizando MAE, RMSE y R².
- Comparación de los resultados frente a un modelo de persistencia.
- Evaluación del desempeño durante concentraciones elevadas de PM2.5.
- Selección de Ridge Regression como modelo candidato principal del experimento, considerando sus métricas de validación y prueba.

**Registro de experimentos y reproducibilidad**

- Configuración del experimento `RAMA_PM25_Actividad5` en MLflow.
- Registro de hiperparámetros, métricas, ejecuciones y artefactos de evaluación.
- Registro y recuperación de los modelos entrenados.
- Identificación y resolución de una incidencia de serialización durante el registro inicial de Random Forest Regressor.
- Conservación del historial de la ejecución fallida y de su posterior recuperación.
- Generación del respaldo `mlflow_rama_final.zip`.
- Verificación de la integridad estructural del respaldo y de su base de datos SQLite.
- Restauración funcional de MLflow en una nueva sesión de Google Colab.
- Actualización de las referencias de almacenamiento en la copia restaurada.
- Recuperación de Ridge Regression mediante el formato `skops`.
- Recuperación de Random Forest Regressor mediante el formato `cloudpickle`.
- Verificación del estado `READY` de ambos modelos definitivos.
- Recuperación de los artefactos de evaluación y comprobación de las métricas originales.
- Recálculo satisfactorio de MAE, RMSE y R² a partir de los 609 registros de prueba de cada modelo.
- Confirmación de que la restauración no requirió volver a entrenar los modelos.

**Validación independiente y documentación técnica**

- Implementación del script independiente `fuentes/train.py`.
- Ejecución satisfactoria del script desde una copia independiente del repositorio.
- Reproducción de las métricas originales dentro de la tolerancia establecida.
- Registro independiente de los modelos en el experimento `RAMA_PM25_TrainScript`.
- Generación de cuatro evidencias documentales mediante consultas a la API de MLflow.
- Exportación de las evidencias en archivos PNG y CSV, agrupados en `evidencias_mlflow.zip`.
- Creación de `requirements.txt` con las dependencias principales del proyecto.
- Instalación satisfactoria de las dependencias en un entorno virtual aislado con Python 3.13.16.
- Verificación de compatibilidad mediante `pip check`, sin conflictos detectados.
- Comprobación de las importaciones de NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, MLflow y Cloudpickle.
- Validación del funcionamiento de Matplotlib mediante el backend `Agg` en el entorno aislado.
- Incorporación del notebook complementario `fuentes/Validacion_Restauracion_MLflow_ABG.ipynb` al repositorio GitHub.
- Actualización de la documentación metodológica y técnica en `README.md` y `CHANGELOG.md`.

**Actividades de cierre académico y entrega:**

El desarrollo experimental y las verificaciones técnicas previstas se encuentran completados. Para concluir formalmente la TAREA 5 se realizarán las siguientes actividades:

- Confirmar documentalmente las unidades de medición y la metodología de agregación del dataset original RAMA, o dejar expresamente documentadas las limitaciones de la información disponible.
- Actualizar el historial de cambios de `CHANGELOG.md` con los resultados definitivos de restauración de MLflow y validación de dependencias.
- Organizar las evidencias documentales, los respaldos y los archivos del repositorio necesarios para la entrega.
- Elaborar el reporte técnico académico, incluyendo descripción del problema, metodología, preparación de datos, entrenamiento, ajuste de hiperparámetros, resultados, análisis crítico, limitaciones, deuda técnica, recomendaciones MLOps, conclusiones y referencias APA.
- Integrar y verificar el archivo ZIP definitivo con los entregables establecidos para la TAREA 5.

Estas actividades forman parte del cierre académico del proyecto y se documentarán como completadas una vez realizadas y verificadas.

**Estado general:** desarrollo técnico y validaciones de reproducibilidad completados; cierre documental y académico en proceso.

### 14. Referencias

Secretaría del Medio Ambiente de la Ciudad de México. (s. f.). *Red Automática de Monitoreo Atmosférico*. Portal de Datos Abiertos de la Ciudad de México. https://www.datos.cdmx.gob.mx/dataset/red-automatica-de-monitoreo-atmosferico

Scikit-learn developers. (s. f.). *Scikit-learn: Machine learning in Python*. https://scikit-learn.org/stable/

MLflow. (s. f.). *MLflow documentation*. https://mlflow.org/docs/latest/

Python Software Foundation. (s. f.). *Python documentation*. https://docs.python.org/3/

Pandas development team. (s. f.). *Pandas documentation*. https://pandas.pydata.org/docs/
