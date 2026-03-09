# Predicción del precio de las acciones de Microsoft con ARIMA en R

## Descripción

Este proyecto aplica técnicas de **análisis de series temporales** para modelar y predecir el precio de las acciones de Microsoft utilizando **modelos ARIMA en R**.

El análisis se realiza a partir del dataset **Microsoft_stock_data.csv**, descargado de Kaggle, que contiene datos históricos del precio de las acciones de Microsoft.

El objetivo del proyecto es:

- Analizar la evolución temporal del precio de la acción
- Construir un modelo ARIMA para realizar predicciones
- Validar el modelo mediante análisis de residuos
- Comparar el modelo ARIMA con el modelo **Prophet**
- Detectar posibles anomalías en la serie temporal

---

# Dataset

Dataset utilizado:

**Microsoft_stock_data.csv**

Contiene información histórica del precio de las acciones de Microsoft, incluyendo:

- Date
- Open
- High
- Low
- Close
- Volume

Para el análisis se crea una nueva variable llamada **trading_price**, calculada como la media de los precios de apertura, cierre, máximo y mínimo:

```
trading_price = (Close + High + Low + Open) / 4
```

Además, se filtran los datos anteriores a **2020**, ya que los valores antiguos presentan precios cercanos a 0 que introducen ruido en el modelo.

---

# Librerías utilizadas

El proyecto utiliza las siguientes librerías de R:

- dplyr
- lubridate
- tseries
- forecast
- lmtest
- ggplot2
- tsibble
- fable
- fable.prophet
- prophet
- tsoutliers

---

# Metodología

El análisis se divide en varias etapas.

---

## 1. Ingesta y preparación de datos

Se carga el archivo CSV y se prepara el dataset:

- Carga de datos
- Creación de la variable **trading_price**
- Filtrado de datos desde el año **2020**
- Conversión de la columna **Date** al formato `Date`
- Conversión de los datos a una serie temporal (`ts`)

Se establece una frecuencia de **252**, ya que los mercados financieros tienen aproximadamente **252 días de trading al año**.

Posteriormente se realiza una **visualización inicial** de la serie para observar tendencias y posibles patrones.

---

## 2. Descomposición de la serie temporal

Se utiliza la función:

```
decompose()
```

para separar la serie en tres componentes:

- Tendencia
- Estacionalidad
- Residuos

Esto permite analizar si existen patrones estacionales dentro de los datos.

---

## 3. Análisis de estacionariedad

Los modelos ARIMA requieren que la serie temporal sea **estacionaria**.

Para comprobarlo se utiliza el **test de Dickey-Fuller aumentado (ADF)**:

```
adf.test()
```

Hipótesis del test:

- H0: la serie no es estacionaria
- H1: la serie es estacionaria

Si la serie no es estacionaria se aplica **diferenciación**.

Para determinar el número de diferencias necesarias se utiliza:

```
ndiffs()
```

Posteriormente se aplica:

```
diff()
```

para transformar la serie en estacionaria.

También se muestra una alternativa eliminando la estacionalidad antes de diferenciar la serie.

---

## 4. Entrenamiento del modelo ARIMA

Se utilizan dos enfoques:

### Modelo automático

Se utiliza:

```
auto.arima()
```

Esta función busca automáticamente los mejores parámetros **(p, d, q)** optimizando el criterio **AIC**.

### Modelo manual

Se analizan los correlogramas:

- **ACF**
- **PACF**

para estimar manualmente los valores de **p** y **q**.

El modelo manual utilizado es:

```
ARIMA(0,1,2)
```

Finalmente se comparan ambos modelos mediante el valor **AIC** y se selecciona el modelo automático.

---

## 5. Validación del modelo

Para validar el modelo se analizan los **residuos** utilizando:

```
checkresiduals()
```

También se aplica el **test de Ljung-Box**.

Hipótesis:

- H0: los residuos son ruido blanco (no hay autocorrelación)
- H1: existe autocorrelación

Para que el modelo sea adecuado los residuos deben:

- tener media cercana a cero
- no presentar autocorrelación
- seguir aproximadamente una distribución normal

---

## 6. Predicción (Forecasting)

Una vez validado el modelo, se realizan predicciones utilizando:

```
forecast()
```

En este caso se predicen los **próximos 500 días** del precio de la acción.

Los resultados se visualizan mediante:

```
autoplot()
```

donde también se muestran los **intervalos de confianza**.

---

# Ampliaciones

## Train/Test Split

Se realiza una separación de los datos en:

- **Training set**: datos para entrenar el modelo
- **Test set**: últimos 90 días (aproximadamente 3 meses)

Posteriormente:

1. Se entrena un modelo ARIMA con los datos de entrenamiento.
2. Se realizan predicciones para los siguientes 90 días.
3. Se comparan con los valores reales del conjunto de test.

Para evaluar el error del modelo se calcula:

**RMSE (Root Mean Squared Error)**

---

## Comparación con Prophet

También se entrena un modelo utilizando la librería **Prophet**.

Para ello:

- se convierten los datos a formato **tsibble**
- se renombran las columnas a **ds** y **y**, como requiere Prophet
- se entrena el modelo
- se generan predicciones para los próximos **500 días**

La comparación visual muestra que **Prophet puede ajustarse mejor a corto plazo**, aunque a largo plazo los intervalos de confianza aumentan.

---

## Detección de anomalías

Se utiliza el paquete:

```
tsoutliers
```

para detectar posibles **outliers en la serie temporal**.

Posteriormente se representan en un gráfico comparando:

- serie original
- serie limpia
- posibles anomalías detectadas

En este caso no se detectan outliers significativos.

---

# Autor

Proyecto académico de análisis de **series temporales y forecasting en R** aplicado a datos financieros.
