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

<img width="890" height="491" alt="image" src="https://github.com/user-attachments/assets/04d36d55-6517-42d9-9907-a67008777466" />

En la gráfica podemos observar que hay una tendencia clara ascendente, con algunos picos de bajada o de subida. Sin embargo, no se puede observar un patrón claro. Aunque posteriormente, al descomponer la serie si podemos ver en la gráfica estacionaria un patrón

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

<img width="890" height="491" alt="image" src="https://github.com/user-attachments/assets/e5f1bfd6-2d8b-4d69-a99c-3635c69e3584" />

Si observamos la gráfica estacional, podemos concluir que hay un patrón que se repite cada año, y es que al iniciar el año, suele caer el valor de las acciones, y en verano suele subir bastante las acciones

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

Vemos que nos sale estos resultados:
Augmented Dickey-Fuller Test

data:  trading_price
Dickey-Fuller = -2.3666, Lag order = 11, p-value = 0.4231
alternative hypothesis: stationary

Observamos que el p-value es bastante mayor que 0.5, por lo que no podemos rechazar la hipótesis nula (H0), por lo que la serie no es estacionaria, teniendo que diferenciar la serie para que lo sea y podamos usar el modelo ARIMA

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

<img width="890" height="491" alt="image" src="https://github.com/user-attachments/assets/79e6f2ea-5031-44c2-aabe-3975680acd2f" />

Vemos que hay uno o más picos en la serie y el resto es 0, por lo que nos encontramos con un modelo de media móvil (MA), es decir el parámetro q no es cero. Una vez sabemos esto, miramos la gráfica PACF para saber el orden o valor del parámetro q

- **PACF**

<img width="890" height="491" alt="image" src="https://github.com/user-attachments/assets/84b1fc63-caa8-45c7-aba0-fb61b332806a" />

Observamos que se convierte en 0 entre el 0.02 y el 0.04, por lo que probaremos entre ese rango de valores (2-4) para obtener un modelo ARIMA óptimo

Después de algunas pruebas, el modelo manual utilizado es:

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

<img width="890" height="491" alt="image" src="https://github.com/user-attachments/assets/c72746ab-2516-4f0c-bbf0-6701dec12d89" />

En primer lugar podemos concluir que hay una distribución normal (como vemos en el tercer gráfico), luego podemos observar la primera gráfica, que simplemente son los residuos del modelo y la segunda es el correlograma del modelo

También se aplica el **test de Ljung-Box**.

Hipótesis:

- H0: los residuos son ruido blanco (no hay autocorrelación)
- H1: existe autocorrelación

Para que el modelo sea adecuado los residuos deben:

- tener media cercana a cero

<img width="422" height="44" alt="image" src="https://github.com/user-attachments/assets/cf238314-cd7f-40dc-8f4a-e29b948edb90" />

Vemos que la media es muy cercana a cero, por lo que este requisito se cumple sin problemas

- no presentar autocorrelación

<img width="488" height="170" alt="image" src="https://github.com/user-attachments/assets/b29904a2-e841-429d-8ed0-c74d11b08637" />

En este caso, vemos que el p-value es bastante mayor que 0.5, por lo que no podemos rechazar la hipótesis nula, por lo que no hay una correlación evidente, de manera que se cumple otro de los requisitos para la validación del modelo

- seguir aproximadamente una distribución normal

Hemos visto anteriormente que sigue una distribución normal

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

<img width="890" height="491" alt="image" src="https://github.com/user-attachments/assets/e3284b40-ce6f-4070-ac43-fd2618c7d0c8" />


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

**RMSE (Root Mean Squared Error)**, cuyo valor es de 16.8

Aquí se puede observar la gráfica de test vs predicción

<img width="890" height="491" alt="image" src="https://github.com/user-attachments/assets/c7ff0b42-65d9-4259-9aaa-4e067a9181a3" />

---

## Comparación con Prophet

También se entrena un modelo utilizando la librería **Prophet**.

Para ello:

- se convierten los datos a formato **tsibble**
- se renombran las columnas a **ds** y **y**, como requiere Prophet
- se entrena el modelo
- se generan predicciones para los próximos **500 días**

La comparación visual muestra que **Prophet puede ajustarse mejor a corto plazo**, aunque a largo plazo los intervalos de confianza aumentan.

<img width="890" height="491" alt="image" src="https://github.com/user-attachments/assets/f1ab97d8-be44-4a4b-94a6-16872f8dd03a" />

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

<img width="890" height="491" alt="image" src="https://github.com/user-attachments/assets/ef74e305-9bef-4bb7-af88-43026d8fee68" />

---
# Resultados principales

- El modelo ARIMA seleccionado fue determinado mediante `auto.arima()`.
- Los residuos del modelo cumplen las condiciones de ruido blanco según el test de Ljung-Box.
- El error de predicción obtenido en el conjunto de test fue:

RMSE ≈ 16

Esto indica que el modelo consigue capturar la tendencia general del precio de la acción, aunque existen errores en momentos de alta volatilidad del mercado.

La comparación con Prophet muestra que:

- Prophet se ajusta mejor a corto plazo.
- A largo plazo ambos modelos presentan intervalos de confianza similares.

---
# Autor

Proyecto académico de análisis de **series temporales y forecasting en R** aplicado a datos financieros, hecho por Adrián García García.
