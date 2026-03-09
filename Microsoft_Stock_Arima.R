library(dplyr)
library(lubridate)
library(tseries)
library(forecast)
library(lmtest)
library(ggplot2)
library(tsibble)
library(fable)
library(fable.prophet)
library(prophet)
library(tsoutliers)

# PASO 2. Ingesta y Conversión a Serie Temporal

file_path <- "Microsoft_stock_data.csv"

microsoft_stock_df <- read.csv(file_path)

# Como el dataset tiene columnas del valor de la bolsa en el cierre, apertura
# y los valores más altos y bajos, lo que he hecho es hacer una media entre
# todos ellos y crear una columna llamada "tradin_price", siendo nuestra columna objetivo

microsoft_stock_df <- microsoft_stock_df %>% mutate(trading_price = (Close + High + Low + Open)/4)

# Después de estudiar en profundidad el dataset y viendo videos acerca del funcionamiento
# en profundidad de ARIMA, este dataset desde el año 1986 hasta 2020, tiene valores cercanos a 0
# y luego hay un cambio brusco ascendente en 2020.
# Si a eso lo juntamos con que el modelo ARIMA es muy delicado con los datos donde haya
# alta volatilidad, es decir, donde hay cambios bruscos, he sacado la conclusión de que
# los datos cercanos a 0 no es de utilidad en este caso,
# ya que solo hace que haya ruido en el modelo,
# ya que a dia de hoy, excepto sorpresas, es muy dificil que el valor en bolsa de Microsoft
# esté a 0, por lo que hacemos un filtrado para eliminar los datos desde 1984 hasta 2020

microsoft_stock_df <- microsoft_stock_df %>%
  filter(Date >= as.Date("2020-01-01"))


class(microsoft_stock_df$Date) # Vemos que la fecha es de tipo carácteres

# Con esta función, pasamos la fecha de carácteres a Date

microsoft_stock_df$Date <- ymd(microsoft_stock_df$Date)

class(microsoft_stock_df$Date) # Vemos que ya se ha pasado a Date

# Se establece de frecuencia un valor de 252 ya que en los datos de la bolsa
# no cuenta los fines de semanas ni festivos, por lo que en total quedaría unos 252 días

trading_price <- ts(microsoft_stock_df$trading_price, start = c(2020,1,1), frequency = 252)

plot(trading_price)
print("In the plot, we can see a clear upward trend, with some downward peaks")
print("However, in this plot no clear pattern is observed")

# PASO 3: Descomposición

components <- decompose(trading_price)
plot(components)
print("If we look at the seasonality plot, we see that in this case there are seasonal patterns")
print("with upward peaks in summer and downward peaks when a new year begins")



# PASO 4. Análisis de Estacionariedad

# Para este test tenemos un contraste de hipótesis
# Ho: La serie no es estacionaria
# H1: La serie es estacionaria
adf_test <- adf.test(trading_price)
adf_test

print("We see that the p-value is much greater than 0.05, specifically 0.4231,")
print("therefore, we cannot reject the null hypothesis (the series is not stationary)")

n_diferences <- ndiffs(trading_price) # Con ndiffs vemos cuántas diferenciación hay que hacer
paste("We have to do ", n_diferences, " differences (d)")

trading_price_stationary = diff(trading_price, differences=1)
plot(trading_price_stationary)

# Otra forma de hacer la serie estacionaria

timeseriesseasonallyadjusted <- trading_price-components$seasonal
tsstationary <- diff(timeseriesseasonallyadjusted, differences=2)
plot(tsstationary)


# PASO 5

# Vamos a probar en primer lugar el modelo automático

arima_automatic <- auto.arima(trading_price, trace = TRUE)

print("Below is a summary of the model")
summary(arima_automatic)
coeftest(arima_automatic)

# Veamos la gráfica ACF y PACF para determinar los valores de p y q

# En esta gráfica vemos que hay uno o más picos, pero que el resto es 0,
# por lo que el modelo es de media móvil (q)
acf(trading_price_stationary,lag.max=40)

# Veamos donde el gráfico se hace a 0, que vemos que es entre el 0.01 (q=1) y 0.04 (q=4)
pacf(trading_price_stationary, lag.max = 40)

# El valor q que mejor resultado se obtiene es el q=2
trading_arima <- arima(trading_price, order = c(0,1,2), method = "ML")
summary(trading_arima)
coeftest(trading_arima)

print("We see that both models have MA1 and MA2 terms")
print("(which represent the impact of the errors from one and two previous periods) that are similar")
print("Additionally, they have very similar errors")
print("To compare them, we need to look at the AIC value; the lower it is, the better the model")
print("We see that the automatic ARIMA has one unit less (8542.05) than the manual ARIMA (8543.42),")
print("that is, the difference is minimal, so we can choose either one")
print("but to be fair, we will stick with the automatic one, which has a lower AIC value")

# En clase se dijo que las variables y los print tenía que ser en ingles, pero como
# esto es una explicación algo más técnica, dejo los print comentados en español para facilitar
# la corrección

#print("Vemos que ambos modelos tienen un ma1 y ma2")
#print("(que representan el impacto de los errores de uno y dos periodos anteriores) parecidos")
#print("Además tienen unos errores muy parecidos")
#print("Para compararlos, tenemos que fijarnos en el valor de AIC, cuanto más bajo, mejor es el modelo")
#print("vemos que el arima automático tiene una unidad menos (8542.05) que el arima manual (8543.42),")
#print("es decir, la diferencia es mínima, por lo que podemos elegir cualquiera")
#print("pero para ser justos, nos quedaremos con la automática, que tiene menos valor en AIC")

# PASO 6. Validación del Modelo

# checkresiduals nos permite analizar los residuos del modelo
# además se aplica el test de Ljung-Box, donde:
# - H0: Los datos son independientes (ruido blanco); no hay autocorrelación significativa
# - H1: Los datos muestran autocorrelación, no son aleatorios

residuals <- checkresiduals(arima_automatic)
print("We see that, when applying the Ljung-Box test, the p-value is much greater than 0.05")
print("therefore, we cannot reject the null hypothesis, where the data are independent (white noise)")
print("and do not have significant autocorrelation.")
print("Additionally, we see that the mean is close to zero")
print(mean(arima_automatic$residuals))
print("On the other hand, we can see that it follows a normal distribution")

# Paso 7. Forecasting

# A continuación, vamos a predecir los próximos 500 días del valor de la bolsa de Microsoft

predicts <- forecast(arima_automatic, h=500)

# En la gráfica observamos el intervalo de confianza, que es donde puede rondar los valores predecidos

autoplot(predicts)

#--------------------------
# AMPLIACIÓN

# Train/Test Split

n <- length(trading_price) # Obtenemos la longitud de la serie
training_dataset <- trading_price[1:(n-90)] # Obtenemos desde el principio hasta los últimos 90 días como datos de entrenamiento
test <- trading_price[(n-89):n] # Luego para el test obtenemos solo los últimos 90 días (3 meses)

# Creamos un modelo arima con los datos de entrenamiento
arima_training_automatic <- auto.arima(training_dataset, trace = TRUE)
predicts_training <- forecast(arima_training_automatic, h=90) # Hacemos la predicción con el modelo ARIMA
print(predicts_training)

print(test)

# Creamos un dataframe con los datos de predicción y el test
df <- data.frame(
  time  = 1:length(predicts_training$lower[, 1]), # Obtenemos los valores del intervalo de confianza de la parte baja
  Predicts = as.numeric(predicts_training$lower[, 1]),
  Test = as.numeric(test)
)

# Hacemos un pivot para que aparezca que datos son de predicción y cuales de test
# que servirá ahora para mostrarlo graficamente

df_long <- tidyr::pivot_longer(
  df,
  cols = c(Predicts, Test),
  names_to = "Serie",
  values_to = "Value"
)

# Vemos graficamente que, al principio si que lo predice medianamente bien,
# luego hay un pico en los datos de test, por lo que en ese caso hay ciertos errores
# pero luego, cuando se vuelve a estabilizar, si que lo predice bien de nuevo (dias 70-85)
ggplot(df_long, aes(x = time, y = Value, color = Serie)) +
  geom_line(linewidth = 1) +
  labs(
    x = "DPredicted days",
    y = "Stock price",
    title = "Evolution of testing and predictive actions"
  ) +
  theme_minimal()

RMSE <- sqrt(mean((predicts_training$lower[, 1] - test)^2))

paste("The RMSE error is: ", RMSE)
print("We see that it has an approximate value of 16, even though at the beginning and at the end")
print("the graph doesn't predict it too badly, although it's true that the test data has a peak")
print("this makes the error much higher")

# Comparativa con Prophet

# Nos quedaremos solo con la fecha y el precio de la acción que es lo que nos interesa

microsoft_trading_df <- microsoft_stock_df[, c("Date", "trading_price")]

# Lo convertimos a tabla
microsoft_trading_df <- microsoft_trading_df %>% 
  as_tsibble(index = Date)%>% 
  fill_gaps()

# Cambiamos el nombre de las columnas, ya que lo requiere la librería Prophet
colnames(microsoft_trading_df)[1] <- "ds"
colnames(microsoft_trading_df)[2] <- "y"

# Entrenamos el modelo Prophet
model_prophet <- prophet(microsoft_trading_df)
# Hacemos un dataframe donde se guardará los datos que se prediga
future = make_future_dataframe(model_prophet,periods = 500, freq="day")

#Hacemos la prediccion
forecast = predict(model_prophet,future)
# En este gráfico vemos que, a corto plazo es bastante mejor que nuestro modelo ARIMA
# sin embargo, cuando pasa los meses, el intervalo de confianza va aumentando,
# esto hace que a largo plazo ambos modelos sean practicamente iguales.
# Obviamente predecir valores de las acciones de una empresa es complicado, por lo que es complicado que haya un modelo perfecto

plot(model_prophet, forecast)


# Detección de anomalias

# Utilizamos la función tsoutliers para ver los outliers de la serie
outliers <- tsoutliers(trading_price)

# Lo pasamos a dataframe

out_df <- data.frame(
  time = time(trading_price)[outliers$outliers$index],
  value = trading_price[outliers$outliers$index]
)
# Cuando lo vemos en el gráfico, vemos que en este caso, no hay ningún outlier,
# en caso de que lo hubiera, saldría con puntos azules

autoplot(tsclean(trading_price), series = "clean", color = "red", lwd = 0.9) +
  autolayer(trading_price, series = "original", color = "gray", lwd = 1) +
  geom_point(
    data = out_df,
    aes(x = time, y = value),
    color = "blue",
    size = 2
  ) +
  labs(
    x = "Day",
    y = "Stock Microsoft price ($US)"
  )
