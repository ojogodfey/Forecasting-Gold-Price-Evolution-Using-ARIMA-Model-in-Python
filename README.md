# Forecasting-Gold-Price-Evolution-Using-ARIMA-Model-in-Python
This Program Downloads and Forecasts Gold Price Evolution Using ARIMA Model in Python
# -*- coding: utf-8 -*-
""" Final Project: Forecasting Gold Price"""

# Import Relevant Libraries
import warnings
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa.stattools as sm
import matplotlib.pyplot as plt
# Import the Time Series data from Deskstop
df = pd.read_csv('C:\\Users\\Ojo\\Desktop\\gold_price.csv', parse_dates=True, index_col=0)
# Step1: Data Preparation
# Apply Log Transformation to the time series data
df_log = np.log(df)
#Get the first log difference
df_log_diff = df_log-df_log.shift()
# Get the number of values in the time series
series_count = np.count_nonzero(df)
print(series_count)
# 1.1 Test for stationarity
def test_stationarity(timeseries):
    """ This Test function tests for Stationarity of the Time Series Via Visualization"""
    #Determining Rollong Stattistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)
    #Plotting the Rolling Statistics
    plt.plot(timeseries, color='blue', label='Original Data')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.xlabel('Years')
    plt.ylabel('Price Index')
    plt.legend(loc='best')
    plt.title('Price Evolution, Rolling Mean and Std @ Window=12')
    plt.show(block=False)
test_stationarity(df)
# 1.2 Unit Root Test (ADT Test)
def dickey_fuller_test():
    """ This function Test for Stationarity by returning the argumented dicker fuller test result"""
    adf_results = {}
    for col in df.columns.values:
         adf_results[col] = sm.adfuller(df[col], autolag='AIC')
         print(adf_results[col])
dickey_fuller_test()
# 1.3 Log Transform Visualization
def log_transform():
    """This function returns the plot of the log transformed time series\
    and log difference"""
    plt.subplot(211)
    plt.plot(df_log, color='blue', label='Log Transform')
    plt.title('Gold Log Transformed Prices Evolution (1978-2016)')
    plt.legend(loc='best')
    plt.subplot(212)
    plt.plot(df_log_diff, color='red', label='Log Difference')
    plt.legend(loc='best')
    plt.show(block=False)
log_transform()
# Step2: Model Selection
# 2.1 Model Determination via ACF and PACF plots
def acf_pacf(ts):
    """ This function returns the Autocorrelation and Partial Autocorrelations Functions"""
    lag_acf = sm.acf(df_log_diff.dropna(), nlags=20)
    lag_pacf = sm.pacf_ols(df_log_diff.dropna(), nlags=20)
    #Plot ACF
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='blue')
    plt.axhline(y=-1.96/np.sqrt(len(df_log_diff.dropna())), linestyle='--', color='blue')
    plt.axhline(y=1.96/np.sqrt(len(df_log_diff.dropna())), linestyle='--', color='blue')
    plt.title('Autocorrelation Function')
    #Plot PACF
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=-1.96/np.sqrt(len(df_log_diff.dropna())), linestyle='--', color='blue')
    plt.axhline(y=1.96/np.sqrt(len(df_log_diff.dropna())), linestyle='--', color='blue')
    plt.title('Partial Autocorrealtion Function')
    plt.tight_layout()
#acf_pacf(df_log_diff)
def acf_pacf_plots():
    """This function returns the ACF and PACF plot of the log transformed Gold Price Evolution"""
    acf_plot = plot_acf(df_log, lags=20)
    pacf_plot = plot_pacf(df_log, lags=20)
    return acf_plot, pacf_plot
acf_pacf_plots()
# 2.2 Model Optimization of (pdq) via iterations
def optimize_arima():
    """ This function returns different pdq models and their AIC values """
    warnings.filterwarnings('ignore')
    p = 0
    q = 0
    d = 0
    pdq = []
    aic = []
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    arima_mod = ARIMA(df_log, (p,d,q)).fit(transparams=True)
                    x = arima_mod.aic
                    x1 = p,d,q
                    print(x1, x)
                    aic.append(x)
                    pdq.append(x1)
                except:
                    pass
optimize_arima()
# Step3: Models Tesing and Visualization 
def ARIMA_model():
    """ This Function returns AR, MA and the combined ARIMA Model results and Visualization"""
    # AutoRegressive Model
    model = ARIMA(df_log,order=(2,1,0))
    results_ARIMA = model.fit(disp=-1)
    plt.plot(df_log_diff.dropna(), label='Log Difference')
    plt.plot(results_ARIMA.fittedvalues, label='AR Model', color='red')
    plt.title('AutoRegressive Model')
    plt.legend(loc='best')
    plt.show()
    # Moving Average Model
    model = ARIMA(df_log,order=(0,1,2))
    results_ARIMA = model.fit(disp=-1)
    plt.plot(df_log_diff.dropna(), label='Log Difference')
    plt.plot(results_ARIMA.fittedvalues, label='MA Model', color='red')
    plt.title('Moving Average Model')
    plt.legend(loc='best')
    plt.show()
    # AutoRegressive Integrated Moving Average Model
    model = ARIMA(df_log,order=(2,1,2))
    results_ARIMA = model.fit(disp=-1)
    plt.plot(df_log_diff.dropna(), label='Log Difference')
    plt.plot(results_ARIMA.fittedvalues, label='ARIMA Model', color='red')
    plt.title('AutoRegressive Integrated Moving Average')
    plt.legend(loc='best')
    plt.show()
ARIMA_model()
# Step4:Model Validation
def validation_summary():
    """ This Function Returns the Summary of ARIMA (2,1,2) Model Validation"""
    model = ARIMA(df_log, order=(2,1,2))
    results = model.fit()
    print(results.summary())
validation_summary()
def validation_plots():
    """ This Function Returns the plots of ARIMA (2,1,2) Model Validation"""
    model = ARIMA(df_log, order=(2,1,2))
    results = model.fit()
    residuals = pd.DataFrame(results.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
validation_plots()
# Step5: Forecasting
def forecast():
    """This function returns the forecast values of the time series from 2017-2019"""
    # Forecast for the first time difference of the series
    warnings.filterwarnings('ignore')
    arima212 = ARIMA(df_log, (2,1,2)).fit()
    forecast = arima212.predict(start=39, end=42, dynamic='False')
    # Add Predicted differences to the last log transformed value of the time series
    forecast1 = 7.131539 + -0.020183
    forecast2 = forecast1 + 0.051105
    forecast3 = forecast2 + 0.081160
    #forecast of Gold Prices from 2017 - 2019
    yr_2017 = np.exp(forecast1)
    yr_2018 = np.exp(forecast2)
    yr_2019 = np.exp(forecast3)
    print('The log difference forecast for the next three years are\n', forecast)
    print('Forecast for 2017 is', yr_2017,'\nForecast for 2018 is',yr_2018, '\nForecast for 2019 is', yr_2019)
forecast()
