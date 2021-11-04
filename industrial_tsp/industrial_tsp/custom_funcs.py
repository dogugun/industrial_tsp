# custom_funcs.py
import sys
sys.path.insert(1, './')
from config import numeric_cols
import pandas as pd
from matplotlib.pyplot import plot as plt
from statsmodels.tsa.stattools import adfuller


def detrend(series, window_size=20):
    moving_avg = series.rolling(window=window_size).mean()
    series = series - moving_avg
    return series


def difference(series):
    series = series - series.shift()
    return series


def set_vector_date(data, vector_length):
    vector_columns = numeric_cols
    for col in vector_columns:
        col_name = col
        for i in range(vector_length):
            new_col = col_name + '_t-{}'.format(i + 1)
            data[new_col] = data[col].shift(i + 1)
    return data


def set_target_in_ph(data, prediction_horizon):
    ph_col_name = 'target_in_oh_{}'.format(prediction_horizon)
    data[ph_col_name] = data['target'].shift(-1 * prediction_horizon)
    return data

def check_stationarity(timeseries_p, col_name):
    print('Results of Dickey-Fuller Test for {}:'.format(col_name))
    dftest = adfuller(timeseries_p, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)







def crosscorr(datax, datay, lag=0, wrap=False):
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))

