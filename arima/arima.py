import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sm
from matplotlib import pyplot
from numpy import array
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

n = 1
sequence_length = 10


def read_data(path='../training'):
    filepaths = [path + "/" + f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = list()
    for file in filepaths:
        csv = pd.read_csv(file, sep=",", header=0)
        csv.index = pd.DatetimeIndex(csv["start_time"]).to_period('s')
        df2 = csv.loc[:, ["mean_CPU_usage"]]
        df_list.append(df2)
    return df_list


# normalize data
mins = {}
maxs = {}


def normalize_data_minMax(df):
    pd.options.mode.chained_assignment = None
    # find min and max
    for c in df.columns:
        if mins.get(c) is None:
            min = np.min(df[c])
            max = np.max(df[c])
            mins[c] = min
            maxs[c] = max
        elif np.min(df[c]) < mins.get(c):
            mins[c] = np.min(df[c])
        elif maxs[c] > maxs.get(c):
            maxs[c] = maxs.get(c)
    for c in df.columns:
        min = mins[c]
        max = maxs[c]
        value_range = max - min
        df.loc[:, c] = (df.loc[:, c] - min) / value_range
    return df


def naive_ratio(prediction, real_value):
    prediction= array(prediction)
    real_value=array(real_value)
    # Compute the absolute difference between corresponding elements of a and b
    prediction_nr = prediction[1:]
    real_value_nr = real_value[:-1]
    abs_diff_et1 = np.abs(prediction_nr - real_value_nr)
    # Compute the sum of the absolute differences
    sum_abs_diff_et1 = np.sum(abs_diff_et1)
    et1 = (1 / len(prediction_nr)) * sum_abs_diff_et1
    abs_diff = np.abs(prediction - real_value)
    # Compute the sum of the absolute differences
    sum_abs_diff = np.sum(abs_diff)
    et = (1 / len(prediction)) * sum_abs_diff
    return et / et1


def calc_MSE_Accuracy(y_test, y_test_pred):
    print(type(y_test))
    mae = round(sm.mean_absolute_error(y_test, y_test_pred), 5)
    mse = round(sm.mean_squared_error(y_test, y_test_pred), 5)
    r2 = round(sm.r2_score(y_test, y_test_pred), 5)
    print("Mean absolute error =", mae)
    print("Mean squared error =", mse)
    print("R2 score =", r2)
    nr = naive_ratio(y_test_pred, y_test)
    print("Naive ratio =", nr)


def main():
    test_data_files = read_data("../test")
    X = test_data_files[0]
    size = int(len(X) * 0.8)
    train, test = X[0:size], X[size:len(X)]
    df_train = train.values
    df_test = test.values
    model = auto_arima(df_train, maxiter=100)
    order = model.order
    # df_train = normalize_data_minMax(train)
    # df_test = normalize_data_minMax(test)
    history = [x for x in df_train]
    history = history[-sequence_length:]
    predictions = list()
    observations = list()
    # walk-forward validation
    for t in range(int(len(df_test) / n)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        observations.append(df_test[t * n])
        history.extend(df_test[n * t:n * (t + 1)])
        history = history[-sequence_length:]

    # plot forecasts against actual outcomes
    pyplot.plot(observations, color='blue')
    pyplot.plot(predictions, color='red')
    plt.savefig('arima_new.png')
    pyplot.show()
    calc_MSE_Accuracy(observations, predictions)


if __name__ == "__main__":
    main()
